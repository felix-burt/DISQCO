// fm_algo.hpp — FM pass algorithm (homo path, no hetero/Steiner overhead).
// Included once by bindings.cpp; all functions live in an anonymous namespace
// so there are no ODR issues.
#pragma once
#include "fm_state.hpp"
#include <random>
#include <vector>
#include <cstring>

namespace {

// ── 1. Initialise per-edge state from current assignment ──────────────────
void do_map_counts(FMState& state, const FMHyperGraph& hg) {
    int32_t E = hg.E, K = hg.K, Q = hg.Q;
    for (int32_t e = 0; e < E; ++e) {
        for (int32_t k = 0; k < K; ++k) {
            state.root_count(e, k) = 0;
            state.rec_count(e, k)  = 0;
        }
        auto [rb, re] = hg.root_span(e);
        for (const int32_t* p = rb; p != re; ++p) {
            int32_t q = *p % Q, t = *p / Q;
            state.root_count(e, state.assignment[t * Q + q])++;
        }
        auto [reb, ree] = hg.rec_span(e);
        for (const int32_t* p = reb; p != ree; ++p) {
            int32_t q = *p % Q, t = *p / Q;
            state.rec_count(e, state.assignment[t * Q + q])++;
        }
        int32_t config = 0;
        for (int32_t k = 0; k < K; ++k)
            if (state.rec_count(e, k) > 0 && state.root_count(e, k) == 0)
                config |= (1 << k);
        state.edge_config[e] = config;
        state.edge_cost[e]   = __builtin_popcount(config);
    }
}

// ── 2. Analytical gain for (node → dest) using cached per-edge counts ─────
// Equivalent to find_gain_unmapped in FM_methods.py but uses CSR + bitmask.
int32_t compute_gain(const FMState& state, const FMHyperGraph& hg,
                     int32_t node, int32_t src, int32_t dest) {
    int32_t gain = 0;
    auto [eb, ee] = hg.node_edge_span(node);
    for (const int32_t* ep = eb; ep != ee; ++ep) {
        int32_t e        = *ep;
        int32_t old_cost = state.edge_cost[e];
        int32_t conf     = state.edge_config[e];
        int32_t rc_src   = state.root_count(e, src);
        int32_t rcc_src  = state.rec_count(e, src);
        int32_t rc_dst   = state.root_count(e, dest);

        int32_t new_conf = conf;
        if (hg.is_root(node, e)) {
            // rc[src]-=1: bit_src = rcc_src>0 && rc_src==1
            if (rcc_src > 0 && rc_src == 1)
                new_conf |= (1 << src);
            else
                new_conf &= ~(1 << src);
            // rc[dest]+=1: always clears bit
            new_conf &= ~(1 << dest);
        } else {
            // rcc[src]-=1: bit_src = rcc_src>1 && rc_src==0
            if (rcc_src > 1 && rc_src == 0)
                new_conf |= (1 << src);
            else
                new_conf &= ~(1 << src);
            // rcc[dest]+=1: bit_dest = rc_dest==0
            if (rc_dst == 0)
                new_conf |= (1 << dest);
            else
                new_conf &= ~(1 << dest);
        }
        gain += __builtin_popcount(new_conf) - old_cost;
    }
    return gain;
}

// ── 3. Compute all gains and fill bucket structure ────────────────────────
void find_all_gains_and_fill_buckets(FMState& state, const FMHyperGraph& hg) {
    int32_t K = hg.K;
    // Proper clear: reset pos for every item in each bucket, then clear items
    for (auto& b : state.buckets.buckets) {
        for (int32_t a : b.items) b.pos[a] = -1;
        b.items.clear();
    }
    state.buckets.best_idx = -1;

    for (int32_t n = 0; n < hg.N; ++n) {
        if (state.locked[n]) continue;
        int32_t src = state.get_assignment(n);
        for (int32_t dest = 0; dest < K; ++dest) {
            if (dest == src) continue;
            int32_t act = hg.action_id(n, dest);
            int32_t g   = compute_gain(state, hg, n, src, dest);
            state.gains[act] = g;
            state.buckets.insert(act, g);
        }
    }
}

// ── 4. Find best valid action; scan buckets from most-negative gain up ────
// Returns (action_id, gain) or (-1, 0). Marks the chosen node locked.
std::pair<int32_t, int32_t> find_action(FMState& state, const FMHyperGraph& hg,
                                         std::mt19937& rng) {
    int32_t K     = hg.K;
    int32_t nbkts = state.buckets.nbuckets();

    for (int32_t idx = 0; idx < nbkts; ++idx) {
        IndexedSet& bkt = state.buckets.buckets[idx];
        if (bkt.empty()) continue;

        int32_t gain = state.buckets.idx_to_gain(idx);
        int32_t n    = bkt.size();
        int32_t start = (n > 1) ? static_cast<int32_t>(rng() % static_cast<uint32_t>(n)) : 0;

        for (int32_t i = 0; i < n; ++i) {
            int32_t action = bkt.items[(start + i) % n];
            int32_t node   = action / K;
            int32_t dest   = action % K;
            int32_t t      = node  / hg.Q;

            if (!state.locked[node] && state.space(t, dest) > 0) {
                state.locked[node] = true;
                return {action, gain};
            }
        }
    }
    return {-1, 0};
}

// ── 5. Apply move and update affected neighbour gains incrementally ────────
//
// delta_arr[N*K] is an external work buffer (all-zero on entry, zero-reset
// on exit).  dirty_actions accumulates which actions were touched so we can
// apply them all at the end and reset delta_arr without a full clear.
void apply_move_and_update(FMState& state, const FMHyperGraph& hg,
                            int32_t action,
                            std::vector<int32_t>& delta_arr,
                            std::vector<int32_t>& dirty_actions) {
    int32_t K    = hg.K;
    int32_t node = action / K;
    int32_t dest = action % K;
    int32_t src  = state.get_assignment(node);
    int32_t t    = node / hg.Q;

    state.set_assignment(node, dest);
    state.dec_space(t, dest);
    state.inc_space(t, src);

    auto [eb, ee] = hg.node_edge_span(node);
    for (const int32_t* ep = eb; ep != ee; ++ep) {
        int32_t e = *ep;

        // Pre-move values (live in state before we update at end of edge block)
        const int32_t* base_rc  = &state.edge_root_counts[e * K];
        const int32_t* base_rcc = &state.edge_rec_counts[e * K];

        int32_t pre_rc_src  = base_rc[src];
        int32_t pre_rc_dst  = base_rc[dest];
        int32_t pre_rcc_src = base_rcc[src];
        int32_t pre_rcc_dst = base_rcc[dest];
        int32_t pre_conf    = state.edge_config[e];
        int32_t pre_cost    = state.edge_cost[e];

        bool is_rpin = !hg.is_root(node, e);  // node is a receiver pin

        // Post-move counts (current node's move only)
        int32_t post_rc_src  = is_rpin ? pre_rc_src  : pre_rc_src  - 1;
        int32_t post_rc_dst  = is_rpin ? pre_rc_dst  : pre_rc_dst  + 1;
        int32_t post_rcc_src = is_rpin ? pre_rcc_src - 1 : pre_rcc_src;
        int32_t post_rcc_dst = is_rpin ? pre_rcc_dst + 1 : pre_rcc_dst;

        // Post-move config (only src and dest bits change)
        int32_t post_conf = pre_conf;
        if (post_rcc_src > 0 && post_rc_src == 0) post_conf |= (1 << src);
        else                                        post_conf &= ~(1 << src);
        if (post_rcc_dst > 0 && post_rc_dst == 0) post_conf |= (1 << dest);
        else                                        post_conf &= ~(1 << dest);
        int32_t post_cost = __builtin_popcount(post_conf);

        // Helpers: post counts at partition k
        auto post_rc = [&](int32_t k) -> int32_t {
            return (k == src) ? post_rc_src : (k == dest) ? post_rc_dst : base_rc[k];
        };
        auto post_rcc = [&](int32_t k) -> int32_t {
            return (k == src) ? post_rcc_src : (k == dest) ? post_rcc_dst : base_rcc[k];
        };

        // ── delta gains for root-pin neighbours ─────────────────────────
        {
            auto [rb, re] = hg.root_span(e);
            for (const int32_t* rp = rb; rp != re; ++rp) {
                int32_t nbr = *rp;
                if (state.locked[nbr]) continue;
                int32_t ns = state.get_assignment(nbr);

                for (int32_t nd = 0; nd < K; ++nd) {
                    if (nd == ns) continue;
                    int32_t na = hg.action_id(nbr, nd);

                    // cost_b: pre state + nbr's root move (rc[ns]-=1, rc[nd]+=1)
                    int32_t conf_b = pre_conf;
                    if (base_rcc[ns] > 0 && base_rc[ns] == 1)
                        conf_b |= (1 << ns); else conf_b &= ~(1 << ns);
                    conf_b &= ~(1 << nd);                   // rc[nd]+=1 always clears
                    int32_t cost_b = __builtin_popcount(conf_b);

                    // cost_ab: post state + nbr's root move
                    int32_t conf_ab = post_conf;
                    if (post_rcc(ns) > 0 && post_rc(ns) == 1)
                        conf_ab |= (1 << ns); else conf_ab &= ~(1 << ns);
                    conf_ab &= ~(1 << nd);
                    int32_t cost_ab = __builtin_popcount(conf_ab);

                    int32_t delta = post_cost - pre_cost - cost_ab + cost_b;
                    if (delta != 0) {
                        if (delta_arr[na] == 0) dirty_actions.push_back(na);
                        delta_arr[na] += delta;
                    }
                }
            }
        }

        // ── delta gains for receiver-pin neighbours ──────────────────────
        {
            auto [reb, ree] = hg.rec_span(e);
            for (const int32_t* rp = reb; rp != ree; ++rp) {
                int32_t nbr = *rp;
                if (state.locked[nbr]) continue;
                int32_t ns = state.get_assignment(nbr);

                for (int32_t nd = 0; nd < K; ++nd) {
                    if (nd == ns) continue;
                    int32_t na = hg.action_id(nbr, nd);

                    // cost_b: pre state + nbr's rec move (rcc[ns]-=1, rcc[nd]+=1)
                    int32_t conf_b = pre_conf;
                    if (base_rcc[ns] > 1 && base_rc[ns] == 0)
                        conf_b |= (1 << ns); else conf_b &= ~(1 << ns);
                    if (base_rc[nd] == 0)
                        conf_b |= (1 << nd); else conf_b &= ~(1 << nd);
                    int32_t cost_b = __builtin_popcount(conf_b);

                    // cost_ab: post state + nbr's rec move
                    int32_t conf_ab = post_conf;
                    if (post_rcc(ns) > 1 && post_rc(ns) == 0)
                        conf_ab |= (1 << ns); else conf_ab &= ~(1 << ns);
                    if (post_rc(nd) == 0)
                        conf_ab |= (1 << nd); else conf_ab &= ~(1 << nd);
                    int32_t cost_ab = __builtin_popcount(conf_ab);

                    int32_t delta = post_cost - pre_cost - cost_ab + cost_b;
                    if (delta != 0) {
                        if (delta_arr[na] == 0) dirty_actions.push_back(na);
                        delta_arr[na] += delta;
                    }
                }
            }
        }

        // Update edge counts and config in state
        state.edge_root_counts[e * K + src]  = post_rc_src;
        state.edge_root_counts[e * K + dest] = post_rc_dst;
        state.edge_rec_counts[e * K + src]   = post_rcc_src;
        state.edge_rec_counts[e * K + dest]  = post_rcc_dst;
        state.edge_config[e] = post_conf;
        state.edge_cost[e]   = post_cost;
    }

    // Apply accumulated delta gains to gain table and buckets
    for (int32_t na : dirty_actions) {
        int32_t delta = delta_arr[na];
        delta_arr[na] = 0;  // reset for next move
        if (delta == 0) continue;

        int32_t old_g = state.gains[na];
        int32_t new_g = old_g - delta;
        state.gains[na] = new_g;
        if (state.buckets.action_in_bucket(na, old_g))
            state.buckets.move(na, old_g, new_g);
    }
    dirty_actions.clear();
}

// ── 6. FM pass: returns (best_assignment, last_assignment, best_gain, last_gain)
struct FMPassResult {
    std::vector<int32_t> best_asgn;  // flat [D*Q], assignment at best prefix
    std::vector<int32_t> last_asgn;  // flat [D*Q], assignment after all moves
    int32_t best_gain;
    int32_t last_gain;
};

FMPassResult run_fm_pass(const FMHyperGraph& hg,
                          const int32_t* initial_asgn,  // [D*Q] input (not modified)
                          const int32_t* qpu_sizes,
                          int32_t max_gain,
                          int32_t limit,
                          uint32_t seed) {
    int32_t N = hg.N;

    // Working copy of assignment (modified in-place during pass)
    std::vector<int32_t> work(initial_asgn, initial_asgn + N);

    FMState state(hg, work.data(), max_gain, qpu_sizes);
    do_map_counts(state, hg);
    find_all_gains_and_fill_buckets(state, hg);

    std::mt19937 rng(seed);
    std::vector<int32_t> moves;      // action_id per move
    std::vector<int32_t> cum_gains;  // cumulative gain after each move
    cum_gains.push_back(0);

    // Work buffers for incremental update (avoid per-call allocation)
    std::vector<int32_t> delta_arr(hg.num_actions(), 0);
    std::vector<int32_t> dirty_actions;

    int32_t cumulative = 0;

    for (int32_t iter = 0; iter < limit; ++iter) {
        auto [action, gain] = find_action(state, hg, rng);
        if (action < 0) break;

        apply_move_and_update(state, hg, action, delta_arr, dirty_actions);

        cumulative += gain;
        moves.push_back(action);
        cum_gains.push_back(cumulative);
    }

    // Find best prefix index (minimum cumulative gain)
    int32_t best_idx  = 0;
    int32_t best_gain = 0;
    for (int32_t i = 1; i < (int32_t)cum_gains.size(); ++i) {
        if (cum_gains[i] < best_gain) {
            best_gain = cum_gains[i];
            best_idx  = i;
        }
    }

    // last assignment = work (which was updated in-place throughout)
    std::vector<int32_t> last_asgn(work.begin(), work.end());

    // best assignment = replay moves[0..best_idx) from initial
    std::vector<int32_t> best_asgn(initial_asgn, initial_asgn + N);
    for (int32_t i = 0; i < best_idx; ++i) {
        int32_t act  = moves[i];
        int32_t node = act / hg.K;
        int32_t d    = act % hg.K;
        int32_t q    = node % hg.Q;
        int32_t t_   = node / hg.Q;
        best_asgn[t_ * hg.Q + q] = d;
    }

    return FMPassResult{best_asgn, last_asgn, best_gain, cum_gains.back()};
}

} // anonymous namespace
