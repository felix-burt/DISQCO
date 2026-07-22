#pragma once
#include "hgraph.hpp"
#include "buckets.hpp"
#include <vector>
#include <cstdint>
#include <cstring>

// Mutable state for one FM pass.
// Reset at the start of each pass; the gain table and bucket structure are
// rebuilt from the assignment and hypergraph each time.
struct FMState {
    const FMHyperGraph& hg;

    // Assignment: flat int32 array of length D*Q.
    // Pointer into the numpy buffer — no copy, written back in-place.
    int32_t* assignment;

    // Per-edge mutable attributes — recomputed at pass start via
    // map_counts_and_configs, then updated incrementally.
    std::vector<int32_t> edge_cost;          // [E]
    std::vector<int32_t> edge_config;        // [E]  K-bit bitmask
    std::vector<int32_t> edge_root_counts;   // [E * K]  flat: [e*K + k]
    std::vector<int32_t> edge_rec_counts;    // [E * K]

    // Gain table: gains[node_id * K + dest] for every (node, destination) pair.
    std::vector<int32_t> gains;              // [N * K]

    // Lock flags: true once a node has been moved this pass.
    std::vector<bool> locked;                // [N]

    // Free slots per (timestep, partition): spaces[t * K + k].
    std::vector<int32_t> spaces;             // [D * K]

    // Bucket structure over gain range [-max_gain, +max_gain].
    BucketArray buckets;
    int32_t max_gain;

    explicit FMState(const FMHyperGraph& hg_, int32_t* assignment_, int32_t max_gain_,
                     const int32_t* qpu_sizes,
                     const int32_t* active_nodes = nullptr, int32_t n_active = 0)
        : hg(hg_), assignment(assignment_), max_gain(max_gain_)
    {
        int32_t E = hg.E, K = hg.K, N = hg.N, D = hg.D;

        edge_cost.assign(E, 0);
        edge_config.assign(E, 0);
        edge_root_counts.assign(E * K, 0);
        edge_rec_counts.assign(E * K, 0);
        gains.assign(N * K, 0);
        spaces.assign(D * K, 0);

        // Initialise spaces from qpu_sizes.
        for (int32_t t = 0; t < D; ++t)
            for (int32_t k = 0; k < K; ++k)
                spaces[t * K + k] = qpu_sizes[k];

        if (active_nodes != nullptr && n_active > 0) {
            // Subtract only active nodes and lock everything else.
            locked.assign(N, true);
            for (int32_t i = 0; i < n_active; ++i) {
                int32_t n = active_nodes[i];
                locked[n] = false;
                int32_t t = n / hg.Q;
                spaces[t * K + assignment[n]]--;
            }
        } else {
            // All nodes active: subtract all and start unlocked.
            locked.assign(N, false);
            for (int32_t t = 0; t < D; ++t)
                for (int32_t q = 0; q < hg.Q; ++q)
                    spaces[t * K + assignment[t * hg.Q + q]]--;
        }

        buckets.init(max_gain, hg.num_actions());
    }

    // --- between-pass reset ---
    // Reset only locked + spaces from a new assignment vector.
    // Gains and BucketArray are NOT touched: gains are overwritten by
    // find_all_gains_and_fill_buckets, and BucketArray is cleared there too.
    void reset_for_next_pass(const int32_t* new_asgn,
                              const int32_t* qpu_sizes,
                              const int32_t* active_nodes = nullptr,
                              int32_t n_active = 0) {
        int32_t K = hg.K, Q = hg.Q;
        // Copy new assignment into our working buffer.
        std::copy(new_asgn, new_asgn + hg.N, assignment);

        if (active_nodes != nullptr && n_active > 0) {
            // Unlock active nodes, reset their timestep spaces.
            // First re-fill spaces for each active timestep, then subtract.
            // Use a small scratch set to avoid redundant fills.
            std::vector<bool> t_seen(hg.D, false);
            for (int32_t i = 0; i < n_active; ++i) {
                int32_t t = active_nodes[i] / Q;
                if (!t_seen[t]) {
                    t_seen[t] = true;
                    for (int32_t k = 0; k < K; ++k)
                        spaces[t * K + k] = qpu_sizes[k];
                }
                locked[active_nodes[i]] = false;
            }
            for (int32_t i = 0; i < n_active; ++i) {
                int32_t n = active_nodes[i];
                spaces[(n / Q) * K + assignment[n]]--;
            }
        } else {
            std::fill(locked.begin(), locked.end(), false);
            for (int32_t t = 0; t < hg.D; ++t)
                for (int32_t k = 0; k < K; ++k)
                    spaces[t * K + k] = qpu_sizes[k];
            for (int32_t t = 0; t < hg.D; ++t)
                for (int32_t q = 0; q < Q; ++q)
                    spaces[t * K + assignment[t * Q + q]]--;
        }
    }

    // --- accessors ---

    int32_t get_assignment(int32_t node) const {
        int32_t q = node % hg.Q, t = node / hg.Q;
        return assignment[t * hg.Q + q];
    }
    void set_assignment(int32_t node, int32_t part) {
        int32_t q = node % hg.Q, t = node / hg.Q;
        assignment[t * hg.Q + q] = part;
    }

    int32_t& root_count(int32_t e, int32_t k) { return edge_root_counts[e * hg.K + k]; }
    int32_t& rec_count (int32_t e, int32_t k) { return edge_rec_counts [e * hg.K + k]; }
    int32_t  root_count(int32_t e, int32_t k) const { return edge_root_counts[e * hg.K + k]; }
    int32_t  rec_count (int32_t e, int32_t k) const { return edge_rec_counts [e * hg.K + k]; }

    int32_t space(int32_t t, int32_t k) const { return spaces[t * hg.K + k]; }
    void dec_space(int32_t t, int32_t k) { spaces[t * hg.K + k]--; }
    void inc_space(int32_t t, int32_t k) { spaces[t * hg.K + k]++; }

    // Recompute config bitmask and cost for edge e from current counts.
    // config bit k is set iff rec_count[k] > 0 && root_count[k] == 0.
    void recompute_config_and_cost(int32_t e) {
        int32_t config = 0;
        for (int32_t k = 0; k < hg.K; ++k)
            if (rec_count(e, k) > 0 && root_count(e, k) == 0)
                config |= (1 << k);
        edge_config[e] = config;
        edge_cost[e]   = __builtin_popcount(config);
    }
};
