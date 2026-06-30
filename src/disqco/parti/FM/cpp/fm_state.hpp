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
                     const int32_t* qpu_sizes)
        : hg(hg_), assignment(assignment_), max_gain(max_gain_)
    {
        int32_t E = hg.E, K = hg.K, N = hg.N, D = hg.D;

        edge_cost.assign(E, 0);
        edge_config.assign(E, 0);
        edge_root_counts.assign(E * K, 0);
        edge_rec_counts.assign(E * K, 0);
        gains.assign(N * K, 0);
        locked.assign(N, false);
        spaces.assign(D * K, 0);

        // Initialise spaces from qpu_sizes then subtract assigned nodes
        for (int32_t t = 0; t < D; ++t)
            for (int32_t k = 0; k < K; ++k)
                spaces[t * K + k] = qpu_sizes[k];
        for (int32_t t = 0; t < D; ++t)
            for (int32_t q = 0; q < hg.Q; ++q) {
                int32_t part = assignment[t * hg.Q + q];
                spaces[t * K + part]--;
            }

        buckets.init(max_gain, hg.num_actions());
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
