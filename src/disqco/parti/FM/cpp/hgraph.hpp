#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

// Read-only hypergraph used by the FM pass.
// Built once from the Python QuantumCircuitHyperGraph and cached.
//
// Node encoding:  node_id = t * Q + q   (matches numpy assignment[t][q])
// Action encoding: action_id = node_id * K + dest
//
// Adjacency stored in CSR (Compressed Sparse Row) format for cache-friendly
// sequential reads in the inner loop of find_gain / take_action_and_update.
struct FMHyperGraph {
    int32_t Q;   // num_qubits
    int32_t D;   // depth
    int32_t E;   // num_edges
    int32_t K;   // num_partitions
    int32_t N;   // Q * D  (total nodes)

    // --- root pins per edge (CSR) ---
    // root_pins[root_offsets[e] .. root_offsets[e+1]) = node IDs in root set of edge e
    std::vector<int32_t> root_pins;
    std::vector<int32_t> root_offsets;  // length E+1

    // --- receiver pins per edge (CSR) ---
    std::vector<int32_t> rec_pins;
    std::vector<int32_t> rec_offsets;   // length E+1

    // --- incident edges per node (CSR) ---
    // node_edges[node_offsets[n] .. node_offsets[n+1]) = edge IDs incident to node n
    std::vector<int32_t> node_edges;
    std::vector<int32_t> node_offsets;  // length N+1

    // --- per-pin role flag ---
    // root_flag[root_offsets[e] + i] = 1 (always, for symmetry with rec_flag)
    // rec_flag[rec_offsets[e]  + i] = 1 (always)
    // Used in take_action_and_update to know which count array to decrement.
    // Stored as parallel bool arrays into the same CSR structures.
    // (Alternatively encoded implicitly: all entries in root_pins are roots.)
    // We keep separate arrays so the inner loop can use a single unified pin list
    // when needed.

    // Helpers
    int32_t node_id(int32_t q, int32_t t) const { return t * Q + q; }
    int32_t action_id(int32_t node, int32_t dest) const { return node * K + dest; }
    int32_t num_actions() const { return N * K; }

    // Span of root pins for edge e
    std::pair<const int32_t*, const int32_t*> root_span(int32_t e) const {
        return { root_pins.data() + root_offsets[e],
                 root_pins.data() + root_offsets[e + 1] };
    }
    // Span of receiver pins for edge e
    std::pair<const int32_t*, const int32_t*> rec_span(int32_t e) const {
        return { rec_pins.data() + rec_offsets[e],
                 rec_pins.data() + rec_offsets[e + 1] };
    }
    // Span of incident edges for node n
    std::pair<const int32_t*, const int32_t*> node_edge_span(int32_t n) const {
        return { node_edges.data() + node_offsets[n],
                 node_edges.data() + node_offsets[n + 1] };
    }

    int32_t root_degree(int32_t e) const { return root_offsets[e+1] - root_offsets[e]; }
    int32_t rec_degree(int32_t e)  const { return rec_offsets[e+1]  - rec_offsets[e];  }
    int32_t node_degree(int32_t n) const { return node_offsets[n+1] - node_offsets[n]; }

    // Returns true if node n is a root pin of edge e.
    // Used in take_action_and_update to choose which count array to update.
    // (Linear scan over root pins — degree is small in practice.)
    bool is_root(int32_t n, int32_t e) const {
        auto [begin, end] = root_span(e);
        for (const int32_t* p = begin; p != end; ++p)
            if (*p == n) return true;
        return false;
    }
};
