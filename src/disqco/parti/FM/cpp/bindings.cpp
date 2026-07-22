#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <cstring>
#include <numeric>
#include <unordered_set>

#include "hgraph.hpp"
#include "indexed_set.hpp"
#include "buckets.hpp"
#include "fm_state.hpp"
#include "fm_algo.hpp"

namespace py = pybind11;
using arr_i32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

// ---------------------------------------------------------------------------
// Build FMHyperGraph from numpy arrays prepared by the Python conversion helper.
// All arrays are int32.  Shapes:
//   root_pins     [total_root_pins]
//   root_offsets  [E+1]
//   rec_pins      [total_rec_pins]
//   rec_offsets   [E+1]
//   node_edges    [total_node_edge_entries]
//   node_offsets  [N+1]
// ---------------------------------------------------------------------------
FMHyperGraph build_hgraph(
    int32_t Q, int32_t D, int32_t K,
    arr_i32 root_pins_arr,   arr_i32 root_offsets_arr,
    arr_i32 rec_pins_arr,    arr_i32 rec_offsets_arr,
    arr_i32 node_edges_arr,  arr_i32 node_offsets_arr)
{
    FMHyperGraph hg;
    hg.Q = Q;
    hg.D = D;
    hg.K = K;
    hg.N = Q * D;
    hg.E = static_cast<int32_t>(root_offsets_arr.size()) - 1;

    auto copy_vec = [](arr_i32& a) {
        auto buf = a.request();
        const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
        return std::vector<int32_t>(ptr, ptr + buf.size);
    };

    hg.root_pins    = copy_vec(root_pins_arr);
    hg.root_offsets = copy_vec(root_offsets_arr);
    hg.rec_pins     = copy_vec(rec_pins_arr);
    hg.rec_offsets  = copy_vec(rec_offsets_arr);
    hg.node_edges   = copy_vec(node_edges_arr);
    hg.node_offsets = copy_vec(node_offsets_arr);

    return hg;
}

// ---------------------------------------------------------------------------
// map_counts_and_configs: populate per-edge counts and configs from assignment.
// Exposed so the Python side can call it for validation against the Python path.
// ---------------------------------------------------------------------------
void map_counts_and_configs(
    FMHyperGraph& hg,
    arr_i32 assignment_arr,
    py::array_t<int32_t>& out_root_counts,
    py::array_t<int32_t>& out_rec_counts,
    py::array_t<int32_t>& out_configs,
    py::array_t<int32_t>& out_costs)
{
    auto asgn = assignment_arr.unchecked<2>();   // shape [D, Q]
    auto rc   = out_root_counts.mutable_unchecked<2>(); // shape [E, K]
    auto rcc  = out_rec_counts.mutable_unchecked<2>();
    auto cfg  = out_configs.mutable_unchecked<1>();
    auto cst  = out_costs.mutable_unchecked<1>();

    int32_t E = hg.E, K = hg.K;

    for (int32_t e = 0; e < E; ++e) {
        // zero counts
        for (int32_t k = 0; k < K; ++k) { rc(e,k) = 0; rcc(e,k) = 0; }

        // root pins
        auto [rb, re] = hg.root_span(e);
        for (const int32_t* p = rb; p != re; ++p) {
            int32_t q = *p % hg.Q, t = *p / hg.Q;
            rc(e, asgn(t, q))++;
        }
        // receiver pins
        auto [reb, ree] = hg.rec_span(e);
        for (const int32_t* p = reb; p != ree; ++p) {
            int32_t q = *p % hg.Q, t = *p / hg.Q;
            rcc(e, asgn(t, q))++;
        }

        // config bitmask: bit k set iff rec_count[k]>0 && root_count[k]==0
        int32_t config = 0;
        for (int32_t k = 0; k < K; ++k)
            if (rcc(e,k) > 0 && rc(e,k) == 0)
                config |= (1 << k);
        cfg(e) = config;
        cst(e) = __builtin_popcount(config);
    }
}

// ---------------------------------------------------------------------------
// calculate_full_cost: sum of edge costs for the current assignment.
// ---------------------------------------------------------------------------
int32_t calculate_full_cost(const FMHyperGraph& hg, arr_i32 assignment_arr) {
    auto asgn = assignment_arr.unchecked<2>(); // [D, Q]
    int32_t total = 0;
    for (int32_t e = 0; e < hg.E; ++e) {
        int32_t config = 0;
        // root counts
        std::vector<int32_t> rc(hg.K, 0), rcc(hg.K, 0);
        auto [rb, re] = hg.root_span(e);
        for (const int32_t* p = rb; p != re; ++p) {
            int32_t q = *p % hg.Q, t = *p / hg.Q;
            rc[asgn(t, q)]++;
        }
        auto [reb, ree] = hg.rec_span(e);
        for (const int32_t* p = reb; p != ree; ++p) {
            int32_t q = *p % hg.Q, t = *p / hg.Q;
            rcc[asgn(t, q)]++;
        }
        for (int32_t k = 0; k < hg.K; ++k)
            if (rcc[k] > 0 && rc[k] == 0) config |= (1 << k);
        total += __builtin_popcount(config);
    }
    return total;
}

// ---------------------------------------------------------------------------
// fm_pass: run one full FM pass and return best/last assignments + gains.
// assignment_arr is read-only; results are returned as fresh numpy arrays.
// ---------------------------------------------------------------------------
py::dict fm_pass(
    FMHyperGraph& hg,
    arr_i32 assignment_arr,
    arr_i32 qpu_sizes_arr,
    int32_t max_gain,
    int32_t limit,
    py::object active_nodes_obj = py::none())
{
    // Read-only input buffers
    auto asgn_buf = assignment_arr.request();
    const int32_t* asgn_ptr = static_cast<const int32_t*>(asgn_buf.ptr);
    int32_t N = hg.N, D = hg.D, Q = hg.Q;

    auto qsz_buf = qpu_sizes_arr.request();
    const int32_t* qsz_ptr = static_cast<const int32_t*>(qsz_buf.ptr);

    // Optional active-node mask: if provided, only those nodes participate.
    arr_i32 active_nodes_arr;
    const int32_t* active_nodes_ptr = nullptr;
    int32_t n_active = 0;
    if (!active_nodes_obj.is_none()) {
        active_nodes_arr = active_nodes_obj.cast<arr_i32>();
        auto buf = active_nodes_arr.request();
        active_nodes_ptr = static_cast<const int32_t*>(buf.ptr);
        n_active = static_cast<int32_t>(buf.size);
    }

    // Use a thread-local or time-based seed for reproducible stochasticity
    uint32_t seed = static_cast<uint32_t>(std::random_device{}());

    FMPassResult res = run_fm_pass(hg, asgn_ptr, qsz_ptr, max_gain, limit, seed,
                                   active_nodes_ptr, n_active);

    // Build numpy arrays for best and last assignments (shape [D, Q])
    auto make_arr = [&](const std::vector<int32_t>& v) {
        auto arr = py::array_t<int32_t>({D, Q});
        std::memcpy(arr.mutable_data(), v.data(), N * sizeof(int32_t));
        return arr;
    };

    // Return a 3-element list interface compatible with run_FM's assignment_list
    // logic: [initial, best, last] with gains [0, best_gain, last_gain].
    auto initial_arr = py::array_t<int32_t>({D, Q});
    std::memcpy(initial_arr.mutable_data(), asgn_ptr, N * sizeof(int32_t));

    py::list asgn_list;
    asgn_list.append(initial_arr);
    asgn_list.append(make_arr(res.best_asgn));
    asgn_list.append(make_arr(res.last_asgn));

    py::list gain_list;
    gain_list.append(0);
    gain_list.append(res.best_gain);
    gain_list.append(res.last_gain);

    py::dict result;
    result["assignment_list"] = asgn_list;
    result["gain_list"]       = gain_list;
    return result;
}

// ---------------------------------------------------------------------------
// coarsen_one_level: merge a batch of timestep pairs in the CSR hypergraph.
//
// pairs_src / pairs_dst: parallel int32 arrays of length n_pairs.
// Each pair (pairs_src[i], pairs_dst[i]) merges all pins at timestep
// pairs_src[i] into timestep pairs_dst[i].
//
// Returns (new_FMHyperGraph, active_node_ids) where active_node_ids is a
// flat int32 numpy array of node IDs whose timestep was NOT a source
// (i.e. nodes that are actually present in the coarsened level).
// ---------------------------------------------------------------------------
py::tuple coarsen_one_level(
    const FMHyperGraph& hg,
    arr_i32 pairs_src_arr,
    arr_i32 pairs_dst_arr)
{
    int32_t Q = hg.Q, D = hg.D, K = hg.K, E = hg.E;

    auto src_buf = pairs_src_arr.request();
    auto dst_buf = pairs_dst_arr.request();
    int32_t n_pairs = static_cast<int32_t>(src_buf.size);
    const int32_t* src_ptr = static_cast<const int32_t*>(src_buf.ptr);
    const int32_t* dst_ptr = static_cast<const int32_t*>(dst_buf.ptr);

    // t_remap[t]: timestep t maps to this representative after merging.
    std::vector<int32_t> t_remap(D);
    std::iota(t_remap.begin(), t_remap.end(), 0);
    std::vector<bool> is_src(D, false);
    for (int32_t i = 0; i < n_pairs; ++i) {
        t_remap[src_ptr[i]] = dst_ptr[i];
        is_src[src_ptr[i]] = true;
    }

    // Active node IDs: nodes whose timestep is NOT a source.
    std::vector<int32_t> active_ids;
    active_ids.reserve((D - n_pairs) * Q);
    for (int32_t t = 0; t < D; ++t) {
        if (!is_src[t]) {
            for (int32_t q = 0; q < Q; ++q)
                active_ids.push_back(t * Q + q);
        }
    }

    // Build remapped pin sets per edge.  Keep the same E edge slots so that
    // edge indices are stable; trivial edges get empty spans.
    std::vector<int32_t> new_root_pins, new_root_offsets;
    std::vector<int32_t> new_rec_pins,  new_rec_offsets;
    new_root_offsets.reserve(E + 1);
    new_rec_offsets.reserve(E + 1);
    new_root_offsets.push_back(0);
    new_rec_offsets.push_back(0);

    // node_edge_lists[n] = list of edge indices incident to node n.
    std::vector<std::vector<int32_t>> node_edge_lists(hg.N);

    std::unordered_set<int32_t> root_set, rec_set;
    for (int32_t e = 0; e < E; ++e) {
        root_set.clear();
        rec_set.clear();

        {
            auto [rb, re] = hg.root_span(e);
            for (const int32_t* p = rb; p != re; ++p)
                root_set.insert(t_remap[*p / Q] * Q + (*p % Q));
        }
        {
            auto [reb, ree] = hg.rec_span(e);
            for (const int32_t* p = reb; p != ree; ++p)
                rec_set.insert(t_remap[*p / Q] * Q + (*p % Q));
        }

        // Trivial edge: root and receiver sets are identical after remapping.
        if (root_set == rec_set) {
            new_root_offsets.push_back(static_cast<int32_t>(new_root_pins.size()));
            new_rec_offsets.push_back(static_cast<int32_t>(new_rec_pins.size()));
            continue;
        }

        // Add root pins; register edge in node_edge_lists.
        for (int32_t n : root_set) {
            new_root_pins.push_back(n);
            node_edge_lists[n].push_back(e);
        }
        // Add rec pins; add to node_edge_lists only if not already via root.
        for (int32_t n : rec_set) {
            new_rec_pins.push_back(n);
            if (root_set.find(n) == root_set.end())
                node_edge_lists[n].push_back(e);
        }

        new_root_offsets.push_back(static_cast<int32_t>(new_root_pins.size()));
        new_rec_offsets.push_back(static_cast<int32_t>(new_rec_pins.size()));
    }

    // Build node_edges CSR.
    std::vector<int32_t> node_edges, node_offsets;
    node_offsets.reserve(hg.N + 1);
    node_offsets.push_back(0);
    for (int32_t n = 0; n < hg.N; ++n) {
        for (int32_t ei : node_edge_lists[n])
            node_edges.push_back(ei);
        node_offsets.push_back(static_cast<int32_t>(node_edges.size()));
    }

    // Assemble new FMHyperGraph.
    FMHyperGraph new_hg;
    new_hg.Q = Q; new_hg.D = D; new_hg.K = K;
    new_hg.N = hg.N; new_hg.E = E;
    new_hg.root_pins    = std::move(new_root_pins);
    new_hg.root_offsets = std::move(new_root_offsets);
    new_hg.rec_pins     = std::move(new_rec_pins);
    new_hg.rec_offsets  = std::move(new_rec_offsets);
    new_hg.node_edges   = std::move(node_edges);
    new_hg.node_offsets = std::move(node_offsets);

    // Return active_ids as a numpy array.
    auto active_arr = py::array_t<int32_t>(static_cast<py::ssize_t>(active_ids.size()));
    if (!active_ids.empty())
        std::memcpy(active_arr.mutable_data(), active_ids.data(),
                    active_ids.size() * sizeof(int32_t));

    return py::make_tuple(std::move(new_hg), active_arr);
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_fm_cpp, m) {
    m.doc() = "C++ data structures and helpers for FM hypergraph partitioning";

    py::class_<FMHyperGraph>(m, "FMHyperGraph")
        .def_readonly("Q", &FMHyperGraph::Q, "Number of qubits")
        .def_readonly("D", &FMHyperGraph::D, "Depth (timesteps)")
        .def_readonly("E", &FMHyperGraph::E, "Number of hyperedges")
        .def_readonly("K", &FMHyperGraph::K, "Number of partitions")
        .def_readonly("N", &FMHyperGraph::N, "Total nodes (Q * D)")
        .def("node_id",  &FMHyperGraph::node_id,  "Encode (q, t) → node_id")
        .def("action_id",&FMHyperGraph::action_id, "Encode (node_id, dest) → action_id")
        .def("num_actions", &FMHyperGraph::num_actions)
        .def("root_degree", &FMHyperGraph::root_degree, "Root-pin count for edge e")
        .def("rec_degree",  &FMHyperGraph::rec_degree,  "Receiver-pin count for edge e")
        .def("node_degree", &FMHyperGraph::node_degree, "Incident-edge count for node n")
        // Return pin / edge lists as Python lists (for validation only, not hot path)
        .def("get_root_pins", [](const FMHyperGraph& hg, int32_t e) {
            auto [b, end] = hg.root_span(e);
            return std::vector<int32_t>(b, end);
        })
        .def("get_rec_pins", [](const FMHyperGraph& hg, int32_t e) {
            auto [b, end] = hg.rec_span(e);
            return std::vector<int32_t>(b, end);
        })
        .def("get_node_edges", [](const FMHyperGraph& hg, int32_t n) {
            auto [b, end] = hg.node_edge_span(n);
            return std::vector<int32_t>(b, end);
        });

    m.def("build_hgraph", &build_hgraph,
          py::arg("Q"), py::arg("D"), py::arg("K"),
          py::arg("root_pins"),   py::arg("root_offsets"),
          py::arg("rec_pins"),    py::arg("rec_offsets"),
          py::arg("node_edges"),  py::arg("node_offsets"),
          "Build a C++ FMHyperGraph from pre-built CSR numpy arrays.");

    m.def("map_counts_and_configs", &map_counts_and_configs,
          py::arg("hg"), py::arg("assignment"),
          py::arg("out_root_counts"), py::arg("out_rec_counts"),
          py::arg("out_configs"), py::arg("out_costs"),
          "Fill per-edge count/config/cost arrays from the current assignment.");

    m.def("calculate_full_cost", &calculate_full_cost,
          py::arg("hg"), py::arg("assignment"),
          "Compute total communication cost for the current assignment.");

    m.def("coarsen_one_level", &coarsen_one_level,
          py::arg("hg"), py::arg("pairs_src"), py::arg("pairs_dst"),
          "Merge a batch of timestep pairs in the CSR hypergraph.\n"
          "Returns (new_FMHyperGraph, active_node_ids_int32).");

    m.def("fm_pass", &fm_pass,
          py::arg("hg"), py::arg("assignment"),
          py::arg("qpu_sizes"), py::arg("max_gain"), py::arg("limit"),
          py::arg("active_nodes") = py::none(),
          "Run one FM pass. Returns dict with 'assignment_list' ([initial, best, last])\n"
          "and 'gain_list' ([0, best_gain, last_gain]).\n"
          "active_nodes: optional int32 array of node IDs that may be moved;\n"
          "all others are locked (used for coarsened hypergraph levels).");

    m.def("fm_level",
          [](FMHyperGraph& hg,
             arr_i32 assignment_arr,
             arr_i32 qpu_sizes_arr,
             int32_t max_gain,
             int32_t limit,
             py::object active_nodes_obj,
             int32_t n_passes,
             bool stochastic) -> py::dict
          {
              auto asgn_buf = assignment_arr.request();
              const int32_t* asgn_ptr = static_cast<const int32_t*>(asgn_buf.ptr);
              auto qsz_buf  = qpu_sizes_arr.request();
              const int32_t* qsz_ptr  = static_cast<const int32_t*>(qsz_buf.ptr);

              arr_i32 active_nodes_arr;
              const int32_t* active_ptr = nullptr;
              int32_t n_active = 0;
              if (!active_nodes_obj.is_none()) {
                  active_nodes_arr = active_nodes_obj.cast<arr_i32>();
                  auto buf = active_nodes_arr.request();
                  active_ptr = static_cast<const int32_t*>(buf.ptr);
                  n_active   = static_cast<int32_t>(buf.size);
              }

              uint32_t seed = static_cast<uint32_t>(std::random_device{}());
              FMLevelResult res = run_fm_level(hg, asgn_ptr, qsz_ptr, max_gain, limit,
                                               seed, active_ptr, n_active,
                                               n_passes, stochastic);

              int32_t D = hg.D, Q = hg.Q, N = hg.N;
              auto best_arr = py::array_t<int32_t>({D, Q});
              std::memcpy(best_arr.mutable_data(), res.best_asgn.data(), N * sizeof(int32_t));

              py::dict result;
              result["best_assignment"] = best_arr;
              result["cost_deltas"]     = py::cast(res.cost_deltas);
              return result;
          },
          py::arg("hg"), py::arg("assignment"),
          py::arg("qpu_sizes"), py::arg("max_gain"), py::arg("limit"),
          py::arg("active_nodes") = py::none(),
          py::arg("n_passes") = 10,
          py::arg("stochastic") = true,
          "Run n_passes FM passes at one coarsening level, reusing FMState across passes.\n"
          "Returns dict with 'best_assignment' ([D,Q]) and 'cost_deltas' (list of ints).");
}
