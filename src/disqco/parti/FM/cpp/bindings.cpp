#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <random>
#include <cstring>

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
    int32_t limit)
{
    // Read-only input buffers
    auto asgn_buf = assignment_arr.request();
    const int32_t* asgn_ptr = static_cast<const int32_t*>(asgn_buf.ptr);
    int32_t N = hg.N, D = hg.D, Q = hg.Q;

    auto qsz_buf = qpu_sizes_arr.request();
    const int32_t* qsz_ptr = static_cast<const int32_t*>(qsz_buf.ptr);

    // Use a thread-local or time-based seed for reproducible stochasticity
    uint32_t seed = static_cast<uint32_t>(std::random_device{}());

    FMPassResult res = run_fm_pass(hg, asgn_ptr, qsz_ptr, max_gain, limit, seed);

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

    m.def("fm_pass", &fm_pass,
          py::arg("hg"), py::arg("assignment"),
          py::arg("qpu_sizes"), py::arg("max_gain"), py::arg("limit"),
          "Run one FM pass. Returns dict with 'assignment_list' ([initial, best, last])\n"
          "and 'gain_list' ([0, best_gain, last_gain]).");
}
