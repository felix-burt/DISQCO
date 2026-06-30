"""
Validate the C++ FMHyperGraph data structure against the Python hypergraph.

Checks:
  1. Node counts, edge counts, Q/D/K match.
  2. Root and receiver pin sets round-trip correctly for every edge.
  3. Per-node incident-edge sets round-trip correctly for every node.
  4. calculate_full_cost matches the Python implementation on the initial assignment.
  5. map_counts_and_configs matches the Python path for all edges.
"""
import pytest
import numpy as np
import sys

sys.path.insert(0, "src")

pytest.importorskip("disqco._fm_cpp")
from disqco import _fm_cpp

from disqco.circuits.cp_fraction import cp_fraction
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.parti.FM.fiduccia import FiducciaMattheyses
from disqco.parti.FM._fm_cpp_builder import build_cpp_hgraph
from disqco.graphs.hypergraph_methods import calculate_full_cost, map_counts_and_configs


@pytest.fixture(scope="module")
def setup_32():
    circuit = cp_fraction(32, 32, 0.5, seed=42)
    network = QuantumNetwork(qpu_sizes=[8] * 4, qpu_connectivity=None)
    fm = FiducciaMattheyses(circuit, network)
    hg_py = fm.hypergraph
    asgn  = fm.initial_assignment.copy()
    hg_cpp = fm.get_cpp_hg()
    return fm, hg_py, asgn, hg_cpp


def test_extension_importable():
    assert hasattr(_fm_cpp, "build_hgraph")
    assert hasattr(_fm_cpp, "calculate_full_cost")
    assert hasattr(_fm_cpp, "map_counts_and_configs")


def test_cpp_hg_dimensions(setup_32):
    fm, hg_py, asgn, hg_cpp = setup_32
    assert hg_cpp is not False, "C++ extension not available"
    assert hg_cpp.Q == hg_py.num_qubits
    assert hg_cpp.D == hg_py.depth
    assert hg_cpp.K == fm.num_partitions
    assert hg_cpp.N == hg_py.num_qubits * hg_py.depth
    assert hg_cpp.E == len(hg_py.hyperedges)


def test_root_and_rec_pins_roundtrip(setup_32):
    _, hg_py, _, hg_cpp = setup_32
    Q = hg_py.num_qubits
    edge_list = list(hg_py.hyperedges.keys())

    for e_idx, eid in enumerate(edge_list):
        edge_data = hg_py.hyperedges[eid]

        expected_roots = frozenset(
            t * Q + q
            for (q, t) in edge_data['root_set']
        )
        expected_recs = frozenset(
            t * Q + q
            for (q, t) in edge_data['receiver_set']
        )

        got_roots = frozenset(hg_cpp.get_root_pins(e_idx))
        got_recs  = frozenset(hg_cpp.get_rec_pins(e_idx))

        assert got_roots == expected_roots, (
            f"Edge {e_idx} ({eid}): root pins mismatch\n"
            f"  expected {expected_roots}\n  got {got_roots}"
        )
        assert got_recs == expected_recs, (
            f"Edge {e_idx} ({eid}): rec pins mismatch\n"
            f"  expected {expected_recs}\n  got {got_recs}"
        )


def test_node_incident_edges_roundtrip(setup_32):
    _, hg_py, _, hg_cpp = setup_32
    Q = hg_py.num_qubits
    edge_list = list(hg_py.hyperedges.keys())
    eid_to_idx = {eid: i for i, eid in enumerate(edge_list)}

    for node in hg_py.nodes:
        if node[0] == 'dummy':
            continue
        q, t = node
        n = t * Q + q

        expected = frozenset(
            eid_to_idx[eid]
            for eid in hg_py.node2hyperedges[node]
            if eid in eid_to_idx
        )
        got = frozenset(hg_cpp.get_node_edges(n))

        assert got == expected, (
            f"Node ({q},{t}) id={n}: incident edges mismatch\n"
            f"  expected {expected}\n  got {got}"
        )


def test_calculate_full_cost_matches_python(setup_32):
    fm, hg_py, asgn, hg_cpp = setup_32
    costs = {}
    py_cost = calculate_full_cost(
        hg_py, asgn, fm.num_partitions, costs,
        hetero=fm.network.hetero
    )
    cpp_cost = _fm_cpp.calculate_full_cost(hg_cpp, asgn)
    assert cpp_cost == py_cost, (
        f"Cost mismatch: C++={cpp_cost}, Python={py_cost}"
    )


def test_map_counts_and_configs_matches_python(setup_32):
    fm, hg_py, asgn, hg_cpp = setup_32
    E = hg_cpp.E
    K = hg_cpp.K

    # Call the C++ version
    out_rc   = np.zeros((E, K), dtype=np.int32)
    out_rcc  = np.zeros((E, K), dtype=np.int32)
    out_cfg  = np.zeros(E,      dtype=np.int32)
    out_cst  = np.zeros(E,      dtype=np.int32)
    _fm_cpp.map_counts_and_configs(hg_cpp, asgn, out_rc, out_rcc, out_cfg, out_cst)

    # Call the Python version and read back its edge attrs
    costs = {}
    map_counts_and_configs(hg_py, asgn, fm.num_partitions, costs, hetero=False)

    edge_list = list(hg_py.hyperedges.keys())
    for e_idx, eid in enumerate(edge_list):
        attrs = hg_py.hyperedge_attrs[eid]
        py_cost = attrs['cost']
        py_cfg  = attrs['config']

        cpp_cost = int(out_cst[e_idx])
        # config from C++ is a bitmask; from Python it's a list of 0/1
        cpp_cfg_list = [(out_cfg[e_idx] >> k) & 1 for k in range(K)]

        assert cpp_cost == py_cost, (
            f"Edge {e_idx}: cost mismatch C++={cpp_cost} Python={py_cost}"
        )
        assert cpp_cfg_list == list(py_cfg), (
            f"Edge {e_idx}: config mismatch C++={cpp_cfg_list} Python={list(py_cfg)}"
        )


def test_fm_pass_cost_nondeterminism_sanity(setup_32):
    """
    fm_pass must return a valid assignment with cost ≤ initial cost after one pass.
    We run it multiple times to account for stochasticity and check every result
    is a valid assignment (correct shape, partitions in [0, K)).
    """
    fm, hg_py, asgn, hg_cpp = setup_32
    K = fm.num_partitions
    D = hg_py.depth
    Q = hg_py.num_qubits
    qpu_sizes = np.array(list(fm.qpu_sizes.values()), dtype=np.int32)
    max_gain = fm.find_max_gain()
    limit = int(len(hg_py.nodes) * 0.125)

    for _ in range(3):
        result = _fm_cpp.fm_pass(hg_cpp, asgn, qpu_sizes, max_gain, limit)
        assert 'assignment_list' in result and 'gain_list' in result
        asgn_list = result['assignment_list']
        gain_list = result['gain_list']
        assert len(asgn_list) == 3  # [initial, best, last]
        assert len(gain_list) == 3  # [0, best_gain, last_gain]
        assert gain_list[0] == 0

        for arr in asgn_list:
            assert arr.shape == (D, Q)
            assert np.all(arr >= 0) and np.all(arr < K)

        # best_gain ≤ 0 and ≤ last_gain
        assert gain_list[1] <= 0
        assert gain_list[1] <= gain_list[2]


def test_fm_pass_cost_vs_python(setup_32):
    """
    After one C++ fm_pass, the best assignment should have cost ≤ initial cost.
    The C++ and Python paths need not agree exactly (both are stochastic),
    but the C++ best cost must be ≤ initial cost and internally consistent
    (i.e., calculate_full_cost on the returned best_assignment matches best_gain).
    """
    from disqco.graphs.hypergraph_methods import calculate_full_cost

    fm, hg_py, asgn, hg_cpp = setup_32
    K = fm.num_partitions
    qpu_sizes = np.array(list(fm.qpu_sizes.values()), dtype=np.int32)
    max_gain = fm.find_max_gain()
    limit = int(len(hg_py.nodes) * 0.125)

    costs = {}
    initial_cost = calculate_full_cost(
        hg_py, asgn, K, costs, hetero=fm.network.hetero
    )
    cpp_initial_cost = _fm_cpp.calculate_full_cost(hg_cpp, asgn)
    assert cpp_initial_cost == initial_cost

    result = _fm_cpp.fm_pass(hg_cpp, asgn, qpu_sizes, max_gain, limit)
    best_asgn = result['assignment_list'][1]
    best_gain = result['gain_list'][1]

    # Verify: C++ cost of best_asgn == initial_cost + best_gain
    cpp_best_cost = _fm_cpp.calculate_full_cost(hg_cpp, best_asgn)
    expected_best_cost = initial_cost + best_gain
    assert cpp_best_cost == expected_best_cost, (
        f"Cost inconsistency: calculate_full_cost={cpp_best_cost}, "
        f"initial_cost+best_gain={expected_best_cost}"
    )
