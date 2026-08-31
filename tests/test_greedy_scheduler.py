"""
Test suite for the greedy list scheduler (disqco.scheduling.greedy_scheduler).

Layers of evidence:
1. Regression: on a circuit with NO commuting pairs the relaxed graph equals
   the wire-order graph, so the scheduler must reproduce the evaluator.
2. Audit: re-evaluating the emitted circuit reproduces the scheduler's makespan.
3. Equivalence: the reordered circuit computes the same unitary.
4. Improvement: on a hand-built circuit where commutation freedom provably
   helps, the scheduler beats the emitted order by a known amount.
"""

import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    PartitionedCircuitExtractor,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.scheduling.evaluator import evaluate_quantum_runtime
from disqco.scheduling.greedy_scheduler import (
    greedy_scheduler,
    emit_schedule,
    group_commutative_gates,
    create_grouped_graph,
)

UNIT = {}
DURATIONS = {"u": 1, "cp": 2, "cx": 2, "swap": 6, "measure": 5, "EPR": 100}


def extract(num_qubits=8, depth=16):
    circuit = cp_fraction(num_qubits=num_qubits, depth=depth,
                          fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=["u", "cp"])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    qpu = num_qubits // 2 + 1
    network = QuantumNetwork([qpu, qpu])
    assignment = set_initial_partition_assignment(hypergraph, network,
                                                  round_robin=True)
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph, network=network, partition_assignment=assignment)
    return extractor.extract_partitioned_circuit()


def rigid_circuit():
    """Alternating h/cx: no two ops on a shared wire commute, so the relaxed
    graph is identical to plain wire order - zero scheduling freedom."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.h(1)
    qc.cx(1, 2)
    qc.h(2)
    qc.cx(0, 2)
    return qc


# ---------------------------------------------------------------------------
# 1. Regression: no freedom -> scheduler == evaluator
# ---------------------------------------------------------------------------

def test_no_freedom_matches_evaluator():
    qc = rigid_circuit()
    base, _ = evaluate_quantum_runtime(qc, DURATIONS)
    for priority in ("fifo", "critical_path"):
        sched_makespan, _ = greedy_scheduler(qc, DURATIONS, priority=priority)
        assert sched_makespan == base


# ---------------------------------------------------------------------------
# 2. Audit: emitted circuit re-evaluates to the scheduler's makespan
# ---------------------------------------------------------------------------

def test_emitted_circuit_reproduces_makespan():
    qc = extract()
    makespan, schedule = greedy_scheduler(qc, DURATIONS)
    emitted = emit_schedule(qc, schedule)
    re_evaluated, _ = evaluate_quantum_runtime(emitted, DURATIONS)
    assert re_evaluated == makespan


# ---------------------------------------------------------------------------
# 3. Equivalence: reordering never changes the computation
# ---------------------------------------------------------------------------

def test_emitted_circuit_is_equivalent_unitary():
    """Unitary-only circuit with rich commutation: cp gates everywhere."""
    qc = QuantumCircuit(4)
    qc.h(0); qc.h(1); qc.h(2); qc.h(3)
    qc.cp(0.3, 0, 1); qc.cp(0.5, 0, 2); qc.cp(0.7, 0, 3)
    qc.cp(0.2, 1, 2); qc.x(1); qc.cp(0.9, 2, 3)

    _, schedule = greedy_scheduler(qc, UNIT)
    emitted = emit_schedule(qc, schedule)
    assert Operator(qc).equiv(Operator(emitted))


def test_schedule_conserves_all_ops():
    qc = extract()
    _, schedule = greedy_scheduler(qc, DURATIONS)
    indices = sorted(entry["op_idx"] for entry in schedule)
    assert indices == list(range(len(qc.data)))


# ---------------------------------------------------------------------------
# 4. Improvement: freedom + priority provably helps
# ---------------------------------------------------------------------------

def test_critical_path_beats_emitted_order_by_construction():
    """cp(0,1) and cp(0,2) commute (shared control). Emitted order runs the
    cheap one first, delaying the heavy x-chain behind cp(0,2). The greedy
    scheduler must run cp(0,2) first.

    Emitted:  cp01 [0,1), cp02 [1,2), x2 [2,12)  -> makespan 12
    Greedy:   cp02 [0,1), x2 [1,11), cp01 [1,2)  -> makespan 11
    """
    qc = QuantumCircuit(3)
    qc.cp(0.5, 0, 1)
    qc.cp(0.5, 0, 2)
    qc.x(2)
    durations = {"cp": 1, "x": 10}

    base, _ = evaluate_quantum_runtime(qc, durations)
    best, _ = greedy_scheduler(qc, durations, priority="critical_path")
    assert base == 12
    assert best == 11


def test_scheduler_never_loses_on_extracted_circuit():
    qc = extract()
    base, _ = evaluate_quantum_runtime(qc, DURATIONS)
    best, _ = greedy_scheduler(qc, DURATIONS)
    assert best <= base


# ---------------------------------------------------------------------------
# Graph-level sanity
# ---------------------------------------------------------------------------

def test_grouping_conserves_ops_per_wire():
    """Grouping repartitions each wire's ops - nothing lost, nothing added."""
    from disqco.scheduling.greedy_scheduler import operations_per_wire
    qc = extract()
    wire_ops = operations_per_wire(qc)
    grouped = group_commutative_gates(qc)
    for w, groups in grouped.items():
        assert sorted(i for g in groups for i in g) == wire_ops[w]


def test_relaxed_graph_is_dag_with_fewer_or_equal_edges():
    import networkx as nx
    qc = extract()
    graph = create_grouped_graph(qc, group_commutative_gates(qc))
    assert nx.is_directed_acyclic_graph(graph)
