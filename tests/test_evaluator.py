"""
Test suite for the makespan evaluator (disqco.scheduling.evaluator).

The evaluator is an ASAP critical-path walk: every op starts as soon as all
its wires are free. It measures, never reorders.
"""

import networkx as nx
import pytest
from qiskit import QuantumCircuit, transpile

from disqco import (
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    PartitionedCircuitExtractor,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.scheduling.evaluator import evaluate_quantum_runtime

UNIT = {}


def makespan(*args, **kwargs):
    """Unwrap just the number from the (makespan, schedule) return."""
    return evaluate_quantum_runtime(*args, **kwargs)[0]


def extract(num_qubits=8, depth=16, qpu_topologies=None, comm_sizes=None):
    circuit = cp_fraction(num_qubits=num_qubits, depth=depth,
                          fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=["u", "cp"])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    qpu = num_qubits // 2 + 1
    network = QuantumNetwork([qpu, qpu], comm_sizes=comm_sizes,
                             qpu_topologies=qpu_topologies)
    assignment = set_initial_partition_assignment(hypergraph, network,
                                                  round_robin=True)
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph, network=network, partition_assignment=assignment)
    return extractor.extract_partitioned_circuit()


def test_parallel_chains_overlap():
    """Two 2-gate chains on disjoint qubits: makespan 2, not 4."""
    qc = QuantumCircuit(2)
    qc.h(0); qc.h(0)
    qc.h(1); qc.h(1)
    assert makespan(qc, UNIT) == 2


def test_dependent_chain_serialises():
    """A cx joins the wires: everything after it waits for both."""
    qc = QuantumCircuit(2)
    qc.h(0)          # q0: 0 -> 1
    qc.cx(0, 1)      # needs q0(1), q1(0) -> starts 1, ends 2
    qc.h(1)          # q1: 2 -> 3
    assert makespan(qc, UNIT) == 3


def test_durations_are_respected():
    """A slow gate stretches only the path through it."""
    qc = QuantumCircuit(2)
    qc.h(0)          # 0 -> 1
    qc.x(1)          # 0 -> 5 with x: 5
    assert makespan(qc, {"x": 5, "h": 1}) == 5


def test_barrier_costs_nothing():
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.barrier()
    qc.h(0)
    assert makespan(qc, UNIT) == 2


def test_classical_dependency_serialises_when_included():
    """A measurement feeding a c_if creates a dependency across qubits
    that share no quantum wire - only visible in clbit mode."""
    qc = QuantumCircuit(2, 1)
    qc.h(0)
    qc.measure(0, 0)                 # writes clbit 0
    qc.x(1).c_if(qc.cregs[0], 1)     # waits on clbit 0

    with_cl = makespan(qc, UNIT, include_clbits=True)
    without = makespan(qc, UNIT, include_clbits=False)
    assert with_cl == 3   # h -> measure -> conditioned x, serialised
    assert without == 2   # x(1) detaches; critical path is h -> measure
    assert with_cl > without


def test_unit_durations_match_depth_on_classical_free_circuit():
    """Qiskit's depth() IS unit-duration ASAP - on a circuit with no
    classical bits the evaluator must reproduce it exactly."""
    circuit = cp_fraction(num_qubits=8, depth=16, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=["u", "cp"])
    assert makespan(circuit, UNIT) == circuit.depth()


def test_extracted_circuit_evaluates():
    qc = extract()
    full = makespan(qc, UNIT)
    assert full > 0
    # clbit dependencies can only add constraints, never remove them
    qubit_only = makespan(qc, UNIT, include_clbits=False)
    assert full >= qubit_only


def test_epr_duration_dominates_makespan():
    """Raising the EPR cost must strictly stretch a circuit that uses EPR."""
    qc = extract()
    assert qc.count_ops().get("EPR", 0) > 0
    fast = makespan(qc, {"EPR": 1})
    slow = makespan(qc, {"EPR": 100})
    assert slow > fast


def test_port_topology_costs_runtime():
    """The routed (ports) extraction should not be faster than the
    unconstrained one under identical durations."""
    def topo(n):
        t = nx.path_graph(n)
        comm = list(range(n, n + 4))
        for c in comm:
            t.add_edge(n - 1, c)
        for i, a in enumerate(comm):
            for b in comm[i + 1:]:
                t.add_edge(a, b)
        return t

    durations = {"EPR": 20, "swap": 6, "cp": 2, "u": 1, "measure": 5}
    plain = makespan(extract(), durations)
    qpu = 8 // 2 + 1
    ported = makespan(
        extract(qpu_topologies={0: topo(qpu), 1: topo(qpu)},
                comm_sizes=[4, 4]),
        durations)
    assert ported >= plain

