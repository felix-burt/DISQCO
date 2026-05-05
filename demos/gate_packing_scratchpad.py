"""
Distribute circuits using DISQCO directly, without bosonic-sdk.
"""
import numpy as np
import qasmpi
import qiskit.qasm2
from qiskit import QuantumCircuit, transpile
from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
from disqco.parti import FiducciaMattheyses

_BENCH_CIRCUITS = [
    ("deutsch_n2",      2, 1),
    ("toffoli_n3",      2, 2),
    ("adder_n4",        2, 2),
    ("bell_n4",         2, 2),
    ("qaoa_n6",         3, 3),
    ("ising_n10",       2, 5),
    ("qft_n18",         2, 9),
    ("dnn_n16",         2, 8),
    ("cc_n12",          2, 6),
    ("bv_n14",          2, 7),
]

NUM_PASSES = 10


def load_qiskit(name: str) -> QuantumCircuit:
    raw = qasmpi.get_circuit(name)
    return qiskit.qasm2.loads(raw)

def partition(qc: QuantumCircuit, nodes: int, qubits_per_node: int) -> np.ndarray:
    network = QuantumNetwork.create([qubits_per_node] * nodes, "all_to_all")
    hypergraph = QuantumCircuitHyperGraph(qc, group_gates=False)
    initial = set_initial_partition_assignment(hypergraph, network)
    partitioner = FiducciaMattheyses(
        qc, network, initial, hypergraph=hypergraph, group_gates=False
    )
    results = partitioner.multilevel_partition(
        coarsener=HypergraphCoarsener().coarsen_recursive_batches_mapped,
        passes_per_level=NUM_PASSES,
    )
    assignment = np.asarray(results["best_assignment"], dtype=int)
    if assignment.ndim == 1:
        assignment = assignment.reshape(1, -1)
    return assignment, network


def extract(qc: QuantumCircuit, network, assignment: np.ndarray) -> QuantumCircuit:
    hypergraph = QuantumCircuitHyperGraph(qc)
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph, network=network, partition_assignment=assignment
    )
    return extractor.extract_partitioned_circuit()


def qubits_per_node(qc: QuantumCircuit) -> dict[int, int]:
    """Read node qubit counts from DISQCO's Q<n>_/C<n>_ register naming."""
    counts: dict[int, int] = {}
    for reg in qc.qregs:
        name = reg.name
        if not name or name[0] not in {"Q", "C"}:
            continue
        digits = ""
        for ch in name[1:]:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            continue
        node = int(digits)
        counts[node] = counts.get(node, 0) + reg.size
    return counts


for name, nodes, qpn in _BENCH_CIRCUITS:
    print(f"  {name} ...", end=" ", flush=True)
    try:
        qc = load_qiskit(name)
        assignment, network = partition(qc, nodes, qpn)
        extracted = extract(qc, network, assignment)
        actual = qubits_per_node(extracted)
        print(f"nodes={nodes}, qubits_per_node={qpn} → actual: {actual}", flush=True)
    except Exception as e:
        print(f"FAIL: {e}", flush=True)
