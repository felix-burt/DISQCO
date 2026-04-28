"""
Test suite for circuit extraction functionality.

Tests the PartitionedCircuitExtractor class for extracting distributed quantum
circuits from hypergraphs with both initial (unoptimized) and optimized assignments.

After the bosonic refactor, DISQCO consumes a `bosonic_model.Circuit` and produces a
`bosonic_model.DistributedCircuit`. Tests build a Qiskit `QuantumCircuit` for ergonomic
construction, then convert to bosonic via `CircuitConverters.from_qiskit` before passing
into DISQCO. The cross-QPU "EPR" count is read from `DistributedCircuit.coupling_map()`.
"""

from pathlib import Path

import pytest
import numpy as np
from bosonic_converters import CircuitConverters
from bosonic_model import DistributedCircuit
from qiskit import QuantumCircuit, qasm2, transpile

from disqco import QuantumNetwork, QuantumCircuitHyperGraph, PartitionedCircuitExtractor
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
from disqco import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "circuits"


def _to_bosonic(qc: QuantumCircuit):
    """Helper: convert a Qiskit QuantumCircuit into a bosonic Circuit for DISQCO."""
    return CircuitConverters.from_qiskit(qc)


def _epr_count(distributed: DistributedCircuit) -> int:
    """Sum of cross-QPU remote-link instructions in a DistributedCircuit."""
    return sum(distributed.coupling_map().values())


@pytest.fixture
def test_circuit_qiskit():
    """Create a test circuit (Qiskit-side) for extraction."""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_circuit(test_circuit_qiskit):
    """Bosonic Circuit equivalent of the Qiskit test circuit."""
    return _to_bosonic(test_circuit_qiskit)


@pytest.fixture
def test_network():
    """Create a 2-QPU network for testing."""
    return QuantumNetwork.create([5, 5], 'all_to_all')


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from test circuit."""
    return QuantumCircuitHyperGraph(test_circuit)


@pytest.fixture
def initial_assignment(test_hypergraph, test_network):
    """Create initial unoptimized assignment."""
    return set_initial_partition_assignment(test_hypergraph, test_network)


@pytest.fixture
def optimized_assignment(test_circuit, test_network):
    """Create optimized assignment using FM partitioner."""
    partitioner = FiducciaMattheyses(test_circuit, network=test_network)
    results = partitioner.partition(num_passes=5)
    return results['best_assignment']


def test_circuit_extractor_import():
    from disqco import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_from_circuit_extraction_module():
    from disqco.circuit_extraction import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_instantiation(test_hypergraph, test_network, initial_assignment):
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment,
    )
    assert extractor is not None
    assert extractor.graph is test_hypergraph
    assert extractor.network is test_network
    assert np.array_equal(extractor.partition_assignment, initial_assignment.tolist())


def test_extract_circuit_with_initial_assignment(test_hypergraph, test_network, initial_assignment):
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment,
    )
    distributed = extractor.extract_partitioned_circuit()

    assert isinstance(distributed, DistributedCircuit)
    assert sorted(distributed.qubits_per_node.keys()) == [0, 1]
    assert all(len(circ.instructions) > 0 for circ in distributed.circuits.values())


def test_extract_circuit_with_optimized_assignment(test_hypergraph, test_network, optimized_assignment):
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment,
    )
    distributed = extractor.extract_partitioned_circuit()

    assert isinstance(distributed, DistributedCircuit)
    assert sorted(distributed.qubits_per_node.keys()) == [0, 1]


def test_compare_epr_counts_initial_vs_optimized(test_hypergraph, test_network,
                                                 initial_assignment, optimized_assignment):
    """The optimised partition should never use more EPR pairs than the initial one."""
    dist_initial = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment,
    ).extract_partitioned_circuit()

    dist_optimized = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment,
    ).extract_partitioned_circuit()

    epr_initial = _epr_count(dist_initial)
    epr_optimized = _epr_count(dist_optimized)
    assert epr_optimized <= epr_initial


def test_extracted_circuit_structure(test_hypergraph, test_network, initial_assignment):
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment,
    )
    distributed = extractor.extract_partitioned_circuit()

    # Each per-node circuit must have at least one qreg and a shared cl_global creg.
    for node, circ in distributed.circuits.items():
        assert len(circ.qregs) > 0
        assert len(circ.cregs) > 0
        assert any(name.startswith("cl_global") for name in circ.cregs)
        assert "result" in circ.cregs


def test_extraction_with_different_networks(test_hypergraph, initial_assignment):
    linear_net = QuantumNetwork.create([5, 5], 'linear')
    dist_linear = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=linear_net,
        partition_assignment=initial_assignment,
    ).extract_partitioned_circuit()
    assert isinstance(dist_linear, DistributedCircuit)

    alltoall_net = QuantumNetwork.create([5, 5], 'all_to_all')
    dist_a2a = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=alltoall_net,
        partition_assignment=initial_assignment,
    ).extract_partitioned_circuit()
    assert isinstance(dist_a2a, DistributedCircuit)


def test_extraction_with_three_partitions(test_circuit):
    network = QuantumNetwork.create([3, 3, 3], 'linear')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)

    distributed = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
    ).extract_partitioned_circuit()

    assert isinstance(distributed, DistributedCircuit)
    assert sorted(distributed.qubits_per_node.keys()) == [0, 1, 2]


def test_extraction_with_four_partitions(test_circuit):
    network = QuantumNetwork.create([2, 2, 2, 2], 'grid')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)

    distributed = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
    ).extract_partitioned_circuit()

    assert isinstance(distributed, DistributedCircuit)
    assert sorted(distributed.qubits_per_node.keys()) == [0, 1, 2, 3]


def test_full_workflow_initial_to_optimized():
    qiskit_circuit = cp_fraction(num_qubits=12, depth=12, fraction=0.5, seed=123)
    qiskit_circuit = transpile(qiskit_circuit, basis_gates=['u', 'cp'])
    circuit = _to_bosonic(qiskit_circuit)

    network = QuantumNetwork.create([5, 5, 5], 'linear')
    hypergraph = QuantumCircuitHyperGraph(circuit)
    initial = set_initial_partition_assignment(hypergraph, network)

    dist_initial = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=initial,
    ).extract_partitioned_circuit()

    partitioner = FiducciaMattheyses(circuit, network=network)
    results = partitioner.partition(num_passes=10)

    dist_optimized = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=results['best_assignment'],
    ).extract_partitioned_circuit()

    assert isinstance(dist_initial, DistributedCircuit)
    assert isinstance(dist_optimized, DistributedCircuit)
    assert _epr_count(dist_optimized) <= _epr_count(dist_initial)


def test_extraction_preserves_circuit_qubit_count(test_circuit_qiskit, test_circuit, test_network):
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, test_network)

    distributed = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=test_network,
        partition_assignment=assignment,
    ).extract_partitioned_circuit()

    # Each logical qubit must map to one of the data registers in the output.
    total_data = 0
    for circ in distributed.circuits.values():
        for reg in circ.qregs.values():
            if reg.name.startswith("Q"):
                total_data += reg.size
    assert total_data >= test_circuit_qiskit.num_qubits


def test_extractor_with_single_partition():
    qiskit_circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    qiskit_circuit = transpile(qiskit_circuit, basis_gates=['u', 'cp'])
    circuit = _to_bosonic(qiskit_circuit)

    network = QuantumNetwork({0: 10})
    hypergraph = QuantumCircuitHyperGraph(circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)

    distributed = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
    ).extract_partitioned_circuit()

    # Single partition: no cross-QPU couplings possible.
    assert _epr_count(distributed) == 0


def test_group_closed_when_all_subgates_applied_immediately():
    """Regression: groups whose last two-qubit gate lands at the current time step must have
    close_group called explicitly, otherwise the root qubit stays marked as grouped and
    subsequent teleportation logic silently skips it, producing an incorrect circuit."""
    qiskit_circuit = qasm2.load(
        FIXTURES_DIR / "variational_n4_transpiled.qasm",
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
    )
    circuit = _to_bosonic(qiskit_circuit)

    hypergraph = QuantumCircuitHyperGraph(circuit)
    network = QuantumNetwork.create([3, 3], "all_to_all")
    initial = set_initial_partition_assignment(hypergraph, network)
    partitioner = FiducciaMattheyses(
        circuit,
        network,
        initial,
        hypergraph=hypergraph,
    )
    results = partitioner.multilevel_partition(
        coarsener=HypergraphCoarsener().coarsen_recursive_batches_mapped,
        passes_per_level=10,
    )

    distributed = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=results["best_assignment"],
    ).extract_partitioned_circuit()

    assert isinstance(distributed, DistributedCircuit)
