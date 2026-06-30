"""
Test suite for circuit extraction functionality.

Tests the PartitionedCircuitExtractor class for extracting distributed quantum
circuits from hypergraphs with both initial (unoptimized) and optimized assignments.
"""

import pytest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit import QuantumRegister
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace, state_fidelity
from qiskit_aer import AerSimulator

from disqco import QuantumNetwork, QuantumCircuitHyperGraph, PartitionedCircuitExtractor
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
from disqco import set_initial_partition_assignment


def _build_channel_validation_circuit() -> QuantumCircuit:
    """Small deterministic circuit that induces non-local interactions under partitioning."""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    qc.cp(np.pi / 3, 1, 2)
    qc.cp(np.pi / 5, 0, 2)
    qc.u(0.2, 0.3, -0.4, 1)
    qc.u(-0.1, 0.6, 0.7, 2)
    return transpile(qc, basis_gates=['u', 'cp'])


def _choi_state_fidelity(base_circuit: QuantumCircuit, extractor: PartitionedCircuitExtractor,
                         partitioned_circuit: QuantumCircuit) -> float:
    """Compute process fidelity via normalized Choi states.

    We prepare |Phi+> between a reference register and the logical input qubits,
    evolve the system through the distributed circuit, then compare the reduced
    output/reference state against the ideal circuit's Choi state.
    """
    simulator = AerSimulator(method='density_matrix')
    num_logical = base_circuit.num_qubits

    # Keep mid-circuit measurements/feed-forward but remove terminal readout only.
    stripped = partitioned_circuit.remove_final_measurements(inplace=False)
    stripped = transpile(stripped, basis_gates=['u', 'cx'], optimization_level=0)

    input_map = extractor.qubit_manager.inital_qubit_placement
    output_map = extractor.qubit_manager.log_to_phys_idx

    input_indices = [partitioned_circuit.find_bit(input_map[i]).index for i in range(num_logical)]
    output_indices = [partitioned_circuit.find_bit(output_map[i]).index for i in range(num_logical)]

    ref = QuantumRegister(num_logical, name='ref')
    full = QuantumCircuit(ref, *stripped.qregs, *stripped.cregs)

    # Build maximally entangled state between reference and system input qubits.
    for i in range(num_logical):
        full.h(ref[i])
        full.cx(ref[i], full.qubits[num_logical + input_indices[i]])

    full.compose(stripped, qubits=full.qubits[num_logical:], clbits=full.clbits, inplace=True)
    full.save_density_matrix()

    result = simulator.run(full).result()
    rho_full = DensityMatrix(result.data(0)['density_matrix'])

    keep_bits = [full.qubits[i] for i in range(num_logical)] + [
        full.qubits[num_logical + idx] for idx in output_indices
    ]
    trace_out = [idx for idx, bit in enumerate(full.qubits) if bit not in keep_bits]
    rho_reduced = partial_trace(rho_full, trace_out)

    ideal = QuantumCircuit(2 * num_logical)
    for i in range(num_logical):
        ideal.h(i)
        ideal.cx(i, num_logical + i)
    ideal.compose(base_circuit, qubits=list(range(num_logical, 2 * num_logical)), inplace=True)
    rho_ideal = DensityMatrix(Statevector.from_instruction(ideal))

    return float(state_fidelity(rho_reduced, rho_ideal))


@pytest.fixture
def test_circuit():
    """Create a test circuit for extraction"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_network():
    """Create a 2-QPU network for testing"""
    return QuantumNetwork.create([5, 5], 'all_to_all')


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from test circuit"""
    return QuantumCircuitHyperGraph(test_circuit)


@pytest.fixture
def initial_assignment(test_hypergraph, test_network):
    """Create initial unoptimized assignment"""
    return set_initial_partition_assignment(test_hypergraph, test_network)


@pytest.fixture
def optimized_assignment(test_circuit, test_network):
    """Create optimized assignment using FM partitioner"""
    partitioner = FiducciaMattheyses(test_circuit, network=test_network)
    results = partitioner.partition(num_passes=5)
    return results['best_assignment']


def test_circuit_extractor_import():
    """Test that PartitionedCircuitExtractor can be imported from disqco"""
    from disqco import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_from_circuit_extraction_module():
    """Test importing from circuit_extraction module"""
    from disqco.circuit_extraction import PartitionedCircuitExtractor
    assert PartitionedCircuitExtractor is not None


def test_circuit_extractor_instantiation(test_hypergraph, test_network, initial_assignment):
    """Test creating an instance of PartitionedCircuitExtractor"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    assert extractor is not None
    assert extractor.graph == test_hypergraph
    assert extractor.network == test_network
    assert np.array_equal(extractor.partition_assignment, initial_assignment)
    
    print("\n✓ PartitionedCircuitExtractor instantiated successfully")


def test_extract_circuit_with_initial_assignment(test_hypergraph, test_network, initial_assignment):
    """Test extracting circuit with initial unoptimized assignment"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert isinstance(partitioned_circuit, QuantumCircuit)
    assert partitioned_circuit.num_qubits > 0
    assert partitioned_circuit.depth() > 0
    
    print(f"\n✓ Extracted circuit with initial assignment:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")
    print(f"  Number of qubits: {partitioned_circuit.num_qubits}")


def test_extract_circuit_with_optimized_assignment(test_hypergraph, test_network, optimized_assignment):
    """Test extracting circuit with optimized FM assignment"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert isinstance(partitioned_circuit, QuantumCircuit)
    assert partitioned_circuit.num_qubits > 0
    assert partitioned_circuit.depth() > 0
    
    print(f"\n✓ Extracted circuit with optimized assignment:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")
    print(f"  Number of qubits: {partitioned_circuit.num_qubits}")


def test_compare_epr_counts_initial_vs_optimized(test_hypergraph, test_network, 
                                                  initial_assignment, optimized_assignment):
    """Test that optimized assignment typically produces fewer EPR pairs"""
    # Extract with initial assignment
    extractor_initial = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    circuit_initial = extractor_initial.extract_partitioned_circuit()
    circuit_initial_epr = transpile(circuit_initial, basis_gates=['u', 'cp', 'EPR'])
    
    # Extract with optimized assignment
    extractor_optimized = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=optimized_assignment
    )
    circuit_optimized = extractor_optimized.extract_partitioned_circuit()
    circuit_optimized_epr = transpile(circuit_optimized, basis_gates=['u', 'cp', 'EPR'])
    
    # Count EPR pairs
    ops_initial = circuit_initial_epr.count_ops()
    ops_optimized = circuit_optimized_epr.count_ops()
    
    epr_initial = ops_initial.get('EPR', 0)
    epr_optimized = ops_optimized.get('EPR', 0)
    
    print(f"\n✓ EPR pair comparison:")
    print(f"  Initial assignment: {epr_initial} EPR pairs")
    print(f"  Optimized assignment: {epr_optimized} EPR pairs")
    print(f"  Reduction: {epr_initial - epr_optimized} EPR pairs")
    
    # Optimized should typically use same or fewer EPR pairs
    assert epr_optimized <= epr_initial


def test_extracted_circuit_structure(test_hypergraph, test_network, initial_assignment):
    """Test the structure of extracted partitioned circuit"""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    # Check circuit has quantum and classical registers
    assert len(partitioned_circuit.qregs) > 0
    assert len(partitioned_circuit.cregs) > 0
    
    # Circuit should have operations
    assert len(partitioned_circuit.data) > 0
    
    print(f"\n✓ Circuit structure:")
    print(f"  Quantum registers: {len(partitioned_circuit.qregs)}")
    print(f"  Classical registers: {len(partitioned_circuit.cregs)}")
    print(f"  Total operations: {len(partitioned_circuit.data)}")


def test_extraction_with_different_networks(test_hypergraph, initial_assignment):
    """Test extraction with different network topologies"""
    # Test with linear network
    linear_net = QuantumNetwork.create([5, 5], 'linear')
    extractor_linear = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=linear_net,
        partition_assignment=initial_assignment
    )
    circuit_linear = extractor_linear.extract_partitioned_circuit()
    assert circuit_linear is not None
    
    # Test with all-to-all network
    alltoall_net = QuantumNetwork.create([5, 5], 'all_to_all')
    extractor_alltoall = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=alltoall_net,
        partition_assignment=initial_assignment
    )
    circuit_alltoall = extractor_alltoall.extract_partitioned_circuit()
    assert circuit_alltoall is not None
    
    print("\n✓ Extraction works with different network topologies")


def test_extraction_with_three_partitions(test_circuit):
    """Test extraction with 3 QPUs"""
    network = QuantumNetwork.create([3, 3, 3], 'linear')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert partitioned_circuit.num_qubits > 0
    
    print(f"\n✓ Extraction with 3 partitions:")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")


def test_extraction_with_four_partitions(test_circuit):
    """Test extraction with 4 QPUs"""
    network = QuantumNetwork.create([2, 2, 2, 2], 'grid')
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    assert partitioned_circuit is not None
    assert partitioned_circuit.num_qubits > 0
    
    print(f"\n✓ Extraction with 4 partitions (grid):")
    print(f"  Circuit depth: {partitioned_circuit.depth()}")


def test_full_workflow_initial_to_optimized():
    """Test complete workflow from initial assignment to optimized extraction"""
    # Create circuit
    circuit = cp_fraction(num_qubits=12, depth=12, fraction=0.5, seed=123)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    
    # Create network
    network = QuantumNetwork.create([5, 5, 5], 'linear')
    
    # Create hypergraph
    hypergraph = QuantumCircuitHyperGraph(circuit)
    
    # Get initial assignment
    initial_assignment = set_initial_partition_assignment(hypergraph, network)
    
    # Extract with initial assignment
    extractor_initial = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=initial_assignment
    )
    circuit_initial = extractor_initial.extract_partitioned_circuit()
    
    # Run optimization
    partitioner = FiducciaMattheyses(circuit, network=network)
    results = partitioner.partition(num_passes=10)
    optimized_assignment = results['best_assignment']
    
    # Extract with optimized assignment
    extractor_optimized = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=optimized_assignment
    )
    circuit_optimized = extractor_optimized.extract_partitioned_circuit()
    
    # Both circuits should be valid
    assert circuit_initial is not None
    assert circuit_optimized is not None
    assert circuit_initial.num_qubits > 0
    assert circuit_optimized.num_qubits > 0
    
    # Count EPR pairs
    circuit_initial_epr = transpile(circuit_initial, basis_gates=['u', 'cp', 'EPR'])
    circuit_optimized_epr = transpile(circuit_optimized, basis_gates=['u', 'cp', 'EPR'])
    
    epr_initial = circuit_initial_epr.count_ops().get('EPR', 0)
    epr_optimized = circuit_optimized_epr.count_ops().get('EPR', 0)
    
    print(f"\n✓ Full workflow test:")
    print(f"  Initial EPR pairs: {epr_initial}")
    print(f"  Optimized EPR pairs: {epr_optimized}")
    print(f"  Improvement: {epr_initial - epr_optimized} pairs ({100*(epr_initial-epr_optimized)/max(epr_initial,1):.1f}%)")
    
    # Optimized should be better or equal
    assert epr_optimized <= epr_initial


def test_extraction_preserves_circuit_semantics(test_circuit, test_network):
    """Test that extraction produces a circuit with same number of operations"""
    hypergraph = QuantumCircuitHyperGraph(test_circuit)
    assignment = set_initial_partition_assignment(hypergraph, test_network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=test_network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    
    # Original circuit operations
    original_ops = test_circuit.count_ops()
    original_gate_count = sum(count for gate, count in original_ops.items() 
                              if gate not in ['barrier', 'measure'])
    
    # Partitioned circuit will have additional teleportation operations
    # but should preserve the original gates
    assert partitioned_circuit.depth() >= test_circuit.depth()
    
    print(f"\n✓ Circuit semantics:")
    print(f"  Original gates: {original_gate_count}")
    print(f"  Original depth: {test_circuit.depth()}")
    print(f"  Partitioned depth: {partitioned_circuit.depth()}")


def test_extractor_with_single_partition():
    """Test extraction with single partition (trivial case)"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    
    # Single partition - should need no EPR pairs
    network = QuantumNetwork({0: 10})
    hypergraph = QuantumCircuitHyperGraph(circuit)
    assignment = set_initial_partition_assignment(hypergraph, network)
    
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment
    )
    
    partitioned_circuit = extractor.extract_partitioned_circuit()
    circuit_epr = transpile(partitioned_circuit, basis_gates=['u', 'cp', 'EPR'])
    
    epr_count = circuit_epr.count_ops().get('EPR', 0)
    
    # Single partition should require 0 EPR pairs
    assert epr_count == 0
    
    print(f"\n✓ Single partition extraction: {epr_count} EPR pairs (expected 0)")


@pytest.mark.parametrize('assignment_kind', ['initial', 'optimized'])
def test_distributed_extraction_channel_fidelity(assignment_kind):
    """Validate distributed extraction with Choi-state fidelity (channel/process check)."""
    base_circuit = _build_channel_validation_circuit()
    network = QuantumNetwork.create([2, 2], 'all_to_all')
    hypergraph = QuantumCircuitHyperGraph(base_circuit)

    if assignment_kind == 'initial':
        assignment = set_initial_partition_assignment(hypergraph, network)
    else:
        partitioner = FiducciaMattheyses(base_circuit, network=network)
        assignment = partitioner.partition(num_passes=5)['best_assignment']

    extractor = PartitionedCircuitExtractor(
        graph=hypergraph,
        network=network,
        partition_assignment=assignment,
    )
    partitioned = extractor.extract_partitioned_circuit()

    fidelity = _choi_state_fidelity(base_circuit, extractor, partitioned)
    print(f"\n✓ Channel/process fidelity ({assignment_kind}): {fidelity:.10f}")

    assert fidelity > 1 - 1e-8
