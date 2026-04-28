"""
Test suite for quantum circuit partitioners
"""

import pytest
from bosonic_converters import CircuitConverters
from qiskit import transpile
from disqco import QuantumNetwork
from disqco.parti import FiducciaMattheyses, GeneticPartitioner, FGPPartitioner
from disqco.circuits.cp_fraction import cp_fraction


@pytest.fixture
def test_circuit():
    """Create a cp_fraction circuit with 32 qubits, fraction 0.5, depth 32 as a bosonic Circuit."""
    circuit = cp_fraction(num_qubits=32, depth=32, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return CircuitConverters.from_qiskit(circuit)


@pytest.fixture
def all_to_all_network():
    """Create an all-to-all network with 4 partitions of 8 qubits each"""
    qpu_sizes = {0 : 9, 1 : 9, 2 : 9, 3 : 9}
    # No qpu_connectivity specified = all-to-all
    network = QuantumNetwork(qpu_sizes)
    return network


def test_fiduccia_mattheyses_instantiation(test_circuit, all_to_all_network):
    """Test creating an instance of FiducciaMattheyses partitioner"""
    partitioner = FiducciaMattheyses(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    assert partitioner is not None
    assert partitioner.num_qubits == 32
    assert partitioner.num_partitions == 4
    assert partitioner.hypergraph is not None


def test_genetic_partitioner_instantiation(test_circuit, all_to_all_network):
    """Test creating an instance of GeneticPartitioner"""
    partitioner = GeneticPartitioner(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    assert partitioner is not None
    assert partitioner.circuit.qubits() == 32
    assert len(partitioner.network.qpu_sizes) == 4


def test_fgp_partitioner_instantiation(test_circuit, all_to_all_network):
    """Test creating an instance of FGPPartitioner"""
    partitioner = FGPPartitioner(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    assert partitioner is not None
    assert partitioner.num_qubits == 32
    assert partitioner.num_qpus == 4
    assert not partitioner.network.hetero  # Verify it's all-to-all


def test_all_partitioners_have_partition_method(test_circuit, all_to_all_network):
    """Test that all partitioners have a partition method"""
    partitioners = [
        FiducciaMattheyses(test_circuit, all_to_all_network),
        GeneticPartitioner(test_circuit, all_to_all_network),
        FGPPartitioner(test_circuit, all_to_all_network)
    ]
    
    for partitioner in partitioners:
        assert hasattr(partitioner, 'partition')
        assert callable(partitioner.partition)


def test_fiduccia_mattheyses_partition(test_circuit, all_to_all_network):
    """Test running FiducciaMattheyses partition method"""
    partitioner = FiducciaMattheyses(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    results = partitioner.partition(num_passes=5)
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    assert results['best_assignment'] is not None
    print(f"\nFiducciaMattheyses - Cost: {results['best_cost']}")


def test_genetic_partitioner_partition(test_circuit, all_to_all_network):
    """Test running GeneticPartitioner partition method"""
    partitioner = GeneticPartitioner(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    results = partitioner.partition(
        population_size=10,
        generations=5,
        mutation_rate=0.1
    )
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    assert results['best_assignment'] is not None
    print(f"\nGeneticPartitioner - Cost: {results['best_cost']}")


def test_fgp_partitioner_partition(test_circuit, all_to_all_network):
    """Test running FGPPartitioner partition method"""
    partitioner = FGPPartitioner(
        circuit=test_circuit,
        network=all_to_all_network
    )
    
    results = partitioner.partition()
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert 'mapping' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    assert results['best_assignment'] is not None
    print(f"\nFGPPartitioner - Cost: {results['best_cost']}, Assignment shape: {results['best_assignment'].shape}")


def test_factory_create_fiduccia_mattheyses(test_circuit, all_to_all_network):
    """Test creating FiducciaMattheyses via factory method"""
    from disqco.parti.partitioner import QuantumCircuitPartitioner
    
    # Test various string specifiers
    for partitioner_type in ['fm', 'FM', 'Fiduccia', 'FiducciaMattheyses']:
        partitioner = QuantumCircuitPartitioner.create(
            partitioner_type, 
            test_circuit, 
            all_to_all_network
        )
        assert isinstance(partitioner, FiducciaMattheyses)
        assert partitioner.num_qubits == 32


def test_factory_create_genetic(test_circuit, all_to_all_network):
    """Test creating GeneticPartitioner via factory method"""
    from disqco.parti.partitioner import QuantumCircuitPartitioner
    
    # Test various string specifiers
    for partitioner_type in ['genetic', 'GENETIC', 'ga', 'GA']:
        partitioner = QuantumCircuitPartitioner.create(
            partitioner_type, 
            test_circuit, 
            all_to_all_network
        )
        assert isinstance(partitioner, GeneticPartitioner)
        assert partitioner.circuit.qubits() == 32


def test_factory_create_fgp(test_circuit, all_to_all_network):
    """Test creating FGPPartitioner via factory method"""
    from disqco.parti.partitioner import QuantumCircuitPartitioner
    
    # Test various string specifiers
    for partitioner_type in ['fgp', 'FGP']:
        partitioner = QuantumCircuitPartitioner.create(
            partitioner_type, 
            test_circuit, 
            all_to_all_network
        )
        assert isinstance(partitioner, FGPPartitioner)
        assert partitioner.num_qubits == 32


def test_factory_invalid_type(test_circuit, all_to_all_network):
    """Test that factory raises ValueError for invalid partitioner type"""
    from disqco.parti.partitioner import QuantumCircuitPartitioner
    
    with pytest.raises(ValueError, match="Unknown partitioner type"):
        QuantumCircuitPartitioner.create(
            'invalid_type', 
            test_circuit, 
            all_to_all_network
        )


def test_factory_with_kwargs(test_circuit, all_to_all_network):
    """Test that factory correctly passes kwargs to partitioner constructors"""
    from disqco.parti.partitioner import QuantumCircuitPartitioner
    
    # Test with FGP which has specific kwargs
    partitioner = QuantumCircuitPartitioner.create(
        'fgp', 
        test_circuit, 
        all_to_all_network,
        remove_singles=False,
        choose_initial=True
    )
    
    assert isinstance(partitioner, FGPPartitioner)
    assert partitioner.remove_singles == False
    assert partitioner.choose_initial == True
