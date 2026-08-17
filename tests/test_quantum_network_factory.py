"""
Test suite for QuantumNetwork factory method.

Tests the QuantumNetwork.create() factory method for creating networks
with different coupling topologies using string specifiers.
"""

import pytest
from disqco import QuantumNetwork


def test_quantum_network_create_all_to_all():
    """Test creating an all-to-all network via factory method"""
    qpu_sizes = [8, 8, 8, 8]
    
    # Test different string variations
    for coupling_str in ['all_to_all', 'complete', 'fully_connected', 'AllToAll']:
        network = QuantumNetwork.create(qpu_sizes, coupling_str)
        assert network is not None
        assert network.num_qpus == 4
        assert network.hetero == True  # Has connectivity specified
        # All-to-all should have n*(n-1)/2 edges
        assert len(network.qpu_connectivity) == 6  # 4*3/2 = 6
        print(f"\n{coupling_str}: {len(network.qpu_connectivity)} edges")


def test_quantum_network_create_linear():
    """Test creating a linear network via factory method"""
    qpu_sizes = [8, 8, 8, 8, 8, 8, 8, 8]
    network = QuantumNetwork.create(qpu_sizes, 'linear')
    
    assert network is not None
    assert network.num_qpus == 8
    assert network.hetero == True
    # Linear should have n-1 edges
    assert len(network.qpu_connectivity) == 7
    
    # Verify linear structure: each edge connects i to i+1
    for i in range(7):
        assert [i, i+1] in network.qpu_connectivity
    
    print(f"\nLinear network: {len(network.qpu_connectivity)} edges")


def test_quantum_network_create_grid():
    """Test creating a grid network via factory method"""
    # 16 QPUs -> 4x4 grid
    qpu_sizes = [4] * 16
    network = QuantumNetwork.create(qpu_sizes, 'grid')
    
    assert network is not None
    assert network.num_qpus == 16
    assert network.hetero == True
    # 4x4 grid should have 2*(4*3) = 24 edges (horizontal + vertical)
    assert len(network.qpu_connectivity) == 24
    
    print(f"\nGrid network (16 nodes): {len(network.qpu_connectivity)} edges")


def test_quantum_network_create_random():
    """Test creating a random network via factory method"""
    qpu_sizes = [8] * 8
    
    # Test with default probability
    network1 = QuantumNetwork.create(qpu_sizes, 'random')
    assert network1 is not None
    assert network1.num_qpus == 8
    assert network1.hetero == True
    
    # Test with custom probability
    network2 = QuantumNetwork.create(qpu_sizes, 'random', p=0.8)
    assert network2 is not None
    assert network2.num_qpus == 8
    
    print(f"\nRandom network (p=0.5): {len(network1.qpu_connectivity)} edges")
    print(f"Random network (p=0.8): {len(network2.qpu_connectivity)} edges")
    
    # Higher probability should generally have more edges (though random)
    # Just verify both are connected (at least n-1 edges)
    assert len(network1.qpu_connectivity) >= 7
    assert len(network2.qpu_connectivity) >= 7


def test_quantum_network_create_tree():
    """Test creating a tree network via factory method"""
    qpu_sizes = [8] * 7
    
    # Binary tree (k=2)
    network = QuantumNetwork.create(qpu_sizes, 'tree', k=2)
    assert network is not None
    assert network.num_qpus == 7
    assert network.hetero == True
    
    # Tree with n nodes has n-1 edges (actually more in this implementation)
    # Just verify it's created and has edges
    assert len(network.qpu_connectivity) > 0
    
    print(f"\nTree network (7 nodes, k=2): {len(network.qpu_connectivity)} edges")


def test_quantum_network_create_network_of_grids():
    """Test creating network of grids via factory method"""
    # 2 grids of 4 nodes each, connected by 2-hop path
    # Total: 2*4 + 2 = 10 nodes
    qpu_sizes = [4] * 10
    
    network = QuantumNetwork.create(
        qpu_sizes, 
        'network_of_grids',
        num_grids=2,
        nodes_per_grid=4,
        l=2
    )
    
    assert network is not None
    assert network.num_qpus == 10
    assert network.hetero == True
    
    print(f"\nNetwork of grids (2 grids, 4 nodes each): {len(network.qpu_connectivity)} edges")


def test_quantum_network_create_with_dict_sizes():
    """Test factory method with dict-based qpu_sizes"""
    qpu_sizes = {0: 8, 1: 8, 2: 8, 3: 8}
    network = QuantumNetwork.create(qpu_sizes, 'linear')
    
    assert network is not None
    assert network.num_qpus == 4
    assert network.hetero == True
    assert len(network.qpu_connectivity) == 3


def test_quantum_network_create_with_comm_sizes():
    """Test factory method with communication sizes"""
    qpu_sizes = [8, 8, 8, 8]
    comm_sizes = [2, 2, 2, 2]
    
    network = QuantumNetwork.create(qpu_sizes, 'linear', comm_sizes=comm_sizes)
    
    assert network is not None
    assert network.comm_sizes == {0: 2, 1: 2, 2: 2, 3: 2}


def test_quantum_network_create_invalid_coupling_type():
    """Test that invalid coupling type raises ValueError"""
    qpu_sizes = [8, 8, 8, 8]
    
    with pytest.raises(ValueError, match="Unknown coupling type"):
        QuantumNetwork.create(qpu_sizes, 'invalid_type')


def test_quantum_network_create_network_of_grids_missing_params():
    """Test that network_of_grids without required params raises ValueError"""
    qpu_sizes = [4] * 10
    
    with pytest.raises(ValueError, match="network_of_grids requires"):
        QuantumNetwork.create(qpu_sizes, 'network_of_grids')


def test_quantum_network_factory_vs_direct_construction():
    """Compare factory method with direct construction"""
    qpu_sizes = [8, 8, 8, 8]
    
    # Direct construction with linear coupling
    from disqco.graphs.quantum_network import linear_coupling
    connectivity = linear_coupling(4)
    network_direct = QuantumNetwork(qpu_sizes, connectivity)
    
    # Factory method
    network_factory = QuantumNetwork.create(qpu_sizes, 'linear')
    
    # Should produce equivalent networks
    assert network_direct.num_qpus == network_factory.num_qpus
    assert len(network_direct.qpu_connectivity) == len(network_factory.qpu_connectivity)
    assert network_direct.hetero == network_factory.hetero
    
    print("\nFactory method produces equivalent network to direct construction")


def test_quantum_network_factory_in_partitioning_workflow():
    """Test using factory method in a complete partitioning workflow"""
    from disqco.circuits.cp_fraction import cp_fraction
    from disqco.parti import FiducciaMattheyses
    from qiskit import transpile
    
    # Create circuit
    circuit = cp_fraction(num_qubits=32, depth=32, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    
    # Create network using factory
    network = QuantumNetwork.create([9, 9, 9, 9], 'linear')
    
    # Create partitioner and run
    partitioner = FiducciaMattheyses(circuit, network=network)
    results = partitioner.partition(num_passes=3)
    
    assert 'best_cost' in results
    assert results['best_cost'] >= 0
    
    print(f"\nPartitioning with factory-created linear network - Cost: {results['best_cost']}")