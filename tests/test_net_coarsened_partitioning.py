"""
Test suite for network coarsened partitioning.

Network coarsened partitioning combines:
1. Network coarsening - recursively coarsen the network topology
2. Network cutting - divide coarsened network into sub-networks 
3. Subgraph partitioning - create subgraphs with dummy nodes
4. Multilevel FM - apply FM algorithm with temporal coarsening for each subgraph
5. Solution stitching - combine partitioned subgraphs back into complete solution
"""

import pytest
import numpy as np
from bosonic_converters import CircuitConverters
from qiskit import QuantumCircuit, transpile

from disqco import QuantumNetwork
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
from disqco.coarsening import HypergraphCoarsener, NetworkCoarsener
from disqco.parti.FM.net_coarsened_partitioning import (
    run_full_net_coarsened_FM, 
    check_assignment_validity
)


@pytest.fixture
def linear_network():
    """Create a linear network topology for testing"""
    num_qpus = 8
    qpu_capacity = 5
    qpu_sizes = [qpu_capacity] * num_qpus
    
    # Use factory method to create linear network
    return QuantumNetwork.create(qpu_sizes, 'linear')


@pytest.fixture
def grid_network():
    """Create a grid network topology for testing"""
    num_qpus = 16
    qpu_capacity = 4
    qpu_sizes = [qpu_capacity] * num_qpus
    
    # Use factory method to create grid network
    return QuantumNetwork.create(qpu_sizes, 'grid')


@pytest.fixture
def test_circuit_small():
    """Create a small test bosonic Circuit."""
    num_qubits = 32
    circuit = cp_fraction(num_qubits, depth=32, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return CircuitConverters.from_qiskit(circuit)


@pytest.fixture
def test_circuit_large():
    """Create a larger test bosonic Circuit."""
    num_qubits = 64
    circuit = cp_fraction(num_qubits, depth=64, fraction=0.5, seed=123)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return CircuitConverters.from_qiskit(circuit)


def test_net_coarsened_partitioning_direct_call(test_circuit_small, linear_network):
    """Test net coarsened partitioning by directly calling run_full_net_coarsened_FM"""
    from disqco import QuantumCircuitHyperGraph
    
    hypergraph = QuantumCircuitHyperGraph(test_circuit_small)
    num_qubits = test_circuit_small.qubits()
    
    results = run_full_net_coarsened_FM(
        hypergraph=hypergraph,
        num_qubits=num_qubits,
        network=linear_network,
        coarsening_factor=2,
        passes_per_level=5,
        use_multiprocessing=False,
        ML_internal_level_limit=50
    )
    
    # Validate results structure
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    
    # Validate assignment
    final_assignment = results['best_assignment']
    is_valid = check_assignment_validity(
        final_assignment, 
        linear_network.qpu_sizes, 
        hypergraph
    )
    assert is_valid, "Final assignment violates capacity constraints"
    
    print(f"\nDirect call net coarsened partitioning - Cost: {results['best_cost']}")


def test_net_coarsened_partitioning_via_partitioner(test_circuit_small, linear_network):
    """Test net coarsened partitioning via FiducciaMattheyses partitioner class"""
    partitioner = FiducciaMattheyses(test_circuit_small, network=linear_network)
    
    # Get hypergraph coarsener
    hypergraph_coarsener = HypergraphCoarsener().coarsen_recursive_subgraph_batch
    
    results = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        hypergraph_coarsener=hypergraph_coarsener,
        passes_per_level=5,
        use_multiprocessing=False
    )
    
    # Validate results
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    
    # Validate assignment
    final_assignment = results['best_assignment']
    is_valid = check_assignment_validity(
        final_assignment,
        linear_network.qpu_sizes,
        partitioner.hypergraph
    )
    assert is_valid, "Final assignment violates capacity constraints"
    
    print(f"\nPartitioner class net coarsened partitioning - Cost: {results['best_cost']}")


def test_net_coarsened_with_different_coarsening_factors(test_circuit_small, linear_network):
    """Test net coarsened partitioning with different network coarsening factors"""
    partitioner = FiducciaMattheyses(test_circuit_small, network=linear_network)
    
    coarsening_factors = [2, 4]
    costs = []
    
    for cf in coarsening_factors:
        results = partitioner.net_coarsened_partition(
            coarsening_factor=cf,
            passes_per_level=3,
            use_multiprocessing=False,
            ML_internal_level_limit=50
        )
        
        assert results['best_cost'] >= 0
        costs.append(results['best_cost'])
        
        # Validate assignment
        is_valid = check_assignment_validity(
            results['best_assignment'],
            linear_network.qpu_sizes,
            partitioner.hypergraph
        )
        assert is_valid, f"Assignment invalid for coarsening_factor={cf}"
    
    print(f"\nCoarsening factor 2: Cost = {costs[0]}")
    print(f"Coarsening factor 4: Cost = {costs[1]}")


def test_net_coarsened_with_grid_network(test_circuit_small, grid_network):
    """Test net coarsened partitioning with a grid network topology"""
    partitioner = FiducciaMattheyses(test_circuit_small, network=grid_network)
    
    results = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        passes_per_level=5,
        use_multiprocessing=False,
        ML_internal_level_limit=50
    )
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert results['best_cost'] >= 0
    
    # Validate assignment
    is_valid = check_assignment_validity(
        results['best_assignment'],
        grid_network.qpu_sizes,
        partitioner.hypergraph
    )
    assert is_valid, "Assignment invalid for grid network"
    
    print(f"\nGrid network (16 QPUs) - Cost: {results['best_cost']}")


def test_net_coarsened_with_different_hypergraph_coarseners(test_circuit_small, linear_network):
    """Test net coarsened partitioning with different hypergraph coarsening strategies"""
    partitioner = FiducciaMattheyses(test_circuit_small, network=linear_network)
    coarsener_instance = HypergraphCoarsener()
    
    # Test with recursive subgraph batch coarsening (default)
    results_recursive = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        hypergraph_coarsener=coarsener_instance.coarsen_recursive_subgraph_batch,
        passes_per_level=3,
        use_multiprocessing=False
    )
    
    # Test with recursive batches mapped coarsening
    results_mapped = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        hypergraph_coarsener=coarsener_instance.coarsen_recursive_batches_mapped,
        passes_per_level=3,
        use_multiprocessing=False
    )
    
    # Both should produce valid results
    assert results_recursive['best_cost'] >= 0
    assert results_mapped['best_cost'] >= 0
    
    print(f"\nRecursive subgraph batch coarsening - Cost: {results_recursive['best_cost']}")
    print(f"Recursive batches mapped coarsening - Cost: {results_mapped['best_cost']}")


def test_network_coarsening_hierarchy(linear_network):
    """Test that network coarsening creates proper hierarchy"""
    net_coarsener = NetworkCoarsener(linear_network)
    
    # Coarsen with factor of 2
    net_coarsener.coarsen_network_recursive(l=2)
    
    # Should create multiple levels
    assert len(net_coarsener.network_coarse_list) > 1
    
    # Each level should have fewer QPUs than the previous
    for i in range(len(net_coarsener.network_coarse_list) - 1):
        current_level = net_coarsener.network_coarse_list[i]
        next_level = net_coarsener.network_coarse_list[i + 1]
        
        current_qpus = len(current_level.qpu_sizes)
        next_qpus = len(next_level.qpu_sizes)
        
        assert next_qpus <= current_qpus, f"Level {i+1} has more QPUs than level {i}"
    
    print(f"\nNetwork coarsening hierarchy: {[len(net.qpu_sizes) for net in net_coarsener.network_coarse_list]} QPUs")


def test_net_coarsened_large_circuit(test_circuit_large, linear_network):
    """Test net coarsened partitioning with a larger circuit"""
    # Create network with more capacity using factory method
    large_network = QuantumNetwork.create([10] * 8, 'linear')
    
    partitioner = FiducciaMattheyses(test_circuit_large, network=large_network)
    
    results = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        passes_per_level=3,
        use_multiprocessing=False,
        ML_internal_level_limit=30
    )
    
    assert results['best_cost'] >= 0
    
    # Validate assignment
    is_valid = check_assignment_validity(
        results['best_assignment'],
        large_network.qpu_sizes,
        partitioner.hypergraph
    )
    assert is_valid, "Assignment invalid for large circuit"
    
    print(f"\nLarge circuit (64 qubits) - Cost: {results['best_cost']}")


def test_net_coarsened_vs_standard_multilevel(test_circuit_small, linear_network):
    """Compare net coarsened partitioning with standard multilevel partitioning"""
    partitioner = FiducciaMattheyses(test_circuit_small, network=linear_network)
    
    # Net coarsened partitioning
    net_coarsened_results = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        passes_per_level=5,
        use_multiprocessing=False,
        ML_internal_level_limit=50
    )
    
    # Standard multilevel partitioning
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    multilevel_results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=5
    )
    
    # Both should produce valid results
    assert net_coarsened_results['best_cost'] >= 0
    assert multilevel_results['best_cost'] >= 0
    
    print(f"\nNet coarsened partitioning - Cost: {net_coarsened_results['best_cost']}")
    print(f"Standard multilevel partitioning - Cost: {multilevel_results['best_cost']}")
    print(f"Difference: {abs(net_coarsened_results['best_cost'] - multilevel_results['best_cost'])}")


def test_net_coarsened_vs_standard_multilevel_large_circuit(test_circuit_large):
    """Compare net coarsened partitioning with standard multilevel partitioning for large circuit"""
    # Create a network with sufficient capacity for 64 qubits using factory method
    large_network = QuantumNetwork.create([10] * 8, 'linear')
    
    partitioner = FiducciaMattheyses(test_circuit_large, network=large_network)
    
    # Net coarsened partitioning
    net_coarsened_results = partitioner.net_coarsened_partition(
        coarsening_factor=2,
        passes_per_level=5,
        use_multiprocessing=False,
        ML_internal_level_limit=30
    )
    
    # Standard multilevel partitioning
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    multilevel_results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=5
    )
    
    # Both should produce valid results
    assert net_coarsened_results['best_cost'] >= 0
    assert multilevel_results['best_cost'] >= 0
    
    # Validate both assignments
    is_valid_nc = check_assignment_validity(
        net_coarsened_results['best_assignment'],
        large_network.qpu_sizes,
        partitioner.hypergraph
    )
    is_valid_ml = check_assignment_validity(
        multilevel_results['best_assignment'],
        large_network.qpu_sizes,
        partitioner.hypergraph
    )
    assert is_valid_nc, "Net coarsened assignment invalid"
    assert is_valid_ml, "Multilevel assignment invalid"
    
    print(f"\nLarge circuit (64 qubits) comparison:")
    print(f"  Net coarsened partitioning - Cost: {net_coarsened_results['best_cost']}")
    print(f"  Standard multilevel partitioning - Cost: {multilevel_results['best_cost']}")
    print(f"  Difference: {abs(net_coarsened_results['best_cost'] - multilevel_results['best_cost'])}")
    
    # Net coarsened should be competitive with standard multilevel
    # (both methods should find reasonable solutions)
    assert net_coarsened_results['best_cost'] > 0
    assert multilevel_results['best_cost'] > 0
