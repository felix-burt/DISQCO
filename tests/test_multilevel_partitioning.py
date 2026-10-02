"""
Test suite for multilevel partitioning with FiducciaMattheyses
"""

import pytest
from qiskit import transpile
from disqco import QuantumNetwork
from disqco.parti import FiducciaMattheyses
from disqco.coarsening import HypergraphCoarsener
from disqco.circuits.cp_fraction import cp_fraction


@pytest.fixture
def test_circuit():
    """Create a cp_fraction circuit with 16 qubits"""
    circuit = cp_fraction(num_qubits=32, depth=32, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_network():
    """Create an all-to-all network with 4 QPUs"""
    qpu_sizes = {0: 9, 1: 9, 2: 9, 3: 9}
    network = QuantumNetwork(qpu_sizes)
    return network


def test_multilevel_partition_direct_call(test_circuit, test_network):
    """Test multilevel partitioning by directly calling multilevel_partition()"""
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    
    # Get coarsening function
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    
    # Call multilevel_partition directly with coarsener
    results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=5
    )
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert 'cost_list' in results
    assert 'assignment_list' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    print(f"\nDirect multilevel_partition call - Cost: {results['best_cost']}")


def test_multilevel_partition_via_partition_call(test_circuit, test_network):
    """Test multilevel partitioning by calling partition() with hypergraph_coarsener"""
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    
    # Get coarsening function
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    
    # Call partition() with hypergraph_coarsener kwarg
    results = partitioner.partition(
        hypergraph_coarsener=coarsener,
        passes_per_level=5
    )
    
    assert 'best_cost' in results
    assert 'best_assignment' in results
    assert 'cost_list' in results
    assert 'assignment_list' in results
    assert isinstance(results['best_cost'], (int, float))
    assert results['best_cost'] >= 0
    print(f"\nPartition call with hypergraph_coarsener - Cost: {results['best_cost']}")


def test_multilevel_with_different_coarseners(test_circuit, test_network):
    """Test multilevel partitioning with different coarsening strategies"""
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    coarsener_instance = HypergraphCoarsener()
    
    # Test with recursive coarsening
    coarsener_recursive = coarsener_instance.coarsen_recursive_batches_mapped
    results_recursive = partitioner.multilevel_partition(
        coarsener=coarsener_recursive,
        passes_per_level=3
    )
    
    # Test with full (window-based) coarsening
    coarsener_full = coarsener_instance.coarsen_full
    results_full = partitioner.multilevel_partition(
        coarsener=coarsener_full,
        passes_per_level=3,
        num_levels=3
    )
    
    # Test with block coarsening
    coarsener_blocks = coarsener_instance.coarsen_blocks
    results_blocks = partitioner.multilevel_partition(
        coarsener=coarsener_blocks,
        passes_per_level=3,
        num_blocks=4,
        block_size=2
    )
    
    # All should produce valid results
    assert results_recursive['best_cost'] >= 0
    assert results_full['best_cost'] >= 0
    assert results_blocks['best_cost'] >= 0
    
    print(f"\nRecursive coarsening - Cost: {results_recursive['best_cost']}")
    print(f"Full coarsening - Cost: {results_full['best_cost']}")
    print(f"Block coarsening - Cost: {results_blocks['best_cost']}")


def test_multilevel_partition_returns_cost_list(test_circuit, test_network):
    """Test that multilevel partition returns cost progression"""
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped
    
    results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=3
    )
    
    # Should have cost_list tracking improvement
    assert 'cost_list' in results
    assert len(results['cost_list']) > 0
    assert all(isinstance(c, (int, float)) for c in results['cost_list'])
    
    # Should have assignment_list
    assert 'assignment_list' in results
    assert len(results['assignment_list']) == len(results['cost_list'])
    
    print(f"\nCost progression: {results['cost_list']}")


def test_level_limit_is_honored(test_circuit, test_network):
    """Regression test: level_limit must cap the number of levels processed.

    Before the fix, level_limit was never copied into coarsener_kwargs so the
    lookup `coarsener_kwargs.get('level_limit', 100)` always returned 100,
    making the argument a no-op.  After the fix, passing level_limit=2 must
    result in exactly 2 entries in assignment_list / cost_list.
    """
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    coarsener = HypergraphCoarsener().coarsen_recursive_batches_mapped

    results = partitioner.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=3,
        level_limit=2,
    )

    assert len(results['assignment_list']) == 2, (
        f"Expected 2 levels (level_limit=2) but got {len(results['assignment_list'])}"
    )
    assert len(results['cost_list']) == 2, (
        f"Expected 2 cost entries (level_limit=2) but got {len(results['cost_list'])}"
    )

    # Confirm an unrestricted run produces more than 2 levels, proving the
    # coarsener actually generates >2 levels for this circuit.
    partitioner2 = FiducciaMattheyses(test_circuit, test_network)
    results_full = partitioner2.multilevel_partition(
        coarsener=coarsener,
        passes_per_level=3,
    )
    assert len(results_full['assignment_list']) > 2, (
        "Unrestricted run produced <=2 levels; level_limit=2 test is not meaningful"
    )

    print(f"\nlevel_limit=2 → {len(results['assignment_list'])} level(s) processed")
    print(f"Unrestricted   → {len(results_full['assignment_list'])} level(s) processed")


def test_single_level_partition_for_comparison(test_circuit, test_network):
    """Test single-level (no coarsening) partition for comparison"""
    partitioner = FiducciaMattheyses(test_circuit, test_network)
    
    # Call partition without coarsener (single-level)
    results_single = partitioner.partition(num_passes=10)
    
    assert 'best_cost' in results_single
    assert results_single['best_cost'] >= 0
    
    print(f"\nSingle-level partition - Cost: {results_single['best_cost']}")
