"""
Test suite for hypergraph coarsening
"""

import pytest
from bosonic_converters import CircuitConverters
from qiskit import transpile
from disqco import QuantumCircuitHyperGraph
from disqco.coarsening import HypergraphCoarsener
from disqco.circuits.cp_fraction import cp_fraction


@pytest.fixture
def test_circuit():
    """Create a cp_fraction circuit with 16 qubits as a bosonic Circuit."""
    circuit = cp_fraction(num_qubits=16, depth=16, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return CircuitConverters.from_qiskit(circuit)


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from the test circuit"""
    hypergraph = QuantumCircuitHyperGraph(test_circuit, group_gates=True)
    return hypergraph


def test_hypergraph_coarsener_import():
    """Test that HypergraphCoarsener can be imported from disqco.coarsening"""
    from disqco.coarsening import HypergraphCoarsener
    assert HypergraphCoarsener is not None


def test_hypergraph_coarsener_instantiation():
    """Test creating an instance of HypergraphCoarsener"""
    coarsener = HypergraphCoarsener()
    assert coarsener is not None
    assert hasattr(coarsener, 'coarsen_recursive_batches_mapped')
    assert hasattr(coarsener, 'coarsen_full')
    assert hasattr(coarsener, 'coarsen_blocks')


def test_hypergraph_coarsen_recursive(test_hypergraph):
    """Test recursive coarsening on a hypergraph"""
    coarsener = HypergraphCoarsener()
    
    graph_list, mapping_list = coarsener.coarsen_recursive_batches_mapped(
        hypergraph=test_hypergraph
    )
    
    assert len(graph_list) > 0
    assert len(mapping_list) > 0
    assert len(graph_list) == len(mapping_list)
    
    # Verify the graphs are hypergraph objects
    assert all(hasattr(g, 'nodes') for g in graph_list)
    print(f"\nRecursive coarsening created {len(graph_list)} levels")


def test_hypergraph_coarsen_full(test_hypergraph):
    """Test full window-based coarsening on a hypergraph"""
    coarsener = HypergraphCoarsener()
    
    graph_list, mapping_list = coarsener.coarsen_full(
        hypergraph=test_hypergraph,
        num_levels=3
    )
    
    assert len(graph_list) > 0
    assert len(mapping_list) > 0
    # With num_levels=3, we should have at most 4 graphs (original + 3 levels)
    assert len(graph_list) <= 4
    print(f"\nFull coarsening with num_levels=3 created {len(graph_list)} levels")


def test_hypergraph_coarsen_blocks(test_hypergraph):
    """Test block-based coarsening on a hypergraph"""
    coarsener = HypergraphCoarsener()
    
    # coarsen_blocks requires num_blocks parameter
    graph_list, mapping_list = coarsener.coarsen_blocks(
        hypergraph=test_hypergraph,
        num_blocks=4,
        block_size=2
    )
    
    assert len(graph_list) > 0
    assert len(mapping_list) > 0
    print(f"\nBlock coarsening with num_blocks=4, block_size=2 created {len(graph_list)} levels")


def test_hypergraph_coarsener_in_star_import():
    """Test that HypergraphCoarsener is in __all__ for star imports"""
    from disqco import coarsening
    assert 'HypergraphCoarsener' in coarsening.__all__
