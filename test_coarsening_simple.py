#!/usr/bin/env python3
"""
Quick test script for the optimized coarsening functions.
"""

import time
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from qiskit import QuantumCircuit
    from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
    from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the correct directory and dependencies are installed")
    sys.exit(1)

def create_small_test_circuit():
    """Create a small test circuit for verification."""
    circuit = QuantumCircuit(4)
    
    # Add some gates to create structure
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.cx(0, 2)
    
    return circuit

def test_coarsening():
    """Test the coarsening functions."""
    print("Creating test circuit...")
    circuit = create_small_test_circuit()
    
    print("Converting to hypergraph...")
    start_time = time.time()
    hypergraph = QuantumCircuitHyperGraph(circuit=circuit)
    setup_time = time.time() - start_time
    print(f"Hypergraph setup: {setup_time:.3f}s")
    print(f"Initial - Nodes: {len(hypergraph.nodes)}, Hyperedges: {len(hypergraph.hyperedges)}")
    
    coarsener = HypergraphCoarsener()
    
    # Test the optimized coarsening
    print("\nTesting optimized coarsening...")
    start_time = time.time()
    H_list, mapping_list = coarsener.coarsen_recursive_batches_mapped(hypergraph)
    optimized_time = time.time() - start_time
    print(f"Optimized coarsening: {optimized_time:.3f}s")
    print(f"Coarsening levels: {len(H_list)}")
    
    # Print results for each level
    for i, (H, mapping) in enumerate(zip(H_list, mapping_list)):
        print(f"Level {i}: Nodes: {len(H.nodes)}, Hyperedges: {len(H.hyperedges)}, Layers: {len(mapping)}")
    
    # Verify that coarsening actually happened
    if len(H_list) > 1:
        initial_nodes = len(H_list[0].nodes)
        final_nodes = len(H_list[-1].nodes)
        if final_nodes < initial_nodes:
            print(f"\n✓ Coarsening successful: {initial_nodes} → {final_nodes} nodes")
        else:
            print(f"\n✗ Coarsening failed: nodes didn't decrease ({initial_nodes} → {final_nodes})")
    else:
        print("\n✗ No coarsening levels generated")
    
    # Test original method for comparison if available
    if hasattr(coarsener, 'coarsen_recursive_mapped'):
        print("\nTesting regular coarsening for comparison...")
        start_time = time.time()
        H_list_reg, mapping_list_reg = coarsener.coarsen_recursive_mapped(hypergraph, None)
        regular_time = time.time() - start_time
        print(f"Regular coarsening: {regular_time:.3f}s")
        
        if optimized_time > 0:
            speedup = regular_time / optimized_time
            print(f"Speedup: {speedup:.2f}x")
        
        # Compare results
        if len(H_list) == len(H_list_reg):
            print("✓ Both methods produced same number of levels")
        else:
            print(f"✗ Different levels: optimized={len(H_list)}, regular={len(H_list_reg)}")

if __name__ == "__main__":
    test_coarsening()
