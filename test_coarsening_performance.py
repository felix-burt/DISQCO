#!/usr/bin/env python3
"""
Performance testing script for the optimized coarsening functions.
"""

import time
import numpy as np
from src.disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from src.disqco.graphs.coarsening.coarsener import HypergraphCoarsener

def create_test_grid_graph(width=16, height=16, depth=8):
    """Create a test grid graph for performance testing."""
    from qiskit import QuantumCircuit
    
    # Create a simple quantum circuit with a grid structure
    num_qubits = width * height
    circuit = QuantumCircuit(num_qubits)
    
    # Add some gates to create a reasonable hypergraph structure
    for t in range(depth):
        # Add horizontal connections
        for row in range(height):
            for col in range(width - 1):
                q1 = row * width + col
                q2 = row * width + col + 1
                circuit.cx(q1, q2)
        
        # Add vertical connections
        for row in range(height - 1):
            for col in range(width):
                q1 = row * width + col
                q2 = (row + 1) * width + col
                circuit.cx(q1, q2)
    
    return circuit

def benchmark_coarsening():
    """Benchmark the coarsening performance on grid graphs."""
    print("Setting up test circuit...")
    
    # Test with increasingly large grid sizes
    test_sizes = [
        (8, 8, 4),   # 64 qubits, 4 layers
        (16, 16, 4), # 256 qubits, 4 layers  
        (32, 32, 4), # 1024 qubits, 4 layers (might be too large for testing)
    ]
    
    coarsener = HypergraphCoarsener()
    
    for width, height, depth in test_sizes:
        num_qubits = width * height
        print(f"\nTesting grid {width}x{height} ({num_qubits} qubits, {depth} layers)...")
        
        try:
            # Create test circuit
            circuit = create_test_grid_graph(width, height, depth)
            
            # Convert to hypergraph
            print("  Converting to hypergraph...")
            start_time = time.time()
            hypergraph = QuantumCircuitHyperGraph(circuit=circuit)
            setup_time = time.time() - start_time
            print(f"  Hypergraph setup: {setup_time:.3f}s")
            print(f"  Nodes: {len(hypergraph.nodes)}, Hyperedges: {len(hypergraph.hyperedges)}")
            
            # Test the optimized coarsening
            print("  Testing optimized coarsening...")
            start_time = time.time()
            H_list, mapping_list = coarsener.coarsen_recursive_batches_mapped(hypergraph)
            optimized_time = time.time() - start_time
            print(f"  Optimized coarsening: {optimized_time:.3f}s")
            print(f"  Coarsening levels: {len(H_list)}")
            
            # Test the regular coarsening for comparison (if available)
            if hasattr(coarsener, 'coarsen_recursive_mapped'):
                print("  Testing regular coarsening...")
                start_time = time.time()
                H_list_reg, mapping_list_reg = coarsener.coarsen_recursive_mapped(hypergraph, None)
                regular_time = time.time() - start_time
                print(f"  Regular coarsening: {regular_time:.3f}s")
                print(f"  Speedup: {regular_time/optimized_time:.2f}x")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    benchmark_coarsening()
