#!/usr/bin/env python3
"""
Test script to verify that node labels show true coordinates instead of sub-node coordinates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from src.disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from src.disqco.drawing.tikz_drawing import hypergraph_to_tikz_subgraph

def test_true_labeling():
    """Test that nodes are labeled with their true coordinates, not sub-node coordinates."""
    print("=== Testing True Node Labeling ===")
    
    # Create a simple quantum circuit
    circuit = QuantumCircuit(4, 4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.measure_all()
    
    # Create hypergraph
    H = QuantumCircuitHyperGraph(circuit, map_circuit=False)
    
    # Print nodes to see what we're working with
    print("Hypergraph nodes:")
    for node in list(H.nodes)[:10]:  # Show first 10 nodes
        print(f"  {node}")
    
    # Simple assignment: all qubits to partition 0 for first layer, partition 1 for second layer
    num_qubits = 4
    depth = H.depth
    assignment = []
    for t in range(depth):
        if t % 2 == 0:
            assignment.append([0, 0, 1, 1])  # qubits 0,1 -> partition 0, qubits 2,3 -> partition 1
        else:
            assignment.append([1, 1, 0, 0])  # qubits 0,1 -> partition 1, qubits 2,3 -> partition 0
    
    # Assignment map: maps original (q,t) -> subgraph (q',t')
    # For this test, let's say the subgraph remaps:
    # Original (0,0) -> Sub (0,0)
    # Original (1,0) -> Sub (1,0) 
    # Original (2,0) -> Sub (0,1)  # This is the key test - original (2,0) becomes sub (0,1)
    # Original (3,0) -> Sub (1,1)
    assignment_map = {}
    sub_counter = 0
    for t in range(min(depth, 2)):  # Just test first 2 layers
        for q in range(num_qubits):
            if (q, t) in [(0, 0), (1, 0), (2, 0), (3, 0)]:  # Original nodes
                if (q, t) == (2, 0):
                    assignment_map[(q, t)] = (0, 1)  # Remap to different sub-coordinates
                elif (q, t) == (3, 0):
                    assignment_map[(q, t)] = (1, 1)  # Remap to different sub-coordinates  
                else:
                    assignment_map[(q, t)] = (q, t)  # Keep same coordinates
    
    # Node map (for this test, assume 1-to-1 mapping of partitions)
    node_map = {0: 0, 1: 1}
    
    # QPU info
    qpu_info = [2, 2]  # 2 qubits per partition
    
    print(f"\nAssignment map:")
    for orig, sub in assignment_map.items():
        print(f"  Original {orig} -> Sub {sub}")
    
    # Generate TikZ code
    tikz_code = hypergraph_to_tikz_subgraph(
        H=H,
        assignment=assignment,
        qpu_info=qpu_info,
        assignment_map=assignment_map,
        node_map=node_map,
        show_labels=True,
        save=True,
        path="/Users/ftb123/MLQCP_FM/test_labeling_output.tex"
    )
    
    print(f"\nGenerated TikZ code length: {len(tikz_code)} characters")
    
    # Check if the code contains true labels
    # Look for labels that show original coordinates, not sub-coordinates
    print("\nChecking for label content in TikZ code:")
    lines = tikz_code.split('\n')
    label_lines = [line for line in lines if 'label=' in line and '(' in line and ')' in line]
    
    print("Found label lines:")
    for line in label_lines[:5]:  # Show first 5 label lines
        print(f"  {line.strip()}")
    
    # Specifically check if we see original coordinates in labels
    has_original_coords = False
    for line in label_lines:
        if '(2,0)' in line:  # This should show as (2,0) not (0,1)
            has_original_coords = True
            print(f"\n✓ Found original coordinate (2,0) in label: {line.strip()}")
        if '(3,0)' in line:  # This should show as (3,0) not (1,1)  
            has_original_coords = True
            print(f"✓ Found original coordinate (3,0) in label: {line.strip()}")
    
    if has_original_coords:
        print("\n✓ SUCCESS: Labels are showing true original coordinates!")
    else:
        print("\n✗ WARNING: Could not find expected original coordinates in labels")
    
    print(f"\nTikZ output saved to: /Users/ftb123/MLQCP_FM/test_labeling_output.tex")

if __name__ == "__main__":
    print("Testing True Node Labeling in Subgraph Drawing")
    print("=" * 60)
    
    try:
        test_true_labeling()
        print("\n" + "=" * 60)
        print("Test completed!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
