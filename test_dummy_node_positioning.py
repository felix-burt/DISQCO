#!/usr/bin/env python3
"""
Test script to verify that dummy nodes are properly shifted up and have no labels
in the updated subgraph drawing functionality.
"""

import sys
import os
sys.path.append('src')

from qiskit import QuantumCircuit
from src.disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from src.disqco.drawing.tikz_drawing import hypergraph_to_tikz_subgraph

def test_dummy_node_positioning():
    """Test that dummy nodes are shifted up and labels are removed."""
    
    # Create a simple test circuit
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    circuit.rz(0.5, 0)
    circuit.ry(0.3, 1)
    
    # Create hypergraph from circuit
    H = QuantumCircuitHyperGraph(circuit=circuit, map_circuit=False)
    H.num_qubits = 2
    H.depth = 3
    
    # Manually add regular nodes
    H.add_node(0, 0)
    H.add_node(1, 1) 
    H.add_node(0, 2)
    
    # Set node attributes for proper styling
    H.node_attrs[(0, 0)] = {"type": "single-qubit", "name": "rx"}
    H.node_attrs[(1, 1)] = {"type": "single-qubit", "name": "ry"}
    H.node_attrs[(0, 2)] = {"type": "single-qubit", "name": "rz"}
    
    # Manually add dummy nodes  
    H.nodes.add(("dummy", 0))
    H.nodes.add(("dummy", 1, 0))
    H.nodes.add(("dummy", 1, 1))
    
    # Set dummy node attributes
    H.node_attrs[("dummy", 0)] = {"type": "dummy"}
    H.node_attrs[("dummy", 1, 0)] = {"type": "dummy"}
    H.node_attrs[("dummy", 1, 1)] = {"type": "dummy"}
    
    # Add some edges
    H.hyperedges["e1"] = {"root_set": {(0, 0)}, "receiver_set": {("dummy", 0)}}
    H.hyperedges["e2"] = {"root_set": {(1, 1)}, "receiver_set": {("dummy", 1, 0)}}
    H.hyperedges["e3"] = {"root_set": {("dummy", 1, 1)}, "receiver_set": {(0, 2)}}
    
    # Test parameters
    assignment = [0, 1]  # Simple qubit assignment
    qpu_info = [2]  # Single partition with 2 qubits
    assignment_map = {(0, 0): (0, 0), (1, 1): (1, 1), (0, 2): (0, 2)}
    node_map = {0: 0, 1: 0}  # Map partitions
    
    print("Testing dummy node positioning and label removal...")
    
    # Generate TikZ code
    tikz_code = hypergraph_to_tikz_subgraph(
        H=H,
        assignment=assignment,
        qpu_info=qpu_info,
        assignment_map=assignment_map,
        node_map=node_map,
        show_labels=True,  # Regular nodes should have labels, dummy nodes should not
        save=True,
        path="/Users/ftb123/MLQCP_FM/test_dummy_positioning.tex"
    )
    
    print("Generated TikZ code. Analyzing...")
    
    # Analyze the generated code
    lines = tikz_code.split('\n')
    dummy_nodes = []
    regular_nodes = []
    
    for line in lines:
        if '\\node' in line and 'n_dummy' in line:
            dummy_nodes.append(line.strip())
        elif '\\node' in line and 'style=' in line and 'n_dummy' not in line:
            regular_nodes.append(line.strip())
    
    print(f"\nFound {len(dummy_nodes)} dummy nodes:")
    for i, node_line in enumerate(dummy_nodes):
        print(f"  {i+1}. {node_line}")
        
        # Check that dummy nodes don't have labels
        if 'label=' in node_line:
            print(f"    ERROR: Dummy node has label!")
        else:
            print(f"    ✓ No label (correct)")
            
        # Extract position coordinates
        import re
        coords = re.search(r'at \(([-\d.]+),([-\d.]+)\)', node_line)
        if coords:
            x, y = float(coords.group(1)), float(coords.group(2))
            print(f"    Position: ({x:.3f}, {y:.3f})")
    
    print(f"\nFound {len(regular_nodes)} regular nodes:")
    for i, node_line in enumerate(regular_nodes):
        print(f"  {i+1}. {node_line}")
        
        # Extract position coordinates
        coords = re.search(r'at \(([-\d.]+),([-\d.]+)\)', node_line)
        if coords:
            x, y = float(coords.group(1)), float(coords.group(2))
            print(f"    Position: ({x:.3f}, {y:.3f})")
            
            # Check if regular nodes have labels when show_labels=True
            if 'label=' in node_line:
                print(f"    ✓ Has label (correct)")
            else:
                print(f"    Label status: No label")
    
    # Compare positions to verify dummy nodes are shifted up
    if dummy_nodes and regular_nodes:
        print("\nPosition comparison:")
        
        # Extract y-coordinates
        dummy_y_coords = []
        regular_y_coords = []
        
        for node_line in dummy_nodes:
            coords = re.search(r'at \(([-\d.]+),([-\d.]+)\)', node_line)
            if coords:
                dummy_y_coords.append(float(coords.group(2)))
        
        for node_line in regular_nodes:
            coords = re.search(r'at \(([-\d.]+),([-\d.]+)\)', node_line)
            if coords:
                regular_y_coords.append(float(coords.group(2)))
        
        if dummy_y_coords and regular_y_coords:
            min_dummy_y = min(dummy_y_coords)
            max_regular_y = max(regular_y_coords)
            
            print(f"  Dummy nodes Y range: {min(dummy_y_coords):.3f} to {max(dummy_y_coords):.3f}")
            print(f"  Regular nodes Y range: {min(regular_y_coords):.3f} to {max(regular_y_coords):.3f}")
            
            if min_dummy_y > max_regular_y:
                print(f"  ✓ Dummy nodes are shifted up (min dummy Y > max regular Y)")
            else:
                print(f"  Note: Dummy nodes may not all be above regular nodes")
                # Show some specific comparisons
                for dy in dummy_y_coords:
                    shift_detected = any(dy > ry + 0.1 for ry in regular_y_coords)  # 0.1 tolerance
                    if shift_detected:
                        print(f"    ✓ Dummy at Y={dy:.3f} is shifted up")
    
    print(f"\nTikZ file saved to: /Users/ftb123/MLQCP_FM/test_dummy_positioning.tex")
    print("Test completed!")

if __name__ == "__main__":
    test_dummy_node_positioning()
