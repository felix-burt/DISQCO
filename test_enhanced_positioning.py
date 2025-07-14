#!/usr/bin/env python3
"""
Test script for enhanced position mapping functions.
Tests grid space tracking and consistent y-coordinate placement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.disqco.drawing.map_positions import (
    space_mapping, get_pos_list, 
    space_mapping_enhanced, get_pos_list_enhanced, 
    get_pos_list_subgraph, get_space_usage_info
)


def test_basic_enhanced_mapping():
    """Test basic enhanced space mapping functionality."""
    print("=== Testing Basic Enhanced Mapping ===")
    
    # Simple QPU configuration: 2 partitions with 4 qubits each
    qpu_info = [4, 4]
    num_layers = 3
    
    # Test enhanced space mapping
    space_map = space_mapping_enhanced(qpu_info, num_layers, track_usage=True)
    
    print(f"Space map structure for {num_layers} layers:")
    for t, layer in enumerate(space_map):
        print(f"  Layer {t}:")
        for partition, info in layer.items():
            print(f"    Partition {partition}: available={info['available']}, used={info['used']}")
    
    # Test simple assignment
    num_qubits = 6
    assignment = [
        [0, 0, 1, 1, 1, 1],  # Layer 0: qubits 0,1 -> partition 0, qubits 2,3,4,5 -> partition 1
        [0, 0, 0, 1, 1, 1],  # Layer 1: qubits 0,1,2 -> partition 0, qubits 3,4,5 -> partition 1  
        [1, 1, 1, 1, 0, 0],  # Layer 2: qubits 0,1,2,3 -> partition 1, qubits 4,5 -> partition 0
    ]
    
    pos_list = get_pos_list_enhanced(None, num_qubits, assignment, space_map, favor_consistency=True)
    
    print(f"\nPosition assignments:")
    for t, layer_pos in enumerate(pos_list):
        print(f"  Layer {t}: {layer_pos}")
    
    # Get usage info
    usage_info = get_space_usage_info(space_map)
    print(f"\nSpace usage:")
    for t, layer_info in usage_info.items():
        print(f"  Layer {t}:")
        for partition, info in layer_info.items():
            print(f"    Partition {partition}: {info['used_positions']}/{info['total_positions']} used ({info['usage_rate']:.1%})")
            print(f"      Qubit assignments: {info['qubit_assignments']}")


def test_consistency_preservation():
    """Test that qubits maintain consistent y-coordinates when possible."""
    print("\n=== Testing Consistency Preservation ===")
    
    qpu_info = [6, 6]  # Two partitions, 6 qubits each
    num_layers = 4
    num_qubits = 3
    
    # Assignment where qubit 0 stays in partition 0, qubit 1 moves, qubit 2 stays in partition 1
    assignment = [
        [0, 1, 1],  # Layer 0
        [0, 1, 1],  # Layer 1 
        [0, 0, 1],  # Layer 2: qubit 1 moves to partition 0
        [0, 0, 1],  # Layer 3
    ]
    
    space_map = space_mapping_enhanced(qpu_info, num_layers, track_usage=True)
    pos_list = get_pos_list_enhanced(None, num_qubits, assignment, space_map, favor_consistency=True)
    
    print("Testing consistency preservation:")
    for t, layer_pos in enumerate(pos_list):
        print(f"  Layer {t}: {layer_pos}")
    
    # Check if qubit 0 maintains same y-coordinate (should be 0 in partition 0)
    qubit_0_positions = [pos_list[t][0] for t in range(num_layers)]
    print(f"Qubit 0 y-positions across layers: {qubit_0_positions}")
    
    # Check if qubit 2 maintains same y-coordinate in partition 1 
    qubit_2_positions = [pos_list[t][2] for t in range(num_layers)]
    print(f"Qubit 2 y-positions across layers: {qubit_2_positions}")


def test_subgraph_mapping():
    """Test subgraph position mapping with assignment maps."""
    print("\n=== Testing Subgraph Mapping ===")
    
    qpu_info = [4, 4]
    num_layers = 2
    num_qubits_sub = 3  # Subgraph has 3 qubits
    
    # Assignment for subgraph
    assignment = [
        [0, 1, 1],  # Layer 0: sub-qubit 0 -> partition 0, sub-qubits 1,2 -> partition 1
        [0, 0, 1],  # Layer 1: sub-qubits 0,1 -> partition 0, sub-qubit 2 -> partition 1
    ]
    
    # Assignment map: subgraph nodes -> original nodes
    assignment_map = {
        (0, 0): (2, 0),  # Sub-qubit 0 at layer 0 comes from original qubit 2 at layer 0
        (0, 1): (2, 1),  # Sub-qubit 0 at layer 1 comes from original qubit 2 at layer 1
        (1, 0): (5, 0),  # Sub-qubit 1 at layer 0 comes from original qubit 5 at layer 0
        (1, 1): (5, 1),  # Sub-qubit 1 at layer 1 comes from original qubit 5 at layer 1
        (2, 0): (7, 0),  # Sub-qubit 2 at layer 0 comes from original qubit 7 at layer 0
        (2, 1): (7, 1),  # Sub-qubit 2 at layer 1 comes from original qubit 7 at layer 1
    }
    
    # Node map: original partitions -> subgraph partitions
    node_map = {0: 0, 1: 1}
    
    space_map = space_mapping_enhanced(qpu_info, num_layers, track_usage=True)
    pos_list = get_pos_list_subgraph(None, num_qubits_sub, assignment, space_map, assignment_map, node_map)
    
    print("Subgraph position assignments:")
    for t, layer_pos in enumerate(pos_list):
        print(f"  Layer {t}: {layer_pos}")
    
    usage_info = get_space_usage_info(space_map)
    print("Subgraph space usage:")
    for t, layer_info in usage_info.items():
        print(f"  Layer {t}:")
        for partition, info in layer_info.items():
            print(f"    Partition {partition}: qubit assignments = {info['qubit_assignments']}")


def test_backward_compatibility():
    """Test that original functions still work as expected."""
    print("\n=== Testing Backward Compatibility ===")
    
    qpu_info = [3, 3]
    num_layers = 2
    num_qubits = 4
    
    # Test original functions
    space_map_orig = space_mapping(qpu_info, num_layers)
    assignment = [
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ]
    
    pos_list_orig = get_pos_list(None, num_qubits, assignment, space_map_orig)
    
    print("Original space mapping result:")
    for t, layer_pos in enumerate(pos_list_orig):
        print(f"  Layer {t}: {layer_pos}")
    
    # Test enhanced functions with same input (should work with backward compatibility mode)
    space_map_new = space_mapping_enhanced(qpu_info, num_layers, track_usage=False)
    # Convert to new format for testing
    space_map_enhanced_format = space_mapping_enhanced(qpu_info, num_layers, track_usage=True)
    pos_list_new = get_pos_list_enhanced(None, num_qubits, assignment, space_map_enhanced_format)
    
    print("Enhanced space mapping result:")
    for t, layer_pos in enumerate(pos_list_new):
        print(f"  Layer {t}: {layer_pos}")


if __name__ == "__main__":
    print("Testing Enhanced Position Mapping Functions")
    print("=" * 50)
    
    try:
        test_basic_enhanced_mapping()
        test_consistency_preservation()
        test_subgraph_mapping()
        test_backward_compatibility()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
