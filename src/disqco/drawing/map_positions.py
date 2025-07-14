import copy

def space_mapping(qpu_info, num_layers):
    qpu_mapping = {}
    qubit_index = 0
    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    for j, qpu_size in enumerate(qpu_sizes):
        qubit_list = []
        for _ in range(qpu_size):
            qubit_list.append(qubit_index)
            qubit_index += 1
        qpu_mapping[j] = qubit_list
    space_mapping = []
    for t in range(num_layers):
        space_mapping.append(copy.deepcopy(qpu_mapping))
    
    return space_mapping

def get_pos_list(graph, num_qubits, assignment, space_map, assignment_map = None):

    num_layers = len(space_map)
    pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]

    if assignment_map is not None:
        inverse_assignment_map = {}
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    for q in range(len(assignment[0])):
        old_partition = None
        for t in range(len(assignment)):
            node = (q, t)
            if assignment_map is not None:
                if (q,t) not in inverse_assignment_map:
                    continue
                node = inverse_assignment_map[(q,t)]

            # partition = assignment[(q,t)]
            try:
                partition = assignment[node[1]][node[0]]
            except IndexError:
                print(f"IndexError for q={q}, t={t} which maps to {node}. Assignment {assignment}")
            if partition == -1:
                # If the qubit is not assigned to any partition, we can skip it
                continue
            if old_partition is not None:
                if partition == old_partition:
                    if y_pos in space_map[t][partition]:
                        y_pos = pos_list[t-1][q]
                        pos_list[t][q] = y_pos 
                        space_map[t][partition].remove(y_pos)
                    else:
                        qubit_list = space_map[t][partition]
                        y_pos = qubit_list.pop(0)
                        pos_list[t][q] = y_pos

                else:
                    qubit_list = space_map[t][partition]
                    y_pos = qubit_list.pop(0)
                    pos_list[t][q] = y_pos
            else:
                qubit_list = space_map[t][partition]
                y_pos = qubit_list.pop(0)
                pos_list[t][q] = y_pos
            old_partition = partition
    return pos_list

def get_pos_list_ext(graph, num_qubits, assignment, space_map, qpu_sizes, assignment_map = None):

    num_layers = len(space_map)
    pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]

    if assignment_map is not None:
        inverse_assignment_map = {}
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    
    for q in range(num_qubits):
        old_partition = None
        for t in range(num_layers):
            if assignment_map is not None:
                q, t = inverse_assignment_map[(q, t)]
            # partition = assignment[(q,t)]
            partition = assignment[t][q]
            if old_partition is not None:
                if partition == old_partition:
                    if y_pos in space_map[t][partition]:
                        y_pos = pos_list[t-1][q]
                        pos_list[t][q] = y_pos 
                        space_map[t][partition].remove(y_pos)
                    else:
                        qubit_list = space_map[t][partition]
                        y_pos = qubit_list.pop(0)
                        pos_list[t][q] = y_pos

                else:
                    qubit_list = space_map[t][partition]
                    y_pos = qubit_list.pop(0)
                    pos_list[t][q] = y_pos
            else:
                qubit_list = space_map[t][partition]
                y_pos = qubit_list.pop(0)
                pos_list[t][q] = y_pos
            old_partition = partition


    # pos_dict = {}
    # for t in range(len(pos_list)):
    #     for q in range(num_qubits):
    #         pos_dict[(q, t)] = pos_list[t][q]

    # for node in graph.nodes():
    #     if node not in pos_dict:
    #         partition = assignment[node]
    #         if partition == 0:
    #             boundary1 = 0
    #             boundary2 = qpu_sizes[0]
    #         else:
    #             boundary1 = qpu_sizes[partition-1]
    #             boundary2 = qpu_sizes[partition]
    #         position = (boundary1 + boundary2) // 2

    #         pos_dict[node] = position


    return pos_list

# Enhanced versions with grid space tracking and consistent y-coordinate placement

def space_mapping_enhanced(qpu_info, num_layers, track_usage=True):
    """
    Enhanced space mapping that can track usage and maintain available positions.
    
    Args:
        qpu_info: QPU size information (dict or list)
        num_layers: Number of time layers
        track_usage: If True, track which positions are used
    
    Returns:
        dict: Enhanced space mapping with usage tracking
    """
    qpu_mapping = {}
    qubit_index = 0
    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    
    for j, qpu_size in enumerate(qpu_sizes):
        qubit_list = []
        for _ in range(qpu_size):
            qubit_list.append(qubit_index)
            qubit_index += 1
        qpu_mapping[j] = qubit_list
    
    if track_usage:
        # Return enhanced structure with usage tracking
        space_mapping = []
        for t in range(num_layers):
            layer_mapping = {}
            for partition, positions in qpu_mapping.items():
                layer_mapping[partition] = {
                    'available': copy.deepcopy(positions),
                    'used': set(),
                    'qubit_to_y': {}  # Track which y-position each qubit is at
                }
            space_mapping.append(layer_mapping)
        return space_mapping
    else:
        # Original behavior for backward compatibility
        space_mapping = []
        for t in range(num_layers):
            space_mapping.append(copy.deepcopy(qpu_mapping))
        return space_mapping


def get_pos_list_enhanced(graph, num_qubits, assignment, space_map_enhanced, assignment_map=None, favor_consistency=True):
    """
    Enhanced position list generation that favors consistent y-coordinates for nodes
    with the same qubit index and tracks grid space usage.
    
    Args:
        graph: The hypergraph
        num_qubits: Number of qubits
        assignment: Qubit assignment to partitions [layer][qubit] -> partition
        space_map_enhanced: Enhanced space mapping with usage tracking
        assignment_map: Maps overall node indices to subgraph node indices  
        favor_consistency: If True, try to place same qubit at same y across layers
        
    Returns:
        list: Position list [layer][qubit] -> y_position
    """
    num_layers = len(space_map_enhanced)
    pos_list = [[0 for _ in range(num_qubits)] for _ in range(num_layers)]
    
    if assignment_map is not None:
        inverse_assignment_map = {}
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    
    # Track preferred y-positions for each qubit to maintain consistency
    qubit_preferred_y = {}
    
    for q in range(num_qubits):
        old_partition = None
        preferred_y = None
        
        for t in range(num_layers):
            # Apply assignment map if provided
            current_q, current_t = q, t
            if assignment_map is not None and (q, t) in inverse_assignment_map:
                current_q, current_t = inverse_assignment_map[(q, t)]
            
            # Get partition assignment
            if t < len(assignment) and current_q < len(assignment[t]):
                partition = assignment[t][current_q]
            else:
                continue  # Skip if assignment doesn't exist
                
            if partition == -1:
                # If the qubit is not assigned to any partition, skip it
                continue
            
            space_info = space_map_enhanced[t][partition]
            
            # Strategy 1: If we're in the same partition as before, try to keep same y
            if favor_consistency and old_partition == partition and preferred_y is not None:
                if preferred_y in space_info['available']:
                    # We can use the same y-position
                    y_pos = preferred_y
                    space_info['available'].remove(y_pos)
                    space_info['used'].add(y_pos)
                    space_info['qubit_to_y'][current_q] = y_pos
                    pos_list[t][q] = y_pos
                else:
                    # Preferred position not available, get next best
                    if space_info['available']:
                        y_pos = space_info['available'].pop(0)
                        space_info['used'].add(y_pos)
                        space_info['qubit_to_y'][current_q] = y_pos
                        pos_list[t][q] = y_pos
                        # Update preferred position for this qubit
                        preferred_y = y_pos
                        qubit_preferred_y[q] = y_pos
            else:
                # Strategy 2: Try to use a position that's consistent with this qubit's history
                if favor_consistency and q in qubit_preferred_y:
                    target_y = qubit_preferred_y[q]
                    if target_y in space_info['available']:
                        y_pos = target_y
                        space_info['available'].remove(y_pos)
                    else:
                        # Target not available, get any available position
                        if space_info['available']:
                            y_pos = space_info['available'].pop(0)
                        else:
                            # No positions available - this shouldn't happen in normal cases
                            y_pos = 0
                else:
                    # Strategy 3: Just get the next available position
                    if space_info['available']:
                        y_pos = space_info['available'].pop(0)
                    else:
                        # No positions available - fallback
                        y_pos = 0
                
                space_info['used'].add(y_pos)
                space_info['qubit_to_y'][current_q] = y_pos
                pos_list[t][q] = y_pos
                
                # Update preferred position for this qubit
                preferred_y = y_pos
                qubit_preferred_y[q] = y_pos
            
            old_partition = partition
    
    return pos_list


def get_pos_list_subgraph(graph, num_qubits, assignment, space_map_enhanced, assignment_map, node_map, favor_consistency=True):
    """
    Enhanced position list generation specifically for subgraphs, using assignment_map
    to determine sub-nodes and consistent placement.
    
    Args:
        graph: The subgraph hypergraph
        num_qubits: Number of qubits in subgraph
        assignment: Qubit assignment to partitions [layer][qubit] -> partition
        space_map_enhanced: Enhanced space mapping with usage tracking
        assignment_map: Maps overall node indices to subgraph node indices
        node_map: Maps overall partition numbers to subgraph partition numbers
        favor_consistency: If True, try to place same qubit at same y across layers
        
    Returns:
        list: Position list [layer][qubit] -> y_position
    """
    num_layers = len(space_map_enhanced)
    pos_list = [[0 for _ in range(num_qubits)] for _ in range(num_layers)]
    
    # Build inverse assignment map for efficient lookup
    inverse_assignment_map = {}
    if assignment_map is not None:
        for original_node, sub_node in assignment_map.items():
            inverse_assignment_map[sub_node] = original_node
    
    # Track preferred y-positions for each original qubit index to maintain consistency
    original_qubit_preferred_y = {}
    
    for sub_q in range(num_qubits):
        old_partition = None
        preferred_y = None
        
        # Get the original qubit index for this sub-qubit
        original_qubit = None
        if inverse_assignment_map:
            # Look for entries that map to this sub-qubit
            for sub_node, orig_node in inverse_assignment_map.items():
                if isinstance(sub_node, tuple) and len(sub_node) == 2:
                    if sub_node[0] == sub_q:  # This is our qubit
                        original_qubit = orig_node[0] if isinstance(orig_node, tuple) else sub_q
                        break
        
        if original_qubit is None:
            original_qubit = sub_q  # Fallback to sub-qubit index
        
        for t in range(num_layers):
            # Get partition assignment for this sub-qubit
            if t < len(assignment) and sub_q < len(assignment[t]):
                sub_partition = assignment[t][sub_q]
            else:
                continue  # Skip if assignment doesn't exist
                
            if sub_partition == -1:
                # If the qubit is not assigned to any partition, skip it
                continue
            
            # Get the space info for this partition
            if sub_partition not in space_map_enhanced[t]:
                continue  # Skip if partition doesn't exist in space map
                
            space_info = space_map_enhanced[t][sub_partition]
            
            # Strategy 1: Try to maintain consistency with the original qubit's preferred y
            if favor_consistency and original_qubit in original_qubit_preferred_y:
                target_y = original_qubit_preferred_y[original_qubit]
                if target_y in space_info['available']:
                    y_pos = target_y
                    space_info['available'].remove(y_pos)
                    space_info['used'].add(y_pos)
                    space_info['qubit_to_y'][sub_q] = y_pos
                    pos_list[t][sub_q] = y_pos
                    preferred_y = y_pos
                    continue
            
            # Strategy 2: If we're in the same partition as before, try to keep same y
            if favor_consistency and old_partition == sub_partition and preferred_y is not None:
                if preferred_y in space_info['available']:
                    y_pos = preferred_y
                    space_info['available'].remove(y_pos)
                    space_info['used'].add(y_pos)
                    space_info['qubit_to_y'][sub_q] = y_pos
                    pos_list[t][sub_q] = y_pos
                else:
                    # Preferred position not available, get next best
                    if space_info['available']:
                        y_pos = space_info['available'].pop(0)
                        space_info['used'].add(y_pos)
                        space_info['qubit_to_y'][sub_q] = y_pos
                        pos_list[t][sub_q] = y_pos
                        preferred_y = y_pos
                        original_qubit_preferred_y[original_qubit] = y_pos
            else:
                # Strategy 3: Get any available position
                if space_info['available']:
                    y_pos = space_info['available'].pop(0)
                else:
                    # No positions available - fallback
                    y_pos = 0
                
                space_info['used'].add(y_pos)
                space_info['qubit_to_y'][sub_q] = y_pos
                pos_list[t][sub_q] = y_pos
                
                # Update preferred position for this original qubit
                preferred_y = y_pos
                original_qubit_preferred_y[original_qubit] = y_pos
            
            old_partition = sub_partition
    
    return pos_list


def get_space_usage_info(space_map_enhanced):
    """
    Get usage information from an enhanced space map.
    
    Args:
        space_map_enhanced: Enhanced space mapping with usage tracking
        
    Returns:
        dict: Usage statistics per layer and partition
    """
    usage_info = {}
    
    for t, layer in enumerate(space_map_enhanced):
        layer_info = {}
        for partition, space_info in layer.items():
            layer_info[partition] = {
                'total_positions': len(space_info['available']) + len(space_info['used']),
                'used_positions': len(space_info['used']),
                'available_positions': len(space_info['available']),
                'usage_rate': len(space_info['used']) / (len(space_info['available']) + len(space_info['used'])) if (len(space_info['available']) + len(space_info['used'])) > 0 else 0,
                'qubit_assignments': dict(space_info['qubit_to_y'])
            }
        usage_info[t] = layer_info
    
    return usage_info


def find_node_layout(graph, assignment, qpu_sizes, assignment_map=None):

    if isinstance(qpu_sizes, dict):
        qpu_sizes = list(qpu_sizes.values())
    depth = graph.depth

    

    slot_positions = {}
    filled_slots = set()

    node_positions = {}

    print(qpu_sizes)
    all_free_spaces = []

    for t in range(depth):
        free_spaces = {}
        for i, qpu_size in enumerate(qpu_sizes):
            free_spaces[i] = [j for j in range(qpu_size)]
            
        print(f'For time {t}, free spaces: {free_spaces}')
        all_free_spaces.append(free_spaces)

    y_index = 0
    for i, qpu in enumerate(qpu_sizes):
        for k in range(qpu):
            y_index += 1
            slot_positions[(i, k)] = y_index
            print(f'Slot position for (QPU {i}, qubit {k}): {slot_positions[(i, k)]}')

    inverse_assignment_map = {}
    if assignment_map is not None:
        for node in assignment_map:
            inverse_assignment_map[assignment_map[node]] = node
    
    for q in range(len(assignment[0])):
        y = None
        for t in range(depth):
            node = (q, t)
            print(f'Checking sub-node {node}')
            if assignment_map is not None:
                if (q, t) not in inverse_assignment_map:
                    continue
            print(f'Sub node is in the graph: {node}')
            partition = assignment[t][q]
            print(f'Partition for node {node} at time {t}: {partition}')
            if y is None:
                print(f'Node has not been assigned at a previous time-step')
                y = all_free_spaces[t][partition].pop(0)
                print(f'Assigned y={y} to node {node}')
            else:
                print(f'Node has been assigned at a previous time-step, checking for y={y}')
                if y in all_free_spaces[t][partition]:
                    all_free_spaces[t][partition].remove(y)
                else:
                    y = all_free_spaces[t][partition].pop(0)
            
            slot = slot_positions.get((partition, y), None)

            # if slot not in filled_slots:
            #     filled_slots.add(slot)
            # else:
            #     raise ValueError(f"Slot {slot} already filled for node {node} at time {t}")
            
            node_positions[node] = slot
            print(f'Node {node} assigned to slot {slot} at time {t}')
    return node_positions





