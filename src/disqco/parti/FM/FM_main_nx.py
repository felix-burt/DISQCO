from disqco.parti.FM.FM_methods_nx import *

def fm_algorithm(graph: nx.Graph, qpu_sizes: dict, max_iterations=10, move_limit=None, total_graph=None, global_assignment='round_robin', ancilla=False):
    """
    Main FM algorithm loop
    """
    num_partitions = len(qpu_sizes)
    if move_limit is None:
        move_limit = len(graph.nodes())
    else:
        move_limit = min(move_limit, len(graph.nodes()))
    
    # Initialize partition assignment
    if isinstance(global_assignment, str):
        assignment = set_initial_partition_assignment(graph, qpu_sizes, method=global_assignment)
    else:
        assignment = global_assignment

    if total_graph is not None:
        W = calculate_W_matrix(total_graph)
    else:
        W = calculate_W_matrix(graph)
    
    # Max gain is the maximum weighted degree of any node in the graph
    max_weighted_degree = 0
    for node in graph.nodes():
        weighted_degree = sum(graph[node][neighbor].get('weight', 1) for neighbor in graph.neighbors(node))
        max_weighted_degree = max(max_weighted_degree, weighted_degree)
    
    max_gain = int(max_weighted_degree)
    
    best_assignment = assignment.copy()
    best_cut = calculate_cut_size(graph, assignment)
    
    cuts_from_all_passes = [best_cut]

    current_cut = best_cut

    for iteration in range(max_iterations):
        
        # Initialize for this pass
        current_assignment = assignment.copy()
        locked = set()
        if ancilla:
            spaces = find_spaces_with_ancillae(assignment, qpu_sizes, graph)
        else:
            spaces = find_spaces(assignment, qpu_sizes, graph)


        D = find_all_gains(current_assignment, num_partitions, qpu_sizes,W, graph, max_gain=max_gain)
        # Fill buckets with current gains

        buckets = fill_buckets(D, current_assignment,num_partitions, max_gain)

        
        # Track all moves and states during the pass
        moves_history = []
        states_history = [current_assignment.copy()]
        gains_history = []
        cumulative_gains = [0]
        
        # Main FM pass - continue until all nodes are locked
        move_count = 0
        while move_count < move_limit:
            # Find best valid action
            result = find_action(graph, buckets, current_assignment, spaces, max_gain, random=False, ancilla=ancilla)
            
            if result[0] is None:  # No valid moves found
                break
                
            node, destination, gain = result
            if node is None:
                break
            source = current_assignment[node]
            
            # Skip if node is already locked
            if node in locked:
                continue
            # Record the move
            moves_history.append((node, source, destination, gain))
            gains_history.append(gain)
            
            # Make the move using corrected function
            current_assignment = take_action_and_update(
                graph, node, source, destination, 
                current_assignment, num_partitions, W, D, 
                buckets, max_gain, locked, spaces=spaces, ancillae=ancilla)
            
            lock_node(node, locked)
            
            # Record state and cumulative gain
            states_history.append(current_assignment.copy())
            cumulative_gains.append(cumulative_gains[-1] + gain)
            move_count += 1

        # Find the best state during this pass
        if cumulative_gains:
            best_gain_index = np.argmax(cumulative_gains)
            best_cumulative_gain = cumulative_gains[best_gain_index]
            best_state_in_pass = states_history[best_gain_index]
            
            
            if iteration % 1 == 0:
                # Roll back to the best state found during this pass
                current_assignment = best_state_in_pass
                # Check if this pass improved the overall solution
                current_cut = current_cut - best_cumulative_gain

            else:
                # Skip the roll back, keep the current assignment
                current_cut = current_cut - cumulative_gains[-1]

            cuts_from_all_passes.append(current_cut)
            if current_cut < best_cut:
                best_cut = current_cut
                best_assignment = current_assignment.copy()
                assignment = current_assignment.copy()

            
            
        
    return best_assignment, best_cut, cuts_from_all_passes

def recursive_fm_algorithm(graph: nx.Graph,  target_partitions: int, initial_assignment=None, max_iterations=10, move_limit=None, depth=0):
    """
    Recursive FM algorithm using sparse assignment approach
    
    Args:
        graph: NetworkX graph to partition
        target_partitions: Desired number of final partitions
        max_iterations: Max iterations for each FM call
        move_limit: Limit on moves per FM pass
        depth: Current recursion depth (for logging)
        global_assignment: Full assignment array for the original graph (sparse approach)
        partition_offset: Offset to add to partition IDs for this subgraph
    
    Returns:
        final_assignment: Full assignment array (only for root call)
        total_cut: Total cut size across all partitions
        partition_subgraphs: List of subgraphs for each partition
    """
    indent = "  " * depth
    print(f"{indent}Recursive FM: Processing graph with {len(graph.nodes())} nodes, target: {target_partitions} partitions")
    
    if initial_assignment is None:
        initial_assignment = set_initial_partition_assignment(graph, {0: len(graph.nodes())}, method='round_robin')
    
    active_partitions = set([p for p in initial_assignment if p >= 0])
    current_num_partitions = len(active_partitions)

    print("Current num partitions:", current_num_partitions)
    print("Target partitions:", target_partitions)
    if current_num_partitions >= target_partitions:
        return initial_assignment, calculate_cut_size(graph, initial_assignment), [graph.subgraph([node for i, node in enumerate(graph.nodes()) if initial_assignment[i] == p]) for p in active_partitions]
    
    current_assignment = initial_assignment.copy()


    for i, p in enumerate(active_partitions):
        subgraph_nodes = [node for i, node in enumerate(graph.nodes()) if initial_assignment[i] == p]
        subgraph = graph.copy()

        for node in graph.nodes():
            if node not in subgraph_nodes:
                subgraph.remove_node(node)
        


        sub_qpu_sizes = {p+i: len(subgraph.nodes()) // 2 + len(subgraph.nodes()) % 2 + 1, p+i+1: len(subgraph.nodes()) // 2 + 1}
        # Recursively bisect the subgraph

        current_assignment = set_sparse_assignment(subgraph, graph, current_assignment, sub_qpu_sizes)



        assignment, cut, _ = fm_algorithm(subgraph, sub_qpu_sizes, max_iterations, move_limit, total_graph=graph, global_assignment=current_assignment)

    return recursive_fm_algorithm(graph, target_partitions, assignment, max_iterations, move_limit, depth + 1)
