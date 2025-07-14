import time
from disqco.graphs.hypergraph_methods import *
from disqco.parti.FM.FM_methods import *
from networkx import diameter

def FM_pass_hetero(hypergraph,
            max_gain,
            assignment,
            num_partitions,
            qpu_info, 
            costs, 
            limit, 
            active_nodes,
            network = None,
            node_map = None,
            assignment_map = None,
            dummy_nodes = {}):
        
        num_qubits = hypergraph.num_qubits
        depth = hypergraph.depth

        spaces = find_spaces(num_qubits, depth, assignment,qpu_info)
        hypergraph = map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, network, costs, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)

        lock_dict = {node: False for node in active_nodes}
  
        max_time = 0
        for node in active_nodes:
            if node[1] > max_time:
                max_time = node[1]

        array = find_all_gains_hetero(hypergraph,active_nodes,assignment,num_partitions,costs, network=network,node_map=node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
        buckets = fill_buckets(array,max_gain)
        gain_list = []
        gain_list.append(0)
        assignment_list = []
        assignment_list.append(assignment)
        cumulative_gain = 0
        action = 0
        h = 0
        while h < limit:
            action, gain = find_action(buckets,lock_dict,spaces,max_gain)
            if action == None:
                break
            cumulative_gain += gain
            gain_list.append(cumulative_gain)
            node = (action[1],action[0])
            destination = action[2]
            if assignment_map is not None:
                sub_node = assignment_map[node]
                source = assignment[sub_node[1]][sub_node[0]]
            else:
                source = assignment[node[1]][node[0]]

            assignment_new, array, buckets = take_action_and_update_hetero(hypergraph,
                                                                           node,
                                                                           destination,
                                                                           array,
                                                                           buckets,
                                                                           num_partitions,
                                                                           lock_dict, 
                                                                           assignment, 
                                                                           costs, 
                                                                           network = network, 
                                                                           node_map = node_map, 
                                                                           assignment_map=assignment_map)

            # assignment_new, array, buckets = take_action_and_update_dict_simple(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs)

            update_spaces(node,source,destination,spaces)
            lock_dict = lock_node(node,lock_dict)

            
            assignment = assignment_new
            assignment_list.append(assignment)
            h += 1
        return assignment_list, gain_list

def run_FM_hetero(
    hypergraph,
    initial_assignment,
    qpu_info,
    num_partitions,
    limit = None,
    max_gain=None,
    passes=100,
    stochastic=True,
    active_nodes=None,
    log = False,
    add_initial = False,
    costs = None,
    network = None,
    node_map = None,
    assignment_map = None,
    dummy_nodes = {}
):  
    if network is None:
        network = QuantumNetwork(qpu_info)
    
    if active_nodes is None:
        active_nodes = hypergraph.nodes

    if costs is None and num_partitions < 12:
        configs = get_all_configs(num_partitions, hetero=True)
        costs, edge_trees = get_all_costs_hetero(network, configs, node_map=node_map)
    else:
        costs = {}

    if limit is None:
        limit = len(active_nodes) * 0.125

    initial_assignment = np.array(initial_assignment)
    initial_cost = calculate_full_cost_hetero(hypergraph, initial_assignment, num_partitions, costs, network = network, node_map = node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)

    if active_nodes is not None:
        active_nodes = hypergraph.nodes
    
    if log:
        print("Initial cost:", initial_cost)
    
    if max_gain is None:
        max_gain = 4*diameter(network.qpu_graph)

    if isinstance(qpu_info, dict):
        qpu_info = {i: size for i, size in qpu_info.items()}
    else:
        qpu_info = {i: qpu_info[i] for i in range(len(qpu_info))}


    cost = initial_cost
    cost_list = []
    best_assignments = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)
    # print("Starting FM passes...")
    for n in range(passes):
        # print(f"Pass number: {n}")
        assignment_list, gain_list = FM_pass_hetero(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, 
            active_nodes = active_nodes, network = network, 
            node_map = node_map, assignment_map=assignment_map,
            dummy_nodes=dummy_nodes
        )

        # Decide how to pick new assignment depending on stochastic or not
        if stochastic:
            if n % 2 == 0:
                # Exploratory approach
                initial_assignment = assignment_list[-1]
                cost += gain_list[-1]
            else:
                # Exploitative approach
                idx_best = np.argmin(gain_list)
                initial_assignment = assignment_list[idx_best]
                cost += min(gain_list)
        else:
            # purely pick the best
            idx_best = np.argmin(gain_list)
            initial_assignment = assignment_list[idx_best]
            cost += min(gain_list)

        # print(f"Running cost after pass {n}:", cost)
        cost_list.append(cost)
        best_assignments.append(initial_assignment)

    # 5) Identify best assignment across all passes
    idx_global_best = np.argmin(cost_list)
    final_assignment = best_assignments[idx_global_best]
    final_cost = cost_list[idx_global_best]

    if log:
        print("All passes complete.")
        print("Final cost:", final_cost)

    # Or re-check cost on final assignment:
    # total_cost = calculate_full_cost(hypergraph, final_assignment, num_partitions, costs)
    # if log:
    #     print("Verified final cost:", total_cost)

    return final_cost, final_assignment, cost_list

def FM_pass_hetero_dummy(hypergraph,
            max_gain,
            assignment,
            num_partitions,
            qpu_info, 
            costs, 
            limit, 
            active_nodes,
            network = None,
            node_map = None,
            assignment_map = None,
            dummy_nodes = {}):
        
        num_qubits = hypergraph.num_qubits
        depth = hypergraph.depth

        spaces = find_spaces(num_qubits, depth, assignment, qpu_info, graph = hypergraph, assignment_map= assignment_map)
        # if costs is None or costs == {}:
        #     nm = node_map
        # else:
        #     nm = None
        hypergraph = map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, network, costs, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)

  
        max_time = 0
        for node in active_nodes:
            if node[1] > max_time:
                max_time = node[1]

        active_nodes = active_nodes - dummy_nodes

        # print("Active nodes:", active_nodes)

        lock_dict = {node: False for node in active_nodes}

        lock_dict.update({node: True for node in dummy_nodes})

        # print("Lock dict:", lock_dict)

        array = find_all_gains_hetero(hypergraph,active_nodes,assignment,num_partitions, costs, network=network, node_map=node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
        buckets = fill_buckets(array,max_gain)
        gain_list = []
        gain_list.append(0)
        assignment_list = []
        assignment_list.append(assignment)
        cumulative_gain = 0
        action = 0
        h = 0
        while h < limit:
            action, gain = find_action(buckets,lock_dict,spaces,max_gain)
            if action == None:
                break
            cumulative_gain += gain
            gain_list.append(cumulative_gain)
            node = (action[1],action[0])
            destination = action[2]
            
            if assignment_map is not None:
                sub_node = assignment_map[node]
                source = assignment[sub_node[1]][sub_node[0]]
            else:
                source = assignment[node[1]][node[0]]

            assignment_new, array, buckets = take_action_and_update_hetero(hypergraph,
                                                                           node,
                                                                           destination,
                                                                           array,
                                                                           buckets,
                                                                           num_partitions,
                                                                           lock_dict, 
                                                                           assignment, 
                                                                           costs, 
                                                                           network = network, 
                                                                           node_map = node_map, 
                                                                           assignment_map=assignment_map)

            # assignment_new, array, buckets = take_action_and_update_dict_simple(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs)

            update_spaces(node,source,destination,spaces)
            lock_dict = lock_node(node,lock_dict)

            
            assignment = assignment_new
            assignment_list.append(assignment)
            h += 1
        return assignment_list, gain_list

def run_FM_hetero_dummy(
    hypergraph,
    initial_assignment,
    qpu_info,
    num_partitions,
    limit = None,
    max_gain=None,
    passes=100,
    stochastic=True,
    active_nodes=None,
    log = False,
    add_initial = False,
    costs = None,
    network = None,
    node_map = None,
    assignment_map = None,
    dummy_nodes = {}
):  
    if network is None:
        network = QuantumNetwork(qpu_info)
    
    if active_nodes is None:
        active_nodes = hypergraph.nodes - dummy_nodes

    if costs is None and num_partitions < 12:
        configs = get_all_configs(len(node_map), hetero=True)
        costs, edge_trees = get_all_costs_hetero(network, configs, node_map=node_map)
    
    # costs = {}
    
    # else:

    if limit is None:
        limit = len(active_nodes) * 0.125

    initial_assignment = np.array(initial_assignment)
    initial_cost = calculate_full_cost_hetero(hypergraph, initial_assignment, num_partitions, costs, network = network, node_map = node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes)

    
    if log:
        print("Initial cost:", initial_cost)
    
    if max_gain is None:
        max_gain = 4*diameter(network.qpu_graph)

    cost = initial_cost
    cost_list = []
    best_assignments = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)
    # print("Starting FM passes...")
    # print("Assignment node (16, 23):", initial_assignment[23][16])
    for n in range(passes):
        # print(f"Pass number: {n}")
        assignment_list, gain_list = FM_pass_hetero_dummy(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, active_nodes = active_nodes, network = network, node_map = node_map, assignment_map=assignment_map, dummy_nodes=dummy_nodes
        )

        # Decide how to pick new assignment depending on stochastic or not
        if stochastic:
            if n % 2 == 0:
                # Exploratory approach
                initial_assignment = assignment_list[-1]
                cost += gain_list[-1]
            else:
                # Exploitative approach
                idx_best = np.argmin(gain_list)
                initial_assignment = assignment_list[idx_best]
                cost += min(gain_list)
        else:
            # purely pick the best
            idx_best = np.argmin(gain_list)
            initial_assignment = assignment_list[idx_best]
            cost += min(gain_list)

        # print(f"Running cost after pass {n}:", cost)
        cost_list.append(cost)
        best_assignments.append(initial_assignment)

    # 5) Identify best assignment across all passes
    idx_global_best = np.argmin(cost_list)
    final_assignment = best_assignments[idx_global_best]
    final_cost = cost_list[idx_global_best]

    if log:
        print("All passes complete.")
        print("Final cost:", final_cost)

    # Or re-check cost on final assignment:
    # total_cost = calculate_full_cost(hypergraph, final_assignment, num_partitions, costs)
    # if log:
    #     print("Verified final cost:", total_cost)

    return final_cost, final_assignment, cost_list
