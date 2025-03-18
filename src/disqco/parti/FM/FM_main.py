import time
from disqco.graphs.hypergraph_methods import *
from disqco.partitioning.FM.FM_methods import *
import copy

def FM_pass(hypergraph,
            max_gain,
            assignment,
            num_partitions,
            qpu_info, 
            costs, 
            limit, 
            active_nodes):
        
        spaces = find_spaces(assignment,qpu_info)
        hypergraph = map_counts_and_configs2(hypergraph,assignment,num_partitions,costs)

        lock_dict = {node: False for node in active_nodes}
  
        max_time = 0
        for node in active_nodes:
            if node[1] > max_time:
                max_time = node[1]

        array = find_all_gains_dict(hypergraph,active_nodes,assignment,num_partitions,costs,log=False)
        buckets = fill_buckets_from_dict(array,max_gain)
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
            source = assignment[node[1]][node[0]]

            assignment_new, array, buckets = take_action_and_update_dict_counts(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs)

            # assignment_new, array, buckets = take_action_and_update_dict_simple(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs)

            update_spaces(node,source,destination,spaces)
            lock_dict = lock_node(node,lock_dict)

            
            assignment = assignment_new
            assignment_list.append(assignment)
            h += 1
        return assignment_list, gain_list

def run_FM(
    hypergraph,
    initial_assignment,
    qpu_info,
    num_partitions,
    limit,
    max_gain=None,
    passes=100,
    stochastic=True,
    active_nodes=None,
    log = False,
    add_initial = False,
    costs = None
):
    if costs is None:
        configs = get_all_configs(num_partitions)
        costs = get_all_costs(configs)

    initial_assignment = np.array(initial_assignment)

    initial_cost = calculate_full_cost(hypergraph, initial_assignment, num_partitions, costs)
    if active_nodes is not None:
        active_nodes = hypergraph.nodes
    
    if log:
        print("Initial cost:", initial_cost)
    cost = initial_cost
    cost_list = []
    best_assignments = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)
    # print("Starting FM passes...")
    for n in range(passes):
        # print(f"Pass number: {n}")
        assignment_list, gain_list = FM_pass(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, active_nodes = active_nodes
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

def run_FM_bench(
    hypergraph,
    initial_assignment,
    qpu_info,
    num_partitions,
    limit,
    max_gain=None,
    passes=100,
    stochastic=True,
    active_nodes=None,
    log = False,
    add_initial = False,
    costs = None
):
    if costs is None:
        configs = get_all_configs(num_partitions)
        costs = get_all_costs(configs)

    initial_assignment = np.array(initial_assignment)

    initial_cost = calculate_full_cost(hypergraph, initial_assignment, num_partitions, costs)
    if active_nodes is not None:
        active_nodes = hypergraph.nodes
    
    if log:
        print("Initial cost:", initial_cost)
    cost = initial_cost
    cost_list = []
    best_assignments = []
    time_list = []
    if add_initial:
        cost_list.append(cost)
        best_assignments.append(initial_assignment)
    # print("Starting FM passes...")
    for n in range(passes):
        # print(f"Pass number: {n}")
        start = time.time()
        assignment_list, gain_list = FM_pass(
            hypergraph, max_gain, initial_assignment,
            num_partitions, qpu_info, costs, limit, active_nodes = active_nodes
        )
        end = time.time()
        time_list.append(end-start)

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

    return final_cost, final_assignment, cost_list, time_list
