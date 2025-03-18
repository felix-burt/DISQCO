import numpy as np
from disqco.partitioning.FM.FM_main import run_FM
from disqco.partitioning.FM.FM_methods import get_all_configs, get_all_costs, calculate_full_cost
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
import time

def refine_assignment(level, num_levels, assignment, mapping_list):
    new_assignment = assignment
    if level < num_levels -1:
        mapping = mapping_list[level]
        for super_node_t in mapping:
            for t in mapping[super_node_t]:
                new_assignment[t] = assignment[super_node_t]
    return new_assignment

def find_max_gain(mapping_list, level):
    largest_node = 1
    for s_node in mapping_list[level]:
        length = len(mapping_list[level][s_node])
        if length > largest_node:
            largest_node = length
    return 2 * largest_node + 2

def multilevel_FM(coarsened_hypergraphs,
                mapping_list, 
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list = [10]*10, 
                stochastic = True, 
                lock_nodes = False,
                log = False,
                add_initial = False,
                costs = None,
                level_limit = None):

    num_partitions = len(qpu_info)
    if costs is None:
        configs = get_all_configs(num_partitions)
        costs = get_all_costs(configs)

    list_of_assignments = []
    list_of_assignments.append(initial_assignment)

    list_of_costs = []

    initial_cost = calculate_full_cost(coarsened_hypergraphs[-1], initial_assignment, num_partitions, costs=costs)
    list_of_costs.append(initial_cost)
    best_cost = initial_cost

    list_of_times = []

    list_of_times.append(0)

    graph_list = coarsened_hypergraphs[::-1]
    active_nodes = graph_list[0].nodes

    mapping_list = mapping_list[::-1]

    graph_list = graph_list[:level_limit]
    mapping_list = mapping_list[:level_limit]

    for i, graph in enumerate(graph_list):
        if lock_nodes:
            new_nodes = {node for node in graph.nodes if node not in active_nodes}
            active_nodes = new_nodes
        else:
            active_nodes = graph.nodes

        if limit is None:
            limit = len(active_nodes)

        max_gain = find_max_gain(mapping_list, i)

        passes = pass_list[i]
        start = time.time()
        best_cost_pass, best_assignment, _ = run_FM(
            hypergraph=graph,            # This stage's coarsened hypergraph
            initial_assignment=initial_assignment,
            qpu_info=qpu_info,
            num_partitions=num_partitions,
            limit=limit,
            max_gain=max_gain,
            passes=passes,
            stochastic=stochastic,
            active_nodes=active_nodes,
            log = log,
            add_initial=add_initial,
            costs=costs
        )
        end = time.time()
        level_time = end - start
        list_of_times.append(level_time)
        
        if best_cost_pass < best_cost:
        # Keep track of the result
            best_cost = best_cost_pass
            assignment = best_assignment
        else:
            assignment = initial_assignment
        

        if log:
            print(f'Best cost at level {i}: {best_cost}')

        refined_assignment = refine_assignment(i, len(graph_list), assignment, mapping_list)
        initial_assignment = refined_assignment

        list_of_assignments.append(initial_assignment)
        list_of_costs.append(best_cost)

    return list_of_assignments, list_of_costs, list_of_times

from disqco.graphs.hypergraph_methods import get_all_costs_hetero, calculate_full_cost_hetero
from disqco.partitioning.FM.FM_hetero import run_FM_hetero
from networkx import diameter
import copy

def multilevel_FM_hetero(coarsened_hypergraphs,
                mapping_list, 
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list = [10]*10, 
                stochastic = True, 
                lock_nodes = False,
                log = False,
                add_initial = False,
                costs = None,
                level_limit = None,
                network = None):

    num_partitions = len(qpu_info)
    if costs is None:
        configs = get_all_configs(num_partitions)
        costs, edge_tree = get_all_costs_hetero(network, configs)

    list_of_assignments = []
    list_of_assignments.append(initial_assignment)

    list_of_costs = []

    initial_cost = calculate_full_cost_hetero(coarsened_hypergraphs[-1], initial_assignment, num_partitions, costs=costs)
    list_of_costs.append(initial_cost)
    best_cost = initial_cost

    list_of_times = []

    list_of_times.append(0)

    graph_list = coarsened_hypergraphs[::-1]
    active_nodes = graph_list[0].nodes

    mapping_list = mapping_list[::-1]

    graph_list = graph_list[:level_limit]
    mapping_list = mapping_list[:level_limit]

    for i, graph in enumerate(graph_list):
        if lock_nodes:
            new_nodes = {node for node in graph.nodes if node not in active_nodes}
            active_nodes = new_nodes
        else:
            active_nodes = graph.nodes

        if limit is None:
            limit = len(active_nodes)

        network_diameter = diameter(network.qpu_graph)
        max_gain = find_max_gain(mapping_list, i)*network_diameter

        passes = pass_list[i]
        start = time.time()
        best_cost_pass, best_assignment, _ = run_FM_hetero(
            hypergraph=graph,            # This stage's coarsened hypergraph
            initial_assignment=initial_assignment,
            qpu_info=qpu_info,
            num_partitions=num_partitions,
            limit=limit,
            max_gain=max_gain,
            passes=passes, 
            stochastic=stochastic,
            active_nodes=active_nodes,
            log = log,
            add_initial=add_initial,
            costs=costs,
            network=network
        )
        end = time.time()
        level_time = end - start
        list_of_times.append(level_time)
        
        if best_cost_pass < best_cost:
        # Keep track of the result
            best_cost = best_cost_pass
            assignment = best_assignment
        else:
            assignment = initial_assignment
        

        if log:
            print(f'Best cost at level {i}: {best_cost}')

        refined_assignment = refine_assignment(i, len(graph_list), assignment, mapping_list)
        initial_assignment = refined_assignment

        list_of_assignments.append(initial_assignment)
        list_of_costs.append(best_cost)

    return list_of_assignments, list_of_costs, list_of_times

def MLFM_window(graph,
                num_levels,
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list= None, 
                stochastic=True, 
                lock_nodes=False,
                log = False,
                add_initial = False,
                costs = None):

    coarsener = HypergraphCoarsener()

    graph_list, mapping_list = coarsener.coarsen_full(hypergraph=graph, num_levels=num_levels)
    if limit == None:
        num_qubits = graph.num_qubits
        depth = graph.depth
        limit = (1/16)*num_qubits*depth
    if pass_list == None:
        pass_list = [10]*(len(graph_list)+2)


    assignment_list, cost_list, time_list = multilevel_FM(coarsened_hypergraphs=graph_list,
                                            mapping_list=mapping_list,
                                            initial_assignment=initial_assignment,
                                            qpu_info=qpu_info,
                                            limit = limit,
                                            pass_list=pass_list,
                                            stochastic=stochastic,
                                            lock_nodes=lock_nodes,
                                            log=log,
                                            add_initial=add_initial,
                                            costs=costs)
    
    return assignment_list, cost_list, time_list

def MLFM_blocks(graph,
                num_levels,
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list= None, 
                stochastic=True, 
                lock_nodes=False,
                log = False,
                add_initial = False,
                costs = None):

    coarsener = HypergraphCoarsener()

    if limit == None:
        num_qubits = graph.num_qubits
        depth = graph.depth
        limit = (1/16)*num_qubits*depth

    graph_list, mapping_list = coarsener.coarsen_blocks(hypergraph=graph, num_blocks=None, block_size=num_levels)

    if pass_list == None:
        pass_list = [10]*(len(graph_list)+2)

    
    assignment_list, cost_list, time_list = multilevel_FM(coarsened_hypergraphs=graph_list,
                                            mapping_list=mapping_list,
                                            initial_assignment=initial_assignment,
                                            qpu_info=qpu_info,
                                            limit = limit,
                                            pass_list=pass_list,
                                            stochastic=stochastic,
                                            lock_nodes=lock_nodes,
                                            log=log,
                                            add_initial=add_initial,
                                            costs=costs)

    return assignment_list, cost_list, time_list

def MLFM_recursive(graph,
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list= None, 
                stochastic = True, 
                lock_nodes = False,
                log = False,
                add_initial = False,
                costs = None,
                level_limit = None):

    coarsener = HypergraphCoarsener()
    graph_list, mapping_list = coarsener.coarsen_recursive_batches(graph)

    if limit is not None:
        if limit == 'qubit':
            num_qubits = graph.num_qubits
            limit = num_qubits
        elif limit == 'full':
            limit = len(graph.nodes)
    
    if pass_list is None:
        pass_list = [10]*(len(graph_list))

    if level_limit is None:
        level_limit = len(graph_list)

    assignment_list, cost_list, time_list = multilevel_FM(graph_list,
                                            mapping_list,
                                            initial_assignment=initial_assignment,  
                                            qpu_info= qpu_info, 
                                            limit = limit, 
                                            pass_list= pass_list,
                                            stochastic = stochastic, 
                                            lock_nodes = lock_nodes,
                                            log = log,
                                            add_initial = add_initial,
                                            costs = costs,
                                            level_limit = level_limit)

    return assignment_list, cost_list, time_list

def MLFM_recursive_hetero(graph,
                initial_assignment,  
                qpu_info, 
                limit = None, 
                pass_list= None, 
                stochastic = True, 
                lock_nodes = False,
                log = False,
                add_initial = False,
                costs = None,
                level_limit = None,
                network = None):

    coarsener = HypergraphCoarsener()
    graph_list, mapping_list = coarsener.coarsen_recursive_batches(graph)

    if limit is not None:
        if limit == 'qubit':
            num_qubits = graph.num_qubits
            limit = num_qubits
        elif limit == 'full':
            limit = len(graph.nodes)
    
    if pass_list is None:
        pass_list = [10]*(len(graph_list))

    if level_limit is None:
        level_limit = len(graph_list)

    assignment_list, cost_list, time_list = multilevel_FM_hetero(graph_list,
                                            mapping_list,
                                            initial_assignment=initial_assignment,  
                                            qpu_info= qpu_info, 
                                            limit = limit, 
                                            pass_list= pass_list,
                                            stochastic = stochastic, 
                                            lock_nodes = lock_nodes,
                                            log = log,
                                            add_initial = add_initial,
                                            costs = costs,
                                            level_limit = level_limit,
                                            network=network)

    return assignment_list, cost_list, time_list

