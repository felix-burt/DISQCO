from disqco.graphs.coarsening.network_coarsener import NetworkCoarsener
from disqco.parti.FM.FM_hetero import run_FM_hetero_dummy
import multiprocessing as mp
from disqco.parti.FM.FM_methods import set_initial_sub_partitions, order_nodes, map_assignment, calculate_full_cost_hetero
from disqco.graphs.GCP_hypergraph import SubGraphManager
from disqco.parti.FM.multilevel_FM import MLFM_recursive_hetero_mapped
import numpy as np
from disqco.parti.FM.partition_and_build import partition_and_build_subgraphs
from copy import deepcopy

def stitch_solution(subgraphs, sub_assignments, node_maps, assignment_maps, num_qubits):
    final_assignment = [[None for _ in range(num_qubits)] for _ in range(len(sub_assignments[0]))]

    for i, sub_ass in enumerate(sub_assignments):
        ass_map = assignment_maps[i]
        subgraph = subgraphs[i]
        node_map = node_maps[i]
        for node in subgraph.nodes:
            if node[0] == 'dummy':
                continue
            q, t = node  # Assuming node is a tuple (q, t)
            sub_node = ass_map[(q,t)]
            ass = sub_ass[sub_node[1]][sub_node[0]]
            final_assignment[t][q] = node_map[ass]
    
    return final_assignment

def run_net_coarsened_FM(graph, initial_network, l=4, multiprocessing=True, level_limit = None):

    net_coarsener = NetworkCoarsener(initial_network)
    initial_graph = deepcopy(graph)

    net_coarsener.coarsen_network_recursive(l=l)

    for i in range(len(net_coarsener.network_coarse_list)-1):

        network_coarse = net_coarsener.network_coarse_list[-1]
        network_coarse.active_nodes = set([node for node in network_coarse.qpu_graph.nodes])

    network_level_list = []
    network_level_list.append([[network_coarse, set([key for key in network_coarse.mapping])]])
    networks = network_level_list[0]

    for i in range(len(net_coarsener.network_coarse_list)-1):
        networks = net_coarsener.cut_network(network_level_list[i], level=i)
        network_level_list.append(networks)

    sub_graph_manager = SubGraphManager(initial_graph)
    pool = mp.Pool(processes=mp.cpu_count())

    subgraphs = [graph]

    for level, network_list in enumerate(network_level_list):

        networks = network_list
        sub_assignments = []
        subgraph_list = []
        node_maps = []
        qpu_size_list = []  
        sub_partitions_list = []
        dummy_node_list = []
        node_maps = []
        index_list = []

        node_list_list = []
        assignment_map_list = []

        for g in subgraphs:
            node_list = order_nodes(g)
            max_qubits_layer = max([len(layer) for layer in node_list])
            g.num_qubits = max_qubits_layer
            assignment_map, sorted_node_list = map_assignment(node_list)
            node_list_list.append(sorted_node_list)
            assignment_map_list.append(assignment_map)

        for i, network_info in enumerate(networks):
            network = network_info[0]
            active_nodes = network_info[1]
            index_list.append(i*len(active_nodes))

            qpu_sizes = {qpu : network.qpu_graph.nodes[qpu]['size'] for qpu in active_nodes}

            qpu_size_list.append(qpu_sizes)
            node_list = order_nodes(subgraphs[i])
            sub_partitions = set_initial_sub_partitions(network, node_list, active_nodes)

            sub_partitions_list.append(sub_partitions)
            subnet, active_nodes = networks[i]

            subnet.qpu_sizes = qpu_size_list[i]
            k = 0
            node_map = {}
            for node in subnet.qpu_graph.nodes:
                if node in active_nodes:
                    node_map[k] = node
                    k += 1

            node_maps.append(node_map)
            subgraph = subgraphs[i]

            dummy_nodes = set()
            for node in subgraph.nodes:
                if node[0] == 'dummy':
                    dummy_nodes.add(node)
                    qpu = node[2]
                    dummy_counter = node[3]
                    node_map[k+dummy_counter] = qpu
            dummy_node_list.append(dummy_nodes) 


        arg_list = [(subgraphs[i],
                    sub_partitions_list[i],
                    qpu_size_list[i],
                    len(networks[i][1]),
                    None,
                    None,
                    50,
                    True,
                    None,
                    False,
                    False,
                    None,
                    networks[i][0],
                    node_maps[i],
                    assignment_map_list[i],
                    dummy_node_list[i],
                    node_list_list[i],
                    level,
                    network_level_list,
                    sub_graph_manager,
                    subgraph_list,
                    sub_assignments,
                    index_list[i]) for i in range(len(networks))
                    ]
    
        if multiprocessing:
            results = pool.starmap(partition_and_build_subgraphs, arg_list)
        else:
            results = []
            node_maps = node_maps
            assignment_map_list = assignment_map_list
            for args in arg_list:
                result = partition_and_build_subgraphs(*args)
                results.append(result)

        
        if level == len(network_level_list)-1:
            subgraph_list = subgraphs
            sub_assignments = [result[0] for result in results]
            new_sub_assignment_list = []
            for i in range(len(sub_assignments)):
                new_sub_assignment_list += sub_assignments[i]
            sub_assignments = new_sub_assignment_list

        else:
            subgraph_list = [result[1] for result in results]
            new_subgraph_list = []
            for i in range(len(subgraph_list)):
                new_subgraph_list += subgraph_list[i]
            subgraph_list = new_subgraph_list
            
        
        subgraphs = subgraph_list
    

    
    if multiprocessing:
        pool.close()
        pool.join()

    num_partitions = len(initial_network.qpu_graph.nodes)
    final_assignment = stitch_solution(subgraphs, sub_assignments[0:len(node_maps)], node_maps, assignment_map_list, initial_graph.num_qubits)    
    cost = calculate_full_cost_hetero(initial_graph, final_assignment, num_partitions, network=initial_network)
    
    return cost, final_assignment