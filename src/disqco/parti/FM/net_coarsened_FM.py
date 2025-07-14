from disqco.graphs.coarsening.network_coarsener import NetworkCoarsener
from disqco.parti.FM.FM_hetero import run_FM_hetero_dummy
import multiprocessing as mp
from disqco.parti.FM.FM_methods import set_initial_sub_partitions, order_nodes, map_assignment, calculate_full_cost_hetero
from disqco.graphs.GCP_hypergraph import SubGraphManager
from disqco.parti.FM.multilevel_FM import MLFM_recursive_hetero_mapped
import numpy as np
from disqco.parti.FM.partition_and_build import partition_and_build_subgraphs
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import time

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
            if ass == -1:
                continue
            final_assignment[t][q] = node_map[ass]
    for k in range(len(final_assignment)):
        for j in range(len(final_assignment[k])):
            if final_assignment[k][j] is None:
                final_assignment[k][j] = final_assignment[k-1][j]
    return final_assignment

def run_net_coarsened_FM(graph, initial_network, l=4, multiprocessing=True, level_limit = None, passes_per_level=10):

    net_coarsener = NetworkCoarsener(initial_network)
    initial_graph = deepcopy(graph)

    start = time.time()

    net_coarsener.coarsen_network_recursive(l=l)

    stop = time.time()

    # print(f"Time to coarsen network: {stop - start:.2f} seconds")

    for i in range(len(net_coarsener.network_coarse_list)-1):

        network_coarse = net_coarsener.network_coarse_list[-1]
        network_coarse.active_nodes = set([node for node in network_coarse.qpu_graph.nodes])

    network_level_list = []
    network_level_list.append([[network_coarse, set([key for key in network_coarse.mapping])]])
    networks = network_level_list[0]

    start = time.time()

    for i in range(len(net_coarsener.network_coarse_list)-1):
        networks = net_coarsener.cut_network(network_level_list[i], level=i)
        network_level_list.append(networks)

    stop = time.time()

    # print(f"Time to cut networks: {stop - start:.2f} seconds")

    sub_graph_manager = SubGraphManager(initial_graph)
    pool = mp.Pool(processes=mp.cpu_count())

    subgraphs = [graph]

    initial_node_list = [[i for i in range(initial_graph.num_qubits)] for t in range(initial_graph.depth)]
    node_list = initial_node_list



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
            
            if level == 0:
                node_list = initial_node_list
            else:
                node_list = order_nodes(g)


            # for layer in node_list:
            #     print(f'Node list layer: {layer}')
            max_qubits_layer = max([len(layer) for layer in node_list])
            g.num_qubits = max_qubits_layer
            assignment_map, sorted_node_list = map_assignment(node_list)
            # for layer in sorted_node_list:
            #     print(f'Sorted node list layer: {layer}')
            #     print(f'Layer length: {len(layer)}')
            node_list_list.append(sorted_node_list)
            assignment_map_list.append(assignment_map)

        for i, network_info in enumerate(networks):
            network = network_info[0]
            net_graph = network.qpu_graph
            # nx.draw(net_graph, with_labels=True)
            # plt.show()
            active_nodes = network_info[1]
            index_list.append(i*len(active_nodes))

            qpu_sizes = {qpu : network.qpu_graph.nodes[qpu]['size'] for qpu in active_nodes}
            # print(f"QPU sizes for network {i}: {qpu_sizes}")
            qpu_size_list.append(qpu_sizes)
            # node_list = order_nodes(subgraphs[i])
            node_list = node_list_list[i]
            sub_partitions = set_initial_sub_partitions(network, node_list, active_nodes,assignment_map_list[i])

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
                    passes_per_level,
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
                    index_list[i],
                    level_limit) for i in range(len(networks))
                    ]
    
        if multiprocessing:
            results = pool.starmap(partition_and_build_subgraphs, arg_list)
        else:
            results = []
            node_maps = node_maps
            assignment_map_list = assignment_map_list
            start = time.time()
            for args in arg_list:
                result = partition_and_build_subgraphs(*args)
                results.append(result)
            stop = time.time()
            # print(f"Time to partition and build all subgraphs for level {level}: {stop - start:.2f} seconds")

        sub_assignments = [result[0] for result in results]
        # for i in range(len(sub_assignments)):
        #     sub_assignment = sub_assignments[i][0]
        #     for j in range(len(sub_assignment)):
        #         print(f"Sub assignment {i}, layer {j}: {sub_assignment[j]}")
        #         num_0s = sum([1 for x in sub_assignment[j] if x == 0])
        #         print(f"Number of 0s in layer {j}: {num_0s}")
        #         num_1s = sum([1 for x in sub_assignment[j] if x == 1])
        #         print(f"Number of 1s in layer {j}: {num_1s}")
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



    # done = False
    # qpu_size_list = [initial_network.qpu_sizes[key] for key in initial_network.qpu_sizes]
    # print(qpu_size_list)
    # while not done:
    #     done, final_assignment = post_process_assignment(final_assignment, qpu_size_list, num_partitions)

    return cost, final_assignment


def post_process_assignment(final_assignment, qpu_sizes, num_partitions):
    """
    Post-process the final assignment to ensure that no partition exceeds its QPU size limit.
    If a partition exceeds its limit, move a qubit from the previous layer to the current layer.
    """
    for t, layer in enumerate(final_assignment):
        for j in range(num_partitions):
            num_js = layer.count(j)
            if num_js > qpu_sizes[j]:
                qubits_in_j = set([idx for idx, val in enumerate(layer) if val == j])
                qubits_in_j_prev_layer = set([idx for idx, val in enumerate(final_assignment[t-1]) if val == j])
                # Find element in current layer that is not in the previous layer
                for idx in qubits_in_j:
                    if idx not in qubits_in_j_prev_layer:
                        qubit_to_move = idx
                        final_assignment[t][qubit_to_move] = final_assignment[t-1][qubit_to_move]
                        return False, final_assignment  # Return False to indicate that a qubit was moved
                # assign partition from previous layer to current layer
    print(f"All capacities are satisfied.")
    return True, final_assignment