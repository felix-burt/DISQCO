import numpy as np
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.GCP_hypergraph_extended import HyperGraph, HyperEdge
import copy
import heapq


def set_initial_partitions_extended(network : QuantumNetwork, graph, num_qubits: int, depth: int, invert=False) -> tuple[np.ndarray, np.ndarray]:
    static_partition = []
    qpu_info = network.qpu_sizes
    num_partitions = len(qpu_info)
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)
    print(static_partition)
    static_partition = static_partition[:num_qubits]

    qubit_assignment = np.zeros((num_qubits, depth), dtype=int)
    for i in range(num_qubits):
        for j in range(depth):
            qubit_assignment[i][j] = static_partition[i]
    
    gate_assignment = np.zeros((num_qubits,num_qubits, depth), dtype=int)
    for i in range(num_qubits):
        for j in range(num_qubits):
            for k in range(depth):
                gate_assignment[i][j][k] = -1

    for node in graph.gate_nodes:
        i = node[0]
        j = node[1]
        k = node[2]
        
        partner1 = (node[0], node[2])
        partner2 = (node[1], node[2])

        # Randomly choose one of the partners
        # if np.random.random() < 0.5:
        #     gate_assignment[i][j][k] = qubit_assignment[partner1]
        # else:
        #     gate_assignment[i][j][k] = qubit_assignment[partner2]

        gate_assignment[i][j][k] = qubit_assignment[partner1]
    
    return qubit_assignment, gate_assignment



def get_edge_count(graph : HyperGraph, edge : HyperEdge, qubit_assignment, gate_assignment, num_partitions):
    counts = np.zeros(num_partitions)
    nodes_in_edge = edge.vertices
    for node in nodes_in_edge:
        if len(node) == 2:
            counts[qubit_assignment[node]] += 1
        else:
            counts[gate_assignment[node]] += 1
    return counts

def config_from_counts(counts):
    config = np.zeros(counts.shape)
    for i in range(len(counts)):
        if counts[i] > 0:
            config[i] = 1
    return config

def get_edge_config(graph : HyperGraph, edge : HyperEdge, qubit_assignment, gate_assignment, num_partitions):
    counts = get_edge_count(graph, edge, qubit_assignment, gate_assignment, num_partitions)

    config = config_from_counts(counts)
    return config

def calculate_full_cost(graph : HyperGraph, qubit_assignment, gate_assignment, num_partitions):
    full_cost = 0
    for edge in graph.hyperedges():
        config = get_edge_config(graph, edge, qubit_assignment, gate_assignment, num_partitions)
        cost = np.sum(config) - 1
        full_cost += cost
    return full_cost

def assign_counts_and_config(graph : HyperGraph, edge : HyperEdge, qubit_assignment, gate_assignment, num_partitions):
    counts = get_edge_count(graph, edge, qubit_assignment, gate_assignment, num_partitions)
    config = config_from_counts(counts)
    edge_key = edge.key

    graph.set_hyperedge_attrs(edge_key, attr_key='counts', attr = counts)
    graph.set_hyperedge_attrs(edge_key, attr_key='config', attr = config)
    graph.set_hyperedge_attrs(edge_key, attr_key='cost', attr = np.sum(config) - 1)



def calculate_gain(counts, config, source, destination):
    counts[source] -= 1
    counts[destination] += 1
    net_gain = 0
    if counts[source] == 0:
        config[source] = 0
        net_gain -= 1
    if counts[destination] == 1:
        config[destination] = 1
        net_gain += 1
        
    return net_gain, counts, config

def assign_all_counts_and_configs(graph : HyperGraph, qubit_assignment, gate_assignment, num_partitions):
    for edge in graph.hyperedges():
        assign_counts_and_config(graph, edge, qubit_assignment, gate_assignment, num_partitions)

def find_gain(graph : HyperGraph, node, source, destination):
    
    incident_edges = graph.incident(node)
    gain : int = 0
    print(f'Finding gain for {node} moving from {source} to {destination}')
    for edge in incident_edges:
        counts = graph.get_hyperedge_attrs(edge, 'counts').copy()
        config = graph.get_hyperedge_attrs(edge, 'config').copy()  

        contrib_gain, new_counts, new_config = calculate_gain(counts, config, source, destination)
        gain += contrib_gain
    # print(f'Total gain {gain}')
    return gain

def find_gain_raw(graph : HyperGraph, node, qubit_assignment, gate_assignment, destination, num_partitions):
    
    incident_edges = graph.incident(node)
    gain : int = 0
    if len(node) == 2:
        for edge_key in incident_edges:
            source = qubit_assignment[node]
            edge = graph._hyperedges[edge_key]
            counts = get_edge_count(graph, edge, qubit_assignment, gate_assignment, num_partitions)

            counts[destination] += 1
            counts[source] -= 1

            if counts[source] == 0:
                gain -= 1
            if counts[destination] == 1:
                gain += 1
    else:
        source = gate_assignment[node]
        for edge_key in incident_edges:
            edge = graph._hyperedges[edge_key]
            counts = get_edge_count(graph, edge, qubit_assignment, gate_assignment, num_partitions)
            counts[destination] += 1
            counts[source] -= 1

            if counts[source] == 0:
                gain -= 1
            if counts[destination] == 1:
                gain += 1


    return gain


def find_all_gains(graph : HyperGraph, qubit_assignment, gate_assignment, num_partitions, locked = set()):
    gain_dict = {}
    for node in set(graph.nodes()) - locked:
        if len(node) == 2:
            source = qubit_assignment[node]
        else:
            source = gate_assignment[node]
        destinations = set(range(num_partitions)) - {source}
        for i in destinations:
            gain = find_gain_raw(graph, node, qubit_assignment, gate_assignment, i, num_partitions)
            gain_dict[(node, i)] = gain
    return gain_dict

def find_spaces(graph : HyperGraph, qubit_assignment, depth, qpu_sizes):
    spaces = np.array([qpu_sizes.copy() for _ in range(depth)])
    for node in graph.qubit_nodes:
        t = node[1]
        k = qubit_assignment[node]
        spaces[t][k] -= 1
    return spaces

def is_gate_node(node):
    if len(node) == 3:
        return True
    return False

def is_valid(action, node, spaces):
    node = action[0]
    if is_gate_node(node):
        return True
    destination = action[1]
    time = node[1]
    if spaces[time][destination] == 0:
        return False
    else:
        return True

def choose_action(buckets, qubit_assignment, locked, max_gain, spaces):
    for i in range(-max_gain, max_gain+1):
        action_set = buckets.get(i, set([]))
        for action in action_set:
            node = action[0]
            if is_valid(action, qubit_assignment, spaces) and node not in locked:
                gain = i
                action_set.remove(action)
                return action, gain
            
    return None


                    
def take_action_and_update(graph, action, qubit_assignment, gate_assignment, gain_dict, buckets, locked, num_partitions):
    node = action[0]
    destination = action[1]
    if len(node) == 2:
        source = qubit_assignment[node]
    else:
        source = gate_assignment[node]
    locked.add(node)
    updates = {}
    edges = graph.incident(node)
    print(f'Updating for {node} moving from {source} to {destination}')
    for edge_key in graph.incident(node):
        print(f'Edge {edge_key}')
        edge = graph._hyperedges[edge_key]
        counts = graph.get_hyperedge_attrs(edge_key, 'counts')
        config = graph.get_hyperedge_attrs(edge_key, 'config')

        old_gain_contrib, new_counts, new_config = calculate_gain(counts.copy(), config.copy(), source, destination)
        nodes_to_update = edge.vertices - locked
        print(f'Nodes to update {nodes_to_update}')
        for next_node in nodes_to_update:
            for next_destination in range(num_partitions):
                if (next_node, next_destination) not in gain_dict:
                    continue

                if len(next_node) == 2:
                    next_node_source = qubit_assignment[next_node]
                else:
                    next_node_source = gate_assignment[next_node]
                gain_contrib, _, _ = calculate_gain(new_counts.copy(), new_config.copy(), next_node_source, next_destination)

                if (next_node, next_destination) not in updates:
                    updates[(next_node, next_destination)] = 0
                
                updates[(next_node, next_destination)] += gain_contrib


        graph.set_hyperedge_attrs(edge_key, attr_key='counts', attr=new_counts)
        graph.set_hyperedge_attrs(edge_key, attr_key='config', attr=new_config)
        graph.set_hyperedge_attrs(edge_key, attr_key='cost', attr=np.sum(new_config) - 1)

    if len(node) == 2:
        qubit_assignment[node] = destination
    else:
        gate_assignment[node] = destination


    for node, destination in updates:
        old_gain = gain_dict[(node, destination)]
        new_gain = updates[(node, destination)]

        buckets[old_gain].remove((node, destination))
        buckets[new_gain].add((node, destination))

        gain_dict[(node, destination)] = new_gain

    return gain_dict, buckets

def take_action_and_update_neighbours(graph, action, qubit_assignment, gate_assignment, gain_dict, buckets, locked, num_partitions):
    node = action[0]
    destination = action[1]
    if len(node) == 2:
        source = qubit_assignment[node]

    else:
        source = gate_assignment[node]

    locked.add(node)
    neighbours = graph.node_neighbours[node]
    
    if len(node) == 2:
        qubit_assignment[node] = destination
    else:
        gate_assignment[node] = destination
    # print(f'Updating for {node} moving from {source} to {destination}')
    # print(f'Neighbours to update {neighbours}')

    # for edge_key in graph.incident(node):
    #     counts = graph.get_hyperedge_attrs(edge_key, 'counts')
    #     config = graph.get_hyperedge_attrs(edge_key, 'config')
    #     old_gain_contrib, new_counts, new_config = calculate_gain(counts.copy(), config.copy(), source, destination)
    #     graph.set_hyperedge_attrs(edge_key, attr_key='counts', attr=new_counts)
    #     graph.set_hyperedge_attrs(edge_key, attr_key='config', attr=new_config)
    #     graph.set_hyperedge_attrs(edge_key, attr_key='cost', attr=np.sum(new_config) - 1)
    for neighbour in neighbours:
        for destination in range(num_partitions):
            # print(f'Checking {neighbour} moving from {partition_assignment[neighbour]} to {destination}')
            if (neighbour, destination) not in gain_dict or neighbour in locked:
                # print(f'Not in gain dict')
                continue
            # print(f'Updating for {neighbour}')
            old_gain = gain_dict[(neighbour, destination)]
            # print(f'Old gain {old_gain}')
            gain = find_gain_raw(graph, neighbour, qubit_assignment, gate_assignment, destination, num_partitions)
            # print(f'New gain {gain}')
            gain_dict[(neighbour, destination)] = gain
            buckets[old_gain].remove((neighbour, destination))
            buckets[gain].add((neighbour, destination))
    
    for k in range(num_partitions):
        if (node, k) in gain_dict:
            old_gain = gain_dict[(node, k)]
            if (node, k) in buckets[old_gain]:
                buckets[old_gain].remove((node, k))
            del gain_dict[(node, k)]

    return gain_dict, buckets


def update_spaces(node, source, destination, spaces):
    if is_gate_node(node):
        return spaces
    # print(f'Updating spaces for {node} moving from {source} to {destination}')
    t = node[1]
    spaces[t][source] += 1
    spaces[t][destination] -= 1
    # print(f'Spaces: {spaces}')
    return spaces
