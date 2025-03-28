import copy
import random
import numpy as np
from disqco.graphs.hypergraph_methods import *
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
import time

def set_initial_partitions(network : QuantumNetwork, num_qubits: int, depth: int, invert=False) -> list:
    static_partition = []
    qpu_info = network.qpu_sizes
    num_partitions = len(qpu_info)
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)
    static_partition = static_partition[:num_qubits]
    full_partitions = np.zeros((depth,len(static_partition)),dtype=int)
    for n in range(depth):
        layer = np.array(static_partition,dtype=int)
        full_partitions[n] = layer
    return full_partitions.tolist()

def set_initial_partitions_dict(network : QuantumNetwork, num_qubits : int, depth : int, invert: bool = False) -> dict[tuple[int,int] : int]:
    """
    Greedy method to assign qubits to partitions. Assigns occording to logical index, fill each partition
    in order. If invert is True, assigns in reverse order.
    network: quantum network object
    num_qubits: number of logical qubits in the circuit
    depth: number of time steps in the circuit
    """
    static_partition = []
    qpu_info = network.qpu_sizes
    num_partitions = len(qpu_info)
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)

    static_partition = static_partition[:num_qubits]
    partition_assignment = {}

    for n in range(depth):
        for k in range(num_qubits):
            partition_assignment[(k,n)] = static_partition[k]
    
    return partition_assignment

def find_spaces(num_qubits: int, depth: int, assignment : dict[tuple[int,int] : int], network : QuantumNetwork) -> dict[int : int]:
    """
    Find the number of free qubits in each partition at each time step.
    num_qubits: number of logical qubits in the circuit
    assignment: function that maps qubits to partitions
    network: quantum network object
    """
    qpu_sizes = network.qpu_sizes
    num_partitions = len(qpu_sizes)
    spaces = {}
    for t in range(depth):
        spaces[t] = [qpu_sizes[k] for k in range(num_partitions)]
        for q in range(num_qubits):
            # spaces[t][assignment[(q,t)]] -= 1
            spaces[t][assignment[t][q]] -= 1
    return spaces

def check_valid(node : tuple[int,int], destination: int, spaces: dict[int : int]) -> bool:
    """
    Check if the destination partition has free data qubit slots.
    node: tuple of (qubit index, time step)
    destination: destination partition
    spaces: dictionary of free qubit slots in each partition at each time step
    """
    t = node[1]
    valid = False
    if spaces[t][destination] > 0:
        valid = True
    return valid

def move_node(node: tuple[int,int], destination: int, assignment: dict[tuple[int,int] : int]) -> dict[tuple[int,int] : int]:
    """ 
    Move a node to a new destination partition by updating the assignment.
    node: tuple of (qubit index, time step)
    destination: destination partition
    assignment: function that maps qubits to partitions
    """
    t = node[1]
    q = node[0]
    assignment_new = copy.deepcopy(assignment)  # Use deepcopy to avoid modifying the original assignment
    # assignment_new[(q,t)] = destination
    assignment_new[t][q] = destination

    return assignment_new

def config_from_counts(root_counts,rec_counts):
    config = []
    for x,y in zip(root_counts,rec_counts):
        if y > 0 and x < 1:
            config.append(1)
        else:
            config.append(0)
    return tuple(config)

def map_hedge_to_config(hypergraph,hedge,assignment,num_partitions):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
    # root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    config = config_from_counts(root_counts,rec_counts)
    return config

def find_gain(graph : QuantumCircuitHyperGraph, node: tuple[int,int], destination: int, assignment: dict[tuple[int,int] : int], num_partitions: int, costs: dict):
    assignment_new = move_node(node, destination, assignment)
    edges = graph.node2hyperedges[node]
    gain = 0
    for edge in edges:
        cost1 = graph.get_hyperedge_attribute(edge,'cost')
        config2 = map_hedge_to_config(graph, edge, assignment_new, num_partitions)
        cost2 = costs[config2]
        gain += cost2 - cost1
    return gain

def find_gain_h(hypergraph,node,destination,assignment,num_partitions,costs):
    assignment_new = move_node(node,destination,assignment)
    edges = hypergraph.node2hyperedges[node]
    gain = 0
    for edge in edges:
        cost1 = hypergraph.get_hyperedge_attribute(edge,'cost')
        root_config, rec_config = map_hedge_to_config(hypergraph,edge,assignment_new,num_partitions)
        cost2 = costs[(root_config,rec_config)]
        gain += cost2 - cost1
    return gain

def find_all_gains(hypergraph,nodes,assignment,num_partitions,costs,log=None):
    array = {}
    for node in nodes:
        for k in range(num_partitions):
            destination = assignment[node[1]][node[0]]
            # destination = assignment[node]
            if destination != k:
                gain = find_gain(hypergraph,node,k,assignment,num_partitions, costs)
                array[(node[1],node[0],k)] = gain
    return array

def find_all_gains_h(hypergraph,nodes,assignment,num_partitions,costs):
    array = {}
    for node in nodes:
        for k in range(num_partitions):
            destination = assignment[node[1]][node[0]]
            # destination = assignment[node]
            if destination != k:
                gain = find_gain_h(hypergraph,node,k,assignment,num_partitions, costs)
                array[(node[1],node[0],k)] = gain
    return array

def fill_buckets(array, max_gain):
    buckets = {}
    for i in range(-max_gain,max_gain+1):
        buckets[i] = set()
    for action in array.keys():
        gain = array[action]
        buckets[gain].add(action)
    return buckets

def update_counts(counts,node,destination,assignment):
    # partition = assignment[node]
    partition = assignment[node[1]][node[0]]
    new_counts = copy.deepcopy(list(counts))
    new_counts[partition] -= 1
    new_counts[destination] += 1
    return tuple(new_counts), partition

def update_config_from_counts(config,root_counts,rec_counts,partition,destination):

    new_config = copy.deepcopy(list(config))

    if rec_counts[partition] == 0:
        new_config[partition] = 0
    else:
        if root_counts[partition] == 0:
            new_config[partition] = 1
        else:
            new_config[partition] = 0
    
    if rec_counts[destination] == 0:
        new_config[destination] = 0
    else:
        if root_counts[destination] == 0:
            new_config[destination] = 1
        else:
            new_config[destination] = 0
    
    return tuple(new_config)

def update_config(old_config, new_counts, source, destination):
    new_config = copy.deepcopy(list(old_config))
    if new_counts[source] == 0:
        new_config[source] = 0
    if new_counts[destination] > 0:
        new_config[destination] = 1
    return tuple(new_config)

def find_member_random(set):
    member = random.choice(list(set))
    return member

def find_action(buckets,lock_dict,spaces,max_gain):
    for i in range(-max_gain,max_gain+1):
        bucket = buckets[i].copy()
        length = len(bucket)
        while length > 0:
            action = find_member_random(bucket)
            node = (action[1],action[0])
            destination = action[2]
            if check_valid(node,destination,spaces):
                if lock_dict[node] == False:
                    lock_dict[node] = True
                    gain = i
                    bucket.remove(action)
                    return action, gain
                else:
                    bucket.remove(action)
                    length -= 1
            else:
                bucket.remove(action)
                length -= 1
    return None, None

def update_spaces(node,source,destination,spaces):
    t = node[1]
    spaces[t][destination] -= 1
    spaces[t][source] += 1

def update_full_config(source,destination,full_config,root_config,rec_config):

    new_full_config = copy.deepcopy(list(full_config))

    if root_config[source] == 0 and rec_config[source] == 1:
        new_full_config[source] = 1
    else:
        new_full_config[source] = 0
    if root_config[destination] == 0 and rec_config[destination] == 1:
        new_full_config[destination] = 1
    else:
        new_full_config[destination] = 0
    return tuple(new_full_config)

def take_action_and_update_old(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
    assignment_new = move_node(node,destination,assignment)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:
        
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        # print("Info", info)
        cost = info['cost']
        cost_new = hedge_to_cost(hypergraph,edge,assignment_new,num_partitions,costs)
        
        root_counts = info['root_counts']
        rec_counts = info['rec_counts']
        if node in root_set:
            root_counts_new, source = update_counts(root_counts,node,destination,assignment)
            root_config_new = update_config(info['root_config'],root_counts_new,source,destination)
            rec_counts_new = tuple(copy.deepcopy(list(rec_counts)))
            rec_config_new = tuple(copy.deepcopy(list(info['rec_config'])))
        elif node in rec_set:
            rec_counts_new, source = update_counts(rec_counts,node,destination,assignment)
            rec_config_new = update_config(info['rec_config'],rec_counts_new,source,destination)
            root_counts_new = tuple(copy.deepcopy(list(root_counts)))
            root_config_new = tuple(copy.deepcopy(list(info['root_config'])))
        
        conf = info['config']
        cost = costs[conf]
        conf_a = get_full_config(root_config_new,rec_config_new)
        cost_a = cost_new
        root_counts_pre = root_counts
        rec_counts_pre = rec_counts
        
        root_config = info['root_config']
        rec_config = info['rec_config']

        root_counts_a = root_counts_new
        root_config_a = root_config_new
        rec_counts_a = rec_counts_new
        rec_config_a = rec_config_new

        for next_root_node in root_set:
            source = assignment[next_root_node[1]][next_root_node[0]] 
            # source = assignment[next_root_node]
            if not lock_dict[next_root_node]:
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        next_root_config_b = update_config(root_config, next_root_counts_b, source1, next_destination)

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, assignment_new)
                        next_root_config_ab = update_config(root_config_a, next_root_counts_ab, source2, next_destination)

                        full_config_b = update_full_config(source1, next_destination, conf, next_root_config_b, rec_config)
                        full_config_ab = update_full_config(source2, next_destination, conf_a, next_root_config_ab, rec_config_a)

                        delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain

        for next_rec_node in rec_set:
            source = assignment[next_rec_node[1]][next_rec_node[0]]
            # source = assignment[next_rec_node]
            if not lock_dict[next_rec_node]:
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                        
                        next_rec_counts_b, source1 = update_counts(rec_counts_pre, next_rec_node, next_destination, assignment)
                        next_rec_config_b = update_config(rec_config, next_rec_counts_b, source1, next_destination)
                        
                        next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, assignment_new)
                        next_rec_config_ab = update_config(rec_config_a, next_rec_counts_ab, source2, next_destination)

                        full_config_b = update_full_config(source1, next_destination, conf, root_config, next_rec_config_b)
                        full_config_ab = update_full_config(source2, next_destination, conf_a, root_config_a, next_rec_config_ab)

                        delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain
            
        hypergraph.set_hyperedge_attribute(edge, 'cost', cost_new)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config_new)
        hypergraph.set_hyperedge_attribute(edge, 'config', conf_a)
            

    for action in delta_gains:
        i = delta_gains[action]
        old_gain = array[action]
        if action in buckets[old_gain]:
            buckets[old_gain].remove(action)
            buckets[old_gain - i].add(action)
            
        array[action] -= i
    return assignment_new, array, buckets

def take_action_and_update_simple(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
    assignment_new = move_node(node,destination,assignment)
    # print("Destination", destination)
    node_set = set()
    edges = hypergraph.node2hyperedges[node]

    for edge in edges:
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        nodes = root_set.union(rec_set)
        node_set = node_set.union(nodes)

    for node in node_set:
        if node not in lock_dict:
            node_assignment = assignment[node[1]][node[0]]
            # node_assignment = assignment[node]
            for dest in range(num_partitions):
                if node_assignment != dest:
                    gain = find_gain(hypergraph,node, dest,assignment_new,num_partitions,costs)
                    old_gain = array[(node[1], node[0], dest)]
                    array[(node[1], node[0], dest)] = gain
                    if (node[1], node[0], dest) in buckets[old_gain]:
                        buckets[old_gain].remove((node[1], node[0], dest))
                        buckets[gain].add((node[1], node[0], dest))

    return assignment_new, array, buckets

def take_action_and_update(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
    assignment_new = move_node(node,destination,assignment)
    # print("Destination", destination)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:
        
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        # print("Info", info)
        cost = info['cost']
        # cost_new = hedge_to_cost(hypergraph,edge,assignment_new,num_partitions,costs)
        config = info['config']
        
        root_counts = info['root_counts']
        # print("Root counts", root_counts)
        rec_counts = info['rec_counts']
        # print("Receiver counts", rec_counts)
        if node in root_set:
            root_counts_new, source = update_counts(root_counts,node,destination,assignment)
            # print("Root counts new", root_counts_new)
            # root_config_new = update_config(info['root_config'],root_counts_new,source,destination)
            # print("Root config new", root_config_new)
            rec_counts_new = tuple(copy.deepcopy(list(rec_counts)))
            config_new = update_config_from_counts(config,root_counts_new,rec_counts_new,source,destination)
            # rec_config_new = tuple(copy.deepcopy(list(info['rec_config'])))
        elif node in rec_set:
            rec_counts_new, source = update_counts(rec_counts,node,destination,assignment)
            # print("Receiver counts new", rec_counts_new)
            # rec_config_new = update_config(info['rec_config'],rec_counts_new,source,destination)
            # print("Receiver config new", rec_config_new)
            root_counts_new = tuple(copy.deepcopy(list(root_counts)))
            # root_config_new = tuple(copy.deepcopy(list(info['root_config'])))
            config_new = update_config_from_counts(config,root_counts_new,rec_counts_new,source,destination)
        
        conf = info['config']
        cost = costs[conf]
        conf_a = config_new
        cost_a = costs[conf_a]
        root_counts_pre = root_counts
        rec_counts_pre = rec_counts

        root_counts_a = root_counts_new
        rec_counts_a = rec_counts_new

        for next_root_node in root_set:
            # print(f'Next root node {next_root_node}')
            source = assignment[next_root_node[1]][next_root_node[0]]
            # source = assignment[next_root_node]
            if not lock_dict[next_root_node]:
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        full_config_b = update_config_from_counts(conf,next_root_counts_b,rec_counts_pre,source1,next_destination)

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, assignment_new)
                        full_config_ab = update_config_from_counts(conf_a,next_root_counts_ab,rec_counts_a,source2,next_destination)


                        delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain

        for next_rec_node in rec_set:
            # print(f'Next receiver node {next_rec_node}')
            # source = assignment[next_rec_node]
            source = assignment[next_rec_node[1]][next_rec_node[0]]
            if not lock_dict[next_rec_node]:
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                        
                        next_rec_counts_b, source1 = update_counts(rec_counts_pre, next_rec_node, next_destination, assignment)
                        full_config_b = update_config_from_counts(conf,root_counts_pre,next_rec_counts_b,source1,next_destination)

                        
                        next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, assignment_new)
                        full_config_ab = update_config_from_counts(conf_a,root_counts_a,next_rec_counts_ab,source2,next_destination)


                        delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain
            
        hypergraph.set_hyperedge_attribute(edge, 'cost', cost_a)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'config', conf_a)
            

    for action in delta_gains:
        # print(f'Action {action} Gain change {delta_gains[action]}')
        i = delta_gains[action]
        old_gain = array[action]
        # print(f'Old gain {old_gain}')
        # print(f'New gain {old_gain - i}')
        if action in buckets[old_gain]:
            # print(f'Old gain in bucket - remove and add to {old_gain - i}')
            buckets[old_gain].remove(action)
            buckets[old_gain - i].add(action)
            
        array[action] -= i
    return assignment_new, array, buckets

def take_action_and_update_hetero(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
    assignment_new = move_node(node,destination,assignment)
    # print("Destination", destination)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:
        
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        # print("Info", info)
        cost = info['cost']
        cost_new = hedge_to_cost_hetero(hypergraph,edge,assignment_new,num_partitions,costs)
        
        root_counts = info['root_counts']
        # print("Root counts", root_counts)
        rec_counts = info['rec_counts']
        # print("Receiver counts", rec_counts)
        if node in root_set:
            root_counts_new, source = update_counts(root_counts,node,destination,assignment)
            # print("Root counts new", root_counts_new)
            root_config_new = update_config(info['root_config'],root_counts_new,source,destination)
            # print("Root config new", root_config_new)
            rec_counts_new = tuple(copy.deepcopy(list(rec_counts)))
            rec_config_new = tuple(copy.deepcopy(list(info['rec_config'])))
        elif node in rec_set:
            rec_counts_new, source = update_counts(rec_counts,node,destination,assignment)
            # print("Receiver counts new", rec_counts_new)
            rec_config_new = update_config(info['rec_config'],rec_counts_new,source,destination)
            # print("Receiver config new", rec_config_new)
            root_counts_new = tuple(copy.deepcopy(list(root_counts)))
            root_config_new = tuple(copy.deepcopy(list(info['root_config'])))
        
        cost_a = cost_new
        root_counts_pre = root_counts
        rec_counts_pre = rec_counts
        
        root_config = info['root_config']
        rec_config = info['rec_config']

        root_counts_a = root_counts_new
        root_config_a = root_config_new
        rec_counts_a = rec_counts_new
        rec_config_a = rec_config_new

        for next_root_node in root_set:
            # print(f'Next root node {next_root_node}')
            source = assignment[next_root_node[1]][next_root_node[0]]
            # source = assignment[next_root_node]
            if not lock_dict[next_root_node]:
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        next_root_config_b = update_config(root_config, next_root_counts_b, source1, next_destination)

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, assignment_new)
                        next_root_config_ab = update_config(root_config_a, next_root_counts_ab, source2, next_destination)

                        cost_b = costs[(next_root_config_b,rec_config)]
                        cost_ab = costs[(next_root_config_ab,rec_config_a)]

                        delta_gain = cost_a - cost - cost_ab + cost_b

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain

        for next_rec_node in rec_set:
            # print(f'Next receiver node {next_rec_node}')
            source = assignment[next_rec_node[1]][next_rec_node[0]]
            # source = assignment[next_rec_node]
            if not lock_dict[next_rec_node]:
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                        
                        next_rec_counts_b, source1 = update_counts(rec_counts_pre, next_rec_node, next_destination, assignment)
                        next_rec_config_b = update_config(rec_config, next_rec_counts_b, source1, next_destination)
                        
                        next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, assignment_new)
                        next_rec_config_ab = update_config(rec_config_a, next_rec_counts_ab, source2, next_destination)

                        cost_b = costs[(root_config,next_rec_config_b)]
                        cost_ab = costs[(root_config_a,next_rec_config_ab)]

                        delta_gain = cost_a - cost - cost_ab + cost_b

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain
            
        hypergraph.set_hyperedge_attribute(edge, 'cost', cost_new)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config_new)
            

    for action in delta_gains:
        # print(f'Action {action} Gain change {delta_gains[action]}')
        i = delta_gains[action]
        old_gain = array[action]
        # print(f'Old gain {old_gain}')
        # print(f'New gain {old_gain - i}')
        if action in buckets[old_gain]:
            # print(f'Old gain in bucket - remove and add to {old_gain - i}')
            buckets[old_gain].remove(action)
            buckets[old_gain - i].add(action)
            
        array[action] -= i
    return assignment_new, array, buckets

def lock_node(node,lock_dict):
    lock_dict[node] = True
    return lock_dict

def assignment_to_list(assignment, num_qubits, depth):
        assignment_list = []
        for t in range(depth):
            layer = []
            for j in range(num_qubits):
                qpu =  assignment[(j,t)]
                layer.append(qpu)
            assignment_list.append(layer)
        return assignment_list