import copy
import random
import numpy as np
from disqco.graphs.hypergraph_methods import *

def set_initial_partitions(qpu_info, num_qubits, num_layers,num_partitions, reduced = False, invert=False):
    static_partition = []
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)
    if reduced:
        static_partition = static_partition[:num_qubits]
    full_partitions = np.zeros((num_layers,len(static_partition)),dtype=int)
    for n in range(num_layers):
        layer = np.array(static_partition,dtype=int)
        full_partitions[n] = layer
    return full_partitions.tolist()

def find_spaces(assignment,qpu_info):
    num_partitions = len(qpu_info)
    spaces = {}
    for t in range(len(assignment)):
        spaces[t] = [qpu_info[k] for k in range(num_partitions)]
        for q in range(len(assignment[t])):
            spaces[t][assignment[t][q]] -= 1
    return spaces

def check_valid(node,destination,spaces):
    t = node[1]
    valid = False
    if spaces[t][destination] > 0:
        valid = True
    return valid

def move_node(node,destination,assignment):
    t = node[1]
    qb = node[0]
    assignment_new = assignment.copy()
    assignment_new[t][qb] = destination
    return assignment_new

def find_gain(hypergraph,node,destination,assignment,num_partitions,costs,log=False):
    assignment_new = move_node(node,destination,assignment)
    edges = hypergraph.node2hyperedges[node]
    gain = 0
    if log:
        print("Node", node, "Destination", destination)
    for edge in edges:
        
        cost1 = hypergraph.get_hyperedge_attribute(edge,'cost')

        config2 = map_hedge_to_config2(hypergraph,edge,assignment_new,num_partitions)
        cost2 = costs[config2]
        gain += cost2 - cost1
    if log:
        print(f'Total gain = {gain}')
    return gain

def find_gain_h(hypergraph,node,destination,assignment,num_partitions,costs,log=False):
    assignment_new = move_node(node,destination,assignment)
    edges = hypergraph.node2hyperedges[node]
    gain = 0
    if log:
        print("Node", node, "Destination", destination)
    for edge in edges:
        
        cost1 = hypergraph.get_hyperedge_attribute(edge,'cost')

        root_config, rec_config = map_hedge_to_config(hypergraph,edge,assignment_new,num_partitions)
        cost2 = costs[(root_config,rec_config)]
        gain += cost2 - cost1
    if log:
        print(f'Total gain = {gain}')
    return gain

def find_all_gains(hypergraph,assignment,num_qubits,num_layers,num_partitions,costs,log=False):
    array = [[[0 for _ in range(num_partitions)] for _ in range(num_qubits)] for _ in range(num_layers)]
    for i in range(num_layers):
        for j in range(num_qubits):
            node = (j,i)
            for k in range(num_partitions):
                gain = find_gain(hypergraph,node,k,assignment,num_partitions,costs, log=log)
                array[i][j][k] = gain
    return array

def find_all_gains_dict(hypergraph,nodes,assignment,num_partitions,costs,log=False):
    array = {}
    for node in nodes:
        for k in range(num_partitions):
            if assignment[node[1]][node[0]] != k:
                gain = find_gain(hypergraph,node,k,assignment,num_partitions, costs, log=log)
                array[(node[1],node[0],k)] = gain
    return array

def find_all_gains_dict_h(hypergraph,nodes,assignment,num_partitions,costs,log=False):
    array = {}
    for node in nodes:
        for k in range(num_partitions):
            if assignment[node[1]][node[0]] != k:
                gain = find_gain_h(hypergraph,node,k,assignment,num_partitions, costs, log=log)
                array[(node[1],node[0],k)] = gain
    return array

def fill_buckets_from_dict(array, max_gain):
    buckets = {}
    for i in range(-max_gain,max_gain+1):
        buckets[i] = set()
    for action in array.keys():
        gain = array[action]
        buckets[gain].add(action)
    return buckets

def fill_buckets(array, max_gain,assignment,lock_dict):
    buckets = {}
    for i in range(-max_gain,max_gain+1):
        buckets[i] = set()
    for i in range(len(array)):
        for j in range(len(array[i])):
            for k in range(len(array[i][j])):
                gain = array[i][j][k]
                if gain is not None:
                    node_assignment = assignment[i][j]
                    if k != node_assignment and lock_dict[(j,i)] == False:
                        buckets[gain].add((i,j,k))
    return buckets

def update_counts(counts,node,destination,assignment):
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
        # print(f'Gain bucket {i} has {length} members')
        while length > 0:
            # print(f'Members left {length}')
            action = find_member_random(bucket)
            # print(action)
            node = (action[1],action[0])
            destination = action[2]
            if check_valid(node,destination,spaces):
                # print("Valid")
                if lock_dict[node] == False:
                    # print("Not locked")
                    lock_dict[node] = True
                    gain = i
                    bucket.remove(action)
                    return action, gain
                else:
                    # print("Locked")
                    bucket.remove(action)
                    # buckets[i].remove(action)
                    length -= 1
            else:
                # print("Invalid")
                bucket.remove(action)
                # buckets[i].remove(action)
                length -= 1
    return None, None

def take_action(hypergraph,node,destination,num_partitions,assignment,costs):
    assignment_new = move_node(node,destination,assignment)
    edge_costs = {}
    edge_costs_new = {}
    edge_root_counts_new = {}
    edge_rec_counts_new = {}
    edge_root_configs_new = {}
    edge_rec_configs_new = {}
    # print("Node", node)
    # print("Destination", destination)
    for edge in hypergraph.node2hyperedges[node]:
        # print("Edge", edge)
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        # print("Info", info)
        cost = info['cost']
        cost_new = hedge_to_cost(hypergraph,edge,assignment_new,num_partitions,costs)
        edge_costs[edge] = cost
        edge_costs_new[edge] = cost_new

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

                                   
        edge_root_counts_new[edge] = root_counts_new
        edge_root_configs_new[edge] = root_config_new

        edge_rec_counts_new[edge] = rec_counts_new
        edge_rec_configs_new[edge] = rec_config_new



    return assignment_new, edge_costs, edge_costs_new, edge_root_counts_new, edge_rec_counts_new, edge_root_configs_new, edge_rec_configs_new

def take_action_simple(node,destination,assignment):
    assignment_new = move_node(node,destination,assignment)
    return assignment_new

def update_spaces(node,source,destination,spaces):
    t = node[1]
    spaces[t][destination] -= 1
    spaces[t][source] += 1

def update_full_config(source,destination,full_config,root_config,rec_config):
    # print("Update config", full_config)
    new_full_config = copy.deepcopy(list(full_config))
    # print(full_config)
    # print("Root config", root_config)
    # print("Rec config", rec_config)
    if root_config[source] == 0 and rec_config[source] == 1:
        new_full_config[source] = 1
    else:
        new_full_config[source] = 0
    if root_config[destination] == 0 and rec_config[destination] == 1:
        new_full_config[destination] = 1
    else:
        new_full_config[destination] = 0
    # print("New full config", new_full_config)
    return tuple(new_full_config)

def update_array(hypergraph,array,node,edge_costs,edge_costs_new, edge_root_counts_new, edge_rec_counts_new, edge_root_configs_new, edge_rec_configs_new,destination,assignment,new_assignment,lock_dict,buckets,num_partitions,costs):
    # print(f'Update array for node {node}')
    edges_affected = hypergraph.node2hyperedges[node]
    # print(f'Edges affected {edges_affected}')
    # print(f'Edge costs: {edge_costs}')
    # print(f'Edge costs new: {edge_costs_new}')
    # print(f'Edge root counts new: {edge_root_counts_new}')
    # print(f'Edge rec counts new: {edge_rec_counts_new}')
    # print(f'Edge root configs new: {edge_root_configs_new}')
    # print(f'Edge rec configs new: {edge_rec_configs_new}')
    delta_gains = {}
    for edge in edges_affected:
        # print(f'Edge {edge}')
        info = hypergraph.hyperedge_attrs[edge]
        # print(f'Info {info}')
        nodes = hypergraph.hyperedges[edge]
        # print(f'Node sets {nodes}')

        root_set = nodes['root_set']
        rec_set = nodes['receiver_set']

        conf = info['config']
        # print(f'Configuration before a and b {conf}')
        cost = edge_costs[edge]
        # print(f'Cost before a and b{cost}')

        conf_a = get_full_config(edge_root_configs_new[edge],edge_rec_configs_new[edge])
        # print(f'Configuration after a before b {conf_a}')
        cost_a = edge_costs_new[edge]
        # print(f'Cost after a before b {cost_a}')
        
        root_counts_pre = info['root_counts']
        # print(f'Root counts before a and b {root_counts_pre}')
        rec_counts_pre = info['rec_counts']
        # print(f'Receiver counts before a and b {rec_counts_pre}')

        root_config = info['root_config']
        # print(f'Root config before a and b {root_config}')
        rec_config = info['rec_config']
        # print(f'Receiver config before a and b {rec_config}')

        root_counts_a = edge_root_counts_new[edge]
        # print(f'Root counts after a {root_counts_a}')
        root_config_a = edge_root_configs_new[edge]
        # print(f'Root config after a {root_config_a}')

        rec_counts_a = edge_rec_counts_new[edge]
        # print(f'Receiver counts after a {rec_counts_a}')
        rec_config_a = edge_rec_configs_new[edge]
        # print(f'Receiver config after a {rec_config_a}')

        for next_root_node in root_set:
            # print(f'Next root node {next_root_node}')
            source = assignment[next_root_node[1]][next_root_node[0]]
            if not lock_dict[next_root_node]:
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)
                        # print(f'Destination {next_destination}')

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        next_root_config_b = update_config(root_config, next_root_counts_b, source1, next_destination)

                        # print(f'Root counts after b {next_root_counts_b}')
                        # print(f'Root config after b {next_root_config_b}')

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, new_assignment)
                        next_root_config_ab = update_config(root_config_a, next_root_counts_ab, source2, next_destination)

                        # print(f'Root counts after ab {next_root_counts_ab}')
                        # print(f'Root config after ab {next_root_config_ab}')

                        full_config_b = update_full_config(source1, next_destination, conf, next_root_config_b, rec_config)
                        # print(f'Full config after b {full_config_b}')

                        full_config_ab = update_full_config(source2, next_destination, conf_a, next_root_config_ab, rec_config_a)
                        # print(f'Full config after ab {full_config_ab}')
                        # print(f'Original cost {cost}')
                        # print(f'Cost after a {cost_a}')
                        # print(f'Cost after b {costs[full_config_b]}')
                        # print(f'Cost after ab {costs[full_config_ab]}')
                        delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]
                        # print(f'Delta gain {delta_gain}')
                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain

            for next_rec_node in rec_set:
                # print(f'Next receiver node {next_rec_node}')
                source = assignment[next_rec_node[1]][next_rec_node[0]]
                if not lock_dict[next_rec_node]:
                    # print('Not locked')
                    for next_destination in range(num_partitions):
                        if source != next_destination:
                            next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                            # print(f'Destination {next_destination}')

                            next_rec_counts_b, source1 = update_counts(rec_counts_pre, next_rec_node, next_destination, assignment)
                            next_rec_config_b = update_config(rec_config, next_rec_counts_b, source1, next_destination)

                            # print(f'Receiver counts after b {next_rec_counts_b}')
                            # print(f'Receiver config after b {next_rec_config_b}')

                            next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, new_assignment)
                            next_rec_config_ab = update_config(rec_config_a, next_rec_counts_ab, source2, next_destination)

                            # print(f'Receiver counts after ab {next_rec_counts_ab}')
                            # print(f'Receiver config after ab {next_rec_config_ab}')

                            full_config_b = update_full_config(source1, next_destination, conf, root_config, next_rec_config_b)
                            # print(f'Full config after b {full_config_b}')
                            full_config_ab = update_full_config(source2, next_destination, conf_a, root_config_a, next_rec_config_ab)
                            # print(f'Full config after ab {full_config_ab}')

                            # print(f'Original cost {cost}')
                            # print(f'Cost after a {cost_a}')
                            # print(f'Cost after b {costs[full_config_b]}')
                            # print(f'Cost after ab {costs[full_config_ab]}')
                            delta_gain = cost_a - cost - costs[full_config_ab] + costs[full_config_b]
                            # print(f'Delta gain {delta_gain}')
                            if next_action in delta_gains:
                                delta_gains[next_action] += delta_gain
                            else:
                                delta_gains[next_action] = delta_gain

    for action in delta_gains:
        # print(f'Action {action} Gain change {delta_gains[action]}')
        i = delta_gains[action]
        old_gain = array[action[0]][action[1]][action[2]]
        # print(f'Old gain {old_gain}')
        # print(f'New gain {old_gain - i}')
        if action in buckets[old_gain]:
            # print(f'Old gain in bucket - remove and add to {old_gain - i}')
            buckets[old_gain].remove(action)
            buckets[old_gain - i].add(action)
            
        array[action[0]][action[1]][action[2]] -= i
    return array, buckets

def update_array_simple(hypergraph,array,node,assignment,lock_dict,buckets,num_partitions,costs):
    # print(f'Update array for node {node}')
    edges_affected = hypergraph.node2hyperedges[node]
    node_set = set()
    for edge in edges_affected:
        nodes = hypergraph.hyperedges[edge]
        root_set = nodes['root_set']
        rec_set = nodes['receiver_set']

        all_nodes = root_set.union(rec_set)
        for node in all_nodes:
            node_set.add(node)
    
    for node in node_set:
        if not lock_dict[node]:
            for dest in range(num_partitions):
                old_gain = array[node[1]][node[0]][dest]
                action = (node[1],node[0],dest)
                gain = find_gain(hypergraph,node,dest,assignment,num_partitions,costs)
                if action in buckets[old_gain]:
                    # print(f'Old gain in bucket - remove and add to {old_gain - i}')
                    buckets[old_gain].remove(action)
                    buckets[gain].add(action) 
                array[action[0]][action[1]][action[2]] = gain
    return array, buckets

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
        cost_new = hedge_to_cost(hypergraph,edge,assignment_new,num_partitions,costs)
        
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
            # print(f'Next root node {next_root_node}')
            source = assignment[next_root_node[1]][next_root_node[0]]
            if not lock_dict[next_root_node]:
                # print('Not locked')
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
            # print(f'Next receiver node {next_rec_node}')
            source = assignment[next_rec_node[1]][next_rec_node[0]]
            if not lock_dict[next_rec_node]:
                # print('Not locked')
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
        # print(f'Action {action} Gain change {delta_gains[action]}')
        i = delta_gains[action]
        old_gain = array[action[0]][action[1]][action[2]]
        # print(f'Old gain {old_gain}')
        # print(f'New gain {old_gain - i}')
        if action in buckets[old_gain]:
            # print(f'Old gain in bucket - remove and add to {old_gain - i}')
            buckets[old_gain].remove(action)
            buckets[old_gain - i].add(action)
            
        array[action[0]][action[1]][action[2]] -= i
    return assignment_new, array, buckets

def take_action_and_update_dict(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
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

def take_action_and_update_dict_simple(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
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
            for dest in range(num_partitions):
                if node_assignment != dest:
                    gain = find_gain(hypergraph,node, dest,assignment_new,num_partitions,costs)
                    old_gain = array[(node[1], node[0], dest)]
                    array[(node[1], node[0], dest)] = gain
                    if (node[1], node[0], dest) in buckets[old_gain]:
                        buckets[old_gain].remove((node[1], node[0], dest))
                        buckets[gain].add((node[1], node[0], dest))

    return assignment_new, array, buckets

def take_action_and_update_dict_counts(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
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

def take_action_and_update_dict_hetero(hypergraph,node,destination,array,buckets,num_partitions,lock_dict,assignment,costs):
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

def check_array(hypergraph,array,assignment,num_qubits,num_partitions,costs,lock_dict):
    corrected_array = find_all_gains(hypergraph,assignment,num_qubits, hypergraph.depth, num_partitions, costs)
    for i in range(len(array)):
        for j in range(len(array[i])):
            if not lock_dict[(j,i)]:
                for k in range(len(array[i][j])):

                    correct_value = corrected_array[i][j][k]
                    value = array[i][j][k]
                    if value != correct_value:
                        
                        print(f'Error - gain for moving qubit {j} at time {i} to partition {k} should give {correct_value}')
                        print(f'Instead gives {value}')
                        check_value = find_gain(hypergraph,(j,i),k,assignment,num_partitions,costs,log=True)
                        print(check_value)
                        # print(buckets[correct_value])
                        # print(buckets[value])

                        # print(log_strings[(i,j,k)])

def lock_node(node,lock_dict):
    lock_dict[node] = True
    return lock_dict

def expand_assignment(assignment,qpu_info):
    counts = [copy.deepcopy(qpu_info) for _ in range(len(assignment))]
    new_assignment = np.zeros((len(assignment),np.sum(qpu_info)),dtype=int)

    for t in range(len(assignment)):
        for q in range(len(assignment[0])):
            partition = assignment[t][q]
            counts[t][partition] -= 1
        layer = assignment[t].copy()
        # print(layer)
        # print(counts[t])
        for i, count in enumerate(counts[t]):
            for k in range(count):
                layer = np.concatenate((layer,np.array([i])))
        # print(layer)
        new_assignment[t] = layer
    return np.array(new_assignment)