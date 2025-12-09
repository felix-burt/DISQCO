import copy
import random
import numpy as np
from disqco.graphs.hypergraph_methods import *
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph
import time

def set_initial_partition_assignment(graph : QuantumCircuitHyperGraph,
                                    network : QuantumNetwork, 
                                    invert=False, 
                                    randomise_static=False, 
                                    randomise_full=False, 
                                    node_map=None,
                                    round_robin=False) -> np.ndarray:
    num_qubits = graph.num_qubits
    depth = graph.depth
    static_partition = []
    qpu_info = network.qpu_sizes
    num_partitions = len(qpu_info)
    if node_map is not None:
        inverse_node_map = {v: k for k, v in node_map.items()}
    if isinstance(qpu_info, dict):
        qpu_info = list(qpu_info.values())
    if not round_robin:
        for n in range(num_partitions):
            for k in range(qpu_info[n]):
                if invert == False:
                    qpu_index = inverse_node_map[n] if node_map else n
                    static_partition.append(qpu_index)
                else:
                    static_partition.append(num_partitions-qpu_index-1)
    else:
        for t in range(depth):
            static_partition = []
            for n in range(num_qubits):
                qpu_index = inverse_node_map[n] if node_map else n
                if invert == False:
                    static_partition.append(qpu_index % num_partitions)
                else:
                    static_partition.append(num_partitions - (qpu_index % num_partitions) - 1)

    if randomise_static:
        np.random.permutation(static_partition)
    static_partition_reduced = static_partition[:num_qubits]
    full_partitions = np.zeros((depth,len(static_partition_reduced)),dtype=int)
    for n in range(depth):
        if randomise_full:
            layer = np.array(np.random.permutation(static_partition_reduced), dtype=int)
        else:
            layer = np.array(static_partition_reduced,dtype=int)
        full_partitions[n] = layer
    return full_partitions

def set_initial_partitions_dict(network : QuantumNetwork, num_qubits : int, depth : int, invert: bool = False) -> dict[tuple[int,int], int]:
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

# def find_spaces(num_qubits: int, depth: int, assignment : np.ndarray, qpu_sizes: dict[int, int], graph : QuantumCircuitHyperGraph | None = None) -> dict[int, int]:
#     """
#     Find the number of free qubits in each partition at each time step.
#     num_qubits: number of logical qubits in the circuit
#     assignment: function that maps qubits to partitions
#     network: quantum network object
#     """
#     spaces = {}

#     for t in range(depth):
#         if isinstance(qpu_sizes, dict):
#             keys = list(sorted(qpu_sizes.keys()))
#             spaces[t] = [qpu_sizes[k] for k in keys]
#         else:
#             spaces[t] = [size for size in qpu_sizes]

#         for q in range(num_qubits):
#             # spaces[t][assignment[(q,t)]] -= 1
#             spaces[t][assignment[t][q]] -= 1

    
#     return spaces

def find_spaces(assignment : np.ndarray, qpu_sizes: dict[int, int], graph : QuantumCircuitHyperGraph) -> dict[int, int]:
    """
    Find the number of free qubits in each partition at each time step.
    num_qubits: number of logical qubits in the circuit
    assignment: function that maps qubits to partitions
    network: quantum network object
    """
    depth = graph.depth
    spaces = {}

    for t in range(depth):
        spaces[t] = {qpu : value for qpu, value in qpu_sizes.items()}
    
    if graph is not None:
        for node in graph.nodes:
            if node[0] == 'dummy':
                continue
            q,t = node
            part = assignment[t][q]
            spaces[t][part] -= 1

    
    return spaces

def check_valid(node : tuple[int,int], destination: int, spaces: dict[int, int]) -> bool:
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

def move_node(node: tuple[int,int], 
              destination: int, 
              assignment: dict[tuple[int,int] : int], 
              ) -> dict[tuple[int,int] : int]:
    """ 
    Move a node to a new destination partition by updating the assignment.
    node: tuple of (qubit index, time step)
    destination: destination partition
    assignment: function that maps qubits to partitions
    """

    sub_node = node
    t = sub_node[1]
    q = sub_node[0]

    assignment_new = assignment.copy()  # Use deepcopy to avoid modifying the original assignment
    # assignment_new[(q,t)] = destination
    assignment_new[t][q] = destination

    return assignment_new

def config_from_counts(root_counts,rec_counts):
    # config = np.zeros(len(root_counts),dtype=int)
    config = []
    for (x,y) in zip(root_counts,rec_counts):
        if y > 0 and x < 1:
            config.append(1)
        else:
            config.append(0)  # Changed from config[i] to config.append(0)
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
        # start = time.time()
        cost1 = graph.get_hyperedge_attribute(edge,'cost')
        # stop = time.time()
        # print(f"Time taken for cost1: {stop - start} seconds")
        # start = time.time()
        config2 = map_hedge_to_config(graph, edge, assignment_new, num_partitions)
        # stop = time.time()
        # print(f"Time taken for config2: {stop - start} seconds")
        # start = time.time()
        if config2 not in costs:
            cost2 = config_to_cost(config2)
            costs[config2] = cost2
        else:
            cost2 = costs[config2]
        # stop = time.time()
        # print(f"Time taken for cost2: {stop - start} seconds")
        # cost2 = get_cost(config2,costs)
        gain += cost2 - cost1
    return gain

def find_gain_unmapped(graph : QuantumCircuitHyperGraph, node: tuple[int,int], destination: int, assignment: dict[tuple[int,int] : int], num_partitions: int, costs: dict):

    edges = graph.node2hyperedges[node]
    gain = 0
    source = assignment[node[1]][node[0]]
    for edge in edges:
        # start = time.time()
        root_counts, rec_counts = hedge_k_counts(graph,edge,assignment,num_partitions,set_attrs=False)

        config1 = list(config_from_counts(root_counts, rec_counts)) 

        if node in graph.hyperedges[edge]['root_set']:

            root_counts[source] -= 1
            root_counts[destination] += 1

            config2 = update_config_from_counts(config1, root_counts, rec_counts, source, destination)
        else:
            rec_counts[source] -= 1
            rec_counts[destination] += 1

            config2 = update_config_from_counts(config1, root_counts, rec_counts, source, destination)
        
        gain += costs[tuple(config2)] - costs[tuple(config1)]



    return gain

def find_gain_h(hypergraph, 
                node, 
                destination, 
                assignment, 
                num_partitions, 
                costs = {}, 
                network : QuantumNetwork = None, 
                node_map = None, 
                dummy_nodes = {}):
    
    assignment_new = move_node(node,destination, assignment)
    edges = hypergraph.node2hyperedges[node]
    gain = 0
    for edge in edges:
        cost1 = hypergraph.get_hyperedge_attribute(edge,'cost')
        root_config, rec_config = map_hedge_to_configs(hypergraph, edge, assignment_new, num_partitions, node_map=node_map, dummy_nodes=dummy_nodes)
        if (root_config, rec_config) not in costs:
            _, cost2 = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = cost2
        else:
            cost2 = costs[(root_config,rec_config)]
        gain += cost2 - cost1
    return gain

def find_all_gains(graph : QuantumCircuitHyperGraph, 
                   assignment: np.ndarray, 
                   num_partitions: int, 
                   costs: dict, 
                   network : QuantumNetwork = None,
                   **kwargs):
    
    hetero = network.hetero
    if hetero:
        node_map = kwargs.get('node_map', None)
        dummy_nodes = kwargs.get('dummy_nodes', set())
        active_qpu_nodes = kwargs.get('active_qpu_nodes', set())
        return find_all_gains_hetero(graph,
                                     assignment,
                                     num_partitions,
                                     network=network,
                                     active_qpu_nodes=active_qpu_nodes,
                                     costs=costs,
                                     node_map=node_map,
                                     dummy_nodes=dummy_nodes)
    else:
        return find_all_gains_homo(graph,
                                   assignment,
                                   num_partitions,
                                   costs)

def find_all_gains_homo(hypergraph,
                        assignment,
                        num_partitions,
                        costs,
                        active_qpu_nodes=set(),
                        dummy_nodes=set()):
    array = {}
    if not active_qpu_nodes:
        active_qpu_nodes = set(range(num_partitions))
    for node in hypergraph.nodes - dummy_nodes:
        q,t = node
        source = assignment[t][q]
        for qpu in active_qpu_nodes - set([source]):
            gain = find_gain(hypergraph,node,qpu,assignment,num_partitions, costs)
            array[(t,q,qpu)] = gain
    return array

def find_all_gains_hetero(hypergraph,
                          assignment,
                          num_partitions,
                          network: QuantumNetwork,
                          active_qpu_nodes=set(),
                          costs={},
                          node_map=None,
                          dummy_nodes={}):
    array = {}
    if not active_qpu_nodes:
        active_qpu_nodes = set(range(num_partitions))
    for node in hypergraph.nodes - dummy_nodes:
        q,t = node
        source = assignment[t][q]
        for k in active_qpu_nodes - set([source]):
            gain = find_gain_h(hypergraph,
                                node,
                                k,
                                assignment,
                                num_partitions,
                                costs = costs,
                                network = network,
                                node_map=node_map,
                                dummy_nodes=dummy_nodes)

            array[(t,q,k)] = gain
    return array

def find_all_gains_hetero_sparse(hypergraph,
                                assignment,
                                num_partitions,
                                costs={},
                                network: QuantumNetwork = None,
                                active_qpu_nodes=set(),
                                node_map=None,
                                dummy_nodes=set()):
    array = {}
    if not active_qpu_nodes:
        active_qpu_nodes = set(range(num_partitions))

    for node in hypergraph.nodes - dummy_nodes:
        q, t = node
        source = assignment[t][q]
        for k in active_qpu_nodes - set([source]):
            gain = find_gain_h(hypergraph,
                               node,
                               k,
                               assignment,
                               num_partitions,
                               costs=costs,
                               network=network,
                               node_map=node_map,
                               dummy_nodes=dummy_nodes)
            array[(t,q,k)] = gain
    return array

def fill_buckets(array, max_gain, boundary_nodes = None):
    buckets = {}
    for i in range(-max_gain,max_gain+1):
        buckets[i] = set()
    for action in array.keys():
        if boundary_nodes is not None:
            if (action[1], action[0]) in boundary_nodes:
                gain = array[action]
                buckets[gain].add(action)
        else:
            gain = array[action]
            buckets[gain].add(action)
    return buckets

def update_counts(counts,
                  node,
                  destination,
                  assignment,
                  node_map=None):
    # partition = assignment[node]
    partition = assignment[node[1]][node[0]]
    # print(f'Unmapped partition: {partition}')
    if node_map is not None:
        partition = node_map[partition]
        # print(f'Mapped partition: {partition}')

    new_counts = counts.copy()
    new_counts[partition] -= 1
    new_counts[destination] += 1

    return new_counts, partition

def increment_index(config, 
                    new_config,
                    source, 
                    destination, 
                    num_partitions):
    index_inc = 0
    if new_config[source] == 0 and config[source] == 1:
        index_inc -= 2**(num_partitions-source-1)
    
    elif new_config[source] == 1 and config[source] == 0:
        index_inc += 2**(num_partitions-source-1)
    
    if new_config[destination] == 1 and config[destination] == 0:
        index_inc += 2**(num_partitions-destination-1)
    elif new_config[destination] == 0 and config[destination] == 1:
        index_inc -= 2**(num_partitions-destination-1)

    return index_inc

def update_config_from_counts(config,
                              root_counts,
                              rec_counts,
                              partition,
                              destination):
    # new_config = copy.deepcopy(list(config))
    new_config = config.copy()

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
    
    return new_config

def update_config(old_config, 
                  new_counts, 
                  source, 
                  destination):
    new_config = copy.deepcopy(list(old_config))
    if new_counts[source] == 0:
        new_config[source] = 0
    if new_counts[destination] > 0:
        new_config[destination] = 1
    return tuple(new_config)

def find_member_random(set):
    member = random.choice(list(set))
    # for member in set: # Choose first element 
    #     break
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

def update_full_config(source,
                       destination,
                       full_config,
                       root_config,
                       rec_config):

    new_full_config = full_config.copy()

    if root_config[source] == 0 and rec_config[source] == 1:
        new_full_config[source] = 1
    else:
        new_full_config[source] = 0
    if root_config[destination] == 0 and rec_config[destination] == 1:
        new_full_config[destination] = 1
    else:
        new_full_config[destination] = 0
    return new_full_config

def take_action_and_update(hypergraph,
                           node,
                           destination,
                           array,
                           buckets,
                           num_partitions,
                           lock_dict,
                           assignment,
                           costs = {},
                           network : QuantumNetwork = None,
                           boundary_nodes = None,
                           boundary_counts = None,
                           **kwargs):
    hetero = kwargs.get('hetero', False)
    sparse = kwargs.get('sparse', False)
    if hetero:
        node_map = kwargs.get('node_map', None)
        if sparse:
            dummy_nodes = kwargs.get('dummy_nodes', set())
            return take_action_and_update_hetero_sparse(hypergraph,
                                                        node,
                                                        destination,
                                                        array,
                                                        buckets,
                                                        lock_dict,
                                                        assignment,
                                                        costs=costs,
                                                        network=network,
                                                        node_map=node_map)
        return take_action_and_update_hetero(hypergraph,
                                             node,
                                             destination,
                                             array,
                                             buckets,
                                             num_partitions,
                                             lock_dict,
                                             assignment,
                                             costs=costs,
                                             network=network,
                                             node_map=node_map)
    else:
        return take_action_and_update_homo(hypergraph,
                                           node,
                                           destination,
                                           array,
                                           buckets,
                                           num_partitions,
                                           lock_dict,
                                           assignment,
                                           costs,
                                           boundary_nodes,
                                           boundary_counts)

def take_action_and_update_homo(hypergraph,
                                node,
                                destination,
                                array,
                                buckets,
                                num_partitions,
                                lock_dict,
                                assignment,
                                costs,
                                boundary_nodes=None,
                                boundary_counts=None):
    
    assignment_new = move_node(node,destination,assignment)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:

        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']

        cost = info['cost']
        conf = info['config']
        root_counts = info['root_counts']
        rec_counts = info['rec_counts']


        if node in root_set:
            root_counts_new, source = update_counts(root_counts,node,destination,assignment)
            rec_counts_new = rec_counts.copy()
            config_new = update_config_from_counts(conf,root_counts_new,rec_counts_new,source,destination)

        elif node in rec_set:
            rec_counts_new, source = update_counts(rec_counts,node,destination,assignment)
            root_counts_new = root_counts.copy()
            config_new = update_config_from_counts(conf,root_counts_new,rec_counts_new,source,destination)

        if tuple(config_new) not in costs:
            cost_a = config_to_cost(config_new,costs)
            costs[tuple(config_new)] = cost_a
        else:
            cost_a = costs[tuple(config_new)]

        conf_a = config_new

        if boundary_nodes is not None:
            if cost == 0 and cost_a == 1:
                for b_node in root_set | rec_set:
                    boundary_counts[b_node] += 1
                    if boundary_counts[b_node] == 1:
                        boundary_nodes.add(b_node)
            if cost_a == 0 and cost > 0:
                for b_node in root_set | rec_set:
                    boundary_counts[b_node] -= 1
                    if boundary_counts[b_node] == 0:
                        boundary_nodes.remove(b_node)

        root_counts_pre = root_counts
        rec_counts_pre = rec_counts

        root_counts_a = root_counts_new
        rec_counts_a = rec_counts_new

        for next_root_node in root_set:
            source = assignment[next_root_node[1]][next_root_node[0]]
            if not lock_dict[next_root_node]:
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        full_config_b = update_config_from_counts(conf,next_root_counts_b,rec_counts_pre,source1,next_destination)

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, assignment_new)
                        full_config_ab = update_config_from_counts(conf_a,next_root_counts_ab,rec_counts_a,source2,next_destination)
                        if tuple(full_config_ab) not in costs:
                            cost_ab = config_to_cost(full_config_ab,costs)
                            costs[tuple(full_config_ab)] = cost_ab
                        else:
                            cost_ab = costs[tuple(full_config_ab)]

                        if tuple(full_config_b) not in costs:
                            cost_b = config_to_cost(full_config_b,costs)
                            costs[tuple(full_config_b)] = cost_b
                        else:
                            cost_b = costs[tuple(full_config_b)]

                        delta_gain = cost_a - cost - cost_ab + cost_b

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
                        full_config_b = update_config_from_counts(conf,root_counts_pre,next_rec_counts_b,source1,next_destination)

                        next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, assignment_new)
                        full_config_ab = update_config_from_counts(conf_a,root_counts_a,next_rec_counts_ab,source2,next_destination)

                        if tuple(full_config_ab) not in costs:
                            cost_ab = config_to_cost(full_config_ab,costs)
                            costs[tuple(full_config_ab)] = cost_ab
                        else:
                            cost_ab = costs[tuple(full_config_ab)]
                        if tuple(full_config_b) not in costs:
                            cost_b = config_to_cost(full_config_b,costs)
                            costs[tuple(full_config_b)] = cost_b
                        else:
                            cost_b = costs[tuple(full_config_b)]

                        delta_gain = cost_a - cost - cost_ab + cost_b



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
        node = (action[1], action[0])
        if action in buckets[old_gain]:
            # print(f'Old gain in bucket - remove and add to {old_gain - i}')
            buckets[old_gain].remove(action)
            if node in boundary_nodes:
                buckets[old_gain - i].add(action)

        elif action not in buckets[old_gain] and node in boundary_nodes:
            buckets[old_gain - i].add(action)
            
        array[action] -= i
    # print(f'New buckets: {buckets}')
    return assignment_new, array, buckets

def take_action_and_update_hetero(hypergraph,
                                  node,
                                  destination,
                                  array,
                                  buckets,
                                  num_partitions,
                                  lock_dict,
                                  assignment,
                                  network: QuantumNetwork,
                                  costs={},
                                  node_map=None):
    assignment_new = move_node(node, destination, assignment)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:
        
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        cost = info['cost'] 
        root_counts = info['root_counts']
        rec_counts = info['rec_counts']
        if node in root_set:
            root_counts_new, source = update_counts(root_counts,node,destination,assignment)
            root_config_new = update_config(info['root_config'],root_counts_new,source,destination)
            rec_counts_new = rec_counts.copy()
            rec_config_new = tuple(copy.deepcopy(list(info['rec_config'])))
        elif node in rec_set:
            rec_counts_new, source = update_counts(rec_counts,node,destination,assignment)
            rec_config_new = update_config(info['rec_config'],rec_counts_new,source,destination)
            root_counts_new = root_counts.copy()
            root_config_new = tuple(copy.deepcopy(list(info['root_config'])))
        
        if (root_config_new, rec_config_new) not in costs:
            _, cost_a = network.steiner_forest(root_config_new, rec_config_new, node_map=node_map)
            costs[(root_config_new, rec_config_new)] = cost_a
        else:
            cost_a = costs[(root_config_new, rec_config_new)]

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
            if next_root_node not in lock_dict:
                continue

            # source = assignment[next_root_node]
            if not lock_dict[next_root_node]:
                next_root_node_sub = next_root_node
                source = assignment[next_root_node_sub[1]][next_root_node_sub[0]]
                # print('Not locked')
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_root_node[1], next_root_node[0], next_destination)

                        next_root_counts_b, source1 = update_counts(root_counts_pre, next_root_node, next_destination, assignment)
                        next_root_config_b = update_config(root_config, next_root_counts_b, source1, next_destination)

                        next_root_counts_ab, source2 = update_counts(root_counts_a, next_root_node, next_destination, assignment_new)
                        next_root_config_ab = update_config(root_config_a, next_root_counts_ab, source2, next_destination)

                        if (next_root_config_b, rec_config) not in costs:
                            _, cost_b = network.steiner_forest(next_root_config_b, rec_config, node_map=node_map)
                            costs[(next_root_config_b, rec_config)] = cost_b
                        else:
                            cost_b = costs[(next_root_config_b, rec_config)]

                        if (next_root_config_ab, rec_config_a) not in costs:
                            _, cost_ab = network.steiner_forest(next_root_config_ab, rec_config_a, node_map=node_map)
                            costs[(next_root_config_ab, rec_config_a)] = cost_ab
                        else:
                            cost_ab = costs[(next_root_config_ab,rec_config_a)]

                        delta_gain = cost_a - cost - cost_ab + cost_b

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain

        for next_rec_node in rec_set:
            if next_rec_node not in lock_dict:
                continue
            if not lock_dict[next_rec_node]:
                next_rec_node_sub = next_rec_node

                source = assignment[next_rec_node_sub[1]][next_rec_node_sub[0]]
                for next_destination in range(num_partitions):
                    if source != next_destination:
                        next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                        next_rec_counts_b, source1 = update_counts(rec_counts_pre, next_rec_node, next_destination, assignment)
                        next_rec_config_b = update_config(rec_config, next_rec_counts_b, source1, next_destination)
                        
                        next_rec_counts_ab, source2 = update_counts(rec_counts_a, next_rec_node, next_destination, assignment_new)
                        next_rec_config_ab = update_config(rec_config_a, next_rec_counts_ab, source2, next_destination)

                        if (root_config, next_rec_config_b) not in costs:
                            _, cost_b = network.steiner_forest(root_config, next_rec_config_b, node_map=node_map)
                            costs[(root_config, next_rec_config_b)] = cost_b
                        else:
                            cost_b = costs[(root_config,next_rec_config_b)]
                        if (root_config_a, next_rec_config_ab) not in costs:
                            _, cost_ab = network.steiner_forest(root_config_a, next_rec_config_ab, node_map=node_map)
                            costs[(root_config_a, next_rec_config_ab)] = cost_ab
                        else:
                            
                            cost_ab = costs[(root_config_a,next_rec_config_ab)]

                        delta_gain = cost_a - cost - cost_ab + cost_b

                        if next_action in delta_gains:
                            delta_gains[next_action] += delta_gain
                        else:
                            delta_gains[next_action] = delta_gain
            
        hypergraph.set_hyperedge_attribute(edge, 'cost', cost_a)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts_new)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config_new)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config_new)
            

    for action in delta_gains:
        i = delta_gains[action]
        old_gain = array[action]
        if action in buckets[old_gain]:
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

def transform_assignment(assignment, mapping, qpu_sizes, node_map):
    new_assignment = copy.deepcopy(assignment)
    mapping_list_sizes_base = {}
    for qpu_coarse in mapping:
        contained_qpus = mapping[qpu_coarse]
        mapping_list_sizes_base[qpu_coarse] = {}
        for qpu in contained_qpus:
            mapping_list_sizes_base[qpu_coarse][qpu] = qpu_sizes[qpu]   
    for t in range(len(assignment)):
        mapping_list_sizes = copy.deepcopy(mapping_list_sizes_base)
        for q in range(len(assignment[t])):
            coarse_qpu = assignment[t][q]
            fine_qpu = node_map[coarse_qpu]
            options = mapping_list_sizes[fine_qpu]
            for choice in options:
                if mapping_list_sizes[fine_qpu][choice] > 0:
                    new_assignment[t][q] = choice
                    mapping_list_sizes[fine_qpu][choice] -= 1
                    if mapping_list_sizes[fine_qpu][choice] == 0:
                        del mapping_list_sizes[fine_qpu][choice]
                break
    return new_assignment

def order_nodes(g : QuantumCircuitHyperGraph):
    depth = g.depth
    node_list = [[] for _ in range(depth)]
    for node in g.nodes:
        if node[0] == "dummy":
            continue
        q, t = node
        node_list[t].append(q)  
    return node_list

def sort_node_list(node_list : list[int]):
    sorted_node_list = []
    for layer in node_list:
        sorted_layer = sorted(layer)
        sorted_node_list.append(sorted_layer)

    return sorted_node_list

def set_initial_sub_partitions(sub_network : QuantumNetwork, node_list : list[list[int]], active_nodes) -> np.ndarray:
    """
    Set the initial partitions for the sub-network.
    """
    assignment = []
    max_qubits_per_layer = 0
    for layer in node_list:
        assignment_layer = []
        num_qubits_layer = len(layer)
        if num_qubits_layer > max_qubits_per_layer:
            max_qubits_per_layer = num_qubits_layer
            
        counter = 0
        for i, (qpu, size) in enumerate(sub_network.qpu_sizes.items()):
            if qpu in active_nodes:
                for j in range(size):
                    assignment_layer.append(counter)
                counter += 1    
        
        assignment.append(assignment_layer)
        

    assignment = np.array([np.array(assignment[j][:max_qubits_per_layer]) for j in range(len(assignment))])

        
    
    return assignment


def refine_assignment(level, num_levels, assignment, mapping_list):
    new_assignment = assignment
    if level < num_levels -1:
        mapping = mapping_list[level]
        for super_node_t in mapping:
            for t in mapping[super_node_t]:
                new_assignment[t] = assignment[super_node_t]
    return new_assignment

def take_action_and_update_hetero_sparse(hypergraph,
                                  node,
                                  destination,
                                  array,
                                  buckets,
                                  lock_dict,
                                  assignment,
                                  costs = {},
                                  network : QuantumNetwork = None,
                                  node_map = None):
    
    assignment_new = move_node(node,destination,assignment)
    delta_gains = {}
    for edge in hypergraph.node2hyperedges[node]:
        
        info = hypergraph.hyperedge_attrs[edge]
        root_set = hypergraph.hyperedges[edge]['root_set']
        rec_set = hypergraph.hyperedges[edge]['receiver_set']
        cost = info['cost']

        root_counts = info['root_counts']
        rec_counts = info['rec_counts']
        destination_mapped = destination
        if node_map is not None:
            destination_mapped = node_map[destination]
        if node in root_set:
            
            root_counts_new, source_mapped = update_counts(root_counts,node,destination_mapped,assignment,node_map=node_map)
            root_config_new = update_config(info['root_config'],root_counts_new,source_mapped,destination_mapped)
            rec_counts_new = rec_counts.copy()
            rec_config_new = tuple(copy.deepcopy(list(info['rec_config'])))
        elif node in rec_set:
            rec_counts_new, source_mapped = update_counts(rec_counts,node,destination_mapped,assignment,node_map=node_map)
            rec_config_new = update_config(info['rec_config'],rec_counts_new,source_mapped,destination_mapped)
            root_counts_new = root_counts.copy()
            root_config_new = tuple(copy.deepcopy(list(info['root_config'])))
        
        if (root_config_new, rec_config_new) not in costs:
            _, cost_a = network.steiner_forest(root_config_new, rec_config_new, node_map=node_map)
            costs[(root_config_new, rec_config_new)] = cost_a
        else:
            cost_a = costs[(root_config_new, rec_config_new)]

        root_counts_pre = root_counts
        rec_counts_pre = rec_counts
        
        root_config = info['root_config']
        rec_config = info['rec_config']

        root_counts_a = root_counts_new
        root_config_a = root_config_new
        rec_counts_a = rec_counts_new
        rec_config_a = rec_config_new
        destination_set = network.active_nodes
        for next_root_node in root_set:
            if next_root_node not in lock_dict:
                continue
            if lock_dict[next_root_node]:
                continue
            source = assignment[next_root_node[1]][next_root_node[0]]
            
            for next_destination in destination_set - {source}:
                if node_map is not None:
                    next_destination_mapped = node_map[next_destination]
                next_action = (next_root_node[1], next_root_node[0], next_destination)

                next_root_counts_b, source1 = update_counts(root_counts_pre, 
                                                            next_root_node, 
                                                            next_destination_mapped, 
                                                            assignment, 
                                                            node_map=node_map)
                next_root_config_b = update_config(root_config, 
                                                    next_root_counts_b, 
                                                    source1, 
                                                    next_destination_mapped)

                next_root_counts_ab, source2 = update_counts(root_counts_a, 
                                                                next_root_node, 
                                                                next_destination_mapped, 
                                                                assignment_new, 
                                                                node_map=node_map)
                next_root_config_ab = update_config(root_config_a, 
                                                        next_root_counts_ab, 
                                                        source2, 
                                                        next_destination_mapped)

                if (next_root_config_b, rec_config) not in costs:
                    _, cost_b = network.steiner_forest(next_root_config_b, rec_config, node_map=node_map)
                    costs[(next_root_config_b, rec_config)] = cost_b
                else:
                    cost_b = costs[(next_root_config_b, rec_config)]

                if (next_root_config_ab, rec_config_a) not in costs:
                    _, cost_ab = network.steiner_forest(next_root_config_ab, rec_config_a, node_map=node_map)
                    costs[(next_root_config_ab, rec_config_a)] = cost_ab
                else:
                    cost_ab = costs[(next_root_config_ab,rec_config_a)]

                delta_gain = cost_a - cost - cost_ab + cost_b

                if next_action in delta_gains:
                    delta_gains[next_action] += delta_gain
                else:
                    delta_gains[next_action] = delta_gain

        for next_rec_node in rec_set:
            if next_rec_node not in lock_dict:
                continue
            if lock_dict[next_rec_node]:
                continue
            next_rec_node_sub = next_rec_node
            source = assignment[next_rec_node_sub[1]][next_rec_node_sub[0]]
            for next_destination in destination_set - {source}:

                if node_map is not None:
                    next_destination_mapped = node_map[next_destination]

                if source != next_destination:
                    next_action = (next_rec_node[1], next_rec_node[0], next_destination)
                    
                    next_rec_counts_b, source1 = update_counts(rec_counts_pre, 
                                                                next_rec_node, 
                                                                next_destination_mapped, 
                                                                assignment, 
                                                                node_map=node_map)
                    next_rec_config_b = update_config(rec_config,
                                                        next_rec_counts_b, 
                                                        source1, 
                                                        next_destination_mapped)

                    next_rec_counts_ab, source2 = update_counts(rec_counts_a, 
                                                                next_rec_node, 
                                                                next_destination_mapped, 
                                                                assignment_new, 
                                                                node_map=node_map)
                    
                    next_rec_config_ab = update_config(rec_config_a, 
                                                        next_rec_counts_ab, 
                                                        source2, 
                                                        next_destination_mapped)

                    if (root_config, next_rec_config_b) not in costs:
                        _, cost_b = network.steiner_forest(root_config, next_rec_config_b, node_map=node_map)
                        costs[(root_config, next_rec_config_b)] = cost_b
                    else:
                        cost_b = costs[(root_config,next_rec_config_b)]
                    if (root_config_a, next_rec_config_ab) not in costs:
                        _, cost_ab = network.steiner_forest(root_config_a, next_rec_config_ab, node_map=node_map)
                        costs[(root_config_a, next_rec_config_ab)] = cost_ab
                    else:
                        
                        cost_ab = costs[(root_config_a,next_rec_config_ab)]

                    delta_gain = cost_a - cost - cost_ab + cost_b

                    if next_action in delta_gains:
                        delta_gains[next_action] += delta_gain
                    else:
                        delta_gains[next_action] = delta_gain
            
        hypergraph.set_hyperedge_attribute(edge, 'cost', cost_a)
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