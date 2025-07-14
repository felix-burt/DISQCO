from itertools import product
import numpy as np
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph

def get_all_configs(num_partitions : int, hetero = False) -> list[tuple[int]]:
    """
    Generates all possible configurations for a given number of partitions."
    """
    # Each configuration is represented as a tuple of 0s and 1s, where 1 indicates
    # that at least one qubit in the edge is assigned to the current partition.
    configs = set(product((0,1),repeat=num_partitions))
    if hetero:
        configs = configs - set([(0,)*num_partitions])

    return list(configs)

def config_to_cost(config : tuple[int]) -> int:
    """
    Converts a configuration tuple to its corresponding cost (assuming all to all connectivity)."
    """
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    return cost

def get_all_costs_hetero(network, 
                         configs : list[tuple[int]], 
                         node_map = None
                         ) -> tuple[dict[tuple[tuple[int],tuple[int]] : int], 
                                    dict[tuple[tuple[int],tuple[int]]] : list[tuple[int]]]:
    """
    Computes the costs and edge forests for all configurations using the provided network."
    """
    costs = {}
    edge_trees = {}
    for root_config in configs:
        for rec_config in configs:
            edges, cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = cost
            edge_trees[(root_config, rec_config)] = edges
    return costs, edge_trees

def get_all_costs(configs : list[tuple[int]]
                  ) -> dict[tuple[int] : int]:
    """
    Computes the costs for all configurations given all-to-all connectivity.
    """

    # costs = np.zeros(len(configs)+1)

    # for config in configs:
    #     cost = config_to_cost(config)
    #     integer = int("".join(map(str, config)), 2)
    #     costs[integer] = cost
    costs = {}
    for config in configs:
        cost = config_to_cost(config)
        costs[tuple(config)] = cost

    return costs

def get_cost(config : tuple[int], costs : np.array) -> int:
        config = list(config)
        config = [str(x) for x in config]
        config = "".join(config)
        config = int(config, 2)
        return costs[config]

def hedge_k_counts(hypergraph,hedge,assignment,num_partitions, set_attrs = False, assignment_map = None, dummy_nodes = {}):
    # root_counts = np.zeros(num_partitions + len(dummy_nodes), dtype=int)
    # rec_counts = np.zeros(num_partitions + len(dummy_nodes), dtype=int)   
    root_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
    rec_counts = [0 for _ in range(num_partitions + len(dummy_nodes))]
    info = hypergraph.hyperedges[hedge]
    root_set = info['root_set']
    receiver_set = info['receiver_set']

    if dummy_nodes == {}:
        for root_node in root_set:
            if assignment_map is not None:
                root_node = assignment_map[root_node]

            partition_root = assignment[root_node[1]][root_node[0]]
            # partition_root = assignment[root_node]

            root_counts[partition_root] += 1
        for rec_node in receiver_set:
            if assignment_map is not None:
                rec_node = assignment_map[rec_node]

            partition_rec = assignment[rec_node[1]][rec_node[0]]

            # partition_rec = assignment[rec_node]
            rec_counts[partition_rec] += 1
        
        # root_counts = tuple(root_counts)
        # rec_counts = tuple(rec_counts)
        if set_attrs:
            hypergraph.set_hyperedge_attribute(hedge, 'root_counts', root_counts)
            hypergraph.set_hyperedge_attribute(hedge, 'rec_counts', rec_counts)
    else:
        for root_node in root_set:
            if root_node not in hypergraph.nodes:
                continue

            if root_node in dummy_nodes:
                # print("Dummy node root", root_node)
                partition_root = num_partitions + root_node[3]
                # print("Partition dummy root", partition_root)
                root_counts[partition_root] += 1
                continue

            if assignment_map is not None:
                root_node = assignment_map[root_node]
            try:
                partition_root = assignment[root_node[1]][root_node[0]]
            except Exception:
                continue
            # partition_root = assignment[root_node]
            root_counts[partition_root] += 1
        for rec_node in receiver_set:
            if rec_node not in hypergraph.nodes:
                continue
            if rec_node in dummy_nodes:
                # print("Dummy node rec", rec_node)
                partition_rec = num_partitions + rec_node[3]
                # print("Partition dummy rec", partition_rec)
                rec_counts[partition_rec] += 1
                continue

            if assignment_map is not None:
                rec_node = assignment_map[rec_node]
                
            try:
                partition_rec = assignment[rec_node[1]][rec_node[0]]
            except Exception:
                # print("Rec node", rec_node)
                # print("Assignment", assignment)
                # print("Assignment shape", np.shape(assignment))
                # print("Assignment map", assignment_map)
                # print("Root set", root_set)
                # print("Receiver set", receiver_set)
                # for node in hypergraph.nodes:
                #     print("Node", node)
                #     print("Mapped node", assignment_map[node] if assignment_map is not None else node)
                continue
            # partition_rec = assignment[rec_node]
            rec_counts[partition_rec] += 1
        
        # root_counts = tuple(root_counts)
        # rec_counts = tuple(rec_counts)
        if set_attrs:
            hypergraph.set_hyperedge_attribute(hedge, 'root_counts', root_counts)
            hypergraph.set_hyperedge_attribute(hedge, 'rec_counts', rec_counts)

    
    return root_counts, rec_counts

def counts_to_configs(root_counts : tuple[int], rec_counts : tuple[int]) -> tuple[tuple[int], tuple[int]]:
    """
    Converts the counts of nodes in each partition to root and rec config tuples."
    """
    root_config = []
    rec_config = []
    for x,y in zip(root_counts,rec_counts):
        if x > 0:
            root_config.append(1)
        else:
            root_config.append(0)
        if y > 0:
            rec_config.append(1)
        else:
            rec_config.append(0)
    return tuple(root_config), tuple(rec_config)

def full_config_from_counts(root_counts : tuple[int], 
                       rec_counts : tuple[int]
                       ) -> tuple[int]:
    """
    Converts the counts of nodes in each partition to full configuration tuple.
    """
    config = []
    for x,y in zip(root_counts,rec_counts):
        if y > 0 and x < 1:
            config.append(1)
        else:
            config.append(0)
    return config

def map_hedge_to_config(hypergraph : QuantumCircuitHyperGraph, 
                          hedge : tuple, 
                          assignment : dict[tuple[int,int]], 
                          num_partitions : int
                          ) -> tuple[int]:
    
    """
    Maps a hyperedge to its full configuration based on the current assignment.
    Uses config_from_counts to skip the intermediate step of counts_to_configs.
    """
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
    config = full_config_from_counts(root_counts,rec_counts)

    return config

def map_hedge_to_configs(hypergraph,hedge,assignment,num_partitions,assignment_map = None, dummy_nodes = {}):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False,assignment_map=assignment_map, dummy_nodes=dummy_nodes)
    root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    # config = config_from_counts(root_counts,rec_counts)
    return root_config,rec_config

def get_full_config(root_config : tuple[int], rec_config : tuple[int]) -> tuple[int]:
    """
    Converts the root and receiver configurations to a full configuration tuple."
    """
    config = list(rec_config)
    for i, element in enumerate(root_config):
        if rec_config[i] == 1:
            config[i] -= element
    return config

def hedge_to_cost(hypergraph : QuantumCircuitHyperGraph, 
                   hedge : tuple, 
                   assignment : dict[tuple[int,int]], 
                   num_partitions : int, 
                   costs : dict[tuple] = {}) -> int:
    """
    Computes the cost of a hyperedge based on its configuration and the current assignment.
    """ 
    config = map_hedge_to_config(hypergraph, hedge, assignment, num_partitions)

    # if config not in costs:
    #     cost = config_to_cost(config)
    #     configint = list(config)
    #     configint = [str(x) for x in configint]
    #     configint = "".join(configint)
    #     costs[config] = cost
    # else:
        # cost = costs[config]
    cost = get_cost(config, costs)
    return cost

def hedge_to_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                         hedge : tuple, 
                         assignment : dict[tuple[int,int]], 
                         num_partitions : int, 
                         costs : dict[tuple] = {},
                         network = None,
                         assignment_map = None,
                            dummy_nodes = {}
                         ) -> int:
    """"
    Computes the cost of a hyperedge based on its configuration and the current assignment."
    """
    root_config, rec_config = map_hedge_to_configs(hypergraph, hedge, assignment, num_partitions, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
    if (root_config, rec_config) not in costs:
        edges, cost = network.steiner_forest(root_config, rec_config)
        costs[(root_config, rec_config)] = cost
    else:
        cost = costs[(root_config, rec_config)]
    return cost

def map_current_costs(hypergraph : QuantumCircuitHyperGraph, 
                      assignment : dict[tuple[int,int]], 
                      num_partitions : int, 
                      costs: dict
                      ) -> None:
    """
    Maps the current costs of all hyperedges to hyperedge attributes based on the current assignment.
    """
    for edge in hypergraph.hyperedges:
        hypergraph.set_hyperedge_attribute(edge, 'cost', hedge_to_cost(hypergraph,edge,assignment,num_partitions,costs))
    return
        
def map_counts_and_configs(hypergraph : QuantumCircuitHyperGraph, 
                            assignment : dict[tuple[int,int]], 
                            num_partitions : int, 
                            costs: dict,
                            **kwargs) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    """
    hetero = kwargs.get('hetero', False)
    if hetero:
        network = kwargs.get('network', None)
        node_map = kwargs.get('node_map', None)
        assignment_map = kwargs.get('assignment_map', None)
        dummy_nodes = kwargs.get('dummy_nodes', {})
        return map_counts_and_configs_hetero(hypergraph, assignment, num_partitions, network, costs=costs, assignment_map=assignment_map, node_map=node_map, dummy_nodes=dummy_nodes)
    else:
        return map_counts_and_configs_homo(hypergraph, assignment, num_partitions, costs=costs)

def map_counts_and_configs_homo(hypergraph : QuantumCircuitHyperGraph, 
                            assignment : dict[tuple[int,int]], 
                            num_partitions : int, 
                            costs: dict) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    """
    for edge in hypergraph.hyperedges:
        # print("Edge", edge)
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)

        config = full_config_from_counts(root_counts, rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'config', config)
        if tuple(config) not in costs:
            cost = config_to_cost(config)
            costs[tuple(config)] = cost
        else:
            cost = costs[tuple(config)]

        hypergraph.set_hyperedge_attribute(edge, 'cost', cost)
    return hypergraph

def map_counts_and_configs_hetero(hypergraph : QuantumCircuitHyperGraph,
                                  assignment : dict[tuple[int,int]],
                                  num_partitions : int,
                                  network,
                                  costs: dict = {},
                                  assignment_map = None,
                                  node_map = None,
                                  dummy_nodes = {}) -> QuantumCircuitHyperGraph:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        root_config, rec_config = counts_to_configs(root_counts,rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
        if (root_config, rec_config) not in costs:
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = edge_cost
        else:
            edge_cost = costs[(root_config, rec_config)]
        hypergraph.set_hyperedge_attribute(edge, 'cost', edge_cost)
    return hypergraph

def calculate_full_cost(hypergraph : QuantumCircuitHyperGraph,
                        assignment,
                        num_partitions : int,
                        costs: dict = {},
                        **kwargs) -> int:
    """
    Wrapper function for computing full cost under either homogeneous (fully connected) or heterogeneous (not fully connected) networks.
    """
    hetero = kwargs.get('hetero', False)
    if hetero:
        network = kwargs.get('network', None)
        node_map = kwargs.get('node_map', None)
        assignment_map = kwargs.get('assignment_map', None)
        dummy_nodes = kwargs.get('dummy_nodes', {})
        return calculate_full_cost_hetero(  hypergraph, 
                                            assignment, 
                                            num_partitions, 
                                            costs=costs, 
                                            network=network,
                                            node_map=node_map, 
                                            assignment_map=assignment_map, 
                                            dummy_nodes=dummy_nodes )
    else:  
        return calculate_full_cost_homo(    hypergraph, 
                                            assignment, 
                                            num_partitions, 
                                            costs=costs )

def calculate_full_cost_homo(hypergraph : QuantumCircuitHyperGraph,
                        assignment : dict[tuple[int,int]],
                        num_partitions : int,
                        costs: dict = {}) -> int:
    """
    Computes the total cost of the hypergraph based on the current assignment and the costs of each configuration.
    """
    cost = 0
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions)
        config = full_config_from_counts(root_counts,rec_counts)
        conf = tuple(config)
        if conf not in costs:
            edge_cost = config_to_cost(config)
            costs[conf] = edge_cost
        else:
            edge_cost = costs[conf]
        cost += edge_cost
    return cost

def calculate_full_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                               assignment : dict[tuple[int,int]],
                               num_partitions : int,
                               costs: dict = {},
                               network = None, 
                               node_map: dict = None,
                               assignment_map = None,
                               dummy_nodes = {}) -> int:
    """
    Computes the total cost of the hypergraph based on the current assignment and the costs of each configuration.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    cost = 0
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, assignment_map=assignment_map, dummy_nodes=dummy_nodes)
        root_config, rec_config = counts_to_configs(root_counts, rec_counts)

        if (root_config, rec_config) in costs:
            edge_cost = costs[(root_config, rec_config)]
        else:
            
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = edge_cost
        cost += edge_cost
    
    return cost