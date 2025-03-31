from itertools import product
import numpy as np
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph

def get_all_configs(num_partitions : int) -> list[tuple[int]]:
    """
    Generates all possible configurations for a given number of partitions."
    """
    # Each configuration is represented as a tuple of 0s and 1s, where 1 indicates
    # that at least one qubit in the edge is assigned to the current partition.
    configs = list(product((0,1),repeat=num_partitions))
    return configs

def config_to_cost(config : tuple[int]) -> int:
    """
    Converts a configuration tuple to its corresponding cost (assuming all to all connectivity)."
    """
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    return cost

def get_all_costs_hetero(network : QuantumNetwork, 
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
    costs = {}
    for config in configs:
        costs[config] = config_to_cost(config)
    return costs

def hedge_k_counts(hypergraph,hedge,assignment,num_partitions, set_attrs = False):
    root_counts = [0 for _ in range(num_partitions)]
    rec_counts = [0 for _ in range(num_partitions)]
    info = hypergraph.hyperedges[hedge]
    root_set = info['root_set']
    receiver_set = info['receiver_set']
    for root_node in root_set:
        partition_root = assignment[root_node[1]][root_node[0]]
        # partition_root = assignment[root_node]
        root_counts[partition_root] += 1
    for rec_node in receiver_set:
        partition_rec = assignment[rec_node[1]][rec_node[0]]
        # partition_rec = assignment[rec_node]
        rec_counts[partition_rec] += 1
    root_counts = tuple(root_counts)
    rec_counts = tuple(rec_counts)
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
    return tuple(config)

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

def map_hedge_to_configs(hypergraph,hedge,assignment,num_partitions):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
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
    return tuple(config)

def hedge_to_cost(hypergraph : QuantumCircuitHyperGraph, 
                   hedge : tuple, 
                   assignment : dict[tuple[int,int]], 
                   num_partitions : int, 
                   costs : dict[tuple] = {}) -> int:
    """
    Computes the cost of a hyperedge based on its configuration and the current assignment.
    """ 
    config = map_hedge_to_config(hypergraph, hedge, assignment, num_partitions)
    if config not in costs:
        cost = config_to_cost(config)
        costs[config] = cost
    else:
        cost = costs[config]
    return cost

def hedge_to_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                         hedge : tuple, 
                         assignment : dict[tuple[int,int]], 
                         num_partitions : int, 
                         costs : dict[tuple] = {},
                         network : QuantumNetwork = None) -> int:
    """"
    Computes the cost of a hyperedge based on its configuration and the current assignment."
    """
    root_config, rec_config = map_hedge_to_configs(hypergraph, hedge, assignment, num_partitions)

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
                            costs: dict) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    """
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph, edge, assignment, num_partitions, set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        config = map_hedge_to_config(hypergraph, edge, assignment, num_partitions)
        hypergraph.set_hyperedge_attribute(edge, 'config', config)
        if config not in costs:
            cost = config_to_cost(config)
            costs[config] = cost
        else:
            cost = costs[config]
            hypergraph.set_hyperedge_attribute(edge, 'cost', cost)
    return hypergraph

def map_counts_and_configs_hetero(hypergraph : QuantumCircuitHyperGraph,
                                  assignment : dict[tuple[int,int]],
                                  num_partitions : int,
                                  network : QuantumNetwork,
                                  costs: dict = {}) -> None:
    """
    Maps the counts and configurations of all hyperedges to hyperedge attributes based on the current assignment.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        root_config, rec_config = counts_to_configs(root_counts,rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
        if (root_config, rec_config) not in costs:
            edges, edge_cost = network.steiner_forest(root_config, rec_config)
            costs[(root_config, rec_config)] = edge_cost
        else:
            edge_cost = costs[(root_config, rec_config)]
        hypergraph.set_hyperedge_attribute(edge, 'cost', edge_cost)
    return hypergraph

def calculate_full_cost(hypergraph : QuantumCircuitHyperGraph,
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
        if config not in costs:
            edge_cost = config_to_cost(config)
            costs[config] = edge_cost
        else:
            edge_cost = costs[config]
        cost += edge_cost
    return cost

def calculate_full_cost_hetero(hypergraph : QuantumCircuitHyperGraph, 
                               assignment : dict[tuple[int,int]],
                               num_partitions : int,
                               costs: dict = {},
                               network: QuantumNetwork = None, 
                               node_map: dict = None) -> int:
    """
    Computes the total cost of the hypergraph based on the current assignment and the costs of each configuration.
    For heterogeneous networks, it uses the network to compute the costs.
    """
    cost = 0
    for edge in hypergraph.hyperedges:

        root_counts,rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions)
        root_config,rec_config = counts_to_configs(root_counts,rec_counts)

        if (root_config, rec_config) in costs:
            edge_cost = costs[(root_config, rec_config)]
        else:
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = edge_cost
        cost += edge_cost
    return cost
