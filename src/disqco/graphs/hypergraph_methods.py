from itertools import product
import numpy as np
from disqco.graphs.quantum_network import QuantumNetwork

def get_all_configs(num_partitions : int) -> list[tuple[int]]:
    configs = list(product((0,1),repeat=num_partitions))
    return configs

def config_to_cost(config : tuple[int]) -> int:
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    return cost

def get_all_costs_hetero(network : QuantumNetwork, configs : list[tuple[int]], node_map = None) -> tuple[dict[tuple[tuple[int],tuple[int]] : int], dict[tuple[tuple[int],tuple[int]]] : list[tuple[int]]]:
    costs = {}
    edge_trees = {}
    for root_config in configs:
        for rec_config in configs:
            edges, cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            costs[(root_config, rec_config)] = cost
            edge_trees[(root_config, rec_config)] = edges
    return costs, edge_trees

def get_all_costs(configs):
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
        root_counts[partition_root] += 1
    for rec_node in receiver_set:
        partition_rec = assignment[rec_node[1]][rec_node[0]]
        rec_counts[partition_rec] += 1
    root_counts = tuple(root_counts)
    rec_counts = tuple(rec_counts)
    if set_attrs:
        hypergraph.set_hyperedge_attribute(hedge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(hedge, 'rec_counts', rec_counts)
    
    return root_counts, rec_counts

def counts_to_configs(root_counts,rec_counts):
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
    root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    return (tuple(root_config),tuple(rec_config))

def map_hedge_to_config2(hypergraph,hedge,assignment,num_partitions):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
    # root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    config = config_from_counts(root_counts,rec_counts)
    return config

def get_full_config(root_config,rec_config):
    config = list(rec_config)
    for i, element in enumerate(root_config):
        if rec_config[i] == 1:
            config[i] -= element
    return tuple(config)

def hedge_to_cost(hypergraph,hedge,assignment,num_partitions,costs):
    root_config,rec_config = map_hedge_to_config(hypergraph,hedge,assignment,num_partitions)
    full_config = get_full_config(root_config,rec_config)
    return costs[full_config]

def hedge_to_cost_hetero(hypergraph,hedge,assignment,num_partitions,costs):
    root_config,rec_config = map_hedge_to_config(hypergraph,hedge,assignment,num_partitions)
    return costs[(root_config,rec_config)]

def hedge_to_cost2(hypergraph,hedge,assignment,num_partitions,costs):
    config = map_hedge_to_config2(hypergraph,hedge,assignment,num_partitions)
    return costs[config]

def map_current_costs(hypergraph,assignment,num_partitions,costs):
    for edge in hypergraph.hyperedges:
        hypergraph.set_hyperedge_attribute(edge, 'cost', hedge_to_cost(hypergraph,edge,assignment,num_partitions,costs))
        
def map_counts_and_configs(hypergraph,assignment,num_partitions,costs):
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        root_config, rec_config = counts_to_configs(root_counts,rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
        config = get_full_config(root_config,rec_config)
        hypergraph.set_hyperedge_attribute(edge, 'config', config)
        hypergraph.set_hyperedge_attribute(edge, 'cost', costs[config])
    return hypergraph

def map_counts_and_configs_hetero(hypergraph,assignment,num_partitions,costs):
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        root_config, rec_config = counts_to_configs(root_counts,rec_counts)
        hypergraph.set_hyperedge_attribute(edge, 'root_config', root_config)
        hypergraph.set_hyperedge_attribute(edge, 'rec_config', rec_config)
        hypergraph.set_hyperedge_attribute(edge, 'cost', costs[(root_config,rec_config)])
    return hypergraph

def map_counts_and_configs2(hypergraph,assignment,num_partitions,costs):
    for edge in hypergraph.hyperedges:
        root_counts, rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions,set_attrs=True)
        hypergraph.set_hyperedge_attribute(edge, 'root_counts', root_counts)
        hypergraph.set_hyperedge_attribute(edge, 'rec_counts', rec_counts)
        config = map_hedge_to_config2(hypergraph,edge,assignment,num_partitions)
        hypergraph.set_hyperedge_attribute(edge, 'config', config)
        hypergraph.set_hyperedge_attribute(edge, 'cost', costs[config])
    return hypergraph

def calculate_full_cost(hypergraph,assignment,num_partitions,costs=None):
    cost = 0
    if costs is None:
        configs = get_all_configs(num_partitions)
        costs = get_all_costs(configs)
    for edge in hypergraph.hyperedges:
        edge_cost = hedge_to_cost(hypergraph,edge,assignment,num_partitions,costs)
        cost += edge_cost
    return cost

def calculate_full_cost_hetero(hypergraph,assignment,num_partitions,costs,network=None, node_map=None):
    cost = 0
    for edge in hypergraph.hyperedges:
        root_counts,rec_counts = hedge_k_counts(hypergraph,edge,assignment,num_partitions)
        root_config,rec_config = counts_to_configs(root_counts,rec_counts)
        if (root_config, rec_config) in costs:
            edge_cost = costs[(root_config,rec_config)]
        else:
            edges, edge_cost = network.steiner_forest(root_config, rec_config, node_map=node_map)
            print("edges", edges)
            print("edge cost", edge_cost)
            costs[(root_config, rec_config)] = edge_cost
        cost += edge_cost
    return cost

