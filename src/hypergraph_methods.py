from itertools import product

def get_all_configs(num_partitions):
    configs = list(product((0,1),repeat=num_partitions))
    return configs

def config_to_cost(config):
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    return cost

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

def map_hedge_to_config(hypergraph,hedge,assignment,num_partitions):
    root_counts,rec_counts = hedge_k_counts(hypergraph,hedge,assignment,num_partitions,set_attrs=False)
    root_config,rec_config = counts_to_configs(root_counts,rec_counts)
    # print(root_config,rec_config)
    return (tuple(root_config),tuple(rec_config))

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

def calculate_full_cost(hypergraph,assignment,num_partitions,costs):
    cost = 0
    for edge in hypergraph.hyperedges:
        edge_cost = hedge_to_cost(hypergraph,edge,assignment,num_partitions,costs)
        cost += edge_cost
    return cost