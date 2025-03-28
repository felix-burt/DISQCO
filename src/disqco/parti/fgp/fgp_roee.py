from disqco.parti.fgp.circuit_to_graph import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def circuit_to_slices(circuit,remove_singles = True):
    "Function to identify interacting qubits at each layer. Returns a list of lists, where each list contains the interacting qubits at each layer."
    operations = circuit_to_gate_layers(circuit)
    interactions = []
    # Find the qubit registers and their sizes
    regs = circuit.qregs
    reg_mapping = {regs[i].name : i for i in range(len(regs))}
    sizes = [reg.size for reg in regs]
    # Scan the operations and find the qubits involved
    for layer in operations:
        current_layer = []
        for op in layer:
            qubits = op[1]
            regs = op[2]
            reg_nums = [reg_mapping[reg] for reg in regs]
            for q in range(len(qubits)):
                start_index = 0
                for n in range(0,reg_nums[q]):
                    start_index += sizes[n]
                qubits[q] = qubits[q] + start_index
            if remove_singles == True:
                # Remove single qubit operations
                if len(qubits) > 1:
                    current_layer.append(qubits)
            else:
                # Can keep the single qubit gates in for easier circuit recovery after partitioning
                current_layer.append(qubits)
        if len(current_layer) > 0:
            interactions.append(current_layer)
    
    return interactions

def exp_decay(m,decay_weight):
    # Exponential decay function for weighting future interactions
    return 2**(-m/decay_weight)

def slices_to_graphs(interactions,qpu_info, decay_func = exp_decay, decay_weight = 1, max_weight = 10000):
    "Convert the list of interactions to a list of graphs, where each graph represents the interactions at a layer with weighted future interactions."
    base_graph = nx.Graph()
    graph_list = []
    for n in range(np.sum(qpu_info)):
        base_graph.add_node(n)
    for n in range(len(interactions)):
        start_layer = interactions[n]
        graph = base_graph.copy()
        for op in start_layer:
            if len(op) > 1:
                # Interactions at current layer have infinite or very large weight (max_weight)
                graph.add_edge(op[0],op[1], weight = max_weight)
        for k in range(n+1,len(interactions)):
            layer = interactions[k]
            for op in layer:
                if len(op) > 1:
                    weight = exp_decay(k-n,decay_weight)
                    if graph.has_edge(op[0],op[1]):
                        graph.edges[op]['weight'] += weight 
                    else:
                        graph.add_edge(op[0],op[1],weight = weight)

        graph_list.append(graph)
    return graph_list

def slices_to_graphs_no_lookahead(interactions,qpu_info):
    "Convert the list of interactions to a list of graphs, where each graph represents the interactions at a layer with weighted future interactions."
    base_graph = nx.Graph()
    graph_list = []
    for n in range(np.sum(qpu_info)):
        base_graph.add_node(n)
    for n in range(len(interactions)):
        start_layer = interactions[n]
        graph = base_graph.copy()
        for op in start_layer:
            if len(op) > 1:
                # Interactions at current layer have infinite or very large weight (max_weight)
                graph.add_edge(op[0],op[1], weight = 1)
        graph_list.append(graph)
    return graph_list


def draw_graph(graph, partition, edge_labels= True):
    "Draw the base graph for interactions at current layer."
    G = graph
    colours = ['b','g','r','c','m','y','k','w']
    color_map = [colours[part] for part in partition]
    pos = nx.circular_layout(G) 
    nx.draw_networkx_nodes(G, pos,node_color=color_map)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    if edge_labels == True:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def draw_graph_no_lookahead(graph, partition, edge_labels= True,max_cost = 10000):
    "Draw the base graph for interactions at current layer with future interactions too"
    G = graph
    colors = {0 : 'red', 1 : 'blue'}
    color_map = [colors[part] for part in partition]
    filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= max_cost]
    H = G.edge_subgraph(filtered_edges).copy()
    pos = nx.circular_layout(H) 
    nx.draw_networkx_nodes(H, pos)
    nx.draw_networkx_edges(H, pos)
    nx.draw_networkx_labels(H, pos)
    if edge_labels == True:
        edge_labels = nx.get_edge_attributes(H, 'weight')
        nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
    plt.show()

def calculate_weighted_cut(graph,partition):
    "Calculate the cut of the graph with the given partition."
    cut = 0
    for edge in graph.edges():
        qubit1 = edge[0]
        qubit2 = edge[1]
        if partition[qubit1] != partition[qubit2]:
            cut += graph.edges()[edge]['weight']
    return cut

def set_initial_partition_fgp(qpu_info,num_partitions,invert=False):
    static_partition = []
    for n in range(num_partitions):
        for k in range(qpu_info[n]):
            if invert == False:
                static_partition.append(n)
            else:
                static_partition.append(num_partitions-n-1)
    return static_partition

def calculate_exchange_cost(graph,partition,action):
    "Calculate the difference in cut if the swap is performed."
    node1 = action[0]
    home1 = partition[node1]
    node2 = action[1]
    home2 = partition[node2]
    cost = 0
    for neighbour in graph.neighbors(node1):
        if neighbour != node2:
            if partition[neighbour] == home1:
                cost += graph.edges()[(node1,neighbour)]['weight']
            elif partition[neighbour] == home2:
                cost -= graph.edges()[(node1,neighbour)]['weight']
    for neighbour in graph.neighbors(node2):
        if neighbour != node1:
            if partition[neighbour] == home2:
                cost += graph.edges()[(node2,neighbour)]['weight']
            elif partition[neighbour] == home1:
                cost -= graph.edges()[(node2,neighbour)]['weight']
    return -cost

def swap(action,partition):
    "Update the partition after nodes are exchanged."
    qubit1 = action[0]
    qubit2 = action[1]
    new_partition = partition.copy()
    store = new_partition[qubit1]
    new_partition[qubit1] = new_partition[qubit2]
    new_partition[qubit2] = store
    return new_partition

def create_action_list(num_qubits):
    "Create a list of all possible swaps."
    swap_list = []
    for i in range(num_qubits):
        for j in range(i+1,num_qubits):
            pair = (i,j)
            swap_list.append(pair)
    return swap_list

def exchange_until_valid(graph,partition,action_list,max_cost = 10000, random_choice = False):
    "Chooses the best swap available until a valid partition is found."
    initial_cut = calculate_weighted_cut(graph,partition)
    cut = initial_cut
    swaps = []
    if initial_cut >= max_cost:
        valid = False
    else: 
        valid = True
    while not valid:
        costs = []
        # Find the exchange cost for each node pair
        for action in action_list:
            exchange_cost = calculate_exchange_cost(graph,partition,action)
            costs.append(exchange_cost)
        costs = np.array(costs)
        costs = np.nan_to_num(costs, nan=-np.inf)
        best_cost = np.max(costs)
        # Add all the best swaps to a list
        best_action_indeces = np.argwhere(costs == best_cost).flatten()
        if isinstance(best_action_indeces,int):
            best_action_index = best_action_indeces
        else:
            if random_choice:
                # Randomly choose between the best swaps
                best_action_index = np.random.choice(best_action_indeces)
            else:
                # Choose the first best swap
                best_action_index = best_action_indeces[0]
        action_taken = action_list[best_action_index]
        swaps.append(action_taken)
        new_partition = swap(action_taken,partition)
        # Calculate to check validity of the new partition
        cut = calculate_weighted_cut(graph,new_partition)
        if cut >= max_cost:
            valid = False
        else: 
            valid = True
        partition = new_partition
        
    return partition, swaps

def find_qubit_first_interactions(interactions):
    "Find the first two qubit interactions between qubits in the circuit."
    for l,layer in enumerate(interactions):
        for interaction in layer:
            if len(interaction) > 1:
                return l

def remove_redundant_teleportations(interactions,partition,mapping):
    "Remove teleportations that occur before any two qubit gates."
    new_partition = partition.copy()
    new_mapping = mapping.copy()
    min_layer = find_qubit_first_interactions(interactions)
    partition_start_layer = partition[min_layer]
    mapping_start_layer = mapping[min_layer]
    for n in range(0,min_layer):
        new_partition[n] = partition_start_layer
        new_mapping[n] = mapping_start_layer
    return new_partition, new_mapping


def fgp_oee(graph_list,initial_partition,action_list,mapping):
    "Perform the partitioning algorithm on the list of sliced graphs. Output a list of partitions for each layer."
    partition = initial_partition.copy()
    full_partition = []
    full_mapping = []
    layer_mapping = mapping
    for n, graph in enumerate(graph_list):
        final_partition, swaps = exchange_until_valid(graph,partition,action_list)
        for nodes in swaps:
            layer_mapping = swap(nodes,layer_mapping)
        partition = final_partition
        full_partition.append(final_partition)
        full_mapping.append(layer_mapping)
    return full_partition, full_mapping

def teleportation_cost(partition, num_qubits_log):
    "Calculate the teleportation cost of the list of partitions."
    cost = 0
    current_part = partition[0][:num_qubits_log]
    for l,layer in enumerate(partition):
        new_part = partition[l][:num_qubits_log]
        for part1,part2 in zip(current_part,new_part):
            if part1 != part2:
                cost += 1
        current_part = new_part
    return cost

def verify_partition(partition,graph_list):
    "Verify the partition by calculating the cut of the graph with the partition."
    cut = 0
    for n, graph in enumerate(graph_list):
        cut += calculate_weighted_cut(graph,partition[n])
    return cut

def create_full_interaction_graph(num_qubits,interactions):
    "Create a static graph for all circuit interactions."
    graph = nx.Graph()
    for n in range(num_qubits):
        graph.add_node(n)
    for layer in interactions:
        for interaction in layer:
            if len(interaction) > 1:
                if not graph.has_edge(interaction[0],interaction[1]):
                    graph.add_edge(interaction[0],interaction[1],weight = 1)
                else:
                    graph.edges()[(interaction[0],interaction[1])]['weight'] += 1
    return graph

def generate_random_candidate(qpu_info: int):
    "Generate a random candidate layer."
    candidate_layer = np.zeros(np.sum(qpu_info),dtype=int)
    counter = 0
    for k in range(len(qpu_info)):
        qpu_size = qpu_info[k]
        for n in range(qpu_size):
            candidate_layer[counter] = k
            counter += 1
    layer = np.random.permutation(candidate_layer)
    return layer

def calculate_static_cut(candidate,graph):
    "Calculate the cut of the static graph with the given partition."
    cut = 0
    for edge in graph.edges():
        weight = graph.edges()[edge]['weight']
        if candidate[edge[0]] != candidate[edge[1]]:
            cut += weight
    return cut

def generate_population(size: int, qpu_info: int):
    population = np.zeros((size,np.sum(qpu_info)),dtype=int)
    for n in range(size):
        population[n] = generate_random_candidate(qpu_info)
    return population

def find_starting_assignment(initial_search_size,qpu_info,graph,num_layers,random = False):
    "Heuristic for finding a good starting partition."
    if random:
    # Generate a population of random candidates and choose the best
        starting_pop = generate_population(initial_search_size,qpu_info)
        sorted_population =sorted(
                starting_pop,
                key=lambda candidate: calculate_static_cut(candidate,graph),
                reverse = False)
        best_candidate = sorted_population[0]
    else:
        best_candidate = set_initial_partition_fgp(qpu_info,num_partitions=len(qpu_info))
    
    mapping = [n for n in range(np.sum(qpu_info))]
    best_cost = calculate_static_cut(best_candidate,graph)
    action_list = create_action_list(np.sum(qpu_info))
    done = False
    # Exchange nodes until no more improvements can be made
    while not done:
        scores = []
        for action in action_list:
            score = calculate_exchange_cost(graph,best_candidate,action)
            new_candidate = swap(action,best_candidate)
            scores.append(score)
        best_action_index = np.argmax(scores)
        if np.max(scores) <= 0:
            done = True
        else:
            best_candidate = swap(action_list[best_action_index],best_candidate)
            mapping = swap(action_list[best_action_index],mapping)
            best_cost = best_cost - score
    return best_candidate, mapping

def calculate_cuts(time_partition,graph):
    "Calculate the cut of dynamic graph with the given partition over time."
    cut = 0
    # Iterate through all edges
    for edge in graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        # Check if the nodes are in different partitions at given time
        if time_partition[node1[1]][node1[0]] != time_partition[node2[1]][node2[0]]:
            cut += 1
    return cut

def main_algorithm(circuit,qpu_info,initial_partition,remove_singles = True,choose_initial = False, intial_search_size = 10000):
    "Main function to run the partitioning algorithm. Function returns a tuple of the list of partitions for each time step and the teleportation cost to implement the partition."
    # Convert the circuit to a list of qubit interactions at each layer
    interactions = circuit_to_slices(circuit,remove_singles=remove_singles)
    # Convert the qubit interactions to a list of graphs
    graph_list = slices_to_graphs(interactions,qpu_info)
    # Create list of actions for swapping qubits
    action_list = create_action_list(np.sum(qpu_info))
    # Create a static graph for all interactions
    full_graph = create_full_interaction_graph(np.sum(qpu_info),interactions)
    # Find the starting partition - either specified as input or found using heuristic
    if choose_initial:
        assignment = initial_partition
        mapping = [n for n in range(np.sum(qpu_info))]
    else:
        assignment, mapping = find_starting_assignment(intial_search_size,qpu_info,full_graph,len(interactions),random = False)
    #  Run the partitioning algorithm
    full_partition, full_mapping = fgp_oee(graph_list,assignment,action_list,mapping)
    # Remove teleportations before first two qubit gate layer
    full_partition, full_mapping = remove_redundant_teleportations(interactions,full_partition,full_mapping)
    # Calculate the teleportation cost of the partition
    final_cost = teleportation_cost(full_partition,circuit.num_qubits)

    return np.array(full_partition), final_cost, full_mapping

def map_to_GCP_graph(circuit,qpu_info,mapping):
    "Visualise the partitioning of the circuit in the GCP framework."
    graph = circuit_to_graph(qpu_info, circuit, group_gates=False)
    mapping.insert(0,mapping[0])
    mapping.append(mapping[-1])

    for n in range(circuit.depth()+2):
        for k in range(np.sum(qpu_info)):
            graph.nodes()[(k,n)]['pos'] = (graph.nodes()[(k,n)]['pos'][0], np.sum(qpu_info) - mapping[n][k])
    return graph

def build_initial_GCP_graph(circuit,qpu_info):
    "Build the initial GCP graph for the circuit."
    graph = circuit_to_graph(qpu_info, circuit, group_gates=False)
    return graph
