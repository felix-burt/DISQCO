import numpy as np
import networkx as nx
from networkx.algorithms import community
import copy
from disqco.parti.FM.FM_main_nx import fm_algorithm
from disqco.parti.FM.FM_methods_nx import set_initial_partition_assignment
from disqco.parti.fgp.fgp_roee import run_initial_OEE


def find_next_start(graph, last_node, unvisited):
    while True:
        neighbors = set(graph.neighbors(last_node))
        candidates = neighbors & unvisited
        if candidates:
            return candidates.pop()
        else:
            # If no unvisited neighbors, return any unvisited node
            return find_next_start(graph, np.random.choice(list(neighbors)), unvisited)
        
def breadth_first_search_partitioning(graph, max_capacity,random_start=False):
    """
    Simple BFS-based graph partitioning. Fill partitions by searching from a starting node until max_capacity is reached,
    use BFS again to find next starting node if there are unvisited nodes remaining.
    """
    
    assignment = np.array([-1] * len(graph.nodes()))
    visited = set()
    queue = []
    
    # Start BFS from an arbitrary node
    if random_start:
        start_node = np.random.choice(list(graph.nodes()))
    else:
        start_node = list(graph.nodes())[-1]  # Consistent starting point for testing
    # start_node = 36 # Trialed as effective starting point for H64f1
    queue.append(start_node)
    visited.add(start_node)
    assignment[start_node] = 0  # Assign to first partition
    partition_size = 0
    current_partition_set = set()
    current_partition = 0
    current_partition_set.add(start_node)
    while queue:
        current_node = queue.pop(0)

        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if partition_size < max_capacity:
                    assignment[neighbor] = current_partition
                    partition_size += 1
                    current_partition_set.add(neighbor)
                else:
                    # Move to next partition
                    current_partition = current_partition + 1
                    queue = [neighbor]
                    assignment[neighbor] = current_partition
                    partition_size = 1
                    current_partition_set = {neighbor}

        unvisited_nodes = set(graph.nodes()) - visited

        if not queue and unvisited_nodes:
            try:
                next_start = find_next_start(graph, current_node, unvisited_nodes)
            except RecursionError:
                next_start = unvisited_nodes.pop()
            assignment[next_start] = current_partition   # Start new partition
            queue.append(next_start)
            visited.add(next_start)
            current_partition_set.add(next_start)
            partition_size += 1

    return assignment



def contraction_partitioning(graph, max_capacity):
    """
    Original contraction partitioning (for performance comparison)
    """
    new_graph = graph.copy()
    for node in new_graph.nodes():
        new_graph.nodes[node]['contained_nodes'] = {node}
    while True:
        matching = list(nx.max_weight_matching(new_graph, maxcardinality=True, weight='weight'))
        contractions = set()

        for u, v in matching:
            if len(new_graph.nodes[u]['contained_nodes']) + len(new_graph.nodes[v]['contained_nodes']) <= max_capacity:
                new_graph.nodes[u]['contained_nodes'].update(new_graph.nodes[v]['contained_nodes'])
            else:
                continue
            contractions.add((u, v))
            new_graph = nx.contracted_edge(new_graph, (u, v), self_loops=False)

        if not contractions:
            break
    
    assignment = np.array([-1] * len(graph.nodes()))
    for part, node in enumerate(new_graph.nodes()):
        for original_node in new_graph.nodes[node]['contained_nodes']:
            assignment[original_node] = part

    return assignment

def fast_contraction_partitioning(graph, max_capacity):
    from disqco.parti.fast_contraction import fast_contraction_partitioning
    return fast_contraction_partitioning(graph, max_capacity)


def set_assignment_from_communities(graph, communities):
    assignment = np.zeros(len(graph.nodes()), dtype=int)
    for i, comm in enumerate(communities):
        for node in comm:
            assignment[node] = i
    return assignment

# def set_initial_assignment(graph, num_parts):
#     assignment = np.zeros(len(graph.nodes()), dtype=int)
#     for i, node in enumerate(graph.nodes()):
#         assignment[node] = i % num_parts
#     return assignment

def calculate_cut_from_communities(graph, communities):
    cut = 0
    counted_edges = set()
    for comm in communities:
        for node in comm:
            for neighbor in graph.neighbors(node):
                if neighbor not in comm:
                    edge = (min(node, neighbor), max(node, neighbor))
                    if edge not in counted_edges:
                        counted_edges.add(edge)
                        cut += 1
    return cut

def calculate_cut_from_assignment(graph, assignment):
    cut = 0
    for edge in graph.edges():
        node, neighbor = edge
        if assignment[node] != assignment[neighbor]:
            cut += 1
    return cut

def add_unit_weights_to_graph(graph):
    for edge in graph.edges():
        graph.edges[edge]["weight"] = 1
    return graph

def find_num_partitions(graph, qpu_sizes):
    """Find the maximum partition size based on the graph size"""
    # For simplicity, we set max_size to the number of nodes in the graph
    max_size = len(graph.nodes) + 2*len(graph.edges)
    num_partitions = max_size // (qpu_sizes[0])
    return num_partitions

def apply_non_uniform_noise(G,noise:float = 5e-4 ,bell_noise:float = 5e-3):
    G = copy.deepcopy(G)
    for u,v in G.edges():
        if G.nodes[u]['QPU'] == G.nodes[v]['QPU']:
            G[u][v]['p'] = noise
        else:
            G[u][v]['p'] = bell_noise
    return G

def cut_nx_graph_max(
    G, qpu_max: int = 25, comms=None, return_comms=False, func="spectral",
    seed=None
):
    ### Inputs
    # - G nx.Graph to partition
    # - QPU_max - max number of vertex in subgraph set
    # - Comms - list of subgraphs as initial inputs (these can be further partions)
    # - return_comms - return the resulting subgraphs selected
    # - seed - random seed for spectral bisection
    # This function does not attempt to find an optimal cut
    # Just bisect the (sub)graphs until there are n segments
    # Spectral - use spectral bisection (eigenvalue partitioning)
    # Kernigham_lin  = min-weight random bisection
    #  spectral seems better but non-optimal
    G = copy.deepcopy(G)
    if func not in ["spectral", "kernighan_lin", "mitis"]:
        raise ValueError("invalid func flag")
    if comms is None:
        comms = [G]
    communities = sorted(
        comms, key=lambda x: len(x.nodes), reverse=True
    )

    if len(G.nodes) <= qpu_max or len(G.nodes) == 1:
        pass  # already done
    elif qpu_max < len(G.nodes) and len(G.nodes) >= 2:
        while len(communities[0].nodes) > 1 and not np.all(
            [len(k.nodes) <= qpu_max for k in communities]
        ):  # keep going while size of communites (|C|) --> n_qpu>=|C|>1
            k_prime = communities.pop(0)
            if func == "spectral":
                left, right = nx.spectral_bisection(k_prime, seed=seed)
            elif func == "kernighan_lin":
                left, right = community.kernighan_lin.kernighan_lin_bisection(
                    k_prime, max_iter=1_000_000
                )
            elif func == "mitis":
                raise NotImplementedError("not done yet sorry")

            communities.append(G.subgraph(left))
            communities.append(G.subgraph(right))
            communities = sorted(
                communities, key=lambda x: len(x.nodes), reverse=True
            )  # set with most nodes - bit of a heuristic choice imo

    else:
        raise ValueError(
            f"can not cut a {len(G.nodes)}-node graph into {qpu_max}-sized segments"
        )
    for device_i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["QPU"] = device_i
            # Node is part of device i for devices in range 0..n
    return G, communities

def spectral_bisection(graph: nx.Graph, max_capacity: int):

    G, comms = cut_nx_graph_max(graph, qpu_max=max_capacity, func="spectral")
    assignment = set_assignment_from_communities(graph, comms)
    return assignment

def kernighan_lin(graph: nx.Graph, max_capacity: int):

    G, comms = cut_nx_graph_max(graph, qpu_max=max_capacity, func="kernighan_lin")
    assignment = set_assignment_from_communities(graph, comms)
    return assignment

def fm_partitioning(graph: nx.Graph, max_capacity: int, initial_assignment='greedy', max_iterations=5):
    total_nodes = len(graph.nodes())
    move_limit = total_nodes // 10

    if isinstance(initial_assignment, str):
        num_partitions = (total_nodes // max_capacity) + 1
        partition_sizes = {i: max_capacity for i in range(num_partitions)}
    else:
        num_partitions = len(set(initial_assignment)) +  2
        partition_sizes = {i: max_capacity for i in range(num_partitions)}
    fm_assignment, fm_cut, fm_cuts_history = fm_algorithm(graph, partition_sizes, global_assignment=initial_assignment, max_iterations=max_iterations, move_limit=move_limit)
    return fm_assignment

# def overall_extreme_exchange(graph: nx.Graph, max_capacity: int, initial_assignment=None):
#     total_nodes = len(graph.nodes())
#     if initial_assignment is None:
#         num_partitions = (total_nodes // max_capacity) + 1
#         partition_sizes = {i: max_capacity for i in range(num_partitions)}
#         initial_assignment = set_initial_partition_assignment(graph, partition_sizes, method='greedy')

#     assignment, _ = run_initial_OEE(graph, initial_partition=initial_assignment, qpu_info=list(partition_sizes.values()))

#     return assignment
def re_sort_partitions(assignment, p, new_count, partition_counts, sorted_partitions):

    for i, part in enumerate(sorted_partitions):
        if new_count >= partition_counts[part]:
            break
    new_index = i
    sorted_partitions.insert(new_index, p)
    print('Resorted partitions:', sorted_partitions)

def squeeze_assignment(assignment, max_capacity, graph=None, verbose=False):
    """
    Iteratively merge underfilled partitions, prioritizing pairs with the most edges between them.
    
    Args:
        assignment: numpy array of partition assignments
        max_capacity: maximum allowed partition size
        graph: NetworkX graph (required for edge-based merging)
        verbose: whether to print merge operations
    
    Returns:
        assignment: updated assignment with merged partitions
    """
    if graph is None:
        raise ValueError("Graph is required for edge-based partition merging")
    
    assignment = assignment.copy()  # Don't modify original
    
    if verbose:
        unique_partitions = np.unique(assignment)
        print(f"Starting with partitions: {unique_partitions}")
    
    iteration = 0
    while True:
        iteration += 1
        if verbose:
            print(f"\n--- Iteration {iteration} ---")
        
        # Get current partition counts (only for partitions that exist)
        partition_counts = np.bincount(assignment)
        non_empty_partitions = np.where(partition_counts > 0)[0]
        
        if len(non_empty_partitions) <= 1:
            if verbose:
                print("Only one partition remaining, stopping.")
            break
        
        # Find all pairs of partitions that can be merged (combined size <= max_capacity)
        valid_pairs = []
        for i, p1 in enumerate(non_empty_partitions):
            for p2 in non_empty_partitions[i+1:]:
                size1, size2 = partition_counts[p1], partition_counts[p2]
                if size1 + size2 <= max_capacity:
                    valid_pairs.append((p1, p2, size1, size2))
        
        if not valid_pairs:
            if verbose:
                print("No valid pairs found for merging.")
            break
        
        # Calculate edge count between each valid pair
        pair_edge_counts = []
        for p1, p2, size1, size2 in valid_pairs:
            # Get nodes in each partition
            nodes_p1 = set(np.where(assignment == p1)[0])
            nodes_p2 = set(np.where(assignment == p2)[0])
            
            # Count edges between the two partitions
            edge_count = 0
            for u, v in graph.edges():
                if (u in nodes_p1 and v in nodes_p2) or (u in nodes_p2 and v in nodes_p1):
                    edge_count += 1
            
            pair_edge_counts.append((p1, p2, size1, size2, edge_count))
        
        # Sort by edge count (descending) - merge pairs with most connections first
        pair_edge_counts.sort(key=lambda x: x[4], reverse=True)
        
        if verbose:
            print("Valid merge candidates (sorted by edge count):")
            for p1, p2, size1, size2, edge_count in pair_edge_counts[:5]:  # Show top 5
                print(f"  Partitions {p1}({size1}) + {p2}({size2}) = {size1+size2}, edges: {edge_count}")
        
        # Take the pair with the most edges between them
        best_pair = pair_edge_counts[0]
        p1, p2, size1, size2, edge_count = best_pair
        
        # Merge: move all nodes from higher ID partition to lower ID partition
        target_partition = min(p1, p2)
        source_partition = max(p1, p2)
        
        # Reassign all nodes from source to target
        assignment[assignment == source_partition] = target_partition
        
        if verbose:
            print(f"✓ Merged partition {source_partition}(size {partition_counts[source_partition]}) into partition {target_partition}(size {partition_counts[target_partition]}) - {edge_count} connecting edges")
        
        # Continue to next iteration
    
    if verbose:
        final_counts = np.bincount(assignment)
        non_empty_final = np.where(final_counts > 0)[0]
        print(f"\nFinal partition counts: {dict(zip(non_empty_final, final_counts[non_empty_final]))}")
        print(f"Total partitions: {len(non_empty_final)}")
    
    return assignment

def compact_partition_ids(assignment):
    """
    Renumber partitions to be consecutive starting from 0.
    This is useful after squeeze_assignment which may leave gaps in partition numbering.
    
    Args:
        assignment: numpy array of partition assignments
    
    Returns:
        assignment: compacted assignment with consecutive partition IDs
    """
    unique_partitions = np.unique(assignment)
    mapping = {old_id: new_id for new_id, old_id in enumerate(unique_partitions)}
    
    new_assignment = assignment.copy()
    for old_id, new_id in mapping.items():
        new_assignment[assignment == old_id] = new_id
    
    return new_assignment