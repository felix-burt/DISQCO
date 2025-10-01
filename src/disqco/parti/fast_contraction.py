"""
Highly optimized contraction partitioning methods
"""
import numpy as np
import networkx as nx
from collections import defaultdict
import heapq

class FastContractionPartitioner:
    """
    Optimized contraction partitioning that minimizes graph operations
    """
    
    def __init__(self, graph):
        self.original_graph = graph
        self.nodes = list(graph.nodes())
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # Build adjacency representation for faster access
        self.adj_matrix = np.zeros((self.n_nodes, self.n_nodes))
        self.edge_weights = {}
        
        for u, v, data in graph.edges(data=True):
            u_idx, v_idx = self.node_to_idx[u], self.node_to_idx[v]
            weight = data.get('weight', 1)
            self.adj_matrix[u_idx, v_idx] = weight
            self.adj_matrix[v_idx, u_idx] = weight  # Undirected
            self.edge_weights[(min(u_idx, v_idx), max(u_idx, v_idx))] = weight
    
    def partition_no_graph_ops(self, max_capacity):
        """
        Contraction without creating new NetworkX graphs
        """
        # Union-Find for tracking merges
        parent = list(range(self.n_nodes))
        size = [1] * self.n_nodes
        active = [True] * self.n_nodes  # Track which nodes are still active
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if size[px] < size[py]:
                px, py = py, px
            parent[py] = px
            size[px] += size[py]
            return True
        
        def get_size(x):
            return size[find(x)]
        
        iteration = 0
        while True:
            iteration += 1
            
            # Find maximum weight matching using greedy approach on active nodes
            active_nodes = [i for i in range(self.n_nodes) if active[i]]
            if len(active_nodes) <= 1:
                break
                
            used = set()
            contractions = []
            
            # Create edges list for active nodes only
            edges = []
            for i in active_nodes:
                for j in active_nodes:
                    if i < j and self.adj_matrix[i, j] > 0:
                        edges.append((self.adj_matrix[i, j], i, j))
            
            # Sort by weight (descending) for greedy matching
            edges.sort(reverse=True)
            
            # Greedy matching
            for weight, u, v in edges:
                if u not in used and v not in used:
                    # Check capacity constraint
                    if get_size(u) + get_size(v) <= max_capacity:
                        union(u, v)
                        contractions.append((u, v))
                        used.add(u)
                        used.add(v)
                        # Mark the smaller index as inactive (absorbed)
                        active[max(u, v)] = False
            
            if not contractions:
                break
        
        # Build final assignment
        assignment = np.array([-1] * self.n_nodes)
        component_to_partition = {}
        partition_id = 0
        
        for i in range(self.n_nodes):
            root = find(i)
            if root not in component_to_partition:
                component_to_partition[root] = partition_id
                partition_id += 1
            assignment[self.nodes[i]] = component_to_partition[root]
        
        return assignment

def fast_contraction_partitioning(graph, max_capacity):
    """
    Wrapper function for the fast partitioner
    """
    partitioner = FastContractionPartitioner(graph)
    return partitioner.partition_no_graph_ops(max_capacity)

def lazy_contraction_partitioning(graph, max_capacity):
    """
    Lazy evaluation approach - only contract when necessary
    """
    # Track component sizes without actually contracting
    components = {node: {node} for node in graph.nodes()}
    remaining_nodes = set(graph.nodes())
    
    while len(remaining_nodes) > 1:
        # Build subgraph of remaining nodes
        subgraph = graph.subgraph(remaining_nodes)
        
        if len(subgraph.edges()) == 0:
            break
            
        # Find matching on current subgraph
        try:
            matching = nx.max_weight_matching(subgraph, maxcardinality=True, weight='weight')
        except:
            break
            
        if not matching:
            break
            
        contractions = []
        for u, v in matching:
            # Check if we can merge these components
            u_size = len(components[u])
            v_size = len(components[v])
            
            if u_size + v_size <= max_capacity:
                # Merge v into u
                components[u].update(components[v])
                del components[v]
                remaining_nodes.remove(v)
                contractions.append((u, v))
        
        if not contractions:
            break
    
    # Build assignment
    assignment = np.array([-1] * len(graph.nodes()))
    for part_id, (root, component) in enumerate(components.items()):
        for node in component:
            assignment[node] = part_id
    
    return assignment
