import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from networkx.algorithms.approximation import steiner_tree
from networkx import erdos_renyi_graph
import math as mt
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
from disqco.graphs.hypergraph_methods import map_hedge_to_configs

# Quantumn Network Class
# This class is used to create a quantum network with multiple QPUs
# and their connectivity. It also provides methods to visualize the network
# and to find the minimum spanning tree for a given set of nodes, which
# is used for finding entanglement distribution paths.
class QuantumNetwork():
    def __init__(self, qpu_sizes, qpu_connectivity = None):

        if isinstance(qpu_sizes, list):
            self.qpu_sizes = {}
            for i in range(len(qpu_sizes)):
                self.qpu_sizes[i] = qpu_sizes[i]
        else:
            self.qpu_sizes = qpu_sizes

        if qpu_connectivity is None:
            self.qpu_connectivity = [(i, j) for i in range(len(qpu_sizes)) for j in range(i+1, len(qpu_sizes))]
        else:
            self.qpu_connectivity = qpu_connectivity
        self.qpu_graph = self.create_qpu_graph()
        self.num_qpus = len(self.qpu_sizes)
        self.mapping = {i: set([i]) for i in range(self.num_qpus)}

    def create_qpu_graph(self):
        qpu_graph = nx.Graph()
        for qpu, qpu_size in self.qpu_sizes.items():
            qpu_graph.add_node(qpu, size=qpu_size)
        for i, j in self.qpu_connectivity:
            qpu_graph.add_edge(i, j)
        return qpu_graph
    
    def draw(self,):
        node_sizes = [20*self.qpu_graph.nodes[i]['size'] for i in self.qpu_graph.nodes]
        node_colors = [self.qpu_graph.nodes[i]['color'] if 'color' in self.qpu_graph.nodes[i] else 'green' for i in self.qpu_graph.nodes]
        nx.draw(self.qpu_graph, with_labels=True, node_size=node_sizes, node_color=node_colors)
        plt.show()

    def multi_source_bfs(self, roots, receivers):
        graph = self.qpu_graph

        visited = set()
        parent = dict()   
        queue = deque()

        for r in roots:
            visited.add(r)
            parent[r] = None 
            queue.append(r)

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

        chosen_edges = set()
        
        for t in receivers:
            if t not in visited:
                continue

            cur = t
            while parent[cur] is not None: 
                p = parent[cur]
                chosen_edges.add(tuple(sorted((p, cur))))
                cur = p
        
        return chosen_edges

    def steiner_forest(self, root_config, rec_config, node_map = None):

        if node_map is not None:
            root_nodes = [node_map[i] for i in range(len(root_config)) if root_config[i] == 1]
            rec_nodes = [node_map[i] for i in range(len(rec_config)) if rec_config[i] == 1]
        else:
            root_nodes = [i for i, element in enumerate(root_config) if element == 1]
            rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]
        
        steiner_g = steiner_tree(self.qpu_graph, root_nodes)
        node_set = set(steiner_g.nodes())
        source_nodes = list(node_set.union(root_nodes))
        edges = self.multi_source_bfs(source_nodes, rec_nodes)
        # edges = self.multi_source_bfs(root_nodes, rec_nodes)
        
        cost = len(edges)
        
        return edges, cost
    
    def get_full_tree(self, graph : QuantumCircuitHyperGraph, 
                      edge : tuple[int,int], 
                      assignment : list[list[int]], 
                      num_partitions: int) -> nx.Graph:
        """
        Get the full tree of edges in network required to cover gates in the edge.
        This is used to find the entanglement distribution paths.

        :param graph: The hypergraph representing the quantum circuit.
        :type graph: QuantumCircuitHyperGraph
        :param edge: The edge in the hypergraph representing the gate.
        :type edge: tuple[int,int]
        :param assignment: The assignment of qubits to QPUs.
        :type assignment: list[list[int]]
        :return: A set of edges representing the full tree.
        :rtype: set[tuple[int,int]]
        """
        if edge not in graph.hyperedges:
            edge = (edge[1], edge[0])
            if edge not in graph.hyperedges:
                edge = edge[1]
                if edge not in graph.hyperedges:
                    raise ValueError(f"Edge {edge} not found in hypergraph.")
        root_config, rec_config = map_hedge_to_configs(hypergraph=graph, 
                                                       hedge=edge, 
                                                       assignment=assignment, 
                                                       num_partitions=num_partitions)

        root_nodes = [i for i, element in enumerate(root_config) if element == 1]
        rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]

        steiner_g = steiner_tree(self.qpu_graph, root_nodes)
        node_set = set(steiner_g.nodes())
        source_nodes = list(node_set.union(root_nodes))
        edges = self.multi_source_bfs(source_nodes, rec_nodes)

        all_network_edges = edges.union(steiner_g.edges())

        tree = nx.Graph()
        tree.add_edges_from(all_network_edges)

        return tree

    def copy(self):
        return QuantumNetwork(self.qpu_sizes, self.qpu_connectivity)

def random_coupling(N, p):
    """
    Generates a connected graph with N nodes and edge probability p.

    :param N: Number of nodes in the graph.
    :type N: int
    :param p: Probability of edge creation between nodes.
    :type p: float
    :returns: A list of edges in the format [[node1, node2], ...].
    :rtype: list
    """
    while True:
        graph = erdos_renyi_graph(N, p)
        if nx.is_connected(graph):
            coupling = [[i,j] for i in range(N) for j in range(N) if i != j and graph.has_edge(i,j)]
            return coupling

def grid_coupling(N):
    """
    Create an adjacency list for a grid-like connection of N nodes.

    If N is a perfect square, it uses sqrt(N) x sqrt(N).
    Otherwise, it finds rows x cols such that rows * cols >= N
    and arranges the nodes accordingly.

    Returns:
        A list of edges in the format [[node1, node2], ...].
    """
    # Compute (approx) number of rows and columns
    root = int(mt.isqrt(N))  # isqrt gives the integer sqrt floor
    if root * root == N:
        rows, cols = root, root
    else:
        # We want rows * cols >= N, with rows ~ cols ~ sqrt(N)
        # Simple approach: start with rows = int(sqrt(N)) and
        # increment cols until rows * cols >= N.
        rows = root
        # One strategy: determine a minimal 'cols' so that rows * cols >= N
        # If that doesn't work, increment rows as needed.
        if rows * root >= N:
            cols = root
        else:
            cols = root + 1
            if rows * cols < N:  # Still not enough
                rows += 1
    
    edges = []
    node_index = lambda r, c: r * cols + c

    for r in range(rows):
        for c in range(cols):
            current_node = node_index(r, c)
            # Stop if we've reached all N nodes
            if current_node >= N:
                break

            # Connect to the right neighbor if within bounds and within N
            if c < cols - 1:
                right_node = node_index(r, c + 1)
                if right_node < N:
                    edges.append([current_node, right_node])

            # Connect to the bottom neighbor if within bounds and within N
            if r < rows - 1:
                bottom_node = node_index(r + 1, c)
                if bottom_node < N:
                    edges.append([current_node, bottom_node])

    return edges

def linear_coupling(N):
    """
    Create a linear coupling for N nodes.

    Returns:
        A list of edges in the format [[node1, node2], ...].
    """
    edges = []
    for i in range(N - 1):
        edges.append([i, i + 1])
    return edges


# coupling = [[i,j] for i in range(int(num_partitions/2)-1) for j in range(int(num_partitions/2)-1) if i != j]

# coupling += [[i,j] for i in range(int(num_partitions/2)+1,num_partitions) for j in range(int(num_partitions/2)+1,num_partitions) if i != j]
# coupling += [[int(num_partitions/2)-2, int(num_partitions/2)-1]]
# coupling += [[int(num_partitions/2)-1, int(num_partitions/2)]]
# coupling += [[int(num_partitions/2), int(num_partitions/2)+1]]
# coupling = None