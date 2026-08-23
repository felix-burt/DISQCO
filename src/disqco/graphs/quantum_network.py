import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
from networkx.algorithms.approximation import steiner_tree
from networkx import erdos_renyi_graph
import math as mt

# Quantumn Network Class
# This class is used to create a quantum network with multiple QPUs
# and their connectivity. It also provides methods to visualize the network
# and to find the minimum spanning tree for a given set of nodes, which
# is used for finding entanglement distribution paths.
class QuantumNetwork():
    def __init__(self, qpu_sizes, qpu_connectivity = None, comm_sizes = None, qpu_topologies = None, comm_links = None):

        if isinstance(qpu_sizes, list):
            self.qpu_sizes = {}
            for i in range(len(qpu_sizes)):
                self.qpu_sizes[i] = qpu_sizes[i]
        else:
            self.qpu_sizes = qpu_sizes

        if qpu_connectivity is None:
            self.hetero = False
            self.qpu_connectivity = [(i, j) for i in range(len(qpu_sizes)) for j in range(i+1, len(qpu_sizes))]
        else:
            self.qpu_connectivity = qpu_connectivity
            self.hetero = True

        self.qpu_graph = self.create_qpu_graph()
        self.num_qpus = len(self.qpu_sizes)
        self.mapping = {i: set([i]) for i in range(self.num_qpus)}
        self.active_nodes = set(self.qpu_sizes.keys())

        if comm_sizes is None:
            self.comm_sizes = {i: 1 for i in range(self.num_qpus)}
        else:
            if isinstance(comm_sizes, list):
                self.comm_sizes = {i: comm_sizes[i] for i in range(len(comm_sizes))}
            else:
                self.comm_sizes = comm_sizes

        self.qpu_topologies = qpu_topologies if qpu_topologies is not None else {}

        self.comm_links = comm_links if comm_links is not None else {}

        self._validate_topologies()
        self._validate_comm_links()

    def _validate_topologies(self):
        if not isinstance(self.qpu_topologies, dict):
            raise TypeError("qpu_topologies must be a dictionary mapping QPU indices to networkx Graphs.")

        for qpu, qubit_graph in self.qpu_topologies.items():
            if qpu not in self.qpu_sizes:
                raise ValueError(f"QPU index {qpu} in qpu_topologies is not present in qpu_sizes.")
            if not isinstance(qubit_graph, nx.Graph):
                raise TypeError(f"Topology for QPU {qpu} must be a networkx Graph.")
            if not nx.is_connected(qubit_graph):
                raise ValueError(f"Topology for QPU {qpu} is not connected.")

            n = self.qpu_sizes[qpu]
            c = self.comm_sizes[qpu]
            data_nodes = set(range(n))
            full_nodes = set(range(n + c))
            if set(qubit_graph.nodes) not in (data_nodes, full_nodes):
                raise ValueError(f"Topology for QPU {qpu} contains nodes outside the valid range [0, {n + c - 1}].")

    def _validate_comm_links(self):
        if not isinstance(self.comm_links, dict):
            raise TypeError(...)
        for qpu, links in self.comm_links.items():
            if qpu not in self.qpu_sizes:
                raise ValueError(f"QPU {qpu} in comm_links is not present in qpu_sizes.")
            for k, neighbor in links.items():
                if not 0 <= k < self.comm_sizes[qpu]:
                    raise ValueError(f"Comm index {k} for QPU {qpu} out of range (comm_sizes = {self.comm_sizes[qpu]}).")
                if neighbor not in self.qpu_sizes:
                    raise ValueError(f"Comm {k} of QPU {qpu} bound to unknown QPU {neighbor}.")
                if not self.qpu_graph.has_edge(qpu, neighbor):
                    raise ValueError(f"Comm {k} of QPU {qpu} bound to QPU {neighbor}, but no network link exists between them.")

    def comm_node(self, p, k):
        return self.qpu_sizes[p] + k

    def comm_constrained(self, p) -> bool:
        topology = self.qpu_topologies.get(p)
        return topology is not None and len(topology.nodes)  == self.qpu_sizes[p] + self.comm_sizes[p]

    def comm_qubits_for_link(self, p, neighbor) -> list[int]:
        bound = [k for k, nb in self.comm_links.get(p, {}).items() if nb == neighbor]
        return bound if bound else list(range(self.comm_sizes[p]))

    @classmethod
    def create(cls, qpu_sizes, coupling_type='all_to_all', comm_sizes=None, qpu_topologies=None, comm_links=None, **kwargs):
        """
        Factory method to create a QuantumNetwork with a specific coupling type.
        
        Args:
            qpu_sizes: List or dict of QPU sizes
            coupling_type: String specifier for network topology. Options:
                - 'all_to_all' or 'complete': Fully connected network (default)
                - 'linear': Linear chain topology
                - 'grid': Grid topology
                - 'random': Random coupling with edge probability p
                - 'tree': Tree topology with branching factor k
                - 'network_of_grids': Network of grid components connected by linear paths
            comm_sizes: Optional communication sizes for QPUs
            qpu_topologies: Optional dict mapping QPU indices to their internal qubit topologies (as networkx Graphs)
            comm_links: Optional dict per QPU, which comm qubit index serves the link to which neighbor QPU: {qpu: {comm_index: neighbor_qpu}}
            **kwargs: Additional arguments for specific coupling types:
                - p: Edge probability for random coupling (default 0.5)
                - k: Branching factor for tree topology (default 2)
                - num_grids: Number of grids for network_of_grids
                - nodes_per_grid: Nodes per grid for network_of_grids
                - l: Path length between grids for network_of_grids
                
        Returns:
            QuantumNetwork instance with specified coupling
            
        Examples:
            >>> # Create all-to-all network
            >>> network = QuantumNetwork.create([8, 8, 8, 8], 'all_to_all')
            >>> # Create linear network
            >>> network = QuantumNetwork.create([8, 8, 8, 8], 'linear')
            >>> # Create grid network
            >>> network = QuantumNetwork.create([4]*16, 'grid')
            >>> # Create random network with p=0.6
            >>> network = QuantumNetwork.create([8]*8, 'random', p=0.6)
        """
        # Determine number of QPUs
        if isinstance(qpu_sizes, list):
            num_qpus = len(qpu_sizes)
        else:
            num_qpus = len(qpu_sizes)
        
        # Normalize coupling_type string
        coupling_type = coupling_type.lower().replace('_', '').replace('-', '')
        
        # Generate connectivity based on type
        if coupling_type in ['alltoall', 'complete', 'fullyconnected']:
            connectivity = all_to_all(num_qpus)
        elif coupling_type == 'linear':
            connectivity = linear_coupling(num_qpus)
        elif coupling_type == 'grid':
            connectivity = grid_coupling(num_qpus)
        elif coupling_type == 'random':
            p = kwargs.get('p', 0.5)
            connectivity = random_coupling(num_qpus, p)
        elif coupling_type == 'tree':
            k = kwargs.get('k', 2)
            connectivity = tree_network(num_qpus, k)
        elif coupling_type == 'networkofgrids':
            num_grids = kwargs.get('num_grids')
            nodes_per_grid = kwargs.get('nodes_per_grid')
            l = kwargs.get('l')
            if num_grids is None or nodes_per_grid is None or l is None:
                raise ValueError(
                    "network_of_grids requires 'num_grids', 'nodes_per_grid', and 'l' parameters"
                )
            connectivity = network_of_grids(num_grids, nodes_per_grid, l)
        else:
            raise ValueError(
                f"Unknown coupling type: '{coupling_type}'. "
                f"Valid options are: 'all_to_all', 'linear', 'grid', 'random', 'tree', 'network_of_grids'"
            )

        return cls(qpu_sizes, qpu_connectivity=connectivity, comm_sizes=comm_sizes, qpu_topologies=qpu_topologies, comm_links=comm_links)

    def create_qpu_graph(self):
        qpu_graph = nx.Graph()
        for qpu, qpu_size in self.qpu_sizes.items():
            qpu_graph.add_node(qpu, size=qpu_size)
        for i, j in self.qpu_connectivity:
            qpu_graph.add_edge(i, j)
        return qpu_graph
    
    def draw(self,):
        num_nodes = len(self.qpu_graph.nodes)
        # Scale node size based on number of nodes
        base_size = max(100, 2000 / num_nodes)
        node_sizes = [base_size * self.qpu_graph.nodes[i]['size'] for i in self.qpu_graph.nodes]
        # Use royalblue for active nodes (default), or custom color if specified
        node_colors = [
            self.qpu_graph.nodes[i]['color'] if 'color' in self.qpu_graph.nodes[i] else 'royalblue' 
            for i in self.qpu_graph.nodes
        ]
        # Create labels with LaTeX formatting
        labels = {i: f"$Q_{{{i}}}$" for i in self.qpu_graph.nodes}
        # Scale font size based on number of nodes
        font_size = max(8, min(16, 120 / num_nodes))
        nx.draw(self.qpu_graph, with_labels=True, labels=labels, 
                node_size=node_sizes, node_color=node_colors,
                edgecolors='k', linewidths=1, font_weight='bold', font_size=font_size)
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
            inverse_node_map = {v: k for k, v in node_map.items()}
            root_nodes = [inverse_node_map[i] for i in range(len(root_config)) if root_config[i] == 1]
            rec_nodes = [inverse_node_map[i] for i in range(len(rec_config)) if rec_config[i] == 1]
        else:
            root_nodes = [i for i, element in enumerate(root_config) if element == 1]
            rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]
        if root_nodes == [] or rec_nodes == []:
            return set(), 0
        # print(f"Root nodes: {root_nodes}, Receiver nodes: {rec_nodes}")

        if root_nodes == rec_nodes:
            # If root and receiver nodes are the same, we can just return the BFS edges
            edges = set()
            cost = 0
            return edges, cost
        if len(root_nodes) == 1:
            source_nodes = root_nodes
        else:
            steiner_g = steiner_tree(self.qpu_graph, root_nodes)
            node_set = set(steiner_g.nodes())
            source_nodes = list(node_set.union(root_nodes))
        edges = self.multi_source_bfs(source_nodes, rec_nodes)
        # edges = self.multi_source_bfs(root_nodes, rec_nodes)
        
        cost = len(edges)
        
        return edges, cost
    
    # def get_full_tree(self, graph : QuantumCircuitHyperGraph, 
    #                   edge : tuple[int,int], 
    #                   assignment : list[list[int]], 
    #                   num_partitions: int) -> nx.Graph:
    #     """
    #     Get the full tree of edges in network required to cover gates in the edge.
    #     This is used to find the entanglement distribution paths.

    #     :param graph: The hypergraph representing the quantum circuit.
    #     :type graph: QuantumCircuitHyperGraph
    #     :param edge: The edge in the hypergraph representing the gate.
    #     :type edge: tuple[int,int]
    #     :param assignment: The assignment of qubits to QPUs.
    #     :type assignment: list[list[int]]
    #     :return: A set of edges representing the full tree.
    #     :rtype: set[tuple[int,int]]
    #     """
    #     if edge not in graph.hyperedges:
    #         edge = (edge[1], edge[0])
    #         if edge not in graph.hyperedges:
    #             edge = edge[1]
    #             if edge not in graph.hyperedges:
    #                 raise ValueError(f"Edge {edge} not found in hypergraph.")
    #     root_config, rec_config = map_hedge_to_configs(hypergraph=graph, 
    #                                                    hedge=edge, 
    #                                                    assignment=assignment, 
    #                                                    num_partitions=num_partitions)

    #     root_nodes = [i for i, element in enumerate(root_config) if element == 1]
    #     rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]

    #     steiner_g = steiner_tree(self.qpu_graph, root_nodes)
    #     node_set = set(steiner_g.nodes())
    #     source_nodes = list(node_set.union(root_nodes))
    #     edges = self.multi_source_bfs(source_nodes, rec_nodes)

    #     all_network_edges = edges.union(steiner_g.edges())

    #     tree = nx.Graph()
    #     tree.add_edges_from(all_network_edges)

    #     return tree

    def get_full_tree(self, 
                    root_p : int,
                    target_partitions : list[int]) -> nx.Graph:
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

        terminal_nodes = [root_p] + target_partitions

        return steiner_tree(G=self.qpu_graph, terminal_nodes=terminal_nodes)


    def copy(self):
        return QuantumNetwork(self.qpu_sizes, self.qpu_connectivity, self.comm_sizes, self.qpu_topologies, self.comm_links)

    def get_costs(self,) -> dict[tuple]:
        """
        Computes the costs for all configurations given connectivity.
        """

        configs = get_all_configs(self.num_qpus, hetero=self.hetero)
        costs = {}
        if self.hetero:
            for root_config in configs:
                for rec_config in configs:
                    edges, cost = self.steiner_forest(root_config, rec_config)
                    costs[(root_config, rec_config)] = cost
        else:
            for config in configs:
                cost = config_to_cost(config)
                costs[tuple(config)] = cost

        return costs
    
    def is_fully_connected(self,) -> bool:
        """
        Check if the network is connected.
        """
        graph = self.qpu_graph
        return nx.is_empty(nx.complement(graph))

    def steiner_forest_from_sets(self, root_config_set, rec_config_set, node_map=None):
        """
        Steiner forest calculation using set-based configs for better performance.
        
        Args:
            root_config_set: Set of partition indices with root nodes
            rec_config_set: Set of partition indices with receiver nodes
            node_map: Optional mapping from partition indices to network positions
            
        Returns:
            tuple: (edges, cost)
        """
        if node_map is not None:
            root_nodes = [node_map[i] for i in root_config_set]
            rec_nodes = [node_map[i] for i in rec_config_set]
        else:
            root_nodes = list(root_config_set)
            rec_nodes = list(rec_config_set)
            
        if not root_nodes or not rec_nodes:
            return set(), 0
        
        
            
        steiner_g = steiner_tree(self.qpu_graph, root_nodes)
        node_set = set(steiner_g.nodes())
        source_nodes = list(node_set.union(root_nodes))
        edges = self.multi_source_bfs(source_nodes, rec_nodes)
        
        cost = len(edges)
        return edges, cost
    
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

def network_of_grids(num_grids, nodes_per_grid, l):
    """
    Construct a network of grid graphs connected by linear paths.

    Args:
        num_grids (int): Number of grid components.
        nodes_per_grid (int): Number of nodes in each grid.
        l (int): Number of hops (edges) in the path connecting consecutive grids.

    Returns:
        List of edges across the entire network.
    """
    all_edges = []
    node_counter = 0
    grid_centers = []

    for i in range(num_grids):
        # Generate grid edges
        grid_edges = grid_coupling(nodes_per_grid)
        # Offset node indices
        offset_edges = [[u + node_counter, v + node_counter] for u, v in grid_edges]
        all_edges.extend(offset_edges)

        # Track a "center" node in the grid to connect bridges (we'll use node 0 of each grid)
        grid_centers.append(node_counter)  # could also pick a more central node
        node_counter += nodes_per_grid

        # Add l-hop path to next grid (if not the last grid)
        if i < num_grids - 1:
            bridge_edges = []
            path_start = grid_centers[-1]
            bridge_nodes = [node_counter + j for j in range(l)]
            path_nodes = [path_start] + bridge_nodes

            for u, v in zip(path_nodes, path_nodes[1:]):
                bridge_edges.append([u, v])
            all_edges.extend(bridge_edges)

            node_counter += l  # reserve node indices for bridge

    return all_edges

def all_to_all(N):
    """
    Create a fully connected network of N nodes.

    Returns:
        A list of edges in the format [[node1, node2], ...].
    """
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append([i, j])
    return edges

def tree_network(N, k=2):
    """
    Create a tree-like network of N nodes. Calculate height of tree as logk(N).
    Each node has k children and the tree is balanced.
    """
    edges = []
    for i in range(1, N):
        edges.append([i, (i - 1) // k])  # Connect each node to its parent
    # The root node is 0, and it has children from 1 to k
    for i in range(k):
        if i < N:
            edges.append([0, i])    
    # Ensure the tree is connected
    if len(edges) < N - 1:
        raise ValueError("Not enough edges to connect all nodes in the tree network.")
 
    return edges

def get_all_configs(num_partitions : int, hetero = False, sparse = False) -> list[set[int]]:
    """
    Generates all possible configurations for a given number of partitions."
    """
    # Each configuration is represented as a tuple of 0s and 1s, where 1 indicates
    # that at least one qubit in the edge is assigned to the current partition.
    from itertools import product
    configs = set(product((0,1),repeat=num_partitions))
    if hetero:
        configs = configs - {tuple([0]*num_partitions)}
    
    if sparse:
        configs_sets = set()
        for config in list(configs):
            config_set = set([idx for idx, val in enumerate(config) if val == 1])
            configs_sets.add(config_set)
        configs = configs_sets

    return list(configs)

def config_to_cost(config : tuple[int], costs : dict[tuple[int], int] | None = None) -> int:
    """
    Converts a configuration tuple to its corresponding cost (assuming all to all connectivity)."
    """
    cost = 0
    for element in config:
        if element == 1:
            cost += 1
    if costs is not None:
        costs[tuple(config)] = cost
        
    return cost