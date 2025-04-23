from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
import networkx as nx
from copy import deepcopy

class NetworkCoarsener:
    """
    Class for coarsening a network graph.
    """

    def __init__(self, network: QuantumNetwork):
        """
        Initialize the NetworkCoarsener with a graph.

        :param graph: The input graph to be coarsened.
        """
        self.initial_network = network

    def merge_nodes(self, G, u, v, current_mapping):
        """
        Merge two nodes in the graph.
        This is a placeholder function and should be implemented based on the specific requirements.
        """
        # Placeholder for merging logic
        # Merge v into u (or vice versa)
        new_size = G.nodes[u]['size'] + G.nodes[v]['size'] - 1
        
        # Union the sets of fine-level QPUs
        current_mapping[u] = current_mapping[u].union(current_mapping[v])
        del current_mapping[v]
        
        # Update size attribute
        G.nodes[u]['size'] = new_size
        
        # Redirect edges that were connected to 'v' so they connect to 'u'
        # (except for self-loops)
        for nbr in list(G.neighbors(v)):
            if nbr != u:
                # Add edge (u, nbr) if not present
                if not G.has_edge(u, nbr):
                    G.add_edge(u, nbr)
        G.remove_node(v)


    def coarsen_network(self, network : QuantumNetwork, current_mapping : dict[int, set[int]] = None, desired_size: int = 10):
        """
        Coarsen the QPU graph by repeatedly merging matched pairs
        until the graph has at most 'desired_size' nodes.
        Returns:
            network_coarse: the coarsened network
            mapping_list: a history of how fine-level QPU nodes were merged
        """
        # Copy to avoid mutating the original
        network_coarse = deepcopy(network)
        # We'll keep track of merges: 
        # current_mapping[node] = set of original (fine) QPUs that have been merged into 'node'
        # In your original code, you might have something like:
        # num_partitions = network_coarse.qpu_graph.number_of_nodes()
        if current_mapping is None:
            current_mapping = {i: {i} for i in network_coarse.qpu_graph.nodes()}
        
        mapping_list = []
        G = network_coarse.qpu_graph

        node_count = G.number_of_nodes()
        while node_count > desired_size:
            # 1. Build an edge-weight dict if you want a size-aware matching
            #    For example, encourage merges of similarly sized nodes:
            weights = {}
            # Iterate over all edges and compute weights based on node sizes
            for (u, v) in G.edges():
                size_u = G.nodes[u]['size']
                size_v = G.nodes[v]['size']
                # negative squared difference => bigger (less negative) for more similar sizes
                weights[(u, v)] = - (size_u - size_v)**2

            # 2. Compute a maximum-cardinality or maximum-weight matching
            #    (set 'maxcardinality=True' if you want the largest number of merges possible;
            #     omit or set it to False if you want a maximum-weight matching that might skip some edges)
            matching = nx.max_weight_matching(G, maxcardinality=False, weight=lambda uv: weights[uv])
            
            # matching is a set of pairs {(u1,v1), (u2,v2), ...}
            if not matching:
                # If no edges matched, we can't coarsen further by pairwise merges,
                # so just break out of the loop
                break
            
            # 3. Merge matched pairs
            merged_nodes = set()
            for (u, v) in matching:
                # Skip if either already merged this iteration
                if u not in G or v not in G or u in merged_nodes or v in merged_nodes:
                    continue
                
                self.merge_nodes(G, u, v, current_mapping)
                

                merged_nodes.add(u)
                merged_nodes.add(v)


            # Keep track of the updated mapping after this batch of merges
            mapping_list.append(deepcopy(current_mapping))
            node_count = G.number_of_nodes()
            
        # Finally, store or update the qpu_sizes in the coarsened network
        network_coarse.qpu_sizes = {n : network_coarse.qpu_graph.nodes[n]['size'] 
                                    for n in network_coarse.qpu_graph.nodes()}
        network_coarse.node_map = {i: node for i, node in enumerate(network_coarse.qpu_graph.nodes())}
        
        return network_coarse, mapping_list[-1]

    def coarsen_network_recursive(self, l : int):
        """
        Coarsen the QPU graph into a number of levels by reducing size by a factor of l"
        Returns:
            network_coarse_list: list of coarsened networks
            mapping_list: a history of how fine-level QPU nodes were merged
        """
        # Copy to avoid mutating the original
        network_coarse = self.initial_network.copy()
        # We'll keep track of merges: 
        # current_mapping[node] = set of original (fine) QPUs that have been merged into 'node'
        # In your original code, you might have something like:
        num_partitions = network_coarse.qpu_graph.number_of_nodes()
        k = num_partitions

        desired_size = int(k / l)

        current_mapping = {i: {i} for i in network_coarse.qpu_graph.nodes()}
        self.initial_network.mapping = deepcopy(current_mapping)
        self.network_coarse_list = [self.initial_network]

        while k > l:
            network_coarse, mapping = self.coarsen_network(network_coarse, current_mapping, desired_size)
            self.network_coarse_list.append(network_coarse)
            network_coarse.mapping = mapping

            k = desired_size
            desired_size = int(k / l)
            current_mapping = deepcopy(mapping)
        
        return self.network_coarse_list

    # def unmerge_nodes(self, g0 : nx.Graph, source_node : int, mapping : dict[int,set[int]], level : int):

    #     parent_network = self.network_coarse_list[-level-2]
    #     g = parent_network.qpu_graph

    #     g0.remove_node(source_node)
    #     sub_nodes = mapping[source_node]

    #     print("Level: ", level)
    #     print("Source node: ", source_node)
    #     print("Sub nodes: ", sub_nodes)
    #     print("Mapping: ", mapping)

    #     print("Nodes in g: ", g.nodes)

    #     for node in sub_nodes:
    #         if node in g.nodes:
    #             print("Node: ", node)
    #             g0.add_node(node, size = g.nodes[node]['size'])
        
    #     for node1 in sub_nodes:
    #         for node2 in sub_nodes:
    #             if node1 != node2 and g.has_edge(node1, node2):
    #                 print("Adding edge: ", node1, node2)
    #                 g0.add_edge(node1, node2)
        

    #             # for coarse_node in mapping:
    #             #     print("Coarse node: ", coarse_node)
    #             #     other_sub_nodes = mapping[coarse_node]
    #             #     print("Other sub nodes: ", other_sub_nodes)
    #             #     for n in other_sub_nodes:
    #             #         print("N: ", n)
    #             #         if g.has_edge(n, node):
    #             #             print("Adding edge: ", n, node)
    #             #             g0.add_edge(node, n)
    #     nodes_to_merge = []
    #     for node in g0.nodes:
    #         if node not in sub_nodes:
    #             nodes_to_merge.append(node)
    #     print("Nodes to merge: ", nodes_to_merge)
    #     merged_nodes = set()

    #     for node1 in sub_nodes:
    #         for node2 in g.nodes:
    #             if node1 != node2 and g.has_edge(node1, node2):
    #                 print("Adding edge: ", node1, node2)
    #                 g0.add_edge(node1, nodes_to_merge[0])

    #     new_mapping = deepcopy(mapping)
    #     for node1 in nodes_to_merge:
    #         for node2 in nodes_to_merge:
    #             if node1 != node2 and g0.has_edge(node1, node2):
    #                 print("Merging nodes: ", node1, node2)
    #                 if node2 not in merged_nodes:
    #                     merged_nodes.add(node2)
    #                     self.merge_nodes(g0, node1, node2, new_mapping)
        


    #     qpu_sizes = {node : g0.nodes[node]['size'] for node in g0.nodes}
    #     g0.remove_edges_from(nx.selfloop_edges(g0))

    #     return g0, qpu_sizes

    def unmerge_nodes(self, g0 : nx.Graph, source_node : int, mapping : dict[int,set[int]], level : int, active_nodes):

        parent_network = self.network_coarse_list[-level-2]
        g = parent_network.qpu_graph
        parent_mapping = parent_network.mapping

        g0.remove_node(source_node)
        sub_nodes = mapping[source_node]

        for node in sub_nodes:
            if node in g.nodes:
                g0.add_node(node, size = g.nodes[node]['size'])

        

                # for coarse_node in mapping:
                #     print("Coarse node: ", coarse_node)
                #     other_sub_nodes = mapping[coarse_node]
                #     print("Other sub nodes: ", other_sub_nodes)
                #     for n in other_sub_nodes:
                #         print("N: ", n)
                #         if g.has_edge(n, node):
                #             print("Adding edge: ", n, node)
                #             g0.add_edge(node, n)

        dummy_nodes = set()
        for node in g0.nodes:
            if node not in sub_nodes:
                dummy_nodes.add(node)


        for node1 in sub_nodes:
            for coarse_node in dummy_nodes:
                contained_nodes = mapping[coarse_node]
                for node2 in contained_nodes:
                    if node1 != node2 and g.has_edge(node1, node2):
                        g0.add_edge(node1, coarse_node)

        for node1 in g0.nodes:
            for node2 in g0.nodes:
                if g.has_edge(node1, node2):
                    g0.add_edge(node1, node2)

        new_mapping = deepcopy(mapping)

        nodes_to_merge = set()

        for node in g0.nodes:
            if node not in sub_nodes and node not in active_nodes:
                nodes_to_merge.add(node)
        
        merged_nodes = []
        for node1 in nodes_to_merge:
            for node2 in nodes_to_merge:
                if node1 != node2 and g0.has_edge(node1, node2):
                    if node2 not in merged_nodes:
                        merged_nodes.append((node1, node2))
                        self.merge_nodes(g0, node1, node2, new_mapping)

        # print("New mapping: ", new_mapping)





        qpu_sizes = {node : g0.nodes[node]['size'] for node in g0.nodes}
        g0.remove_edges_from(nx.selfloop_edges(g0))

        new_active_nodes = set([node for node in sub_nodes if node in g.nodes])

        new_mapping = {node : parent_mapping[node] for node in new_active_nodes}

        dummy_node_mapping = {node : mapping[node] for node in dummy_nodes}

        for node, partner in merged_nodes:
            dummy_node_mapping[node] = dummy_node_mapping[partner].union(dummy_node_mapping[node])
            del dummy_node_mapping[partner]

        
        new_mapping.update(dummy_node_mapping)





        return g0, qpu_sizes, new_mapping, new_active_nodes


    # def cut_network(self, level : int = 0):
    #     """
    #     Cut the network into two sub-networks based on the coarse network.
    #     """
    #     # Create a new network with the same structure as the original
    #     network_list = []
    #     network_coarse = self.network_coarse_list[-level-1]
    #     mapping = network_coarse.mapping
    #     active_nodes = set([sub_node for sub_node in network_coarse.qpu_graph.nodes if sub_node in mapping])
    #     for node in network_coarse.qpu_graph.nodes:
    #         new_graph = network_coarse.qpu_graph.copy()

    #         new_graph, qpu_sizes = self.unmerge_nodes(new_graph, node, mapping, level=level, active_nodes=active_nodes)
    #         connectivity = []

    #         for edge in new_graph.edges:
    #             connectivity.append(edge)
    #         network_new = QuantumNetwork(qpu_sizes, connectivity)
    #         active_nodes = set([sub_node for sub_node in new_graph.nodes if sub_node in mapping[node]])
    #         network_list.append([network_new, active_nodes])
                
    #     return network_list
    
    def cut_network(self, networks_previous_level, level : int = 0):
        """
        Cut the network into two sub-networks based on the coarse network.
        """
        # Create a new network with the same structure as the original
        network_list = []
        # network_coarse = self.network_coarse_list[-level-1]
        # mapping = network_coarse.mapping
        # active_nodes = set([sub_node for sub_node in network_coarse.qpu_graph.nodes if sub_node in mapping])
        for network in networks_previous_level:
            network_coarse = network[0]
            active_nodes = network_coarse.active_nodes
            

            for node in active_nodes:
                new_graph = network_coarse.qpu_graph.copy()
                mapping = network_coarse.mapping
                
                new_graph, qpu_sizes, new_mapping, new_active_nodes = self.unmerge_nodes(new_graph, node, mapping, level=level, active_nodes=active_nodes)
                connectivity = []

                for edge in new_graph.edges:
                    connectivity.append(edge)
                network_new = QuantumNetwork(qpu_sizes, connectivity)
                network_new.mapping = new_mapping
                network_new.active_nodes = new_active_nodes
                network_list.append([network_new, new_active_nodes])
                
        return network_list