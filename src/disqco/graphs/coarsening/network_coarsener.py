from disqco.graphs.quantum_network import QuantumNetwork
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
import networkx as nx
import matplotlib.pyplot as plt
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
        
        # Mark the coarsest level network nodes as level 0

        for node in self.network_coarse_list[-1].qpu_graph.nodes:
            self.network_coarse_list[-1].qpu_graph.nodes[node]['level'] = 0
        
        return self.network_coarse_list
    
    def merge_old_dummies(self, g0, nodes_to_merge, mapping, dummy_nodes):
        """
        Recursively merge old dummy nodes in the graph g0. 
        Old dummy nodes can be merged together or they can be merged with new dummy nodes.
        This function repeats until no more merges are possible.
        
        Args:
            g0: The graph to modify
            nodes_to_merge: Set of old dummy nodes that can be merged
            mapping: The current mapping dictionary
            dummy_nodes: Set of new dummy nodes that old dummies can merge into
        """
        merged_something = True
        
        while merged_something:
            merged_something = False
            edge_list = list(g0.edges())
            
            for edge in edge_list:
                node1, node2 = edge
                
                # Skip if either node has already been removed from the graph
                if node1 not in g0.nodes or node2 not in g0.nodes:
                    continue
                
                # Case 1: Both nodes are old dummy nodes that need to be merged
                if node1 in nodes_to_merge and node2 in nodes_to_merge:

                    self.merge_nodes(g0, node1, node2, mapping)
                    nodes_to_merge.discard(node2)  # Remove the merged node
                    merged_something = True
                    break  # Start over with new edge list
                
                # Case 2: One old dummy node connected to a new dummy node
                elif node1 in nodes_to_merge and node2 in dummy_nodes:
                    self.merge_nodes(g0, node2, node1, mapping)
                    nodes_to_merge.discard(node1)  # Remove the merged node
                    merged_something = True
                    break  # Start over with new edge list
                
                # Case 3: One new dummy node connected to an old dummy node
                elif node1 in dummy_nodes and node2 in nodes_to_merge:
                    self.merge_nodes(g0, node1, node2, mapping)
                    nodes_to_merge.discard(node2)  # Remove the merged node
                    merged_something = True
                    break  # Start over with new edge list
        


    def unmerge_nodes(self, g0 : nx.Graph, source_node : int, mapping : dict[int,set[int]], level : int, active_nodes):


        parent_network = self.network_coarse_list[-level-2]
        g = parent_network.qpu_graph
        parent_mapping = parent_network.mapping
        # Remove the source node from the, this will be added back in with the sub-nodes
        g0.remove_node(source_node)
        # Find the nodes to be expanded - these are the sub-nodes of the source node
        new_active_nodes = mapping[source_node].intersection(g.nodes)

        # Add the sub-nodes to the new graph
        for node in new_active_nodes:
            g0.add_node(node, size = g.nodes[node]['size'], level=level+1)
        
        

        # for node in g0.nodes:
        #     print(f"Node: {node}, node level: {g0.nodes[node]['level']}")

        # Define the dummy nodes, the nodes from the original graph that are not in the sub-nodes
        dummy_nodes = set()
        for node in active_nodes:
            if node not in new_active_nodes:
                dummy_nodes.add(node)
        

        old_dummy_nodes = set(g0.nodes) - dummy_nodes - set(new_active_nodes)

        # Connect sub-nodes to the dummy nodes
        for node1 in new_active_nodes: 
            for coarse_node in dummy_nodes | old_dummy_nodes:
                contained_nodes = mapping[coarse_node]
                for node2 in contained_nodes:
                    if node1 != node2 and g.has_edge(node1, node2):
                        g0.add_edge(node1, coarse_node)



        # Connect sub-nodes to each other
        for node1 in g0.nodes:
            for node2 in g0.nodes:
                if g.has_edge(node1, node2):
                    g0.add_edge(node1, node2)
    

        # Merge old dummy nodes using the recursive function
        nodes_to_merge = old_dummy_nodes.copy()  # Make a copy since the function modifies it
        new_mapping = {node : parent_mapping[node] for node in new_active_nodes}  # Create a new mapping to avoid modifying the original
        old_dummy_mapping = {node : mapping[node] for node in old_dummy_nodes}
        new_mapping.update(old_dummy_mapping)
        dummy_mapping = {node : mapping[node] for node in dummy_nodes}
        new_mapping.update(dummy_mapping)
        self.merge_old_dummies(g0, nodes_to_merge, new_mapping, dummy_nodes)

        # print("New mapping: ", new_mapping)





        qpu_sizes = {node : g0.nodes[node]['size'] for node in g0.nodes}
        g0.remove_edges_from(nx.selfloop_edges(g0))

        # new_active_nodes = set([node for node in sub_nodes if node in g.nodes])


        # new_mapping = {node : parent_mapping[node] for node in new_active_nodes}

        # Build mapping for remaining dummy nodes (both old and new)
        remaining_dummy_nodes = set()
        for node in g0.nodes:
            if node not in new_active_nodes:
                remaining_dummy_nodes.add(node)
        
        # old_dummy_node_mapping = {node : mapping[node] for node in remaining_dummy_nodes}
        # dummy_node_mapping = {node : mapping[node] for node in dummy_nodes}

        # new_mapping.update(old_dummy_node_mapping)
        # new_mapping.update(dummy_node_mapping)

        # nx.draw(g0, with_labels=True)

        # plt.show()






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