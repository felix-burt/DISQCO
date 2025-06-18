import copy
import numpy as np
import time
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
class HypergraphCoarsener:
    """
    A class that provides coarsening methods for a quantum-circuit hypergraph.
    Only coarsens qubit nodes, preserving all gate nodes and their connections.
    """

    def __init__(self):
        pass

    def contract_time_mapped(self, hypergraph, source, target, node_list):
        """
        Contract all qubit nodes at time = source into nodes at time = target,
        preserving all gate nodes and their connections.

        :param hypergraph: A HyperGraph object
        :param source: The time-layer to contract
        :param target: The time-layer to contract into
        :param node_list: List of node IDs for each layer
        :return: A *new* hypergraph object with contracted nodes.
        """
        H_new = hypergraph.copy()
        layer_nodes = [v for v in H_new.nodes() if v[1] == source and len(v) == 2]

        for i, old_node in enumerate(layer_nodes):
            if len(node_list[target]) <= i:
                continue
            new_node = (node_list[target][i], target)
            
            # Skip if either node is not in the graph
            if new_node not in H_new._G or old_node not in H_new._G:
                continue

            # Get all hyperedges containing the old node
            old_edges = list(H_new.incident(old_node))
            
            # Process each hyperedge
            for edge_key in old_edges:
                if edge_key not in H_new._hyperedges:
                    continue
                
                hedge = H_new._hyperedges[edge_key]
                vertices = hedge.vertices
                
                # Skip if this is a size-2 edge and contains the target node
                if len(vertices) == 2 and new_node in vertices:
                    H_new.remove_hyperedge(edge_key)
                    continue

                # Update the hyperedge to use new_node instead of old_node
                if old_node in vertices:
                    vertices.remove(old_node)
                    vertices.add(new_node)
                    
                    # Update the NetworkX graph edges
                    H_new._G.remove_edge(edge_key, old_node)
                    H_new._G.add_edge(edge_key, new_node)
                    
                    # Update incidence lists
                    H_new._del_inc(old_node, edge_key)
                    H_new._add_inc(new_node, edge_key)

            # Update adjacency
            old_neighbors = list(H_new.neighbors(old_node))
            for nbr in old_neighbors:
                if nbr != new_node:  # Avoid self-loops
                    # Add edges between new_node and old neighbors
                    H_new._G.add_edge(new_node, nbr)

            # Remove the old node
            H_new._G.remove_node(old_node)
            
            # Update qubit nodes list
            if old_node in H_new.qubit_nodes:
                H_new.qubit_nodes.remove(old_node)

        return H_new

    def update_mapping(self, mapping, source, target):
        """Update the mapping to reflect merging of source into target."""
        mapping[target] = mapping[target].union(mapping[source])
        del mapping[source]
        return mapping

    def coarsen_recursive_mapped(self, hypergraph, node_list):
        """
        Repeatedly coarsen the hypergraph by contracting layer i into i-1
        in a pairwise fashion, preserving all gate nodes:
        - (depth-1 -> depth-2), (depth-3 -> depth-4), ...
        so that in one pass, roughly half of the layers are merged.
        Continue until only 1 layer remains.

        Returns:
            H_list, mapping_list
        where H_list is a list of intermediate hypergraphs after each pass,
                mapping_list is a list of layer-mappings after each pass.
        """
        H_current = hypergraph.copy()
        H_init = hypergraph.copy()
        depth = hypergraph.depth
    
        mapping = {i: set([i]) for i in range(depth)}

        H_list = [H_init]
        mapping_list = [copy.deepcopy(mapping)]
        
        while True:
            current_layers = sorted(mapping.keys())
            if len(current_layers) <= 1:
                break
                
            pairs_to_merge = []
            rev = list(reversed(current_layers))
            for i in range(0, len(rev)-1, 2):
                source = rev[i]
                target = rev[i+1]
                pairs_to_merge.append((source, target))

            for (src, tgt) in pairs_to_merge:
                H_current = self.contract_time_mapped(H_current, src, tgt, node_list)
                mapping = self.update_mapping(mapping, src, tgt)
            H_current.depth = len(mapping)
            H_list.append(H_current)
            mapping_list.append(copy.deepcopy(mapping))

            current_layers = sorted(mapping.keys())
            if len(current_layers) <= 1:
                break

        return H_list, mapping_list
