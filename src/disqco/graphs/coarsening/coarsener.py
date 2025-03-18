import copy
import numpy as np
import time
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph

class HypergraphCoarsener:
    """
    A class that provides various coarsening methods for a quantum-circuit hypergraph.
    
    The 'hypergraph' argument is a QuantumCircuitHyperGraph-like object with:
      - .nodes (set of (q, t) tuples)
      - .hyperedges (dict, keyed by e_id -> { 'root_set': set(...), 'receiver_set': set(...) })
      - .node2hyperedges (dict, keyed by node -> set of e_ids)
      - .adjacency (dict, keyed by node -> set of neighbor_nodes)
      - .node_attrs (dict, keyed by node -> { ... attributes ... })
      - .copy() method returning a shallow copy of the hypergraph data structure.
      - .remove_hyperedge(e_id) for deleting an edge.

    The methods implemented include:
      - hyper_contract
      - hyper_contract_indexed
      - coarsen_hypergraph_full
      - coarsen_hypergraph_region
      - coarsen_hypergraph_blocks
      - coarsen_hypergraph_blocks_full
    """

    def __init__(self):
        """
        Store a reference (or a copy) of the hypergraph to be coarsened.
        """

    def contract(self, hypergraph, source, target):
        """
        Contract all nodes at time = index1 into nodes at time = index2,
        re-wiring edges, adjacency, and node2hyperedges so that each old_node
        (q, index1) is replaced by new_node (q, index2). Additionally, remove
        the 'time edges' connecting (q, index1) to (q, index±1).

        :param hypergraph: A QuantumCircuitHyperGraph object
        :param index1: The time-layer to contract
        :param index2: The time-layer to contract into (q, index2).
        :return: A *new* hypergraph object with contracted nodes.
        """
        H_new = hypergraph.copy()
        layer_nodes = [v for v in H_new.nodes if v[1] == source]

        for old_node in layer_nodes:
            q, _ = old_node
            new_node = (q, target)

            if new_node not in H_new.nodes:
                H_new.nodes.add(new_node)

            edges_to_remove = []
            for e_id in list(H_new.node2hyperedges[old_node]):
                if e_id not in H_new.hyperedges:
                    continue
                edge_data = H_new.hyperedges[e_id]
                root_s = edge_data["root_set"]
                rec_s = edge_data["receiver_set"]
                all_nodes_in_edge = root_s.union(rec_s)

                if len(all_nodes_in_edge) == 2:
                    if (q, target) in all_nodes_in_edge:
                        edges_to_remove.append(e_id)

            for e_id in edges_to_remove:
                H_new.remove_hyperedge(e_id)

            old_neighbors = list(H_new.adjacency[old_node])
            for nbr in old_neighbors:
                H_new.adjacency[nbr].discard(old_node)
                H_new.adjacency[nbr].add(new_node)
                H_new.adjacency[new_node].add(nbr)

            if old_node in H_new.adjacency:
                del H_new.adjacency[old_node]

            remaining_edges = list(H_new.node2hyperedges[old_node])
            for e_id in remaining_edges:
                H_new.node2hyperedges[old_node].remove(e_id)
                H_new.node2hyperedges[new_node].add(e_id)
                edge_data = H_new.hyperedges[e_id]
                root_s = edge_data['root_set']
                rec_s  = edge_data['receiver_set']
                if old_node in root_s:
                    root_s.remove(old_node)
                    root_s.add(new_node)
                elif old_node in rec_s:
                    rec_s.remove(old_node)
                    rec_s.add(new_node)

            if old_node in H_new.nodes:
                H_new.nodes.remove(old_node)
            
            if old_node in H_new.node_attrs:
                del H_new.node_attrs[old_node]

        return H_new
    
    def contract_weighted(self, hypergraph, source, target):
        """
        Contract all nodes at time = index1 into nodes at time = index2,
        re-wiring edges, adjacency, and node2hyperedges so that each old_node
        (q, index1) is replaced by new_node (q, index2). Additionally, remove
        the 'time edges' connecting (q, index1) to (q, index±1).

        :param hypergraph: A QuantumCircuitHyperGraph object
        :param index1: The time-layer to contract
        :param index2: The time-layer to contract into (q, index2).
        :return: A *new* hypergraph object with contracted nodes.
        """
        H_new = hypergraph.copy()
        layer_nodes = [v for v in H_new.nodes if v[1] == source]

        for old_node in layer_nodes:
            q, _ = old_node
            new_node = (q, target)

            if new_node not in H_new.nodes:
                H_new.nodes.add(new_node)

            edges_to_remove = []
            for e_id in list(H_new.node2hyperedges[old_node]):
                print("Edge ID:", e_id)
                edge_data = H_new.hyperedges[e_id]
                root_s = edge_data["root_set"]
                rec_s = edge_data["receiver_set"]
                new_e_id = (root_s, rec_s)
                if new_e_id not in H_new.hyperedges:
                    H_new.hyperedges[new_e_id] = {
                        "root_set": root_s,
                        "receiver_set": rec_s,
                        "weight": 1
                    }
                else:
                    H_new.hyperedges[new_e_id]["weight"] += 1
                
                all_nodes_in_edge = root_s.union(rec_s)
                if len(all_nodes_in_edge) == 2:
                    if (q, target) in all_nodes_in_edge:
                        edges_to_remove.append(e_id)

            for e_id in edges_to_remove:
                H_new.remove_hyperedge(e_id)

            old_neighbors = list(H_new.adjacency[old_node])
            for nbr in old_neighbors:
                H_new.adjacency[nbr].discard(old_node)
                H_new.adjacency[nbr].add(new_node)
                H_new.adjacency[new_node].add(nbr)

            if old_node in H_new.adjacency:
                del H_new.adjacency[old_node]

            remaining_edges = list(H_new.node2hyperedges[old_node])
            for e_id in remaining_edges:
                H_new.node2hyperedges[old_node].remove(e_id)
                H_new.node2hyperedges[new_node].add(e_id)
                edge_data = H_new.hyperedges[e_id]
                root_s = edge_data['root_set']
                rec_s  = edge_data['receiver_set']
                if old_node in root_s:
                    root_s.remove(old_node)
                    root_s.add(new_node)
                elif old_node in rec_s:
                    rec_s.remove(old_node)
                    rec_s.add(new_node)

            if old_node in H_new.nodes:
                H_new.nodes.remove(old_node)
            
            if old_node in H_new.node_attrs:
                del H_new.node_attrs[old_node]

        return H_new
   
    def update_mapping(self, mapping, source, target):
        mapping[target] = mapping[target].union(mapping[source])
        del mapping[source]
        return mapping

    def coarsen_full(self, hypergraph, num_levels):
        """
        Iteratively coarsen a GCP hypergraph from `depth` down to 0,
        returning a list of hypergraphs at progressively coarser time-layers.

        :param hypergraph: The original (fine) hypergraph.
                          (nodes = {(q, t) | t in [0..depth]})
        :param depth: The maximum time-layer index.
        :return: A list of hypergraphs [H_0, H_1, ..., H_depth]
                where H_0 is the original and H_k is the k-th coarsened graph.
        """
        depth = hypergraph.depth
        layer = depth - 1
        H_list = []
        H_list.append(copy.deepcopy(hypergraph))
        block_size = depth // num_levels
        # Initialise list of contraction mappings for each layer, indicating which node IDs are merged
        mapping_list = []
        mapping = {i: set([i]) for i in range(depth)}
        mapping_list.append(copy.deepcopy(mapping))
        level = 0
        H_current = copy.deepcopy(hypergraph)
        while layer > 0 and level < num_levels:
            H_current, mapping = self.coarsen_region(H_current, mapping, layer, max(layer - block_size,0))
            mapping_list.append(copy.deepcopy(mapping))
            H_list.append(copy.deepcopy(H_current))
            layer  = max(layer-block_size,0)
            level += 1
        
        return H_list, mapping_list

    def coarsen_full_weighted(self, hypergraph, num_levels):
        """
        Iteratively coarsen a GCP hypergraph from `depth` down to 0,
        returning a list of hypergraphs at progressively coarser time-layers.

        :param hypergraph: The original (fine) hypergraph.
                        (nodes = {(q, t) | t in [0..depth]})
        :param depth: The maximum time-layer index.
        :return: A list of hypergraphs [H_0, H_1, ..., H_depth]
                where H_0 is the original and H_k is the k-th coarsened graph.
        """
        depth = hypergraph.depth
        layer = depth - 1
        H_list = []
        H_list.append(copy.deepcopy(hypergraph))
        block_size = depth // num_levels
        # Initialise list of contraction mappings for each layer, indicating which node IDs are merged
        mapping_list = []
        mapping = {i: set([i]) for i in range(depth)}
        mapping_list.append(copy.deepcopy(mapping))
        level = 0
        H_current = copy.deepcopy(hypergraph)
        while layer > 0 and level < num_levels:
            H_current, mapping = self.coarsen_region(H_current, mapping, layer, max(layer - block_size,0))
            mapping_list.append(copy.deepcopy(mapping))
            H_list.append(copy.deepcopy(H_current))
            layer  = max(layer-block_size,0)
            level += 1
        
        return H_list, mapping_list
    
    def coarsen_region(self, graph, mapping, start, stop):
        """
        Coarsen a GCP hypergraph from `start` down to `stop`,
        returning a list of hypergraphs at progressively coarser time-layers.

        :param hypergraph: The original (fine) hypergraph,
                           with nodes = {(q, t) | t in [0..depth]}.
        :param start: The starting time-layer index.
        :param stop: The stopping time-layer index.
        :return: A list of hypergraphs [H_start, H_start-1, ..., H_stop]
                where H_start is the original and H_stop is the final coarsened.
        """
        graph_ = copy.deepcopy(graph)
        
        layer = start
        while layer > stop:
            graph_ = self.contract(graph_, layer, layer - 1)
            mapping = self.update_mapping(mapping, layer, layer - 1)
            layer -= 1
        return graph_, mapping

    def coarsen_blocks(self, hypergraph, num_blocks, block_size = None):
        """
        Coarsen a GCP hypergraph into `num_blocks` blocks, where
        each block is a coarsened version of the previous block.

        :param hypergraph: The original (fine) hypergraph,
                           nodes={(q, t)| t in [0..depth]}.
        :param num_blocks: The number of blocks to coarsen into.
        :return: A list of hypergraphs [H_0, H_1, ..., H_num_blocks]
        """
        # This version is the function you wrote, but it's incomplete or a WIP.
        # We'll leave it as is, or adapt as you wish.
        H_list = [hypergraph]
        H_current = copy.deepcopy(hypergraph)

        depth = hypergraph.depth  # or however you store 'depth'
        if block_size is None:
            block_size = depth // num_blocks
        else:
            num_blocks = depth // block_size

        start_layer = depth - 1
        mapping_list = []
        mapping = {i: set([i]) for i in range(depth)}
        mapping_list.append(copy.deepcopy(mapping))

        while start_layer > depth - block_size:
            layer = start_layer
            while layer > 0:
                H_current = self.contract(H_current, layer, layer -1 )
                mapping = self.update_mapping(mapping, layer, layer - 1)
                layer -= block_size
            mapping_list.append(copy.deepcopy(mapping))
            H_list.append(H_current)
            start_layer -= 1

        remaining_layers = list(mapping.keys())

        length = len(remaining_layers)

        for i in range(0, length-1):
            source = remaining_layers[-i-1]
            target = remaining_layers[-i-2]
            H_current = self.contract(H_current, source, target)
            mapping = self.update_mapping(mapping, source, target)
        mapping_list.append(copy.deepcopy(mapping))
        H_list.append(H_current)
        
        return H_list, mapping_list

    def coarsen_blocks_full(self, hypergraph, mapping):
        """
        Another approach to iterative coarsening, using a 'mapping'
        structure that indicates how layers should be merged. 
        """
        H_list = [hypergraph]
        H_current = copy.deepcopy(hypergraph)

        super_nodes = sorted(list(mapping.keys()), reverse=True)
        mapping_list = [copy.deepcopy(mapping)]

        for i, t in enumerate(super_nodes[:-1]):
            H_current = self.contract(H_current, t, super_nodes[i+1])
            mapping = self.update_mapping(mapping, t, super_nodes[i+1])
            mapping_list.append(copy.deepcopy(mapping))
            H_list.append(H_current)

        return H_list, mapping_list
    
    def coarsen_recursive(self, hypergraph):
        """
        Repeatedly coarsen the hypergraph by contracting layer i into i-1
        in a pairwise fashion:
        - (depth-1 -> depth-2), (depth-3 -> depth-4), ...
        so that in one pass, roughly half of the layers are merged.
        Continue until only 1 layer remains.

        Returns:
            H_list, mapping_list
        where H_list is a list of intermediate hypergraphs after each pass,
                mapping_list is a list of layer-mappings after each pass.
        """
        H_current = copy.deepcopy(hypergraph)
        depth = H_current.depth
        mapping = {i: set([i]) for i in range(depth)}

        H_list = [H_current]
        mapping_list = [copy.deepcopy(mapping)]
        while True:
            # Identify current max layer
            start = time.time()

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

                H_current = self.contract(H_current, src, tgt)

                mapping = self.update_mapping(mapping, src, tgt)

            H_list.append(H_current)
            mapping_list.append(copy.deepcopy(mapping))

            current_layers = sorted(mapping.keys())
            if len(current_layers) <= 1:
                break

        return H_list, mapping_list

    def contract_batch(self, hypergraph, merges):
        """
        merges is a list of (src, tgt) pairs we want to contract in a single pass.
        Return a new hypergraph with all these merges done.

        For layer-based merges, merges might look like:
        [(255->254), (253->252), (251->250), ...]
        """
        circuit = hypergraph.circuit
        # 1) Build the representative map: "which node does X become after merges?"
        rep = {}  # dictionary from old_node -> representative_node
        for v in hypergraph.nodes:
            rep[v] = v  # default: each node is its own rep

        for (src, tgt) in merges:
            for q in range(hypergraph.num_qubits):
            # "src" goes to "tgt"
            # But also if "tgt" itself was mapped somewhere, we need to chain them.
            # e.g. if rep[tgt] = something else, you unify src -> rep[tgt].
                final_tgt = rep[(q,tgt)]
                rep[(q,src)] = final_tgt

        # OPTIONAL: if merges chain, do a second pass to ensure "rep[v] = final representative"
        # e.g. for each node v: while rep[v] != rep[ rep[v] ]: rep[v] = rep[ rep[v] ]
        # This flattens the representative links.

        # 2) Create new data structures
        new_nodes = set()
        new_adjacency = {}
        new_node2hyperedges = {}
        new_hyperedges = {}

        # 3) Build the new node set
        for v in hypergraph.nodes:
            rv = rep[v]  # representative
            new_nodes.add(rv)

        # Initialize adjacency sets, node2hyperedges sets
        for rv in new_nodes:
            new_adjacency[rv] = set()
            new_node2hyperedges[rv] = set()

        # 4) Adjacency
        for v in hypergraph.nodes:
            rv = rep[v]
            for nbr in hypergraph.adjacency[v]:
                rnbr = rep[nbr]
                if rnbr != rv:  # skip self-loops
                    new_adjacency[rv].add(rnbr)
                    # Also add the reverse: new_adjacency[rnbr].add(rv) if undirected
                    new_adjacency[rnbr].add(rv)

        # 5) node2hyperedges
        for v in hypergraph.nodes:
            rv = rep[v]
            for e_id in hypergraph.node2hyperedges[v]:
                new_node2hyperedges[rv].add(e_id)

        # 6) hyperedges
        for e_id, e_data in hypergraph.hyperedges.items():
            # Rebuild root_set, receiver_set with reps
            new_roots = {rep[x] for x in e_data["root_set"]}
            new_recs = {rep[x] for x in e_data["receiver_set"]}
            # Possibly skip if new_roots == new_recs or if it becomes trivial
            # or unify if new_roots intersects new_recs, etc. (application-specific)
            new_hyperedges[e_id] = {
                "root_set": new_roots,
                "receiver_set": new_recs
            }

        # 7) node_attrs
        new_node_attrs = {}
        for rv in new_nodes:
            new_node_attrs[rv] = {}  # or copy from one node as default
        # For each old_node, merge attrs into rep[v]
        for v, attrs in hypergraph.node_attrs.items():
            rv = rep[v]
            # do some merging logic: e.g. sum numeric attributes, union sets, etc.
            # if you just want to overwrite or store a list, you do:
            # new_node_attrs[rv].update(attrs)
            # or something more advanced
            new_node_attrs[rv].update(attrs)

        # 8) Construct the new hypergraph object or mutate the old one
        new_H = QuantumCircuitHyperGraph(circuit=circuit)
        new_H.nodes = new_nodes
        new_H.adjacency = new_adjacency
        new_H.node2hyperedges = new_node2hyperedges
        new_H.hyperedges = new_hyperedges
        new_H.node_attrs = new_node_attrs
        # etc.

        return new_H
    
    def coarsen_recursive_batches(self, hypergraph):
        """
        Repeatedly coarsen the hypergraph by contracting layer i into i-1
        in a pairwise fashion:
        - (depth-1 -> depth-2), (depth-3 -> depth-4), ...
        so that in one pass, roughly half of the layers are merged.
        Continue until only 1 layer remains.

        Returns:
            H_list, mapping_list
        where H_list is a list of intermediate hypergraphs after each pass,
                mapping_list is a list of layer-mappings after each pass.
        """
        H_current = copy.deepcopy(hypergraph)
        depth = H_current.depth
        mapping = {i: set([i]) for i in range(depth)}

        H_list = [H_current]
        mapping_list = [copy.deepcopy(mapping)]
        while True:
            # print("Current mapping:", mapping)
            # Identify current max layer

            current_layers = sorted(mapping.keys())

            # print("Time to sort layers:", time.time() - start)

            if len(current_layers) <= 1:
                break
            pairs_to_merge = []

            rev = list(reversed(current_layers))
            for i in range(0, len(rev)-1, 2):
                source = rev[i]
                target = rev[i+1]
                pairs_to_merge.append((source, target))

            for (src, tgt) in pairs_to_merge:
                mapping = self.update_mapping(mapping, src, tgt)
            H_current = self.contract_batch(H_current, pairs_to_merge)
            # print("Time to perform all contractions:", time.time() - start_outer)

            H_list.append(H_current)
            mapping_list.append(copy.deepcopy(mapping))

            current_layers = sorted(mapping.keys())
            if len(current_layers) <= 1:
                break
        

        return H_list, mapping_list