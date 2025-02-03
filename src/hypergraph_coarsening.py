import copy

def hyper_contract(hypergraph, layer):
    """
    Contract all nodes at time = `layer` into nodes at time = `layer - 1`,
    re-wiring edges, adjacency, and node2hyperedges so that each old_node
    (q, layer) is replaced by new_node (q, layer-1). Additionally, remove
    the 'time edges' connecting (q, layer) to (q, layer±1).

    :param hypergraph: A QuantumCircuitHyperGraph object
    :param layer: The time-layer to contract into (layer - 1).
    :return: A *new* hypergraph object with contracted nodes.
    """

    H_new = hypergraph.copy()  # Work on a copy so we don't mutate the original

    # Identify nodes at the given layer
    layer_nodes = [v for v in H_new.nodes if v[1] == layer]

    for old_node in layer_nodes:
        q, _ = old_node
        new_node = (q, layer - 1)

        # Ensure new_node is in H_new.nodes
        if new_node not in H_new.nodes:
            H_new.nodes.add(new_node)
            # (Optional) copy any desired attributes from old_node to new_node:
            # H_new.node_attrs[new_node] = dict(H_new.node_attrs.get(old_node, {}))

        ##################################################################
        # 1) Identify and remove "time edges" that connect old_node => (q,layer±1)
        ##################################################################
        edges_to_remove = []
        # We'll look at all edges for old_node; some might be ( (q,layer),(q,layer±1) ).
        for e_id in list(H_new.node2hyperedges[old_node]):  # copy list
            if e_id not in H_new.hyperedges:
                continue  # Might have been removed in a previous step
            edge_data = H_new.hyperedges[e_id]
            root_s = edge_data["root_set"]
            rec_s = edge_data["receiver_set"]
            all_nodes_in_edge = root_s.union(rec_s)

            # Check if this edge is exactly a 2-node "time edge" e.g. {(q, layer),(q,layer±1)}
            if len(all_nodes_in_edge) == 2:
                # e.g. it might be {(q, layer),(q, layer+1)}
                if (q, layer - 1) in all_nodes_in_edge:
                    # It's a direct vertical/horizontal neighbor edge
                    edges_to_remove.append(e_id)

        # Remove them
        for e_id in edges_to_remove:
            H_new.remove_hyperedge(e_id)

        ##################################################################
        # 2) Update adjacency: rewire neighbors from old_node -> new_node
        ##################################################################
        old_neighbors = list(H_new.adjacency[old_node])  # snapshot
        for nbr in old_neighbors:
            # Remove old_node from neighbor's adjacency
            H_new.adjacency[nbr].discard(old_node)
            # Add new_node to neighbor's adjacency
            H_new.adjacency[nbr].add(new_node)
            # Add neighbor to new_node's adjacency
            H_new.adjacency[new_node].add(nbr)

        # Remove old_node adjacency entry
        if old_node in H_new.adjacency:
            del H_new.adjacency[old_node]

        ##################################################################
        # 3) Update node2hyperedges & hyperedges to replace old_node w/ new_node
        ##################################################################
        # We re-check since some edges might have been removed in edges_to_remove
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

        ##################################################################
        # 4) Remove old_node from H_new
        ##################################################################
        if old_node in H_new.nodes:
            H_new.nodes.remove(old_node)
        if old_node in H_new.node_attrs:
            del H_new.node_attrs[old_node]

    return H_new

def coarsen_hypergraph_full(hypergraph, depth):
    """
    Iteratively coarsen a GCP hypergraph from `depth` down to 0,
    returning a list of hypergraphs at progressively coarser time-layers.

    :param hypergraph: The original (fine) hypergraph, with
                       nodes = {(q, t) | t in [0..depth]}.
    :param depth: The maximum time-layer index.
    :return: A list of hypergraphs [H_0, H_1, ..., H_depth]
             where H_0 is the original and H_k is the k-th coarsened graph.
    """
    # Start with the fine graph
    layer = depth - 1
    H_list = []
    H_list.append(hypergraph)

    # Make a working copy
    H_current = copy.deepcopy(hypergraph)

    # Contract layer by layer down to 0
    while layer > 0:
        H_current = hyper_contract(H_current, layer)
        H_list.append(H_current)
        layer -= 1

    return H_list

def coarsen_hypergraph_region(hypergraph, start, stop):
    """
    Coarsen a GCP hypergraph from `start` down to `stop`,
    returning a list of hypergraphs at progressively coarser time-layers.

    :param hypergraph: The original (fine) hypergraph, with
                       nodes = {(q, t) | t in [0..depth]}.
    :param start: The starting time-layer index.
    :param stop: The stopping time-layer index.
    :return: A list of hypergraphs [H_start, H_start+1, ..., H_stop]
             where H_start is the original and H_stop is the coarsened graph.
    """
    # Start with the fine graph
    H_list = []
    H_list.append(hypergraph)

    # Make a working copy
    H_current = copy.deepcopy(hypergraph)

    # Contract layer by layer down to `stop`
    layer = start
    while layer > stop:
        H_current = hyper_contract(H_current, layer)
        H_list.append(H_current)
        layer -= 1

    return H_list

def coarsen_hypergraph_blocks(hypergraph, num_blocks):
    """
    Coarsen a GCP hypergraph into `num_blocks` blocks, where
    each block is a coarsened version of the previous block.

    :param hypergraph: The original (fine) hypergraph, with
                       nodes = {(q, t) | t in [0..depth]}.
    :param num_blocks: The number of blocks to coarsen into.
    :return: A list of hypergraphs [H_0, H_1, ..., H_num_blocks]
             where H_0 is the original and H_k is the k-th coarsened graph.
    """
    # Start with the fine graph
    H_list = []
    H_list.append(hypergraph)

    # Make a working copy
    H_current = copy.deepcopy(hypergraph)

    # Contract layer by layer down to 0
    depth = hypergraph.depth
    block_size = depth // num_blocks
    start_layer = depth - 1
    mapping_list = []
    mapping = {i : set([i]) for i in range(depth)}
    mapping_list.append(copy.deepcopy(mapping))
    while start_layer > depth - block_size:
        layer = start_layer
        print(f'Start layer: {layer}')
        while layer > 0:
            print(f'Layer: {layer}')
            H_current = hyper_contract(H_current, layer)
            mapping[layer-1] = mapping[layer-1].union(mapping[layer])
            del mapping[layer]
            print(f'Contracted to layer: {layer-1}')
            print(f'Number of nodes: {len(H_current.nodes)}')
            layer -= block_size
        mapping_list.append(copy.deepcopy(mapping))
        print(f'Mapping: {mapping}')
        H_list.append(H_current)
        print(f'Nodes in block: {sorted(H_current.nodes, key=lambda x: x[1])}')
        start_layer -= 1
    
    print(f'Mapping: {mapping}')
    # for i in range(num_blocks):
    #     H_current = hyper_contract(H_current, 0)
    return H_list, mapping_list


def hyper_contract_indexed(hypergraph, index1, index2):
    """
    Contract all nodes at time = `layer` into nodes at time = `layer - 1`,
    re-wiring edges, adjacency, and node2hyperedges so that each old_node
    (q, layer) is replaced by new_node (q, layer-1). Additionally, remove
    the 'time edges' connecting (q, layer) to (q, layer±1).

    :param hypergraph: A QuantumCircuitHyperGraph object
    :param layer: The time-layer to contract into (layer - 1).
    :return: A *new* hypergraph object with contracted nodes.
    """

    H_new = hypergraph.copy()  # Work on a copy so we don't mutate the original

    # Identify nodes at the given layer
    layer_nodes = [v for v in H_new.nodes if v[1] == index1]

    for old_node in layer_nodes:
        q, _ = old_node
        new_node = (q, index2)

        # Ensure new_node is in H_new.nodes
        if new_node not in H_new.nodes:
            H_new.nodes.add(new_node)
            # (Optional) copy any desired attributes from old_node to new_node:
            # H_new.node_attrs[new_node] = dict(H_new.node_attrs.get(old_node, {}))

        ##################################################################
        # 1) Identify and remove "time edges" that connect old_node => (q,layer±1)
        ##################################################################
        edges_to_remove = []
        # We'll look at all edges for old_node; some might be ( (q,layer),(q,layer±1) ).
        for e_id in list(H_new.node2hyperedges[old_node]):  # copy list
            if e_id not in H_new.hyperedges:
                continue  # Might have been removed in a previous step
            edge_data = H_new.hyperedges[e_id]
            root_s = edge_data["root_set"]
            rec_s = edge_data["receiver_set"]
            all_nodes_in_edge = root_s.union(rec_s)

            # Check if this edge is exactly a 2-node "time edge" e.g. {(q, layer),(q,layer±1)}
            if len(all_nodes_in_edge) == 2:
                # e.g. it might be {(q, layer),(q, layer+1)}
                if (q, index2) in all_nodes_in_edge:
                    # It's a direct vertical/horizontal neighbor edge
                    edges_to_remove.append(e_id)

        # Remove them
        for e_id in edges_to_remove:
            H_new.remove_hyperedge(e_id)

        ##################################################################
        # 2) Update adjacency: rewire neighbors from old_node -> new_node
        ##################################################################
        old_neighbors = list(H_new.adjacency[old_node])  # snapshot
        for nbr in old_neighbors:
            # Remove old_node from neighbor's adjacency
            H_new.adjacency[nbr].discard(old_node)
            # Add new_node to neighbor's adjacency
            H_new.adjacency[nbr].add(new_node)
            # Add neighbor to new_node's adjacency
            H_new.adjacency[new_node].add(nbr)

        # Remove old_node adjacency entry
        if old_node in H_new.adjacency:
            del H_new.adjacency[old_node]

        ##################################################################
        # 3) Update node2hyperedges & hyperedges to replace old_node w/ new_node
        ##################################################################
        # We re-check since some edges might have been removed in edges_to_remove
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

        ##################################################################
        # 4) Remove old_node from H_new
        ##################################################################
        if old_node in H_new.nodes:
            H_new.nodes.remove(old_node)
        if old_node in H_new.node_attrs:
            del H_new.node_attrs[old_node]

    return H_new

def coarsen_hypergraph_blocks_full(hypergraph, mapping):
    """
    Iteratively coarsen a GCP hypergraph from `depth` down to 0,
    returning a list of hypergraphs at progressively coarser time-layers.

    :param hypergraph: The original (fine) hypergraph, with
                       nodes = {(q, t) | t in [0..depth]}.
    :param depth: The maximum time-layer index.
    :return: A list of hypergraphs [H_0, H_1, ..., H_depth]
             where H_0 is the original and H_k is the k-th coarsened graph.
    """
    # Start with the fine graph
    H_list = []
    H_list.append(hypergraph)

    # Make a working copy
    H_current = copy.deepcopy(hypergraph)

    # Contract layer by layer down to 0
    print(mapping)
    super_nodes = sorted(list(mapping.keys()), reverse=True)
    mapping_list = []
    mapping_list.append(copy.deepcopy(mapping))
    for i, t in enumerate(super_nodes[:-1]):
        H_current = hyper_contract_indexed(H_current, t, super_nodes[i+1])
        mapping[super_nodes[i+1]] = mapping[super_nodes[i+1]].union(mapping[t])
        del mapping[t]
        mapping_list.append(copy.deepcopy(mapping))
        H_list.append(H_current)

    return H_list, mapping_list