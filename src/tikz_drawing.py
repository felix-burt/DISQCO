import copy

def space_mapping(qpu_info, num_layers):
    qpu_mapping = {}
    qubit_index = 0
    for j, qpu_size in enumerate(qpu_info):
        qubit_list = []
        for _ in range(qpu_size):
            qubit_list.append(qubit_index)
            qubit_index += 1
        qpu_mapping[j] = qubit_list
    space_mapping = []
    for t in range(num_layers):
        space_mapping.append(copy.deepcopy(qpu_mapping))
    
    return space_mapping

def get_pos_list(num_qubits, assignment, space_map):
    num_layers = len(assignment)
    pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]
    for q in range(num_qubits):
        old_partition = None
        for t in range(num_layers):
            partition = assignment[t][q]
            if old_partition is not None:
                if partition == old_partition:
                    if x_index in space_map[t][partition]:
                        x_index = pos_list[t-1][q]
                        pos_list[t][q] = x_index
                        space_map[t][partition].remove(x_index)
                    else:
                        qubit_list = space_map[t][partition]
                        x_index = qubit_list.pop(0)
                        pos_list[t][q] = x_index

                else:
                    qubit_list = space_map[t][partition]
                    x_index = qubit_list.pop(0)
                    pos_list[t][q] = x_index
            else:
                qubit_list = space_map[t][partition]
                x_index = qubit_list.pop(0)
                pos_list[t][q] = x_index
            old_partition = partition
    return pos_list

def hypergraph_to_tikz(H, num_qubits, assignment, qpu_info, depth, num_qubits_phys, xscale=1.0, yscale=1.0, save=False, path=None):
    """
    Convert a QuantumCircuitHyperGraph 'H' into a TikZ string.
    
    Rules:
    - For each node (q,t):
        1) If t is 0 or max_time => white, smaller shape (e.g., "whiteSmallStyle").
        2) If node 'type' == "two-qubit" => black node ("blackStyle").
        3) If node 'type' == "single-qubit" => grey node ("greyStyle").
        4) Otherwise => invisible ("invisibleStyle").
    - For each hyperedge (keyed by root_node), place a small invisible "edge node"
        slightly offset from root_node, and:
        * Draw a line root_node -> edge_node
        * Draw a line edge_node -> each node in 'receiver_set'
    """
    space_map = space_mapping(qpu_info, depth)
    pos_list = get_pos_list(num_qubits, assignment, space_map)
    tikz_code = []
    tikz_code.append(r"\begin{tikzpicture}")

    # Helper to produce a LaTeX-friendly name from a node tuple
    def node_name(n):
        # n is typically (qubitIndex, timeStep), so e.g. (3,5) => "n_3_5"
        return "n_" + "_".join(str(x) for x in n)

    # Determine the maximum time-layer among H.nodes, or 0 if none
    if H.nodes:
        max_time = max(n[1] for n in H.nodes)
    else:
        max_time = 0

    # Assign a TikZ style name for each node based on layer & 'type'
    def pick_style(node):
        q, t = node
        if t == 0 or t == max_time:
            return "whiteSmallStyle"   # First or last layer => white & smaller
        node_type = H.get_node_attribute(node, 'type', None)
        if node_type == "group" or node_type == "two-qubit":
            return "blackStyle"
        elif node_type == "single-qubit":
            return "greyStyle"
        else:
            return "invisibleStyle"

    # 1) Draw the Nodes
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in H.nodes:
        # Get position, default to (0,0) if missing
        pos = (n[1],num_qubits_phys - pos_list[n[1]][n[0]])
        x, y = pos
        style = pick_style(n)
        tikz_code.append(
            f"    \\node [style={style}] ({node_name(n)}) "
            f"at ({x*xscale},{y*yscale}) {{}};"
        )
    tikz_code.append(r"  \end{pgfonlayer}")

    # 2) Draw Hyperedges
    #    Each hyperedge has a root node (edge_id) and possibly multiple receivers.
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")

    for edge_id, edge_info in H.hyperedges.items():
        # edge_id is the "key" for the hyperedge, presumably a single node (root)
        if isinstance(edge_id[0], int):
            roots = edge_info['root_set']
            root_node = edge_id
            root_t = edge_id[1]
            for node in roots:
                if node[1] < root_t:
                    root_node = node
                    root_t = node[1]

            root_pos = (root_node[1],num_qubits_phys - pos_list[root_node[1]][root_node[0]])
            rx, ry = root_pos

            # Define a new "edge node" slightly offset from the root
            # so lines can fan out from there
            receivers = edge_info["receiver_set"]
            if len(receivers) > 1:
                edge_node_name = "edge_" + node_name(root_node)
                offset_x = rx + 0.3
                offset_y = ry - 0.3
                # Draw the edge node as invisible
                tikz_code.append(
                    f"    \\node [style=invisibleStyle] ({edge_node_name}) "
                    f"at ({offset_x*xscale},{offset_y*yscale}) {{}};"
                )

                # Connect root -> this edge node
                tikz_code.append(
                    f"    \\draw ({node_name(root_node)}) to ({edge_node_name});"
                )
            else:
                edge_node_name = node_name(root_node)

            # Connect edge node -> all receivers
            
            for rnode in receivers:
                tikz_code.append(
                    f"    \\draw [bend right=15] ({edge_node_name}) to ({node_name(rnode)});"
                )
        else:

            root_set = edge_info['root_set']
            for node1 in root_set:
                break
            rec_set = edge_info['receiver_set']
            for node2 in rec_set:
                break
            if node1[0] != node2[0]:
                bend = "[bend right=15]"
            else:
                bend = ""
            tikz_code.append(
                f"    \\draw {bend} ({node_name(node1)}) to ({node_name(node2)});"
            )
        
    tikz_code.append(r"  \end{pgfonlayer}")

    tikz_code.append(r"\end{tikzpicture}")

    # Optionally save to a .tex file
    if save and path is not None:
        with open(path, "w") as f:
            f.write("\n".join(tikz_code))

    return "\n".join(tikz_code)
