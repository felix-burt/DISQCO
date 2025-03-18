
from disqco.drawing.map_positions import space_mapping, get_pos_list

def hypergraph_to_tikz(H, 
                       num_qubits, 
                       assignment, 
                       qpu_info, 
                       depth, 
                       num_qubits_phys, 
                       xscale=None, 
                       yscale=None, 
                       save=False, 
                       path=None):
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

    if xscale is None:
        xscale = 10/depth
    if yscale is None:
        yscale = 6/num_qubits
    space_map = space_mapping(qpu_info, depth)
    pos_list = get_pos_list(num_qubits, assignment, space_map)
    tikz_code = []
    tikz_code.append(r"\begin{tikzpicture}")
    def node_name(n):
        return "n_" + "_".join(str(x) for x in n)

    if H.nodes:
        max_time = max(n[1] for n in H.nodes)
    else:
        max_time = 0

    def pick_style(node):
        q, t = node
        if t == 0 or t == max_time:
            return "whiteSmallStyle"   
        node_type = H.get_node_attribute(node, 'type', None)
        if node_type == "group" or node_type == "two-qubit":
            return "blackStyle"
        elif node_type == "single-qubit":
            return "greyStyle"
        else:
            return "invisibleStyle"
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in H.nodes:
        pos = (n[1],num_qubits_phys - pos_list[n[1]][n[0]])
        x, y = pos
        style = pick_style(n)
        tikz_code.append(
            f"    \\node [style={style}] ({node_name(n)}) "
            f"at ({x*xscale},{y*yscale}) {{}};"
        )
    tikz_code.append(r"  \end{pgfonlayer}")

    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")

    for edge_id, edge_info in H.hyperedges.items():
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
            receivers = edge_info["receiver_set"]
            if len(receivers) > 1:
                edge_node_name = "edge_" + node_name(root_node)
                offset_x = rx + 0.3
                offset_y = ry - 0.3
                tikz_code.append(
                    f"    \\node [style=invisibleStyle] ({edge_node_name}) "
                    f"at ({offset_x*xscale},{offset_y*yscale}) {{}};"
                )
                tikz_code.append(
                    f"    \\draw ({node_name(root_node)}) to ({edge_node_name});"
                )
            else:
                edge_node_name = node_name(root_node)

            
            for rnode in receivers:
                tikz_code.append(
                    f"    \\draw [bend right=15] ({edge_node_name}) to ({node_name(rnode)});"
                )
            for rnode in roots:
                if rnode != root_node:
                    tikz_code.append(
                        f"    \\draw [bend right=15] ({node_name(rnode)}) to ({edge_node_name});")
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

    if save and path is not None:
        with open(path, "w") as f:
            f.write("\n".join(tikz_code))

    return "\n".join(tikz_code)
