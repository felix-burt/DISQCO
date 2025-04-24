from IPython import get_ipython
from disqco.drawing.map_positions import space_mapping, get_pos_list

def hypergraph_to_tikz(
    H,
    assignment,
    qpu_info,
    xscale=None,
    yscale=None,
    save=False,
    path=None,
    invert_colors=False,
    fill_background=True,
    assignment_map= None,
):
    """
    Convert a QuantumCircuitHyperGraph 'H' into a full standalone TikZ/LaTeX document,
    including positions for any 'dummy' nodes (with a 'dummy' attribute).
    """

    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    

    # Basic parameters
    depth = getattr(H, 'depth', 0)
    num_qubits = getattr(H, 'num_qubits', 0)
    num_qubits_phys = sum(qpu_sizes)  # from your code

    # Default scales if not specified
    if xscale is None:
        xscale = 10.0 / depth if depth else 1
    if yscale is None:
        yscale = 6.0 / num_qubits if num_qubits else 1

    # Build the position map for real (qubit,time) nodes
    space_map = space_mapping(qpu_sizes, depth)
    pos_list = get_pos_list(H, num_qubits, assignment, space_map)

    # If no nodes, handle gracefully
    if H.nodes:
        max_time = max(n[1] for n in H.nodes if isinstance(n, tuple) and len(n) == 2)
    else:
        max_time = 0

    # ------------------------------------------------------------
    # 1) If you want to invert the styles for a dark background
    # ------------------------------------------------------------
    if invert_colors:
        edge_color = "white"
        boundary_color = "white"
        white_small_style = r"circle, draw=white, fill=white, scale=0.3"
        black_style       = r"circle, draw=white, fill=black, scale=0.6"
        grey_style        = r"circle, draw=white, fill=gray!50, scale=0.6"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=white"
        dummy_style       = r"circle, draw=white, fill=blue!40, scale=2"  # DUMMY NODES
        background_fill   = "black"
    else:
        edge_color = "black"
        boundary_color = "black"
        white_small_style = r"circle, draw=black, fill=white, scale=0.3"
        black_style       = r"circle, draw=black, fill=black, scale=0.6"
        grey_style        = r"circle, draw=black, fill=gray, scale=0.6"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=none"
        dummy_style       = r"circle, draw=black, fill=blue!20, scale=2"  # DUMMY NODES
        background_fill   = "white"

    if fill_background:
        background_option = f"show background rectangle, background rectangle/.style={{fill={background_fill}}}"
    else:
        background_option = ""

    # ------------------------------------------------------------
    # 2) Function to pick a TikZ style for each node
    # ------------------------------------------------------------
    def pick_style(node):
        # Check if it's a dummy node via attribute
        if H.get_node_attribute(node, 'dummy', False):
            return "dummyStyle"

        # For real circuit nodes:
        q, t = None, None
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
        
        node_type = H.get_node_attribute(node, 'type', None)
        if node_type in ("group", "two-qubit", "root_t"):
            return "blackStyle"
        elif node_type == "single-qubit":
            return "greyStyle"
        else:
            return "invisibleStyle"

    # ------------------------------------------------------------
    # 3) Function to compute (x,y) for each node, including dummy nodes
    # ------------------------------------------------------------
    def pick_position(node):
        # If this node is marked as dummy, place it *above* the real qubits
        # if node[0] == "dummy":
        #     # For example, if node is ("dummy", p, pprime):
        #     # We can parse that tuple to spread them out or place them in a row
        if isinstance(node, tuple) and len(node) == 3 and node[0] == "dummy":
            _, p, pprime = node
            # Example: place them in a single row at y = num_qubits_phys+2
            # and x offset = partition p + some shift
            x = (depth/len(qpu_sizes) *(pprime-1)) * xscale * 1.2  # scale horizontally by pprime
            y = (-2) * yscale * 0.8
            return (x, y)
        # else:
        #     # If for some reason it's a dummy node of a different shape:
        #     return (0, (num_qubits_phys + 2) * yscale)

        # Otherwise, it's a real circuit node: (qubit, time)
        if isinstance(node, tuple) and len(node) == 2:

            if assignment_map is not None:
                q, t = assignment_map[node]
            else:
                q, t = node
            # These must exist in pos_list
            x = t * xscale


  

            y = (num_qubits_phys - pos_list[t][q]) * yscale
            return (x, y)

        # Fallback if unknown
        return (0, 0)

    # A helper to get unique node names in TikZ
    def node_name(n):
        # Flatten the tuple into a string
        return "n_" + "_".join(str(x) for x in n)

    # Begin building the full .tex code
    tikz_code = []
    tikz_code.append(r"\documentclass[tikz,border=2pt]{standalone}")
    tikz_code.append(r"\usepackage{tikz}")
    tikz_code.append(r"\usetikzlibrary{calc,backgrounds}")
    tikz_code.append(r"\pgfdeclarelayer{nodelayer}")
    tikz_code.append(r"\pgfdeclarelayer{edgelayer}")
    tikz_code.append(r"\pgfsetlayers{background,edgelayer,nodelayer,main}")
    tikz_code.append(r"\begin{document}")
    tikz_code.append(rf"\begin{{tikzpicture}}[>=latex, {background_option}]")

    # Define node styles
    tikz_code.append(fr"  \tikzstyle{{whiteSmallStyle}}=[{white_small_style}]")
    tikz_code.append(fr"  \tikzstyle{{blackStyle}}=[{black_style}]")
    tikz_code.append(fr"  \tikzstyle{{greyStyle}}=[{grey_style}]")
    tikz_code.append(fr"  \tikzstyle{{invisibleStyle}}=[{invisible_style}]")
    tikz_code.append(fr"  \tikzstyle{{dummyStyle}}=[{dummy_style}]")  # DUMMY NODES

    # Define an edge style
    tikz_code.append(rf"  \tikzset{{edgeStyle/.style={{draw={edge_color}}}}}")
    tikz_code.append(rf"  \tikzset{{boundaryLine/.style={{draw={boundary_color}, dashed}}}}")

    # --------------- NODES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in H.nodes:
        (x, y) = pick_position(n)
        style = pick_style(n)
        tikz_code.append(
            f"    \\node [style={style}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
        )
    tikz_code.append(r"  \end{pgfonlayer}")

    # --------------- EDGES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")

    for edge_id, edge_info in H.hyperedges.items():
        # The logic for drawing is unchanged,
        # except that if an edge connects a dummy node to a real node,
        # the code simply uses their coordinates from pick_position.
        if isinstance(edge_id[0], int):
            roots = edge_info['root_set']
            root_node = edge_id
            for root in roots:
                if root[0] != "dummy":
                    break
                else:
                    root_node = root
            
            root_t = root_node[1]

            if root_node[0] == "dummy":
                root_t, _ = pick_position(root_node)
            else:   
                root_t = edge_id[1]
                for rt in roots:
                    if isinstance(rt, tuple) and len(rt) == 2:
                        if rt[1] < root_t:
                            root_node = rt
                            root_t = rt[1]

            receivers = edge_info["receiver_set"]
            if len(receivers) > 1:
                edge_node_name = "edge_" + node_name(root_node)
                # We'll place an invisible node near the root_node
                # to fan out edges if there are multiple receivers
                rx, ry = pick_position(root_node)
                rx += 0.3 * xscale
                ry -= 0.3 * yscale
                tikz_code.append(
                    f"    \\node [style=invisibleStyle] ({edge_node_name}) at ({rx:.3f},{ry:.3f}) {{}};"
                )
                tikz_code.append(
                    f"    \\draw [style=edgeStyle] ({node_name(root_node)}) to ({edge_node_name});"
                )
            else:
                edge_node_name = node_name(root_node)

            for rnode in receivers:
                tikz_code.append(
                    f"    \\draw [style=edgeStyle, bend right=15] ({edge_node_name}) to ({node_name(rnode)});"
                )
            # If there's more than one 'root' in root_set, also connect them
            for rnode in roots:
                if rnode != root_node:
                    tikz_code.append(
                        f"    \\draw [style=edgeStyle, bend right=15] ({node_name(rnode)}) to ({edge_node_name});"
                    )

        else:
            # fallback for symbolic edge_id, same logic
            root_set = edge_info['root_set']
            rec_set = edge_info['receiver_set']
            if not root_set or not rec_set:
                continue
            node1 = list(root_set)[0]
            node2 = list(rec_set)[0]
            bend = "[style=edgeStyle, bend right=15]" if node1[0] != node2[0] else "[style=edgeStyle]"
            tikz_code.append(
                f"    \\draw {bend} ({node_name(node1)}) to ({node_name(node2)});"
            )
    tikz_code.append(r"  \end{pgfonlayer}")

    # --------------- BUFFER LAYER ---------------
    tikz_code.append(r"  %--------------- BUFFER LAYER ---------------")
    tikz_code.append(r"  % White boundary nodes at t=-1 and t=max_time+1 for each qubit.")
    buffer_left_time = -1
    buffer_right_time = max_time + 1

    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    inverse_assignment_map = {}

    for qubit in range(num_qubits):
        
        left_x = buffer_left_time * xscale
        left_y = (num_qubits_phys - pos_list[0][qubit]) * yscale
        left_node_name = f"bufL_{qubit}"
        tikz_code.append(
            f"    \\node [style=whiteSmallStyle] ({left_node_name}) "
            f"at ({left_x:.3f},{left_y:.3f}) {{}};"
        )

        right_x = buffer_right_time * xscale
        right_y = (num_qubits_phys - pos_list[max_time][qubit]) * yscale
        right_node_name = f"bufR_{qubit}"
        tikz_code.append(
            f"    \\node [style=whiteSmallStyle] ({right_node_name}) "
            f"at ({right_x:.3f},{right_y:.3f}) {{}};"
        )
    tikz_code.append(r"  \end{pgfonlayer}")
    if assignment_map is not None:
        for node, (q, t) in assignment_map.items():
            inverse_assignment_map[(q, t)] = node
        
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")
    for qubit in range(num_qubits):
        if assignment_map is not None:
            q, t = inverse_assignment_map[(qubit, 0)]
        else:
            q, t = qubit, 0
        if (q, 0) in H.nodes:
            tikz_code.append(
                f"    \\draw [style=edgeStyle] (bufL_{qubit}) to ({node_name((q,0))});"
            )
        if assignment_map is not None:
            q, max_time = inverse_assignment_map[(qubit, max_time)]
        else:
            q, max_time = qubit, max_time
        if (q, max_time) in H.nodes:
            tikz_code.append(
                f"    \\draw [style=edgeStyle] (bufR_{qubit}) to ({node_name((q,max_time))});"
            )
    tikz_code.append(r"  \end{pgfonlayer}")

    # --------------- PARTITION BOUNDARY LINES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")
    for i in range(1, len(qpu_sizes)):
        boundary = sum(qpu_sizes[:i])
        line_y = (num_qubits_phys - boundary + 0.5) * yscale
        left_x = -1.5 * xscale
        right_x = (max_time + 1.5) * xscale
        tikz_code.append(
            f"    \\draw[style=boundaryLine] ({left_x:.3f},{line_y:.3f}) -- ({right_x:.3f},{line_y:.3f});"
        )
    tikz_code.append(r"  \end{pgfonlayer}")

    tikz_code.append(r"\end{tikzpicture}")
    tikz_code.append(r"\end{document}")

    final_code = "\n".join(tikz_code)
    if save and path is not None:
        with open(path, "w") as f:
            f.write(final_code)

    return final_code

def draw_graph_tikz(H, assignment, qpu_info, invert_colors=False, fill_background=True, assignment_map=None):
    """
    Jupyter convenience function to compile & display the TikZ code inline.
    """
    tikz_code = hypergraph_to_tikz(
        H,
        assignment,
        qpu_info,
        save=False,
        invert_colors=invert_colors,
        fill_background=fill_background,
        assignment_map=assignment_map,
    )
    ip = get_ipython()
    args = "-f -r --dpi=150"
    return ip.run_cell_magic('tikz', args, tikz_code)