from IPython import get_ipython
from disqco.drawing.map_positions import space_mapping, get_pos_list

def hypergraph_to_tikz(
    H,
    assignment,
    qpu_info,
    xscale=None,
    yscale=None,
    save=False,
    path=None
):
    """
    Convert a QuantumCircuitHyperGraph 'H' into a full standalone TikZ/LaTeX document.

    - All regular nodes are drawn according to their 'type' attribute.
    - Nodes at t=0 or t=max_time keep their normal style (whiteSmallStyle if no type, or black/grey, etc.).
    - Then we add a *buffer layer* of white nodes at t=-1 and t=max_time+1 for each qubit,
      connecting them (straight line) to the actual boundary node if it exists.

    Returns a string containing the entire LaTeX document.
    """

    # Basic parameters from H
    depth = getattr(H, 'depth', 0)
    num_qubits = getattr(H, 'num_qubits', 0)
    num_qubits_phys = sum(qpu_info)  # from your code

    # Default scales if not specified
    if xscale is None:
        xscale = 10.0 / depth if depth else 1
    if yscale is None:
        yscale = 6.0 / num_qubits if num_qubits else 1

    # Build the position map
    space_map = space_mapping(qpu_info, depth)
    pos_list = get_pos_list(num_qubits, assignment, space_map)

    # If no nodes, handle gracefully
    if H.nodes:
        max_time = max(n[1] for n in H.nodes)
    else:
        max_time = 0

    # --- Function to pick node style
    def pick_style(node):
        q, t = node
        node_type = H.get_node_attribute(node, 'type', None)

        # If t=0 or t=max_time *and* no special type, let them be whiteSmallStyle
        # (But if they do have 'two-qubit','single-qubit','group','root_t', we might override anyway.)
        if t == 0 or t == max_time:
            # If the user wants truly all boundary nodes white, uncomment the next line:
            # return "whiteSmallStyle"
            # Otherwise, do the same logic as normal:
            pass

        if node_type in ("group", "two-qubit", "root_t"):
            return "blackStyle"
        elif node_type == "single-qubit":
            return "greyStyle"
        else:
            # default if not recognized or boundary
            return "invisibleStyle"

    # A helper to get unique node names in TikZ
    def node_name(n):
        # n is (q, t)
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
    tikz_code.append(r"\begin{tikzpicture}[>=latex]")

    # -- Define all styles inline here:
    tikz_code.append(r"  \tikzstyle{whiteSmallStyle}=[circle, draw=black, fill=white, scale=0.3]")
    tikz_code.append(r"  \tikzstyle{blackStyle}=[circle, draw=black, fill=black, scale=0.6]")
    tikz_code.append(r"  \tikzstyle{greyStyle}=[circle, draw=black, fill=gray,  scale=0.6]")
    tikz_code.append(r"  \tikzstyle{invisibleStyle}=[inner sep=0pt, scale=0.1, draw=none]")

    tikz_code.append(r"  %--------------- NODES ---------------")
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")

    # Draw normal nodes
    for n in H.nodes:
        q, t = n
        style = pick_style(n)
        # position = (time, vertical placement)
        x = t * xscale
        y = (num_qubits_phys - pos_list[t][q]) * yscale
        tikz_code.append(
            f"    \\node [style={style}] ({node_name(n)}) "
            f"at ({x:.3f},{y:.3f}) {{}};"
        )
    tikz_code.append(r"  \end{pgfonlayer}")

    tikz_code.append(r"  %--------------- EDGES ---------------")
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")


    # Draw hyperedges
    for edge_id, edge_info in H.hyperedges.items():
        if isinstance(edge_id[0], int):
            # If edge_id is a (q,t) tuple
            roots = edge_info['root_set']
            root_node = edge_id
            root_t = edge_id[1]

            # pick earliest root by time
            for node in roots:
                if node[1] < root_t:
                    root_node = node
                    root_t = node[1]

            receivers = edge_info["receiver_set"]

            # If multiple receivers, place an invisible "edge" node
            if len(receivers) > 1:
                edge_node_name = "edge_" + node_name(root_node)
                rx = (root_node[1] + 0.3) * xscale
                ry = (num_qubits_phys - pos_list[root_node[1]][root_node[0]] - 0.3) * yscale
                tikz_code.append(
                    f"    \\node [style=invisibleStyle] ({edge_node_name}) at ({rx:.3f},{ry:.3f}) {{}};"
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
                        f"    \\draw [bend right=15] ({node_name(rnode)}) to ({edge_node_name});"
                    )

        else:
            # fallback for symbolic edge_id
            root_set = edge_info['root_set']
            rec_set = edge_info['receiver_set']
            if not root_set or not rec_set:
                continue
            node1 = list(root_set)[0]
            node2 = list(rec_set)[0]
            bend = "[bend right=15]" if node1[0] != node2[0] else ""
            tikz_code.append(
                f"    \\draw {bend} ({node_name(node1)}) to ({node_name(node2)});"
            )

    tikz_code.append(r"  \end{pgfonlayer}")

    # ==================================
    # ADD THE EXTRA BUFFER LAYER NODES
    # ==================================
    tikz_code.append(r"  %--------------- BUFFER LAYER ---------------")
    tikz_code.append(r"  % We place white nodes at t=-1 and t=max_time+1 for each qubit.")
    tikz_code.append(r"  % Then connect them to the actual boundary node if it exists.")
    
    buffer_left_time = -1
    buffer_right_time = max_time + 1

    # We'll put these buffer nodes in the same 'nodelayer' for clarity
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for qubit in range(num_qubits):
        # left buffer node
        left_x = buffer_left_time * xscale
        # use the same vertical offset as time=0 for that qubit
        left_y = (num_qubits_phys - pos_list[0][qubit]) * yscale
        left_node_name = f"bufL_{qubit}"

        tikz_code.append(
            f"    \\node [style=whiteSmallStyle] ({left_node_name}) "
            f"at ({left_x:.3f},{left_y:.3f}) {{}};"
        )

        # right buffer node
        right_x = buffer_right_time * xscale
        # use vertical offset as time=max_time for that qubit
        right_y = (num_qubits_phys - pos_list[max_time][qubit]) * yscale
        right_node_name = f"bufR_{qubit}"

        tikz_code.append(
            f"    \\node [style=whiteSmallStyle] ({right_node_name}) "
            f"at ({right_x:.3f},{right_y:.3f}) {{}};"
        )

    tikz_code.append(r"  \end{pgfonlayer}")

    # Now connect these buffer nodes in the edgelayer
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")
    for qubit in range(num_qubits):
        # If there's a real node at (qubit, 0), connect it to buffer-left
        if (qubit, 0) in H.nodes:
            tikz_code.append(
                f"    \\draw (bufL_{qubit}) to ({node_name((qubit,0))});"
            )
        # If there's a real node at (qubit, max_time), connect it to buffer-right
        if (qubit, max_time) in H.nodes:
            tikz_code.append(
                f"    \\draw (bufR_{qubit}) to ({node_name((qubit,max_time))});"
            )
    

    tikz_code.append(r"  \end{pgfonlayer}")

    tikz_code.append(r"  %===== Partition Boundary Lines =====")
    tikz_code.append(r"  \begin{pgfonlayer}{background}")
    # For each boundary i between partition i and i+1,
    # sum up the qubits in partitions up to i
    for i in range(1, len(qpu_info)):
        boundary = sum(qpu_info[:i])  # number of qubits in partitions [0..i-1]
        # vertical position = (num_qubits_phys - boundary)*yscale
        line_y = (num_qubits_phys - boundary + 0.5) * yscale
        # We'll draw from time ~ -0.5 to time ~ max_time+0.5
        left_x = -1.5 * xscale
        right_x = (max_time + 1.5) * xscale

        tikz_code.append(
            f"    \\draw[dashed] ({left_x:.3f},{line_y:.3f}) -- ({right_x:.3f},{line_y:.3f});"
        )
    tikz_code.append(r"  \end{pgfonlayer}")


    # End the picture + document
    tikz_code.append(r"\end{tikzpicture}")
    tikz_code.append(r"\end{document}")

    # Optionally write to disk
    final_code = "\n".join(tikz_code)
    if save and path is not None:
        with open(path, "w") as f:
            f.write(final_code)

    return final_code

def hypergraph_to_tikz_snippet(H, num_qubits, assignment, qpu_info, depth, num_qubits_phys,
                               xscale=None, yscale=None):
    """
    Return ONLY the \begin{tikzpicture}...\end{tikzpicture} code,
    including inline style definitions (via \tikzstyle{} or \tikzset{}).
    Suitable for jupyter-tikz or %tikz magic.
    """

    if xscale is None:
        xscale = 10/depth if depth else 1
    if yscale is None:
        yscale = 6/num_qubits if num_qubits else 1

    space_map = space_mapping(qpu_info, depth)
    pos_list = get_pos_list(num_qubits, assignment, space_map)

    if H.nodes:
        max_time = max(n[1] for n in H.nodes)
    else:
        max_time = 0

    def pick_style(node):
        q, t = node
        if t == 0 or t == max_time:
            return "whiteSmallStyle"
        node_type = H.get_node_attribute(node, 'type', None)
        if node_type in ("group", "two-qubit", "root_t"):
            return "blackStyle"
        elif node_type == "single-qubit":
            return "greyStyle"
        else:
            return "invisibleStyle"

    def node_name(n):
        return "n_" + "_".join(str(x) for x in n)

    # Build the snippet
    tikz_lines = []
    tikz_lines.append(r"\begin{tikzpicture}[>=latex]")
    
    # Define your styles inline here
    tikz_lines.append(r"  \tikzstyle{whiteSmallStyle}=[circle, draw=black, fill=white, scale=0.6]")
    tikz_lines.append(r"  \tikzstyle{blackStyle}=[circle, draw=black, fill=black, scale=0.6]")
    tikz_lines.append(r"  \tikzstyle{greyStyle}=[circle, draw=black, fill=gray, scale=0.6]")
    tikz_lines.append(r"  \tikzstyle{invisibleStyle}=[inner sep=0pt, scale=0.6, draw=none]")

    tikz_lines.append(r"  \pgfdeclarelayer{nodelayer}")
    tikz_lines.append(r"  \pgfdeclarelayer{edgelayer}")
    tikz_lines.append(r"  \pgfsetlayers{edgelayer,nodelayer,main}") 
    # The layering might be optional if you aren't doing advanced layering

    # Node layer
    tikz_lines.append(r"  \begin{pgfonlayer}{nodelayer}")
    for node in H.nodes:
        style = pick_style(node)
        q, t = node
        x = t * xscale
        y = (num_qubits_phys - pos_list[t][q]) * yscale
        tikz_lines.append(
            f"    \\node [style={style}] ({node_name(node)}) "
            f"at ({x:.2f},{y:.2f}) {{}};"
        )
    tikz_lines.append(r"  \end{pgfonlayer}")

    # Edge layer
    tikz_lines.append(r"  \begin{pgfonlayer}{edgelayer}")
    for edge_id, edge_info in H.hyperedges.items():
        if isinstance(edge_id[0], int):
            # your root_node logic
            roots = edge_info['root_set']
            root_node = edge_id
            root_t = edge_id[1]

            # find earliest root
            for rt in roots:
                if rt[1] < root_t:
                    root_node = rt
                    root_t = rt[1]

            receivers = edge_info["receiver_set"]
            if len(receivers) > 1:
                edge_node_name = "edge_" + node_name(root_node)
                offset_x = (root_node[1] + 0.3) * xscale
                offset_y = (num_qubits_phys - pos_list[root_node[1]][root_node[0]] - 0.3) * yscale
                tikz_lines.append(
                    f"    \\node [style=invisibleStyle] ({edge_node_name}) "
                    f"at ({offset_x:.2f},{offset_y:.2f}) {{}};"
                )
                tikz_lines.append(
                    f"    \\draw ({node_name(root_node)}) to ({edge_node_name});"
                )
            else:
                edge_node_name = node_name(root_node)

            for rnode in receivers:
                tikz_lines.append(
                    f"    \\draw [bend right=15] ({edge_node_name}) to ({node_name(rnode)});"
                )
            for rt in roots:
                if rt != root_node:
                    tikz_lines.append(
                        f"    \\draw [bend right=15] ({node_name(rt)}) to ({edge_node_name});"
                    )
        else:
            # fallback for symbolic edge_id
            root_set = edge_info['root_set']
            rec_set = edge_info['receiver_set']
            if not root_set or not rec_set:
                continue
            r1 = list(root_set)[0]
            r2 = list(rec_set)[0]
            bend = "[bend right=15]" if r1[0] != r2[0] else ""
            tikz_lines.append(
                f"    \\draw {bend} ({node_name(r1)}) to ({node_name(r2)});"
            )
    tikz_lines.append(r"  \end{pgfonlayer}")

    tikz_lines.append(r"\end{tikzpicture}")

    code = "\n".join(tikz_lines)
    return code


def draw_graph_tikz(H, assignment, qpu_info):
    tikz_code = hypergraph_to_tikz(
        H,
        assignment,
        qpu_info,
        save=False
    )
    ip = get_ipython()
    args = "-f -r --dpi=150" 
    return ip.run_cell_magic('tikz', args, tikz_code)

