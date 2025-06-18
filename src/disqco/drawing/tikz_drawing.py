from __future__ import annotations
from disqco.drawing.map_positions import space_mapping, get_pos_list, get_pos_list_ext
from typing import Dict, Tuple, Iterable, Hashable, Union, List
from IPython import get_ipython
import numpy as np
from disqco.graphs.GCP_hypergraph_extended import HyperGraph
# 


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
    show_labels=True,
):
    """
    Convert a QuantumCircuitHyperGraph 'H' into a full standalone TikZ/LaTeX document,
    including positions for any 'dummy' nodes (with a 'dummy' attribute).
    
    Args:
        ...
        show_labels (bool): Whether to show node labels (default: True)
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

    # Calculate node scaling based on circuit size
    node_scale = min(0.6, max(0.3, 1.0 / (max(depth, num_qubits) ** 0.5)))
    small_node_scale = node_scale * 0.5
    gate_node_scale = node_scale * 1.2

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
        white_small_style = rf"circle, draw=white, fill=white, scale={small_node_scale}"
        black_style       = rf"circle, draw=white, fill=black, scale={node_scale}"
        white_style       = rf"circle, draw=white, fill=white, scale={node_scale}"
        grey_style        = rf"circle, draw=white, fill=gray!50, scale={node_scale}"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=white"
        dummy_style       = rf"circle, draw=white, fill=blue!40, scale={gate_node_scale}"  # DUMMY NODES
        background_fill   = "black"
    else:
        edge_color = "black"
        boundary_color = "black"
        white_small_style = rf"circle, draw=black, fill=white, scale={small_node_scale}"
        black_style       = rf"circle, draw=black, fill=black, scale={node_scale}"
        white_style       = rf"circle, draw=black, fill=white, scale={node_scale}"
        grey_style        = rf"circle, draw=black, fill=gray, scale={node_scale}"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=none"
        dummy_style       = rf"circle, draw=black, fill=blue!20, scale={gate_node_scale}"  # DUMMY NODES
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
            if H.node_attrs[node].get('name') == "target":
                return "whiteStyle"
            else:
                return "blackStyle"
        elif node_type == "single-qubit":
            # Check if it's an identity gate
            params = H.get_node_attribute(node, 'params', None)
            if params is not None and len(params) > 0:
                if abs(params[0]) < 1e-10:  # If parameter is effectively zero
                    return "invisibleStyle"
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
    tikz_code.append(fr"  \tikzstyle{{whiteStyle}}=[{white_style}]")
    tikz_code.append(fr"  \tikzstyle{{greyStyle}}=[{grey_style}]")
    tikz_code.append(fr"  \tikzstyle{{invisibleStyle}}=[{invisible_style}]")
    tikz_code.append(fr"  \tikzstyle{{dummyStyle}}=[{dummy_style}]")  # DUMMY NODES

    # Define an edge style
    tikz_code.append(rf"  \tikzset{{edgeStyle/.style={{draw={edge_color}}}}}")
    tikz_code.append(rf"  \tikzset{{boundaryLine/.style={{draw={boundary_color}, dashed}}}}")

    # Add this after the other style definitions
    tikz_code.append(r"  \tikzset{nodeLabel/.style={scale=0.5, inner sep=0pt}}")

    # --------------- NODES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in H.nodes:
        (x, y) = pick_position(n)
        style = pick_style(n)
        # Add label for each node
        if isinstance(n, tuple) and len(n) == 2:
            q, t = n
            if show_labels:
                label = f"$({q},{t})$"
                tikz_code.append(
                    f"    \\node [style={style}, label={{[nodeLabel]above right:{label}}}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
            else:
                tikz_code.append(
                    f"    \\node [style={style}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
        else:
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
                ry += 0.3 * yscale
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
            f"    \\node [style=whiteSmallStyle, label={{[nodeLabel]left:$q_{{{qubit}}}$}}] ({left_node_name}) "
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

def draw_graph_tikz(H, assignment, qpu_info, invert_colors=False, fill_background=True, assignment_map=None, show_labels=True,
        tikz_raw = False):
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
        show_labels=show_labels,
    )
    if tikz_raw:
        return tikz_code
    ip = get_ipython()
    args = "-f -r --dpi=150"
    return ip.run_cell_magic('tikz', args, tikz_code)


# ---------------------------------------------------------------------------
# hypergraph_drawing_v2.py  ––  TikZ exporter for the *new* bipartite
# HyperGraph, **with partition‑aware placement** (uses `assignment`).
# ---------------------------------------------------------------------------


# ────────────────────────────────────────────────────────────────────────────
# 1.  Basic helpers
# ────────────────────────────────────────────────────────────────────────────
def hypergraph_to_tikz_v2(
    H : HyperGraph,
    qubit_assignment,
    gate_assignment,
    qpu_info,
    depth,
    num_qubits,
    xscale=None,
    yscale=None,
    save=False,
    path=None,
    invert_colors=False,
    fill_background=True,
    assignment_map= None,
    show_labels=True,
):
    """
    Convert a QuantumCircuitHyperGraph 'H' into a full standalone TikZ/LaTeX document,
    including positions for any 'dummy' nodes (with a 'dummy' attribute).
    
    Args:
        ...
        show_labels (bool): Whether to show node labels (default: True)
    """

    if isinstance(qpu_info, dict):
        qpu_sizes = list(qpu_info.values())
    else:
        qpu_sizes = qpu_info
    
    # Basic parameters
    num_qubits_phys = sum(qpu_sizes)  # from your code

    # Default scales if not specified
    if xscale is None:
        xscale = 10.0 / depth if depth else 1
    if yscale is None:
        yscale = 6.0 / num_qubits if num_qubits else 1

    # Calculate node scaling based on circuit size
    node_scale = min(0.6, max(0.3, 1.0 / (max(depth, num_qubits) ** 0.5)))
    small_node_scale = node_scale * 0.5
    gate_node_scale = node_scale * 1.2


    # Build the position map for real (qubit,time) nodes
    space_map = space_mapping(qpu_sizes, depth)
    pos_list = get_pos_list_ext(H, num_qubits, qubit_assignment.transpose(), space_map, qpu_info)

    # If no nodes, handle gracefully
    if H.nodes():
        max_time = max(n[1] for n in H.nodes() if isinstance(n, tuple) and len(n) == 2)
    else:
        max_time = 0

    # ------------------------------------------------------------
    # 1) If you want to invert the styles for a dark background
    # ------------------------------------------------------------
    if invert_colors:
        edge_color = "white"
        boundary_color = "white"
        white_small_style = rf"circle, draw=white, fill=white, scale={small_node_scale}"
        black_style       = rf"circle, draw=white, fill=black, scale={node_scale}"
        white_style       = rf"circle, draw=white, fill=white, scale={node_scale}"
        grey_style        = rf"circle, draw=white, fill=gray!50, scale={node_scale}"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=white"
        dummy_style       = rf"circle, draw=white, fill=blue!40, scale={gate_node_scale}"  # DUMMY NODES
        background_fill   = "black"
    else:
        edge_color = "black"
        boundary_color = "black"
        white_small_style = rf"circle, draw=black, fill=white, scale={small_node_scale}"
        black_style       = rf"circle, draw=black, fill=black, scale={node_scale}"
        white_style       = rf"circle, draw=black, fill=white, scale={node_scale}"
        grey_style        = rf"circle, draw=black, fill=gray, scale={node_scale}"
        invisible_style   = r"inner sep=0pt, scale=0.1, draw=none"
        dummy_style       = rf"circle, draw=black, fill=blue!20, scale={gate_node_scale}"  # DUMMY NODES
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
        # if H.get_node_attribute(node, 'dummy', False):
        #     return "dummyStyle"

        # For real circuit nodes:
        q, t = None, None
        if isinstance(node, tuple) and len(node) == 2:
            q, t = node
            # Check if it's a single-qubit gate
            if node in H._G.nodes:
                params = H._G.nodes[node].get('params', None)
                if params is not None and len(params) > 0:
                    if abs(params[0]) < 1e-10:  # If parameter is effectively zero
                        return "invisibleStyle"
                    return "greyStyle"
            return "blackStyle"
        elif isinstance(node, tuple) and len(node) == 3 and node[0] != "dummy":
            return "whiteStyle"
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
        
        if isinstance(node, tuple) and len(node) == 3 and node[0] != "dummy":
            # This is a gate node
            x = (node[-1]) * xscale
            # Define height as half way between qubit from node[0] and node[1]

            partner1 = (node[0], node[2])
            partner2 = (node[1], node[2])

            p1 = qubit_assignment[partner1]
            p2 = qubit_assignment[partner2]

            p_node = gate_assignment[node]

            if p_node == p1 and p_node != p2:
                if p2 > p1:
                    buffer = -0.45
                else:
                    buffer = 0.45
                y = (num_qubits_phys - pos_list[partner1[1]][partner1[0]] + buffer) * yscale
            elif p_node == p2 and p_node != p1:
                if p1 > p2:
                    buffer = -0.45
                else:
                    buffer = 0.45
                y = (num_qubits_phys - pos_list[partner2[1]][partner2[0]] + buffer) * yscale
            elif p_node == p1 and p_node == p2:

                y = (num_qubits_phys - (pos_list[partner1[1]][partner1[0]] + pos_list[partner2[1]][partner2[0]])/2) * yscale
            else:

                # Calculate y-offset based on partition number and sizes
                y_offset = 0
                for p in range(p_node):
                    y_offset += qpu_sizes[p]
                # Place node at midpoint of its partition's region
                y = (num_qubits_phys - (y_offset + qpu_sizes[p_node]/2)) * yscale
            return (x, y)

        # Fallback if unknown
        return (0, 0)

    # A helper to get unique node names in TikZ
    def node_name(n):
        # Flatten the tuple into a string
        return "n_" + "_".join(str(x) for x in n)
    
    def is_state_edge(edge):
        # Check if the edge is a state edge
        nodes = list(edge.vertices)
        if len(nodes) == 2:
            if nodes[0][0] == nodes[1][0]:
                return True
    
    def is_gate_edge(edge):
        # Check if the edge is a hyperedge
        for node in edge.vertices:
            if isinstance(node, tuple) and len(node) == 3:
                return True
        return False
    gate_node_positions = {}
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
    tikz_code.append(fr"  \tikzstyle{{whiteStyle}}=[{white_style}]")

    # Define an edge style
    tikz_code.append(rf"  \tikzset{{edgeStyle/.style={{draw={edge_color}}}}}")
    tikz_code.append(rf"  \tikzset{{boundaryLine/.style={{draw={boundary_color}, dashed}}}}")

    # Add this after the other style definitions
    tikz_code.append(r"  \tikzset{nodeLabel/.style={scale=0.5, inner sep=0pt}}")

    # --------------- NODES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")
    for n in H.nodes():
        (x, y) = pick_position(n)
        style = pick_style(n)
        # Check if this node is part of any gate hyperedge by iterating through hyperedges 
        # node is involved in using the incident method 
        # is_gate_hyperedge = False
        # for edge_key in H.incident(n):
        #     edge = H._hyperedges[edge_key]
        #     if is_gate_edge(edge):

        #         is_gate_hyperedge = True
        #         break
        # if is_gate_hyperedge:
        #     style = "invisibleStyle"

        if isinstance(n, tuple) and len(n) == 3:
            gate_node_positions[n] = (x,y)
        # Add label for each node
        if isinstance(n, tuple) and len(n) == 2:
            q, t = n
            if show_labels:
                label = f"$({q},{t})$"
                tikz_code.append(
                    f"    \\node [style={style}, label={{[nodeLabel]above right:{label}}}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
            else:
                tikz_code.append(
                    f"    \\node [style={style}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
        else:
            if show_labels:
                theta = np.round(H._G.nodes[n]['params'][0], 2)
                tikz_code.append(
                    f"    \\node [style={style}, label={{[nodeLabel]above right:{f'{theta}'}}}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
            else:
                tikz_code.append(
                    f"    \\node [style={style}] ({node_name(n)}) at ({x:.3f},{y:.3f}) {{}};"
                )
        

    tikz_code.append(r"  \end{pgfonlayer}")

    # --------------- EDGES ---------------
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")

    for edge in H.hyperedges():
        if is_state_edge(edge):
            nodes = list(edge.vertices)
            node1 = nodes[0]
            node2 = nodes[1]
            # bend = "[style=edgeStyle, bend right=15]" if node1[0] != node2[0] else "[style=edgeStyle]"
            bend = "[style=edgeStyle]"
            tikz_code.append(
                f"    \\draw {bend} ({node_name(node1)}) to ({node_name(node2)});"
            )

        elif is_gate_edge(edge):
            nodes = list(edge.vertices)
            max_time = 0
            min_time = np.inf
            
            # Track gate nodes and their positions
            gate_nodes = []
            start_node = None
            start_y = None
            end_node = None
            
            for node in nodes:
                if isinstance(node, tuple) and len(node) == 2:
                    t = node[-1]
                    if t > max_time:
                        max_time = t
                        end_node = node

                    if t < min_time:
                        min_time = t
                        root_q = node[0]
                        start_node = node
                        # Get y-coordinate of start node
                        start_y = (num_qubits_phys - pos_list[t][root_q]) * yscale
                
                elif isinstance(node, tuple) and len(node) == 3:

                    node_assignment = gate_assignment[node]
                    if node_assignment == 0:
                        y_buffer = 0
                    else:
                        y_buffer = sum(qpu_sizes[:node_assignment])
                    
                    x = node[-1] * xscale
                    y = gate_node_positions[node][1]
                    gate_nodes.append((node, y))

            # Calculate central vertex position
            central_vertex_x = (min_time + (max_time - min_time)/2) * xscale
            
            # First calculate the average y-position of all gate nodes
            if gate_nodes:
                avg_gate_y = sum(y for _, y in gate_nodes) / len(gate_nodes)
                # If average gate position is below start node, place central vertex below
                if avg_gate_y > start_y:
                    central_vertex_y = start_y + 0.25 * yscale
                else:
                    central_vertex_y = start_y - 0.25 * yscale
            else:
                # Default position if no gate nodes
                central_vertex_y = start_y - 0.25 * yscale

            if end_node is None:
                end_node = start_node

            tikz_code.append(
                f"    \\node [style=invisibleStyle] (central_vertex_{root_q}_{min_time}_{max_time}) at ({central_vertex_x:.3f},{central_vertex_y:.3f}) {{}};"
            )
            tikz_code.append(
                f"    \\draw [style=edgeStyle, bend left=15] ({node_name(start_node)}) to (central_vertex_{root_q}_{min_time}_{max_time});"
            )
            tikz_code.append(
                f"    \\draw [style=edgeStyle, bend right=15] (central_vertex_{root_q}_{min_time}_{max_time}) to ({node_name(end_node)});"
            )

            for node in nodes:
                if len(node) == 3:
                    tikz_code.append(
                        f"    \\draw [style=edgeStyle, bend right= 15] ({node_name(node)}) to (central_vertex_{root_q}_{min_time}_{max_time});"
                    )



    tikz_code.append(r"  \end{pgfonlayer}")

    # --------------- BUFFER LAYER ---------------
    tikz_code.append(r"  %--------------- BUFFER LAYER ---------------")
    tikz_code.append(r"  % White boundary nodes at t=-1 and t=max_time+1 for each qubit.")
    buffer_left_time = -1
    buffer_right_time = depth

    tikz_code.append(r"  \begin{pgfonlayer}{nodelayer}")

    for qubit in range(num_qubits):
        left_x = buffer_left_time * xscale
        left_y = (num_qubits_phys - pos_list[0][qubit]) * yscale
        left_node_name = f"bufL_{qubit}"
        tikz_code.append(
            f"    \\node [style=whiteSmallStyle, label={{[nodeLabel]left:$q_{{{qubit}}}$}}] ({left_node_name}) "
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

        
    tikz_code.append(r"  \begin{pgfonlayer}{edgelayer}")
    for qubit in range(num_qubits):
        q, t = qubit, 0
        if (q, 0) in H.nodes():
            tikz_code.append(
                f"    \\draw [style=edgeStyle] (bufL_{qubit}) to ({node_name((q,0))});"
            )
        q, max_time = qubit, depth-1
        if (q, max_time) in H.nodes():
            tikz_code.append(
                f"    \\draw [style=edgeStyle] (bufR_{qubit}) to ({node_name((q,depth-1))});"
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


# ────────────────────────────────────────────────────────────────────────────
# 3.  Notebook convenience wrapper
# ────────────────────────────────────────────────────────────────────────────

def draw_graph_tikz_v2(
    H: "HyperGraph",
    qubit_assignment,
    gate_assignment,
    qpu_info: Union[Iterable[int], Dict[str, int]],
    depth: int,
    num_qubits: int,
    show_labels: bool = True,
    **kwargs,
):
    """Compile & render the TikZ code inline in a Jupyter notebook."""
    code = hypergraph_to_tikz_v2(H, qubit_assignment, gate_assignment, qpu_info, depth=depth, num_qubits=num_qubits, show_labels=show_labels, **kwargs)
    ip = get_ipython()
    return ip.run_cell_magic("tikz", "-f -r --dpi=150", code)


            



    #     # The logic for drawing is unchanged,
    #     # except that if an edge connects a dummy node to a real node,
    #     # the code simply uses their coordinates from pick_position.
    #     if isinstance(edge_id[0], int):
    #         roots = edge_info['root_set']
    #         root_node = edge_id
    #         for root in roots:
    #             if root[0] != "dummy":
    #                 break
    #             else:
    #                 root_node = root
            
    #         root_t = root_node[1]

    #         if root_node[0] == "dummy":
    #             root_t, _ = pick_position(root_node)
    #         else:   
    #             root_t = edge_id[1]
    #             for rt in roots:
    #                 if isinstance(rt, tuple) and len(rt) == 2:
    #                     if rt[1] < root_t:
    #                         root_node = rt
    #                         root_t = rt[1]

    #         receivers = edge_info["receiver_set"]
    #         if len(receivers) > 1:
    #             edge_node_name = "edge_" + node_name(root_node)
    #             # We'll place an invisible node near the root_node
    #             # to fan out edges if there are multiple receivers
    #             rx, ry = pick_position(root_node)
    #             rx += 0.3 * xscale
    #             ry -= 0.3 * yscale
    #             tikz_code.append(
    #                 f"    \\node [style=invisibleStyle] ({edge_node_name}) at ({rx:.3f},{ry:.3f}) {{}};"
    #             )
    #             tikz_code.append(
    #                 f"    \\draw [style=edgeStyle] ({node_name(root_node)}) to ({edge_node_name});"
    #             )
    #         else:
    #             edge_node_name = node_name(root_node)

    #         for rnode in receivers:
    #             tikz_code.append(
    #                 f"    \\draw [style=edgeStyle, bend right=15] ({edge_node_name}) to ({node_name(rnode)});"
    #             )
    #         # If there's more than one 'root' in root_set, also connect them
    #         for rnode in roots:
    #             if rnode != root_node:
    #                 tikz_code.append(
    #                     f"    \\draw [style=edgeStyle, bend right=15] ({node_name(rnode)}) to ({edge_node_name});"
    #                 )

    #     else:
    #         # fallback for symbolic edge_id, same logic
    #         root_set = edge_info['root_set']
    #         rec_set = edge_info['receiver_set']
    #         if not root_set or not rec_set:
    #             continue
    #         node1 = list(root_set)[0]
    #         node2 = list(rec_set)[0]
    #         bend = "[style=edgeStyle, bend right=15]" if node1[0] != node2[0] else "[style=edgeStyle]"
    #         tikz_code.append(
    #             f"    \\draw {bend} ({node_name(node1)}) to ({node_name(node2)});"
    #         )
    # tikz_code.append(r"  \end{pgfonlayer}")


