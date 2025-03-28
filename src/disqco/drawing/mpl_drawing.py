
import matplotlib.pyplot as plt
from disqco.drawing.map_positions import space_mapping, get_pos_list

def hypergraph_to_matplotlib(
    H,
    assignment,
    qpu_info,
    xscale = None,
    yscale = None,
    figsize= (10, 6),
    save= False,
    path= None,
    ax= None
):
    """
    Draw a QuantumCircuitHyperGraph 'H' with Matplotlib, placing a blank
    horizontal row between each partition. The *logical* partition of a node
    is taken from assignment[t][q], so nodes can move between partitions over time.

    qpu_info = [p0_size, p1_size, ... ] says how many "vertical slots" each partition has.
    We'll stack partitions vertically in order: partition 0 at top, then a blank row,
    partition 1, another blank row, etc.
    """
    num_qubits = H.num_qubits
    depth = H.depth
    if xscale is None:
        xscale = 10/depth
    if yscale is None:
        yscale = 6/(sum(qpu_info) + len(qpu_info))



    space_map_ = space_mapping(qpu_info, depth)
    pos_list = get_pos_list(num_qubits, assignment, space_map_)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if H.nodes:
        max_time = max(n[1] for n in H.nodes)
    else:
        max_time = 0

    def pick_style(node):
        """Return (facecolor, edgecolor, marker, size) or None if invisible."""
        q, t = node
        node_type = H.get_node_attribute(node, 'type', None)

        # default style
        facecolor = 'white'
        edgecolor = 'black'
        marker = 'o'
        size = 30

        if node_type in ["group", "two-qubit"]:
            facecolor = "black"
            edgecolor = "black"
            marker = "o"
            size = 30
        elif node_type == "root_t":
            facecolor = "black"
            edgecolor = "black"
            marker = "o"
            size = 30
        elif node_type == "single-qubit":
            facecolor = "gray"
            edgecolor = "black"
            marker = "o"
            size = 30
        else:
            return None
        return facecolor, edgecolor, marker, size

    node_positions = {}
    for n in H.nodes:
        q, t = n
        p = assignment[n]

        local_index = pos_list[t][q]

        base_y = local_index
        y = base_y * yscale

        x = t * xscale

        node_positions[n] = (x, y)

        style = pick_style(n)
        if style is not None:
            facecolor, edgecolor, marker, size = style
            ax.scatter(
                [x],
                [y],
                c=facecolor,
                edgecolors=edgecolor,
                marker=marker,
                s=size,
                zorder=3
            )

    for edge_id, edge_info in H.hyperedges.items():
        if isinstance(edge_id, tuple) and isinstance(edge_id[1], int):
            roots = edge_info.get("root_set", [])
            root_node = edge_id
            root_t = edge_id[1]
            for rnode in roots:
                if isinstance(root_t, int):
                    min_t = root_t
                else:
                    min_t = root_t[1]
                if rnode[1] < min_t:
                    root_node = rnode
                    root_t = rnode[1]

            receivers = edge_info.get("receiver_set", [])
            if root_node in node_positions:
                rx, ry = node_positions[root_node]
            else:
                continue

            if len(receivers) > 1:
                offset_x = rx + 0.3 * xscale
                offset_y = ry - 0.3 * yscale
                ax.plot([rx, offset_x], [ry, offset_y], color="black", zorder=2)
                for rnode in receivers:
                    if rnode in node_positions:
                        rxr, ryr = node_positions[rnode]
                        ax.plot([offset_x, rxr], [offset_y, ryr], color="black", zorder=2)
                roots = edge_info.get("root_set", [])
                for rnode in roots:
                    if rnode in node_positions:
                        rxr, ryr = node_positions[rnode]
                        ax.plot([offset_x, rxr], [offset_y, ryr], color="black", zorder=2)
            elif len(receivers) == 1:
                rnode = list(receivers)[0]
                if rnode in node_positions:
                    rxr, ryr = node_positions[rnode]
                    ax.plot([rx, rxr], [ry, ryr], color="black", zorder=2)
        else:
            root_set = edge_info.get("root_set", [])
            rec_set = edge_info.get("receiver_set", [])
            if not root_set or not rec_set:
                continue
            node1 = list(root_set)[0]
            node2 = list(rec_set)[0]
            if node1 in node_positions and node2 in node_positions:
                x1, y1 = node_positions[node1]
                x2, y2 = node_positions[node2]
                ax.plot([x1, x2], [y1, y2], color="black", zorder=2)

    for n in H.nodes:
        q, t = n
        if n not in node_positions:
            continue
        rx, ry = node_positions[n]

        if t == 0:
            ghost_t = -1
            ghost_x = ghost_t * xscale
            ghost_y = ry
            ax.scatter(
                [ghost_x],
                [ghost_y],
                c="white",
                edgecolors="black",
                marker="o",
                s=20,
                zorder=4
            )
            ax.plot([ghost_x, rx], [ghost_y, ry], color="black", zorder=2)
        elif t == max_time:
            ghost_t = max_time + 1
            ghost_x = ghost_t * xscale
            ghost_y = ry
            ax.scatter(
                [ghost_x],
                [ghost_y],
                c="white",
                edgecolors="black",
                marker="o",
                s=20,
                zorder=4
            )
            ax.plot([rx, ghost_x], [ry, ghost_y], color="black", zorder=2)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Time Layer (scaled by xscale)")
    ax.set_ylabel("Vertical index within assigned partition")
    plt.axis("off")

    if save and path:
        plt.savefig(path, bbox_inches="tight")

    return fig, ax

def draw_graph_mpl(H, assignment, qpu_info):
    fig, ax = hypergraph_to_matplotlib(
        H,
        assignment,
        qpu_info,
        figsize=(10, 6),
        save=False,
        path=None,
        ax=None
    )