# # import copy
# # import matplotlib.pyplot as plt

# # def space_mapping(qpu_info, num_layers):
# #     qpu_mapping = {}
# #     qubit_index = 0
# #     for j, qpu_size in enumerate(qpu_info):
# #         qubit_list = []
# #         for _ in range(qpu_size):
# #             qubit_list.append(qubit_index)
# #             qubit_index += 1
# #         qpu_mapping[j] = qubit_list
# #     space_map = []
# #     for t in range(num_layers):
# #         space_map.append(copy.deepcopy(qpu_mapping))
# #     return space_map

# # def get_pos_list(num_qubits, assignment, space_map):
# #     num_layers = len(assignment)
# #     pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]
# #     for q in range(num_qubits):
# #         old_partition = None
# #         for t in range(num_layers):
# #             partition = assignment[t][q]
# #             if old_partition is not None:
# #                 if partition == old_partition:
# #                     if x_index in space_map[t][partition]:
# #                         x_index = pos_list[t-1][q]
# #                         pos_list[t][q] = x_index
# #                         space_map[t][partition].remove(x_index)
# #                     else:
# #                         qubit_list = space_map[t][partition]
# #                         x_index = qubit_list.pop(0)
# #                         pos_list[t][q] = x_index
# #                 else:
# #                     qubit_list = space_map[t][partition]
# #                     x_index = qubit_list.pop(0)
# #                     pos_list[t][q] = x_index
# #             else:
# #                 qubit_list = space_map[t][partition]
# #                 x_index = qubit_list.pop(0)
# #                 pos_list[t][q] = x_index
# #             old_partition = partition
# #     return pos_list


# # def hypergraph_to_matplotlib(
# #     H,
# #     num_qubits,
# #     assignment,
# #     qpu_info,
# #     depth,
# #     num_qubits_phys,
# #     xscale=1.0,
# #     yscale=1.0,
# #     figsize=(10, 6),
# #     save=False,
# #     path=None,
# #     ax=None
# # ):
# #     """
# #     Draw a QuantumCircuitHyperGraph 'H' with Matplotlib.

# #     1) All nodes (q,t) are styled based on their 'type' attribute.
# #        - If type='two-qubit' or 'group' => black node
# #        - If type='single-qubit' => grey node
# #        - Otherwise => invisible (skip drawing)
# #     2) For nodes at t=0 or t=max_time, we do NOT force them to be small/white:
# #        they use the normal style above.
# #     3) Then we add an extra "ghost" node at t=0 or t=max_time with small
# #        white style, and connect it to the real node.
# #     """

# #     # 1) Compute positions
# #     space_map_ = space_mapping(qpu_info, depth)
# #     pos_list = get_pos_list(num_qubits, assignment, space_map_)

# #     # If user didn't supply an Axes, create one
# #     if ax is None:
# #         fig, ax = plt.subplots(figsize=(10, 6))
# #     else:
# #         fig = ax.figure

# #     # Determine max_time
# #     if H.nodes:
# #         max_time = max(n[1] for n in H.nodes)
# #     else:
# #         max_time = 0

# #     # 2) Helper function to pick style
# #     def pick_style(node):
# #         """Return (facecolor, edgecolor, marker, size) or None if invisible."""
# #         q, t = node
# #         node_type = H.get_node_attribute(node, 'type', None)

# #         # default style
# #         facecolor = 'white'
# #         edgecolor = 'black'
# #         marker = 'o'
# #         size = 30

# #         if node_type in ["group", "two-qubit"]:
# #             facecolor = "black"
# #             edgecolor = "black"
# #             marker = "o"
# #             size = 30
# #         elif node_type == "single-qubit":
# #             facecolor = "gray"
# #             edgecolor = "black"
# #             marker = "o"
# #             size = 30
# #         else:
# #             # 'invisibleStyle' => skip
# #             return None
        
# #         return facecolor, edgecolor, marker, size

# #     # 3) Place real nodes
# #     node_positions = {}
# #     for n in H.nodes:
# #         q, t = n
# #         x = t * xscale
# #         y = (num_qubits_phys - pos_list[t][q]) * yscale
# #         node_positions[n] = (x, y)

# #         style = pick_style(n)
# #         if style is not None:
# #             facecolor, edgecolor, marker, size = style
# #             ax.scatter(
# #                 [x],
# #                 [y],
# #                 c=facecolor,
# #                 edgecolors=edgecolor,
# #                 marker=marker,
# #                 s=size,
# #                 zorder=3
# #             )

# #     # 4) Draw Hyperedges
# #     for edge_id, edge_info in H.hyperedges.items():
# #         # Example logic from your code
# #         if isinstance(edge_id, tuple) and len(edge_id) == 2:
# #             roots = edge_info.get("root_set", [])
# #             root_node = edge_id
# #             root_t = edge_id[1]
# #             # Possibly override root_node with the min t from root_set
# #             for rnode in roots:
# #                 if isinstance(root_t, int):
# #                     min_t = root_t
# #                 else:
# #                     min_t = root_t[1]
# #                 if rnode[1] < min_t:
# #                     root_node = rnode
# #                     root_t = rnode[1]
# #                 # if (isinstance(rnode, tuple)
# #                 #     and len(rnode) == 2
# #                 #     and isinstance(rnode[1], int)
# #                 #     and rnode[1] < root_t):
# #                 #     root_node = rnode
# #                 #     root_t = rnode[1]

# #             receivers = edge_info.get("receiver_set", [])
# #             if root_node in node_positions:
# #                 rx, ry = node_positions[root_node]
# #             else:
# #                 continue

# #             # If multiple receivers, create an offset node, etc.
# #             if len(receivers) > 1:
# #                 offset_x = rx + 0.3 * xscale
# #                 offset_y = ry - 0.3 * yscale
# #                 # root -> offset
# #                 ax.plot([rx, offset_x], [ry, offset_y], color="black", zorder=2)
# #                 # offset -> each receiver
# #                 for rnode in receivers:
# #                     if rnode in node_positions:
# #                         rxr, ryr = node_positions[rnode]
# #                         ax.plot([offset_x, rxr], [offset_y, ryr], color="black", zorder=2)
# #             elif len(receivers) == 1:
# #                 rnode = list(receivers)[0]
# #                 if rnode in node_positions:
# #                     rxr, ryr = node_positions[rnode]
# #                     ax.plot([rx, rxr], [ry, ryr], color="black", zorder=2)
# #         else:
# #             # Possibly the other logic branch
# #             root_set = edge_info.get("root_set", [])
# #             rec_set = edge_info.get("receiver_set", [])
# #             if not root_set or not rec_set:
# #                 continue
# #             node1 = list(root_set)[0]
# #             node2 = list(rec_set)[0]
# #             if node1 in node_positions and node2 in node_positions:
# #                 x1, y1 = node_positions[node1]
# #                 x2, y2 = node_positions[node2]
# #                 ax.plot([x1, x2], [y1, y2], color="black", zorder=2)

# #     # 5) ADD GHOST (EXTRA) SMALL WHITE NODES at t=0 or t=max_time
# #     #    and connect them to the real node at the same (q,t).
# #     for n in H.nodes:
# #         q, t = n
# #         if n not in node_positions:
# #             continue
# #         rx, ry = node_positions[n]

# #         if t == 0:
# #             # Place ghost node one step "before" t=0 => t=-1
# #             ghost_t = -1
# #             ghost_x = ghost_t * xscale
# #             ghost_y = ry  # same vertical
# #             # Plot the ghost node
# #             ghost_size = 20
# #             ax.scatter(
# #                 [ghost_x],
# #                 [ghost_y],
# #                 c="white",
# #                 edgecolors="black",
# #                 marker="o",
# #                 s=ghost_size,
# #                 zorder=4
# #             )
# #             # Draw line from ghost node to the real node (t=0)
# #             ax.plot([ghost_x, rx], [ghost_y, ry], color="black", zorder=2)

# #         elif t == max_time:
# #             # Place ghost node one step "beyond" t=max_time => t=max_time+1
# #             ghost_t = max_time + 1
# #             ghost_x = ghost_t * xscale
# #             ghost_y = ry
# #             ghost_size = 20
# #             ax.scatter(
# #                 [ghost_x],
# #                 [ghost_y],
# #                 c="white",
# #                 edgecolors="black",
# #                 marker="o",
# #                 s=ghost_size,
# #                 zorder=4
# #             )
# #             # Draw line from the real node (t=max_time) to the ghost node
# #             ax.plot([rx, ghost_x], [ry, ghost_y], color="black", zorder=2)

# #     # 6) Tidy up
# #     ax.set_aspect("equal", adjustable="datalim")
# #     ax.set_xlabel("Time Layer (scaled by xscale)")
# #     ax.set_ylabel("Physical Qubit Index (scaled by yscale)")
# #     plt.axis("off")
# #     # optionally invert y if you prefer
# #     # ax.invert_yaxis()

# #     if save and path:
# #         plt.savefig(path, bbox_inches="tight")

# #     return fig, ax

# import copy
# import matplotlib.pyplot as plt

# def space_mapping(qpu_info, num_layers):
#     qpu_mapping = {}
#     qubit_index = 0
#     for j, qpu_size in enumerate(qpu_info):
#         qubit_list = []
#         for _ in range(qpu_size):
#             qubit_list.append(qubit_index)
#             qubit_index += 1
#         qpu_mapping[j] = qubit_list
#     space_map = []
#     for t in range(num_layers):
#         space_map.append(copy.deepcopy(qpu_mapping))
#     return space_map

# def get_pos_list(num_qubits, assignment, space_map):
#     num_layers = len(assignment)
#     pos_list = [[None for _ in range(num_qubits)] for _ in range(num_layers)]
#     for q in range(num_qubits):
#         old_partition = None
#         for t in range(num_layers):
#             partition = assignment[t][q]
#             if old_partition is not None:
#                 if partition == old_partition:
#                     if x_index in space_map[t][partition]:
#                         x_index = pos_list[t-1][q]
#                         pos_list[t][q] = x_index
#                         space_map[t][partition].remove(x_index)
#                     else:
#                         qubit_list = space_map[t][partition]
#                         x_index = qubit_list.pop(0)
#                         pos_list[t][q] = x_index
#                 else:
#                     qubit_list = space_map[t][partition]
#                     x_index = qubit_list.pop(0)
#                     pos_list[t][q] = x_index
#             else:
#                 qubit_list = space_map[t][partition]
#                 x_index = qubit_list.pop(0)
#                 pos_list[t][q] = x_index
#             old_partition = partition
#     return pos_list


# def hypergraph_to_matplotlib(
#     H,
#     num_qubits,
#     assignment,
#     qpu_info,
#     depth,
#     num_qubits_phys,
#     xscale=1.0,
#     yscale=1.0,
#     figsize=(10, 6),
#     save=False,
#     path=None,
#     ax=None
# ):
#     """
#     Draw a QuantumCircuitHyperGraph 'H' with Matplotlib.

#     This version inserts an extra blank horizontal row for each partition
#     as determined by qpu_info. For example, if qpu_info = [4, 4, 8], then
#     partition 0 covers qubits 0..3, partition 1 covers qubits 4..7, etc.
#     Each successive partition is offset one additional row in the y-direction.
#     """

#     # Build an easy lookup for partition index based on qubit q
#     # Example: if qpu_info = [4,4,8], then:
#     #   partition 0 => q in [0..3]
#     #   partition 1 => q in [4..7]
#     #   partition 2 => q in [8..15]
#     part_bounds = []
#     cumulative = 0
#     for size in qpu_info:
#         part_bounds.append((cumulative, cumulative + size))
#         cumulative += size

#     def get_partition_index(q):
#         for i, (start, end) in enumerate(part_bounds):
#             if start <= q < end:
#                 return i
#         # Fallback (shouldn't happen if q < sum(qpu_info)):
#         return 0

#     # 1) Compute positions
#     space_map_ = space_mapping(qpu_info, depth)
#     pos_list = get_pos_list(num_qubits, assignment, space_map_)

#     # If user didn't supply an Axes, create one
#     if ax is None:
#         fig, ax = plt.subplots(figsize=figsize)
#     else:
#         fig = ax.figure

#     # Determine max_time
#     if H.nodes:
#         max_time = max(n[1] for n in H.nodes)
#     else:
#         max_time = 0

#     # 2) Helper function to pick style (facecolor, etc.)
#     def pick_style(node):
#         """Return (facecolor, edgecolor, marker, size) or None if invisible."""
#         q, t = node
#         node_type = H.get_node_attribute(node, 'type', None)

#         # default style
#         facecolor = 'white'
#         edgecolor = 'black'
#         marker = 'o'
#         size = 30

#         if node_type in ["group", "two-qubit"]:
#             facecolor = "black"
#             edgecolor = "black"
#             marker = "o"
#             size = 30
#         elif node_type == "single-qubit":
#             facecolor = "gray"
#             edgecolor = "black"
#             marker = "o"
#             size = 30
#         else:
#             # 'invisibleStyle' => skip
#             return None
        
#         return facecolor, edgecolor, marker, size

#     # 3) Place real nodes
#     node_positions = {}
#     for n in H.nodes:
#         q, t = n
#         # figure out how many qubits remain above this partition
#         # We invert Y as before: (num_qubits_phys - pos_list[t][q])
#         base_y = num_qubits_phys - pos_list[t][q]
#         # find the partition of q => p_idx
#         p_idx = get_partition_index(q)
#         # offset the y by p_idx so each partition is shifted
#         shifted_y = base_y + p_idx
#         x = t * xscale
#         y = shifted_y * yscale

#         node_positions[n] = (x, y)

#         style = pick_style(n)
#         if style is not None:
#             facecolor, edgecolor, marker, size = style
#             ax.scatter(
#                 [x],
#                 [y],
#                 c=facecolor,
#                 edgecolors=edgecolor,
#                 marker=marker,
#                 s=size,
#                 zorder=3
#             )

#     # 4) Draw Hyperedges
#     for edge_id, edge_info in H.hyperedges.items():
#         if isinstance(edge_id, tuple) and len(edge_id) == 2:
#             roots = edge_info.get("root_set", [])
#             root_node = edge_id
#             root_t = edge_id[1]

#             # Possibly override root_node with the min t from root_set
#             for rnode in roots:
#                 if isinstance(root_t, int):
#                     min_t = root_t
#                 else:
#                     min_t = root_t[1]
#                 if rnode[1] < min_t:
#                     root_node = rnode
#                     root_t = rnode[1]

#             receivers = edge_info.get("receiver_set", [])
#             if root_node in node_positions:
#                 rx, ry = node_positions[root_node]
#             else:
#                 continue

#             if len(receivers) > 1:
#                 offset_x = rx + 0.3 * xscale
#                 offset_y = ry - 0.3 * yscale
#                 ax.plot([rx, offset_x], [ry, offset_y], color="black", zorder=2)
#                 for rnode in receivers:
#                     if rnode in node_positions:
#                         rxr, ryr = node_positions[rnode]
#                         ax.plot([offset_x, rxr], [offset_y, ryr], color="black", zorder=2)
#             elif len(receivers) == 1:
#                 rnode = list(receivers)[0]
#                 if rnode in node_positions:
#                     rxr, ryr = node_positions[rnode]
#                     ax.plot([rx, rxr], [ry, ryr], color="black", zorder=2)
#         else:
#             root_set = edge_info.get("root_set", [])
#             rec_set = edge_info.get("receiver_set", [])
#             if not root_set or not rec_set:
#                 continue
#             node1 = list(root_set)[0]
#             node2 = list(rec_set)[0]
#             if node1 in node_positions and node2 in node_positions:
#                 x1, y1 = node_positions[node1]
#                 x2, y2 = node_positions[node2]
#                 ax.plot([x1, x2], [y1, y2], color="black", zorder=2)

#     # 5) Add GHOST nodes at t=0 or t=max_time
#     for n in H.nodes:
#         q, t = n
#         if n not in node_positions:
#             continue
#         rx, ry = node_positions[n]

#         if t == 0:
#             # ghost at t=-1
#             ghost_t = -1
#             ghost_x = ghost_t * xscale
#             ghost_y = ry  # same vertical
#             ghost_size = 20
#             ax.scatter(
#                 [ghost_x],
#                 [ghost_y],
#                 c="white",
#                 edgecolors="black",
#                 marker="o",
#                 s=ghost_size,
#                 zorder=4
#             )
#             ax.plot([ghost_x, rx], [ghost_y, ry], color="black", zorder=2)

#         elif t == max_time:
#             # ghost at t = max_time + 1
#             ghost_t = max_time + 1
#             ghost_x = ghost_t * xscale
#             ghost_y = ry
#             ghost_size = 20
#             ax.scatter(
#                 [ghost_x],
#                 [ghost_y],
#                 c="white",
#                 edgecolors="black",
#                 marker="o",
#                 s=ghost_size,
#                 zorder=4
#             )
#             ax.plot([rx, ghost_x], [ry, ghost_y], color="black", zorder=2)

#     # 6) Tidy up
#     ax.set_aspect("equal", adjustable="datalim")
#     ax.set_xlabel("Time Layer (scaled by xscale)")
#     ax.set_ylabel("Physical Qubit Index (scaled by yscale) + partition offset")
#     plt.axis("off")

#     if save and path:
#         plt.savefig(path, bbox_inches="tight")

#     return fig, ax

import copy
import matplotlib.pyplot as plt

def space_mapping(qpu_info, num_layers):
    qpu_mapping = {}
    qubit_index = 0
    for j, qpu_size in enumerate(qpu_info):
        qubit_list = []
        for _ in range(qpu_size):
            qubit_list.append(qubit_index)
            qubit_index += 1
        qpu_mapping[j] = qubit_list
    space_map = []
    for t in range(num_layers):
        space_map.append(copy.deepcopy(qpu_mapping))
    return space_map

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


def hypergraph_to_matplotlib(
    H,
    num_qubits,
    assignment,
    qpu_info,
    depth,
    # We no longer rely on num_qubits_phys to position nodes. We'll ignore it or
    # keep it if you want a global scaling factor. Let's keep it as a leftover multiplier:
    num_qubits_phys,
    xscale=1.0,
    yscale=1.0,
    figsize=(10, 6),
    save=False,
    path=None,
    ax=None
):
    """
    Draw a QuantumCircuitHyperGraph 'H' with Matplotlib, placing a blank
    horizontal row between each partition. The *logical* partition of a node
    is taken from assignment[t][q], so nodes can move between partitions over time.

    qpu_info = [p0_size, p1_size, ... ] says how many "vertical slots" each partition has.
    We'll stack partitions vertically in order: partition 0 at top, then a blank row,
    partition 1, another blank row, etc.
    """

    # Precompute partition offsets: partition p => sum(qpu_info[:p]) + p
    # so each partition is stacked below the previous + 1 blank row
    partition_offset_list = []
    cumulative = 0
    for i, size in enumerate(qpu_info):
        partition_offset_list.append(cumulative + i)  # i is the blank row
        cumulative += size
    # partition_offset_list[p] => base offset for partition p

    def partition_offset(p):
        return partition_offset_list[p]

    # 1) set up pos_list
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
        # which partition is node (q, t) assigned to?
        p = assignment[t][q]

        # local index within that partition
        local_index = pos_list[t][q]

        # If you want to invert the local index so 0 is top in each partition, you can do:
        # local_index = (qpu_info[p] - 1) - local_index
        # or just use local_index as is.

        # shift by partition_offset
        offset = partition_offset(p)

        # final Y = offset + local_index
        # optionally multiply by any leftover "num_qubits_phys" if you still want that.
        # For consistency with your original approach, let's keep the "phys" as just a scale factor:
        base_y = offset + local_index
        y = base_y * yscale

        # x = t * xscale as normal
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

    # 4) Draw Hyperedges
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

    # 5) Ghost nodes for t=0 or t=max_time
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

    # 6) Tidy up
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Time Layer (scaled by xscale)")
    ax.set_ylabel("Vertical index within assigned partition")
    plt.axis("off")

    if save and path:
        plt.savefig(path, bbox_inches="tight")

    return fig, ax