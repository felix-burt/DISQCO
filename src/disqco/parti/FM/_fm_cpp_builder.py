"""
Convert a Python QuantumCircuitHyperGraph into a C++ FMHyperGraph.

Called once per FiducciaMattheyses lifetime (or per hypergraph change).
Returns an opaque disqco._fm_cpp.FMHyperGraph object, or None if the
C++ extension is not available.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from disqco.graphs.QC_hypergraph import QuantumCircuitHyperGraph


def build_cpp_hgraph(hg: "QuantumCircuitHyperGraph", num_partitions: int):
    """
    Build and return a C++ FMHyperGraph from a Python hypergraph.

    Node encoding:  node_id = t * Q + q
    Action encoding: action_id = node_id * K + dest

    Returns None if disqco._fm_cpp is not importable.
    """
    try:
        from disqco import _fm_cpp
    except ImportError:
        return None

    Q = hg.num_qubits
    D = hg.depth
    K = num_partitions

    # Stable ordering of edges
    edge_list = list(hg.hyperedges.keys())
    E = len(edge_list)

    # --- Build root and receiver pin CSR arrays ---
    root_pins_list: list[int] = []
    root_offsets_list: list[int] = [0]
    rec_pins_list: list[int] = []
    rec_offsets_list: list[int] = [0]

    for eid in edge_list:
        edge_data = hg.hyperedges[eid]
        for node in edge_data['root_set']:
            q, t = node[0], node[1]
            root_pins_list.append(t * Q + q)
        root_offsets_list.append(len(root_pins_list))

        for node in edge_data['receiver_set']:
            q, t = node[0], node[1]
            rec_pins_list.append(t * Q + q)
        rec_offsets_list.append(len(rec_pins_list))

    # --- Build node → incident-edges CSR arrays ---
    N = Q * D
    node_edge_lists: list[list[int]] = [[] for _ in range(N)]
    for e_idx, eid in enumerate(edge_list):
        edge_data = hg.hyperedges[eid]
        seen: set[int] = set()
        for node in edge_data['root_set']:
            q, t = node[0], node[1]
            n = t * Q + q
            if n not in seen:
                node_edge_lists[n].append(e_idx)
                seen.add(n)
        for node in edge_data['receiver_set']:
            q, t = node[0], node[1]
            n = t * Q + q
            if n not in seen:
                node_edge_lists[n].append(e_idx)
                seen.add(n)

    node_edges_list: list[int] = []
    node_offsets_list: list[int] = [0]
    for nelist in node_edge_lists:
        node_edges_list.extend(nelist)
        node_offsets_list.append(len(node_edges_list))

    return _fm_cpp.build_hgraph(
        Q, D, K,
        np.array(root_pins_list,    dtype=np.int32),
        np.array(root_offsets_list, dtype=np.int32),
        np.array(rec_pins_list,     dtype=np.int32),
        np.array(rec_offsets_list,  dtype=np.int32),
        np.array(node_edges_list,   dtype=np.int32),
        np.array(node_offsets_list, dtype=np.int32),
    )
