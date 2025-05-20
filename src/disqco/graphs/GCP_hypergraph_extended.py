from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from typing import Hashable, Iterable, MutableMapping, MutableSet, Union, FrozenSet
import networkx as nx
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph
import math as mt
import numpy as np
from collections import defaultdict

EdgeKey = Hashable
PairKey = frozenset[Hashable]


@dataclass

class HyperEdge:
    """Simple record for an undirected hyper-edge."""
    vertices: set[Hashable]
    key: Hashable = field(default_factory=lambda: uuid.uuid4())
    attrs: dict = field(default_factory=dict)


class HyperGraph:
    """
    Minimal undirected hyper-graph wrapper backed by a NetworkX Graph.

    • Ordinary vertices are regular Graph nodes (`hyperedge=False`)
    • Every hyper-edge is an extra node with attribute `hyperedge=True`
      connected to all incident vertices.
    """
    __slots__ = ("_G", "_hyperedges", "_inc", "node_count", "node_neighbours")

    # --------------------------------------------------------------- #
    # Construction / low-level access                                 #
    # --------------------------------------------------------------- #
    def __init__(self, *, data: nx.Graph | None = None):
        self._G: nx.Graph = data.copy() if data else nx.Graph()
        self._hyperedges: MutableMapping[Hashable, HyperEdge] = {}
        self._inc: defaultdict[Hashable, set[Union[EdgeKey, PairKey]]] = defaultdict(set)

    def nx_graph(self) -> nx.Graph:
        """Return the underlying (mutable) NetworkX graph."""
        return self._G

    # --------------------------------------------------------------- #
    # Vertex interface                                                #
    # --------------------------------------------------------------- #
    def add_node(self, v: Hashable, **attrs) -> None:
        self._G.add_node(v, hyperedge=False, **attrs)

    def add_nodes_from(self, vs: Iterable[Hashable]) -> None:
        for v in vs:
            self.add_node(v)

    def remove_node(self, v: Hashable) -> None:
        if v in self._hyperedges:
            self.remove_hyperedge(v)
        self._G.remove_node(v)

    def _add_inc(self, v: Hashable, key: Union[EdgeKey, PairKey]) -> None:
        self._inc[v].add(key)

    def _del_inc(self, v: Hashable, key: Union[EdgeKey, PairKey]) -> None:
        s = self._inc.get(v)
        if s:
            s.discard(key)
            if not s:
                del self._inc[v]

    def incident(self, v: Hashable):
        """Return *copies* of the incident‑edge key sets (read‑only)."""
        return set(self._inc.get(v, ()))
    # --------------------------------------------------------------- #
    # Hyper-edge interface                                            #
    # --------------------------------------------------------------- #
    def add_hyperedge(
        self,
        vertices: Iterable[Hashable],
        *,
        key: Hashable | None = None,
        **attrs,
    ) -> Hashable:
        """
        Insert a hyper-edge containing *vertices*.
        Returns the edge key.
        """
        v_set = set(vertices)
        key = key if key is not None else uuid.uuid4()

        # edge node
        self._G.add_node(key, hyperedge=True, **attrs)

        # connect edge ↔ each vertex
        self._G.add_edges_from((key, v) for v in v_set)

        self._hyperedges[key] = HyperEdge(v_set, key, attrs)
        for v in v_set:
            self._add_inc(v, key)
        return key

    def remove_hyperedge(self, key: Hashable) -> None:
        for v in self._hyperedges[key].vertices:
            self._del_inc(v, key)
        self._hyperedges.pop(key)
        self._G.remove_node(key)

    def hyperedges(self) -> Iterable[HyperEdge]:
        return self._hyperedges.values()
    
    def nodes(self) -> Iterable[Hashable]:
        """
        All vertices in the hypergraph (excluding hyper-edges).
        """
        return (n for n, d in self._G.nodes(data=True) if not d.get("hyperedge", False))

    # --------------------------------------------------------------- #
    # Graph-like helpers                                              #
    # --------------------------------------------------------------- #
    def neighbors(self, v: Hashable) -> set[Hashable]:
        """
        All vertices one hyper-edge away from *v*.
        (Walk v → e → u for every incident edge e.)
        """
        neigh: MutableSet[Hashable] = set()
        for e in self._G.neighbors(v):
            if self._G.nodes[e].get("hyperedge", False):
                neigh.update(u for u in self._G.neighbors(e) if u != v)
            else:  # ordinary edge (if you decide to add them)
                neigh.add(e)
        return neigh
    
    def remove_node(self, v: Hashable) -> None:
        """
        Delete vertex *v* (or hyper-edge *key*).  
        If *v* is a vertex, every hyper-edge that contains it is updated;
        empty hyper-edges are deleted automatically.
        """
        if v in self._hyperedges:                  # it is a hyper-edge key
            self.remove_hyperedge(v)
        else:                                     # ordinary vertex
            # 1. update the bookkeeping objects
            for key, hedge in list(self._hyperedges.items()):
                if v in hedge.vertices:
                    hedge.vertices.remove(v)
                    if self._G.has_edge(key, v):
                        self._G.remove_edge(key, v)
                    if not hedge.vertices:        # becomes empty → drop it
                        self.remove_hyperedge(key)
            # 2. finally remove the vertex node itself
            self._G.remove_node(v)
        self._inc.pop(v, None)

    # ------------------------------------------------------------------ #
    # NEW: dynamically edit a hyper-edge                                 #
    # ------------------------------------------------------------------ #
    def add_vertex_to_hyperedge(
        self, key: Hashable, v: Hashable, **v_attrs
    ) -> None:
        """Attach vertex *v* to existing hyper-edge *key*."""
        if key not in self._hyperedges:
            raise KeyError(f"Unknown hyper-edge {key!r}")
        if v not in self._G:
            self.add_node(v, **v_attrs)
        if v not in self._hyperedges[key].vertices:
            self._G.add_edge(key, v)
            self._hyperedges[key].vertices.add(v)
        self._add_inc(v, key)

    def remove_vertex_from_hyperedge(
        self, key: Hashable, v: Hashable, *, delete_empty: bool = True
    ) -> None:
        """Detach vertex *v* from hyper-edge *key*."""
        if key not in self._hyperedges:
            raise KeyError(f"Unknown hyper-edge {key!r}")
        hedge = self._hyperedges[key]
        if v not in hedge.vertices:
            return                                      # nothing to do
        hedge.vertices.remove(v)
        if self._G.has_edge(key, v):
            self._G.remove_edge(key, v)
        if not hedge.vertices and delete_empty:
            # hyper-edge is now empty – tidy it up
            self.remove_hyperedge(key)
        self._del_inc(v, key)

    def convert_from_GCP(self, gcp_hypergraph: QuantumCircuitHyperGraph) -> None:
        """
        Convert from GCP hypergraph to this hypergraph.
        """
        num_qubits = gcp_hypergraph.num_qubits
        depth = gcp_hypergraph.depth
        nodes = {}
        hyperedges = []
        for i in range(num_qubits):
            live = False
            for t in range(depth):
                node = (i, t)
                print("Node: ", node)
                node_attrs = gcp_hypergraph.node_attrs[node]
                if node_attrs != {}:
                    node_type = node_attrs["type"]
                else:
                    node_type = 'single-qubit'
                    name = 'i'
                    params = [0, 0, 0]

                    node_attrs = {  'type': node_type,
                                    'name': name,
                                    'params': params    }
                print("Node type: ", node_type)
                if node_type == 'single-qubit':
                    nodes[node] = node_attrs
                    diagonality = self.check_diag_gate(node_attrs)
                    if diagonality and live:
                        hyperedge.add(node)
                        print("Gate is diagonal and root is live: ", node)
                        print("Added node to hyperedge: ", node)
                    else:
                        if live:
                            hyperedges.append(set(hyperedge))
                            hyperedge = set()
                            live = False
                            print("Non diagonal gate, end live group: ", node)
                            print("Hyperedge: ", hyperedge)
                        
                        prev_node = (i, t-1)
                        print("Previous node: ", prev_node)
                        if prev_node in gcp_hypergraph.nodes:
                            print("Previous node exists: ", prev_node)
                            print("Create state hyperedge: ")
                            hyperedge = set()
                            hyperedge.add(prev_node)
                            hyperedge.add(node)
                            hyperedges.append(set(hyperedge))
                            hyperedge = set()
                            live = False
                elif node_type == 'two-qubit':
                    partner = self.find_partner(node, gcp_hypergraph)
                    partner_q = partner[0]
                    print("Gate partner: ", partner)
                    if node_attrs['name'] == 'control':
                        if (partner_q, node[0], t) in nodes:
                            print("Gate node exists: ", (partner_q, node[0], t))
                            gate_node = (partner_q, node[0], t)
                        else:
                            print("Gate node does not exist: ", (partner_q, node[0], t))
                            gate_node = (node[0], partner_q, t)
                            if gate_node not in nodes:
                                print("Create gate node: ", gate_node)
                                edge_attrs = gcp_hypergraph.hyperedge_attrs.get((node,partner), gcp_hypergraph.hyperedge_attrs.get((partner,node)))
                                print("Edge attributes: ", edge_attrs)
                                nodes[gate_node] = {'qubits' : [node[0], partner_q], 'params': edge_attrs['params']}
                        if live: 
                            print("Group is live on root: ", node[0])
                            hyperedge.add(node)
                            print("Added node to hyperedge: ", node)
                            hyperedge.add(gate_node)
                            print("Added gate node to hyperedge: ", gate_node)
                            print("Hyperedge: ", hyperedge)
                        else:

                            print("Group is not live on root: ", node[0])
                            prev_node = (i, t-1)
                            if prev_node in nodes:
                                print("Previous node exists: ", prev_node)
                                hyperedge = set()
                                hyperedge.add(prev_node)
                                hyperedge.add(node)
                                hyperedges.append(set(hyperedge))
                            hyperedge = set()
                            print("Create new hyperedge: ", hyperedge)
                            hyperedge.add(node)
                            hyperedge.add(gate_node)
                            print("Added node to hyperedge: ", node)
                            print("Added gate node to hyperedge: ", gate_node)
                            print("Hyperedge: ", hyperedge)
                            live = True

                    elif node_attrs['name'] == 'target':
                        gate_node = (partner_q, node[0], t)
                        if live:
                            hyperedge.add(node)
                            hyperedge.add(gate_node)
                            live = False
                        else:
                            hyperedge = set()
                            hyperedge.add(node)
                            hyperedge.add(gate_node)
                            hyperedges[node] = hyperedge
                            hyperedge = set()
                            live = False
            if live == True:
                hyperedges.append(set(hyperedge))
        
        for node, attrs in nodes.items():
            if node not in self._G:
                self.add_node(node, key=node, **attrs)

        for hyperedge in hyperedges:
            if len(hyperedge) > 1:
                edge_id = self.find_edge_id(hyperedge)
                self.add_hyperedge(hyperedge, key=edge_id)
        
        self.node_count = len([n for n, d in self._G.nodes(data=True) if not d.get("hyperedge", False)])
        self.node_neighbours = {v: self.neighbors(v) for v in self.nodes()}
                    
    def find_edge_id(self, hyperedge):  
        """
        Find the edge id for a hyperedge.
        """
        t_0 = np.inf
        t_max = -np.inf
        for node in hyperedge:
            if len(node) == 2:
                t = node[1]
                if t < t_0:
                    t_0 = t
                    node_min = node
                if t > t_max:
                    t_max = t
                    node_max = node
        edge_id = (node_min, node_max)
        return edge_id
    
    def find_partner(self, node, gcp_hypergraph):
        """
        Find the partner node for a two-qubit gate.
        """
        neighbors = gcp_hypergraph.adjacency[node]
        for neighbor in neighbors:
            if neighbor[1] == node[1]:
                return neighbor
    
    def get_hyperedge_attrs(self, edge_key, attr_key) -> dict:
        """Return the live attribute dict of hyper‑edge *key*."""
        if edge_key not in self._hyperedges:
            raise KeyError(f"Unknown hyper‑edge {edge_key!r}")
        return self._G.nodes[edge_key][attr_key]          # alias of HyperEdge.attrs

    def set_hyperedge_attrs(self, edge_key, attr_key, attr) -> None:
        """Merge/overwrite attributes on an existing hyper‑edge."""
        if edge_key not in self._hyperedges:
            raise KeyError(f"Unknown hyper‑edge {edge_key!r}")
        self._G.nodes[edge_key].update({attr_key: attr})   # keeps both views consistent
        self._hyperedges[edge_key].attrs.update({attr_key: attr})

    # --------------------------------------------------------------- #
    # Convenience dunder methods                                      #
    # --------------------------------------------------------------- #
    def __contains__(self, item: Hashable) -> bool:
        return item in self._G

    def __len__(self) -> int:
        """Number of *vertex* nodes (exclude hyper-edges)."""
        return sum(
            1 for n, d in self._G.nodes(data=True) if not d.get("hyperedge", False)
        )

    def __iter__(self):
        return (
            n
            for n, d in self._G.nodes(data=True)
            if not d.get("hyperedge", False)
        )
    

    def check_diag_gate(self, gate, include_anti_diags = True):
        "Checks if a gate is diagonal or anti-diagonal"
        name = gate['name']
        if name == 'u' or name == 'u3':
            theta = gate['params'][0]
            if round(theta % mt.pi*2, 2) == round(0, 2):
                return True
            elif round(theta % mt.pi*2, 2) == round(mt.pi/2, 2):
                if include_anti_diags:
                    return True
                else:
                    return False
            else:
                return False
        else:
            if name == 'h':
                return False
            elif name == 'z' or name == 't' or name == 's' or name == 'rz' or name == 'u1':
                return True
            elif name == 'x' or 'y':
                if include_anti_diags:
                    return True
                else:
                    return False
            else:
                return False