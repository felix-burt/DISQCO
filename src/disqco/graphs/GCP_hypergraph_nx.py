from collections import defaultdict
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers, layer_list_to_dict
from disqco.graphs.greedy_gate_grouping import group_distributable_packets
from qiskit import QuantumCircuit
# ---------------- hyperdigraph_nx.py ----------------------------------------
import networkx as nx
from itertools import count

class HyperDiGraphNX:
    """
    Minimal directed hyper-graph on top of networkx.DiGraph.
    Each hyper-edge h = (roots, targets, data) becomes an extra node HE_h.

                   roots ─▶ HE_h ─▶ targets
    """

    _hid = count()                        # running hyper-edge id

    def __init__(self, *, multigraph=False):
        self.g = nx.MultiDiGraph() if multigraph else nx.DiGraph()
        self._edge_node = {}              # hid -> edge-node label

    # ---------- node bookkeeping -------------------------------------------
    def _ensure_node(self, label):
        """Add the node to the graph if it isn't there yet."""
        if label not in self.g:
            self.g.add_node(label)
        return label

    # ---------- public API --------------------------------------------------
    def add_hyperedge(self, roots, targets, data=None):
        """Insert the directed hyper-edge Roots ⟶ Targets, return its id."""
        hid = next(self._hid)
        he  = ('HE', hid)                 # distinguishable tag
        self.g.add_node(he, payload=data)
        self._edge_node[hid] = he

        for r in roots:
            self.g.add_edge(self._ensure_node(r), he)
        for t in targets:
            self.g.add_edge(he, self._ensure_node(t))
        return hid

    def hyper_successors(self, v):
        """Yield every w reachable from v through exactly one hyper-edge."""
        for he in self.g.successors(v):          # v → HE
            if isinstance(he, tuple) and he[0] == 'HE':
                yield from self.g.successors(he) # HE → w

    # utility if you want to hand the raw DiGraph to networkx algorithms
    def underlying_digraph(self):
        return self.g

    def nodes(self, *, include_edge_nodes=False):
        """
        Return an iterable of node labels.

        Parameters
        ----------
        include_edge_nodes : bool, default False
            If False (default) filter out the synthetic 'HE' nodes that
            represent hyper-edges; if True return every node in the
            underlying NetworkX graph.
        """
        if include_edge_nodes:
            return list(self.g.nodes)
        # filter out incidence nodes ('HE', hid)
        return [v for v in self.g.nodes
                if not (isinstance(v, tuple) and v[0] == 'HE')]

    def hyperedges(self):
        """
        Return a list of hyper-edge IDs in the order they were created.
        """
        return list(self._edge_node.keys())
    
# ---------------- quantum_hypergraph_nx.py ----------------------------------
from collections import defaultdict

class QuantumCircuitHyperGraph:
    """
    Identical public interface as your original class, but the connectivity
    lives in a HyperDiGraphNX (→ networkx.DiGraph underneath).
    """

    # ---- construction ------------------------------------------------------
    def __init__(self,
                 circuit,
                 group_gates: bool = True,
                 anti_diag:   bool = True,
                 map_circuit: bool = True):

        self.circuit     = circuit
        self.num_qubits  = circuit.num_qubits
        self.depth       = circuit.depth()

        # NEW: NetworkX-backed hyper-graph
        self.hg = HyperDiGraphNX(multigraph=False)

        # Python dicts for metadata (fast enough, no need to move)
        self.node_attrs       = {}               # (q,t)  -> dict
        self.hyperedge_attrs  = {}               # edgeID -> dict
        self.node2hyperedges  = defaultdict(set)

        if map_circuit:
            self.init_from_circuit(group_gates, anti_diag)

    # ---- tiny helpers ------------------------------------------------------
    def _add_node_if_absent(self, q, t):
        lbl = (q, t)
        self.hg._ensure_node(lbl)
        self.node_attrs.setdefault(lbl, {})
        return lbl

    # ---- public API (same signatures as before) ----------------------------
    def add_node(self, qubit, time):
        return self._add_node_if_absent(qubit, time)

    def add_hyperedge(self, edge_id, root_set, receiver_set):
        self.hg.add_hyperedge(root_set, receiver_set)        # NetworkX part
        self.hyperedge_attrs.setdefault(edge_id, {})         # metadata

        all_nodes = root_set | receiver_set
        for n in all_nodes:
            self.node2hyperedges[n].add(edge_id)

    def add_edge(self, edge_id, node_a, node_b):
        self.add_hyperedge(edge_id, {node_a}, {node_b})

    def neighbors(self, node):
        """All nodes reachable through one hyper-edge (same semantics)."""
        return set(self.hg.hyper_successors(node))

    # ---- attribute helpers (unchanged) -------------------------------------
    def set_node_attribute(self, node, key, value):
        if node not in self.node_attrs:
            raise KeyError(f"{node} not present")
        self.node_attrs[node][key] = value

    def get_node_attribute(self, node, key, default=None):
        return self.node_attrs.get(node, {}).get(key, default)

    def set_hyperedge_attribute(self, edge_id, key, value):
        self.hyperedge_attrs.setdefault(edge_id, {})[key] = value

    def get_hyperedge_attribute(self, edge_id, key, default=None):
        return self.hyperedge_attrs.get(edge_id, {}).get(key, default)
    
    def nodes(self):
        """List of (qubit, time) tuples currently present."""
        return self.hg.nodes()

    def hyperedges(self, data=False):
        """
        List of edge IDs exactly as you passed them when calling
        `add_hyperedge` / `add_edge`.
        """
        if data:
            return [(edge_id, self.hyperedge_attrs[edge_id])
                    for edge_id in self.hyperedges()]
        return list(self.hyperedge_attrs.keys())

    # ---- the rest of your original methods ---------------------------------
    # Wherever the old code did
    #       self.nodes.add(...)
    #       self.hyperedges[edge_id] = ...
    # just call the helpers above.  No other logic changes are required.
    #
    # For brevity those bodies are not repeated here – paste your originals
    # unchanged except that:
    #   * creation of nodes → self._add_node_if_absent(...)
    #   * creation of connections → self.add_edge(...) or self.add_hyperedge(...)
    # Everything else (layer extraction, mapping gates, copying, etc.)
    # stays byte-for-byte the same.

    def init_from_circuit(self, group_gates=True, anti_diag=False, qpu_sizes=None):
        
        self.layers = self.extract_layers(self.circuit, group_gates=group_gates, anti_diag=anti_diag, qpu_sizes=qpu_sizes)
        self.depth = len(self.layers)
        self.add_time_neighbor_edges(self.depth, range(self.num_qubits))
        self.map_circuit_to_hypergraph()

    def extract_layers(self, circuit, group_gates=True, anti_diag=False, qpu_sizes=None):
        layers = circuit_to_gate_layers(circuit, qpu_sizes=qpu_sizes)
        layers = layer_list_to_dict(layers)
        if group_gates:
            layers = group_distributable_packets(layers, group_anti_diags=anti_diag)
        return layers

    def add_time_neighbor_edges(self, depth, qubits):
        """
        For each qubit in qubits, connect (qubit, t) to (qubit, t+1)
        for t in [0, max_time-1].
        """
        for qubit in qubits:
            for t in range(0,depth-1):
                node_a = (qubit, t)
                node_b = (qubit, t + 1)

                self.add_node(qubit, t)
                self.add_node(qubit, t + 1)

                self.add_edge((node_a,node_b), node_a, node_b)
    

    def assign_positions(self, num_qubits_phys):
        """
        Assign a 'pos' attribute to all nodes based on their (qubit, time).
        
        The position is (x, y) = (t, num_qubits_phys - q) 
        for each node (q, t).
        
        :param num_qubits_phys: The total number of physical qubits or 
                                however many 'vertical slots' you want.
        """
        for (q, t) in self.nodes:
            x = t
            y = num_qubits_phys - q
            # Store in node_attrs or via the set_node_attribute function:
            self.set_node_attribute((q, t), "pos", (x, y))
 
    def copy(self):
        """
        Create a new QuantumCircuitHyperGraph that is an identical 
        (shallow) copy of this one, so that modifications to the copy 
        do not affect the original.
        """
        # 1) Create a blank instance (no qubits/depth needed for now)
        new_graph = QuantumCircuitHyperGraph(circuit=self.circuit, map_circuit=False)

        # 2) Copy nodes
        new_graph.nodes = set(self.nodes)

        # 3) Copy hyperedges (including root/receiver sets)
        new_graph.hyperedges = {}
        for edge_id, edge_data in self.hyperedges.items():
            root_copy = set(edge_data['root_set'])
            rec_copy = set(edge_data['receiver_set'])
            new_graph.hyperedges[edge_id] = {
                'root_set': root_copy,
                'receiver_set': rec_copy
            }

        # 4) Copy node2hyperedges
        new_graph.node2hyperedges = defaultdict(set)
        for node, edge_ids in self.node2hyperedges.items():
            new_graph.node2hyperedges[node] = set(edge_ids)

        # 5) Copy adjacency
        new_graph.adjacency = defaultdict(set)
        for node, nbrs in self.adjacency.items():
            new_graph.adjacency[node] = set(nbrs)

        # 6) Copy node_attrs
        new_graph.node_attrs = {}
        for node, attr_dict in self.node_attrs.items():
            new_graph.node_attrs[node] = dict(attr_dict)

        # 7) Copy hyperedge_attrs
        new_graph.hyperedge_attrs = {}
        for edge_id, attr_dict in self.hyperedge_attrs.items():
            new_graph.hyperedge_attrs[edge_id] = dict(attr_dict)

        return new_graph
     
    def map_circuit_to_hypergraph(self,):
        layers_dict = self.layers
        for l in layers_dict:
            layer = layers_dict[l]
            for gate in layer:
                if gate['type'] == 'single-qubit':
                    qubit = gate['qargs'][0]
                    time = l
                    node = self.add_node(qubit,time)
                    self.set_node_attribute(node,'type',gate['type'])
                    self.set_node_attribute(node,'name',gate['name'])
                    self.set_node_attribute(node,'params',gate['params'])
                elif gate['type'] == 'two-qubit':
                    qubit1 = gate['qargs'][0]
                    qubit2 = gate['qargs'][1]
                    time = l
                    node1 = self.add_node(qubit1,time)
                    self.set_node_attribute(node1,'type',gate['type'])
                    self.set_node_attribute(node1,'name','control')
                    node2 = self.add_node(qubit2,time)
                    self.set_node_attribute(node2,'type',gate['type'])
                    if gate['name'] == 'cx' or gate['name'] == 'cu': # May need to specify more
                        self.set_node_attribute(node2,'name','target')
                    else:
                        self.set_node_attribute(node2,'name','control')
                    self.add_edge((node1,node2),node1,node2)
                    self.set_hyperedge_attribute((node1,node2),'type',gate['type'])
                    self.set_hyperedge_attribute((node1,node2),'name',gate['name'])
                    self.set_hyperedge_attribute((node1,node2),'params',gate['params'])
                elif gate['type'] == 'group':
                    root = gate['root']
                    start_time = l
                    root_node = self.add_node(root,start_time)
                    root_set = set()
                    root_set.add(root_node)
                    receiver_set = set()
                    for sub_gate in gate['sub-gates']:
                        if sub_gate['type'] == 'single-qubit':
                            qubit = sub_gate['qargs'][0]
                            time = sub_gate['time']
                            node = self.add_node(qubit,time)
                            root_set.add(node)
                            self.set_node_attribute(node,'type',sub_gate['type'])
                            self.set_node_attribute(node,'name',sub_gate['name'])
                            self.set_node_attribute(node,'params',sub_gate['params'])
                        elif sub_gate['type'] == 'two-qubit':
                            qubit1 = sub_gate['qargs'][0]
                            qubit2 = sub_gate['qargs'][1]
                            time = sub_gate['time']
                            node1 = self.add_node(qubit1,time)
                            root_set.add(node1)
                            if node1 == root_node:
                                type_ = 'group'
                            else:
                                type_ = 'root_t'
                            self.set_node_attribute(node1,'type',type_)
                            self.set_node_attribute(node1,'name','control')
                            node2 = self.add_node(qubit2,time)
                            receiver_set.add(node2)
                            self.set_node_attribute(node2,'type',gate['type'])
                            if sub_gate['name'] == 'cx' or sub_gate['name'] == 'cu': # May need to specify more
                                self.set_node_attribute(node2,'name','target')
                            else:
                                self.set_node_attribute(node2,'name','control')
                    for t in range(start_time,time+1):
                        root_set.add((root,t))
                        if t != start_time:
                            self.set_node_attribute((root,t),'type', 'root_t')
                    self.add_hyperedge(root_node,root_set,receiver_set)
