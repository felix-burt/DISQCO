from collections import defaultdict
from disqco.utils.qiskit_to_op_list import circuit_to_gate_layers, layer_list_to_dict
from disqco.graphs.greedy_gate_grouping import group_distributable_packets
from qiskit import QuantumCircuit
import rustworkx as rx
from itertools import count

class HyperDiGraph:
    """
    A directed hyper‑graph implemented on top of rustworkx.PyDiGraph.
    Each hyper‑edge h = (roots, targets, data) becomes one extra node HE_h.
    """

    _hid = count()                  # running id for hyper‑edges

    def __init__(self, *, multigraph=False):
        self._g           = rx.PyDiGraph(multigraph=multigraph)
        self._idx_of      = {}      # vertex label -> rustworkx index
        self._edge_node   = {}      # hyper‑edge id -> index of HE_h

    # ---- low‑level helpers -----------------------------------------------
    def _index(self, v):
        """Return rustworkx index for vertex label, creating the node if absent."""
        if v not in self._idx_of:
            self._idx_of[v] = self._g.add_node(v)
        return self._idx_of[v]

    # ---- public API -------------------------------------------------------
    def add_hyperedge(self, roots, targets, payload=None):
        """Insert the hyper‑edge Roots ⟶ Targets and return its id."""
        hid   = next(self._hid)
        he_ix = self._g.add_node(("HE", hid, payload))   # tag helps identify later
        self._edge_node[hid] = he_ix

        for r in roots:
            self._g.add_edge(self._index(r), he_ix, None)
        for t in targets:
            self._g.add_edge(he_ix, self._index(t), None)
        return hid

    # ---- convenience views ------------------------------------------------
    def successors(self, v):
        """Yield every w reachable from v through *one* hyper‑edge."""
        v_ix = self._index(v)
        for he in self._g.successors(v_ix):              # v → HE
            for w in self._g.successors(he):             # HE → w
                yield self._g[w]                         # original label

    def underlying_digraph(self):
        """Expose the internal PyDiGraph (read‑only!)."""

        return self._g
    
    # ---------------------------------------------------------------------------
#  prerequisite: the HyperDiGraph wrapper from the previous answer
# ---------------------------------------------------------------------------

import rustworkx as rx
from collections import defaultdict

class QuantumCircuitHyperGraphRx:
    """
    Same public interface as before, but uses HyperDiGraph (→ PyDiGraph inside
    rustworkx) for all connectivity.  Heavy work now runs in compiled Rust.
    """

    # ------------- construction ------------------------------------------------
    def __init__(self,
                 circuit,
                 group_gates: bool = True,
                 anti_diag: bool  = True,
                 map_circuit: bool = True):

        self.circuit   = circuit
        self.num_qubits = circuit.num_qubits
        self.depth      = circuit.depth()

        # NEW: Rust‑backed graph
        self.hg = HyperDiGraph(multigraph=False)

        # still keep cheap Python maps for metadata
        self.node_attrs      = {}               # (q,t)  -> dict
        self.hyperedge_attrs = {}               # edgeID -> dict
        self.node2hyperedges = defaultdict(set) # (q,t)  -> {edgeID}

        if map_circuit:
            self.init_from_circuit(group_gates, anti_diag)

    # ------------- helpers -----------------------------------------------------
    def _add_node_if_absent(self, q, t):
        label = (q, t)
        self.hg._idx(label)                     # creates the node in Rust if new
        self.node_attrs.setdefault(label, {})
        return label

    # ------------- public API (unchanged signature) ---------------------------
    def add_node(self, qubit, time):
        return self._add_node_if_absent(qubit, time)

    def add_hyperedge(self, edge_id, root_set, receiver_set):
        # create the hyper‑edge in the Rust graph
        self.hg.add_hyperedge(root_set, receiver_set)

        # book‑keeping identical to your original version
        self.hyperedge_attrs.setdefault(edge_id, {})
        all_nodes = root_set | receiver_set
        for n in all_nodes:
            self.node2hyperedges[n].add(edge_id)

    def add_edge(self, edge_id, node_a, node_b):
        self.add_hyperedge(edge_id, {node_a}, {node_b})

    # neighbourhood via the Rust iterator (one hop through a hyper‑edge)
    def neighbors(self, node):
        return set(self.hg.hyper_successors(node))

    # ------------ the rest of your old methods ---------------------------------
    # replace every place that formerly manipulated self.nodes or
    # self.hyperedges with the helpers above.  The bodies rarely change;
    # e.g. for attributes we still touch the same Python dictionaries.

    # example: attribute setters are *exactly* the same
    def set_node_attribute(self, node, key, value):
        if node not in self.node_attrs:
            raise KeyError(f"{node} not present")
        self.node_attrs[node][key] = value

    # --------------------------------------------------------------------------
    #  The methods that construct the graph from the QuantumCircuit
    #  (`init_from_circuit`, `map_circuit_to_hypergraph`, …) need only two
    #  surgical edits:
    #     * every time you "create" a node, call self._add_node_if_absent(...)
    #     * every time you "connect" things, call self.add_edge(...) or
    #       self.add_hyperedge(...)
    #  Nothing else changes.
    # --------------------------------------------------------------------------