"""Bosonic-native partitioned circuit extractor.

Extraction emits bosonic-model instructions directly into a
``NativeCircuitBuilder`` without ever constructing a Qiskit ``QuantumCircuit``.
The only remaining Qiskit usage in DISQCO is:
  - ``bosonic_converters.CircuitConverters.to_qiskit`` (simulation utilities in
    ``verification.py``).
  - Test helpers that build input circuits with Qiskit before converting to
    bosonic via ``CircuitConverters.from_qiskit``.
"""
from __future__ import annotations

import copy
import math as mt
from collections import deque

import networkx as nx
import numpy as np

from bosonic_model import Circuit as BosonicCircuit
from bosonic_model import DistributedCircuit
from bosonic_model import Register as BosonicRegister
from bosonic_model.instructions import (
    Condition,
    ConditionalInstruction,
    CpInstruction,
    CxInstruction,
    CzInstruction,
    GateInstruction,
    HInstruction,
    Instruction as BosonicInstruction,
    MeasureInstruction,
    ResetInstruction,
    RzInstruction,
    SdgInstruction,
    SInstruction,
    TdgInstruction,
    TInstruction,
    UInstruction,
    XInstruction,
    YInstruction,
    ZInstruction,
)

from disqco import QuantumCircuitHyperGraph, QuantumNetwork
from disqco.circuit_extraction.DQC_qubit_manager import (
    ClassicalBitManager,
    CommunicationQubitManager,
    DataQubitManager,
    NativeCircuitBuilder,
)


# -------------------------------------------------------------------
# TeleportationManager
# -------------------------------------------------------------------

class TeleportationManager:
    """Emits bosonic-native teleportation primitives into a NativeCircuitBuilder.

    All qubit and cbit arguments are plain ``int`` global indices.
    """

    def __init__(
        self,
        builder: NativeCircuitBuilder,
        hypergraph: QuantumCircuitHyperGraph,
        network: QuantumNetwork,
        qubit_manager: DataQubitManager,
        comm_manager: CommunicationQubitManager,
        creg_manager: ClassicalBitManager,
    ) -> None:
        self.builder = builder
        self.network = network
        self.qubit_manager = qubit_manager
        self.comm_manager = comm_manager
        self.creg_manager = creg_manager
        self.hypergraph = hypergraph

    # ------------------------------------------------------------------
    # Low-level emit helpers
    # ------------------------------------------------------------------

    def _emit_state_transfer(self, q1: int, q2: int, cbit: int) -> None:
        """State-transfer primitive: teleport state from q1 to q2."""
        self.builder.emit(CxInstruction(control=q1, target=q2, qubits=[q1, q2]))
        self.builder.emit(HInstruction(qubit=q1, params=[], qubits=[q1]))
        self.builder.emit(MeasureInstruction(qubit=q1, cbit=cbit, qubits=[q1]))
        self.builder.emit(ResetInstruction(qubit=q1, qubits=[q1]))
        inner = ZInstruction(qubit=q2, params=[], qubits=[q2])
        self.builder.emit(
            ConditionalInstruction(
                condition=Condition(cbit=cbit, value=True),
                op=inner,
                qubits=inner.qubits,
            )
        )

    def _emit_starting_process(
        self, root_q: int, root_comm: int, rec_comm: int, cbit: int
    ) -> None:
        """Cat-entangler starting process: EPR + CX on root + measure + X correction."""
        self.builder.emit(
            GateInstruction(
                name="remote_link_phi_plus",
                qubits=[root_comm, rec_comm],
                params=[],
                opaque=True,
            )
        )
        self.builder.emit(CxInstruction(control=root_q, target=root_comm, qubits=[root_q, root_comm]))
        self.builder.emit(MeasureInstruction(qubit=root_comm, cbit=cbit, qubits=[root_comm]))
        self.builder.emit(ResetInstruction(qubit=root_comm, qubits=[root_comm]))
        inner = XInstruction(qubit=rec_comm, params=[], qubits=[rec_comm])
        self.builder.emit(
            ConditionalInstruction(
                condition=Condition(cbit=cbit, value=True),
                op=inner,
                qubits=inner.qubits,
            )
        )

    def _emit_ending_process(self, target_q: int, link_comm: int, cbit: int) -> None:
        """Cat-entangler ending process: H + measure on link_comm + Z correction on target."""
        self.builder.emit(HInstruction(qubit=link_comm, params=[], qubits=[link_comm]))
        self.builder.emit(MeasureInstruction(qubit=link_comm, cbit=cbit, qubits=[link_comm]))
        self.builder.emit(ResetInstruction(qubit=link_comm, qubits=[link_comm]))
        inner = ZInstruction(qubit=target_q, params=[], qubits=[target_q])
        self.builder.emit(
            ConditionalInstruction(
                condition=Condition(cbit=cbit, value=True),
                op=inner,
                qubits=inner.qubits,
            )
        )

    # ------------------------------------------------------------------
    # Higher-level teleportation operations
    # ------------------------------------------------------------------

    def transfer_state(self, q1: int, q2: int) -> None:
        """Transfer state from q1 to an unused slot q2."""
        cbit = self.creg_manager.allocate_cbit()
        self._emit_state_transfer(q1, q2, cbit)
        self.creg_manager.release_cbit(cbit)

    def entangle_root(self, root_idx: int, p_root: int, p_rec: int) -> None:
        """Entangle the root qubit with a comm qubit in another QPU (starting process)."""
        root_q = self.qubit_manager.log_to_phys_idx[root_idx]
        root_comm = self.comm_manager.find_comm_idx(p_root)
        rec_comm = self.comm_manager.find_comm_idx(p_rec)
        cbit = self.creg_manager.allocate_cbit()
        self._emit_starting_process(root_q, root_comm, rec_comm, cbit)
        self.qubit_manager.groups[root_idx]["linked_qubits"][p_rec] = rec_comm
        self.creg_manager.release_cbit(cbit)
        self.comm_manager.release_comm_qubit(p_root, root_comm)

    def end_entanglement_link(
        self, q_root: int, p_root: int, p_rec: int, p_target: int
    ) -> None:
        """Disentangle the root qubit from a comm qubit in another QPU (ending process)."""
        if p_rec == p_target:
            return
        root_q = self.qubit_manager.log_to_phys_idx[q_root]
        rec_comm = self.qubit_manager.groups[q_root]["linked_qubits"][p_rec]
        if p_root == p_target:
            target_comm = root_q
        else:
            target_comm = self.qubit_manager.groups[q_root]["linked_qubits"][p_target]

        cbit = self.creg_manager.allocate_cbit()
        self._emit_ending_process(target_comm, rec_comm, cbit)
        if self.builder.qubit_kind[rec_comm] == "C":
            self.comm_manager.release_comm_qubit(p_rec, rec_comm)
        else:
            self.qubit_manager.release_data_qubit(p_rec, rec_comm)
        self.creg_manager.release_cbit(cbit)

    def close_group(self, root_idx: int) -> None:
        """Disentangle all links in the group; update root's physical location."""
        group_info = self.qubit_manager.groups[root_idx]
        p_root_init = group_info["init_p_root"]
        final_p_root = group_info["final_p_root"]
        linked_qubits = group_info["linked_qubits"]

        for p, linked_comm in linked_qubits.items():
            if p == final_p_root and p != p_root_init:
                if self.builder.qubit_kind[linked_comm] == "C":
                    try:
                        data_qubit = self.qubit_manager.allocate_data_qubit(p)
                    except Exception as e:
                        print(
                            f"Failed to allocate data qubit in partition {p} "
                            f"for root {root_idx}: {e}"
                        )
                        print(f"All groups: {self.qubit_manager.groups}")
                        raise e
                    self.transfer_state(linked_comm, data_qubit)
                    self.qubit_manager.assign_to_physical(p, data_qubit, root_idx)
                    self.comm_manager.release_comm_qubit(p, linked_comm)
            else:
                if p == p_root_init and p == final_p_root:
                    continue
                self.end_entanglement_link(root_idx, p_root_init, p, final_p_root)
                self.comm_manager.release_comm_qubit(p, linked_comm)

        del self.qubit_manager.groups[root_idx]

    # ------------------------------------------------------------------
    # Teleportation scheduling helpers
    # ------------------------------------------------------------------

    def space_count(self) -> list[int]:
        """Count free data qubit slots per partition."""
        return [len(v) for v in self.qubit_manager.free_data.values()]

    def choose_qubit(self, graph, space_counts):
        for p, space_p in enumerate(space_counts):
            if space_p > 0:
                edges_in = graph.in_edges(p)
                if len(edges_in) == 0:
                    continue
                for edge in edges_in:
                    break
                qubits = graph.get_edge_data(*edge)
                for qubit in qubits:
                    break
                graph.remove_edge(edge[0], edge[1], key=qubit)
                space_counts[edge[0]] += 1
                space_counts[edge[1]] -= 1
                return qubit, edge[0], edge[1]
        return None, None, None

    def get_teleportation_order(
        self,
        assignment1: list,
        assignment2: list,
        num_partitions: int,
        num_qubits: int,
    ) -> list[dict[str, int]]:
        """Determine the order of teleportations between two assignments."""
        graph = nx.MultiDiGraph()
        for p in range(num_partitions):
            graph.add_node(p)
        for q in range(num_qubits):
            if q in self.qubit_manager.groups:
                continue
            p1 = assignment1[q]
            p2 = assignment2[q]
            if p1 != p2:
                graph.add_edge(p1, p2, key=q)

        teleportation_order = []
        space_counts = self.space_count()
        while True:
            qubit, source, destination = self.choose_qubit(graph, space_counts)
            if qubit is None:
                break
            teleportation_order.append(
                {"qubit": qubit, "source": source, "destination": destination}
            )
        if len(graph.edges()) == 0:
            return teleportation_order

        cycles = nx.simple_cycles(graph)
        for cycle in cycles:
            try:
                edges = nx.find_cycle(graph, cycle)
            except nx.NetworkXNoCycle:
                continue
            for (source, destination, qubit) in edges:
                teleportation_order.append(
                    {"qubit": qubit, "source": source, "destination": destination}
                )
                graph.remove_edge(source, destination, key=qubit)
        return teleportation_order

    def swap_qubits_to_physical(
        self, qubit_idx: int, partition: int, data_loc: int
    ) -> bool:
        try:
            data_q = self.qubit_manager.allocate_data_qubit(partition)
            self.transfer_state(data_loc, data_q)
            self.qubit_manager.assign_to_physical(partition, data_q, qubit_idx)
            self.comm_manager.release_comm_qubit(partition, data_loc)
            return True
        except Exception:
            return False

    def teleport_qubits(
        self,
        old_assignment: list[int],
        new_assignment: list[int],
        num_partitions: int,
        num_qubits: int,
    ) -> None:
        """Teleport qubits to transition between two assignments."""
        old_assignment = [int(x) for x in old_assignment]
        new_assignment = [int(x) for x in new_assignment]
        num_partitions = int(num_partitions)
        num_qubits = int(num_qubits)
        teleportation_order = self.get_teleportation_order(
            old_assignment, new_assignment, num_partitions, num_qubits
        )
        remaining_swaps = []
        for teleportation in teleportation_order:
            qubit_idx = int(teleportation["qubit"])
            p_source = int(teleportation["source"])
            p_dest = int(teleportation["destination"])

            comm_qubits = self.entangle_root_on_tree(
                root_q=qubit_idx,
                target_partitions=[p_dest],
                p_root=p_source,
                num_partitions=num_partitions,
                group_gate=False,
            )

            comm_dest = comm_qubits[p_dest]
            cbit = self.creg_manager.allocate_cbit()
            data_q1 = self.qubit_manager.log_to_phys_idx[qubit_idx]
            self._emit_ending_process(comm_dest, data_q1, cbit)
            self.qubit_manager.release_data_qubit(p_source, qubit=data_q1)
            self.creg_manager.release_cbit(cbit)
            success = self.swap_qubits_to_physical(qubit_idx, p_dest, comm_dest)
            if not success:
                remaining_swaps.append((qubit_idx, p_dest, comm_dest))

        while remaining_swaps:
            qubit_idx, p_dest, comm_dest = remaining_swaps.pop(0)
            success = self.swap_qubits_to_physical(int(qubit_idx), int(p_dest), comm_dest)
            if not success:
                remaining_swaps.append((qubit_idx, p_dest, comm_dest))

    def gate_teleport(
        self, root_q: int, rec_q: int, gate: dict, p_root: int, p_rec: int
    ) -> None:
        """Non-local two-qubit gate via gate teleportation along the network tree."""
        comm_qubits = self.entangle_root_on_tree(
            root_q=root_q,
            target_partitions=[p_rec],
            p_root=p_root,
            num_partitions=len(self.network.qpu_sizes),
            group_gate=False,
        )

        comm_q_rec = comm_qubits[p_rec]
        data_q_root = self.qubit_manager.log_to_phys_idx[root_q]
        data_q_rec = self.qubit_manager.log_to_phys_idx[rec_q]

        name = gate["name"]
        params = gate["params"]
        if name == "cp":
            self.builder.emit(
                CpInstruction(
                    control=comm_q_rec,
                    target=data_q_rec,
                    lam=params[0],
                    params=params,
                    qubits=[comm_q_rec, data_q_rec],
                )
            )
        elif name == "cx":
            self.builder.emit(
                CxInstruction(control=comm_q_rec, target=data_q_rec, qubits=[comm_q_rec, data_q_rec])
            )
        elif name == "cz":
            self.builder.emit(
                CzInstruction(control=comm_q_rec, target=data_q_rec, qubits=[comm_q_rec, data_q_rec])
            )

        cbit = self.creg_manager.allocate_cbit()
        self._emit_ending_process(data_q_root, comm_q_rec, cbit)
        self.comm_manager.release_comm_qubit(p_rec, comm_q_rec)
        self.creg_manager.release_cbit(cbit)

    # ------------------------------------------------------------------
    # k-fold entanglement on the network tree
    # ------------------------------------------------------------------

    def entangle_root_on_tree(
        self,
        root_q: int,
        target_partitions: list[int],
        p_root: int,
        num_partitions: int,
        group_gate: bool = False,
    ) -> dict[int, int]:
        undirected_tree = self.network.get_full_tree(
            root_p=p_root, target_partitions=target_partitions
        )
        if undirected_tree:
            directed_tree = nx.DiGraph()
            visited: set[int] = set()
            queue: deque[int] = deque([p_root])
            while queue:
                parent = queue.popleft()
                visited.add(parent)
                for child in undirected_tree.neighbors(parent):
                    if child not in visited:
                        directed_tree.add_edge(parent, child)
                        queue.append(child)
            node_in_comm = self.build_k_fold_starting_process(
                root_q, p_root, target_partitions, directed_tree
            )
            if group_gate:
                for p, comm_qubit in node_in_comm.items():
                    self.qubit_manager.groups[root_q]["linked_qubits"][p] = comm_qubit
            return node_in_comm
        return {}

    def build_k_fold_starting_process(
        self,
        root_q: int,
        p_root: int,
        target_partitions: list[int],
        tree: nx.DiGraph,
    ) -> dict[int, int]:
        """Build the k-fold starting process along *tree*, returning node_in_comm."""
        edges_to_comms: dict[tuple[int, int], tuple[int, int]] = {}
        for p0, p1 in tree.edges():
            comm0 = self.comm_manager.find_comm_idx(p0)
            comm1 = self.comm_manager.find_comm_idx(p1)
            self.builder.emit(
                GateInstruction(
                    name="remote_link_phi_plus",
                    qubits=[comm0, comm1],
                    params=[],
                    opaque=True,
                )
            )
            edges_to_comms[(p0, p1)] = (comm0, comm1)

        root_q_phys = self.qubit_manager.log_to_phys_idx[root_q]
        node_paths: dict[int, list] = {p_root: []}
        node_cbits: dict[int, list] = {p_root: []}
        node_in_comm: dict[int, int] = {p_root: root_q_phys}
        queue: deque[int] = deque([p_root])

        correction_info: list[tuple[int, list[int], int]] = []
        while queue:
            current = queue.popleft()
            children = list(tree.successors(current))
            in_qubit = node_in_comm[current]
            for child in children:
                comm_current, comm_child = edges_to_comms[(current, child)]
                self.builder.emit(
                    CxInstruction(control=in_qubit, target=comm_current, qubits=[in_qubit, comm_current])
                )
                cbit = self.creg_manager.allocate_cbit()
                self.builder.emit(MeasureInstruction(qubit=comm_current, cbit=cbit, qubits=[comm_current]))
                self.builder.emit(ResetInstruction(qubit=comm_current, qubits=[comm_current]))
                node_paths[child] = node_paths[current] + [child]
                node_cbits[child] = node_cbits[current] + [cbit]
                node_in_comm[child] = comm_child
                correction_info.append((comm_child, list(node_cbits[child]), cbit))
                queue.append(child)
                self.comm_manager.release_comm_qubit(current, comm_current)

        for comm_child, cbits, last_cbit in correction_info:
            for cb in cbits:
                inner = XInstruction(qubit=comm_child, params=[], qubits=[comm_child])
                self.builder.emit(
                    ConditionalInstruction(
                        condition=Condition(cbit=cb, value=True),
                        op=inner,
                        qubits=inner.qubits,
                    )
                )
            self.creg_manager.release_cbit(last_cbit)

        all_nodes = set(tree.nodes())
        target_set = set(target_partitions)
        aux_nodes = [n for n in all_nodes if n not in target_set.union({p_root})]
        for aux in aux_nodes:
            min_path = None
            min_target = None
            for tgt in target_partitions + [p_root]:
                try:
                    path = nx.shortest_path(tree, source=aux, target=tgt)
                    if min_path is None or len(path) < len(min_path):
                        min_path = path
                        min_target = tgt
                except nx.NetworkXNoPath:
                    continue
            if min_path is None:
                continue

            local_epr = node_in_comm[aux]
            if min_target == p_root:
                live_epr = root_q_phys
            else:
                live_epr = node_in_comm[min_target]

            cbit = self.creg_manager.allocate_cbit()
            self._emit_ending_process(live_epr, local_epr, cbit)
            self.comm_manager.release_comm_qubit(aux, local_epr)
            self.creg_manager.release_cbit(cbit)
            del node_in_comm[aux]

        return node_in_comm


# -------------------------------------------------------------------
# PartitionedCircuitExtractor
# -------------------------------------------------------------------

class PartitionedCircuitExtractor:
    """Extract a partitioned quantum circuit as a bosonic DistributedCircuit.

    Extraction emits bosonic instructions directly; no Qiskit QuantumCircuit is
    constructed internally.  The output format (DistributedCircuit) is unchanged.
    """

    def __init__(
        self,
        graph: QuantumCircuitHyperGraph,
        network: QuantumNetwork,
        partition_assignment: np.ndarray,
    ) -> None:
        self.layer_dict = graph.layers
        self.layer_dict = self.remove_empty_groups()
        self.partition_assignment = partition_assignment.tolist()
        self.num_qubits = graph.num_qubits
        self.qpu_info = network.qpu_sizes
        self.comm_info = network.comm_sizes
        self.depth = graph.depth
        self.num_partitions = len(self.qpu_info)
        self.graph = graph
        self.network = network
        self.basis_gates = graph.basis_gates

        # Build the native circuit builder and allocate all registers.
        # Registers are interleaved by QPU: Q0_q, C0_0, Q1_q, C1_0, ...
        # This ordering matches the former reorder_registers_by_index output.
        self.builder = NativeCircuitBuilder()
        partition_data: dict[int, list[int]] = {}
        comm_data: dict[int, list[int]] = {}
        for i in range(self.num_partitions):
            q_idx = self.builder.add_qreg(f"Q{i}_q", self.qpu_info[i], "Q", i)
            partition_data[i] = q_idx
            c_idx = self.builder.add_qreg(f"C{i}_0", self.comm_info[i], "C", i)
            comm_data[i] = c_idx

        cl_global_size = max(self.num_qubits, 16)
        cl_global_indices = self.builder.add_creg("cl_global", cl_global_size)
        result_indices = self.builder.add_creg("result", self.num_qubits)
        self.result_reg_base = result_indices[0]

        # Create managers
        self.qubit_manager = DataQubitManager(
            partition_data,
            self.num_qubits,
            self.partition_assignment,
            self.builder,
        )
        self.comm_manager = CommunicationQubitManager(comm_data, self.builder)
        self.creg_manager = ClassicalBitManager(
            self.builder, cl_global_indices[0], cl_global_size
        )

        self.teleportation_manager = TeleportationManager(
            self.builder,
            self.graph,
            self.network,
            self.qubit_manager,
            self.comm_manager,
            self.creg_manager,
        )

        self.current_assignment = self.partition_assignment[0]

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def remove_empty_groups(self) -> dict[int, list[dict]]:
        new_layers = copy.deepcopy(self.layer_dict)
        for i, layer in new_layers.items():
            for gate in layer[:]:
                if gate["type"] == "group":
                    if len(gate["sub-gates"]) == 1:
                        new_gate = gate["sub-gates"].pop(0)
                        t = new_gate["time"]
                        del new_gate["time"]
                        new_layers[t].append(new_gate)
                        layer.remove(gate)
                    elif len(gate["sub-gates"]) == 0:
                        layer.remove(gate)
        return new_layers

    # ------------------------------------------------------------------
    # Gate emission helpers
    # ------------------------------------------------------------------

    def _emit_single_qubit_gate(self, name: str, params: list, phys: int) -> None:
        """Emit a single-qubit gate onto physical qubit *phys*."""
        if name in ("u", "u3"):
            self.builder.emit(
                UInstruction(
                    qubit=phys,
                    theta=params[0],
                    phi=params[1],
                    lam=params[2],
                    params=params,
                    qubits=[phys],
                )
            )
        elif name == "h":
            self.builder.emit(HInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "x":
            self.builder.emit(XInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "y":
            self.builder.emit(YInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "z":
            self.builder.emit(ZInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "s":
            self.builder.emit(SInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "sdg":
            self.builder.emit(SdgInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "t":
            self.builder.emit(TInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "tdg":
            self.builder.emit(TdgInstruction(qubit=phys, params=[], qubits=[phys]))
        elif name == "rz":
            self.builder.emit(
                RzInstruction(qubit=phys, phi=params[0], params=params, qubits=[phys])
            )
        else:
            self.builder.emit(
                GateInstruction(name=name, qubits=[phys], params=list(params), opaque=True)
            )

    def _emit_two_qubit_gate(
        self, name: str, params: list, phys0: int, phys1: int
    ) -> None:
        """Emit a two-qubit gate using physical qubit indices."""
        if name == "cx":
            self.builder.emit(
                CxInstruction(control=phys0, target=phys1, qubits=[phys0, phys1])
            )
        elif name == "cz":
            self.builder.emit(
                CzInstruction(control=phys0, target=phys1, qubits=[phys0, phys1])
            )
        elif name == "cp":
            self.builder.emit(
                CpInstruction(
                    control=phys0,
                    target=phys1,
                    lam=params[0],
                    params=params,
                    qubits=[phys0, phys1],
                )
            )
        else:
            self.builder.emit(
                GateInstruction(
                    name=name, qubits=[phys0, phys1], params=list(params), opaque=True
                )
            )

    # ------------------------------------------------------------------
    # Gate application (logical -> physical)
    # ------------------------------------------------------------------

    def apply_single_qubit_gate(self, gate: dict) -> None:
        q = gate["qargs"][0]
        phys = self.qubit_manager.log_to_phys_idx[q]
        self._emit_single_qubit_gate(gate["name"], gate["params"], phys)

    def apply_local_two_qubit_gate(self, gate: dict) -> None:
        """Apply a two-qubit gate whose qargs are logical qubit indices."""
        qubit0, qubit1 = gate["qargs"]
        phys0 = self.qubit_manager.log_to_phys_idx[qubit0]
        phys1 = self.qubit_manager.log_to_phys_idx[qubit1]
        self._emit_two_qubit_gate(gate["name"], gate["params"], phys0, phys1)

    def check_qpus_local(self, phys0: int, phys1: int) -> bool:
        """Return True if both physical qubits belong to the same QPU."""
        return self.builder.qubit_partition[phys0] == self.builder.qubit_partition[phys1]

    def find_common_part(self, qubit0: int, qubit1: int) -> int:
        """Find a common partition where both qubits have entanglement links."""
        if qubit1 in self.qubit_manager.groups:
            possible_partitions1 = set(self.qubit_manager.groups[qubit1]["linked_qubits"].keys())
        else:
            possible_partitions1 = {self.current_assignment[qubit1]}

        if qubit0 in self.qubit_manager.groups:
            possible_partitions0 = set(self.qubit_manager.groups[qubit0]["linked_qubits"].keys())
        else:
            possible_partitions0 = {self.current_assignment[qubit0]}

        for p in possible_partitions1.intersection(possible_partitions0):
            return p
        return -1

    def apply_non_local_two_qubit_gate(
        self, gate: dict, p_root: int, p1: int
    ) -> None:
        """Apply a two-qubit gate that spans QPUs via previously-established links."""
        root_q, q1 = gate["qargs"]
        common_part = self.find_common_part(root_q, q1)
        if common_part == -1:
            common_part = p1

        if (
            root_q in self.qubit_manager.groups
            and common_part in self.qubit_manager.groups[root_q]["linked_qubits"]
        ):
            root_q_phys = self.qubit_manager.groups[root_q]["linked_qubits"][common_part]
        else:
            root_q_phys = self.qubit_manager.log_to_phys_idx[root_q]

        if (
            q1 in self.qubit_manager.groups
            and common_part in self.qubit_manager.groups[q1]["linked_qubits"]
        ):
            q1_phys = self.qubit_manager.groups[q1]["linked_qubits"][common_part]
        else:
            q1_phys = self.qubit_manager.log_to_phys_idx[q1]

        if not self.check_qpus_local(root_q_phys, q1_phys):
            print(f"Non-local two-qubit gate {gate} cannot be applied locally.")
            print("Root qubit:", root_q, "Q1 qubit:", q1)
            print("Mapped root qubit:", root_q_phys, "Mapped Q1 qubit:", q1_phys)
            print("Data qubit q1:", self.qubit_manager.log_to_phys_idx[q1])
            print("Current assignment:", self.current_assignment)
            print("Qubit manager groups:", self.qubit_manager.groups)
            raise ValueError(
                f"Non-local two-qubit gate {gate} cannot be applied locally."
            )

        self._emit_two_qubit_gate(gate["name"], gate["params"], root_q_phys, q1_phys)

        if gate["time"] == self.qubit_manager.groups[root_q]["final_time"]:
            try:
                self.teleportation_manager.close_group(root_q)
            except Exception as e:
                print(f"Error closing group {root_q}: {e}")
                print(f"Qubit manager groups: {self.qubit_manager.groups}")
                print(f"Gate: {gate}")
                print(f"Free data qubits: {self.qubit_manager.free_data}")
                print(f"In use data qubits: {self.qubit_manager.in_use_data}")
                print(f"Free comm qubits: {self.comm_manager.free_comm}")
                print(f"In use comm qubits: {self.comm_manager.in_use_comm}")
                for i in range(len(self.current_assignment)):
                    print(f"Qubit {i} should be in partition {self.current_assignment[i]}")
                    data_i = self.qubit_manager.log_to_phys_idx[i]
                    print(f"Qubit {i} is on physical qubit {data_i}")
                    if self.builder.qubit_partition[data_i] == int(self.current_assignment[i]):
                        print(f"Qubit {i} is in the correct partition")
                    else:
                        print(f"Qubit {i} is in the wrong partition")
                raise e

    # ------------------------------------------------------------------
    # Gate diagonal / group handling
    # ------------------------------------------------------------------

    def check_diag_gate(self, gate: dict) -> str:
        name = gate["name"]
        if name in ("u", "u3"):
            theta = gate["params"][0]
            if round(theta % (mt.pi * 2), 2) == round(0, 2):
                return "diagonal"
            elif round(theta % (mt.pi * 2), 2) == round(mt.pi / 2, 2):
                return "anti-diagonal"
            else:
                return "non-diagonal"
        if name == "h":
            return "non-diagonal"
        if name in ("z", "t", "s", "rz", "u1", "tdg", "sdg"):
            return "diagonal"
        if name in ("x", "y"):
            return "anti-diagonal"
        return "non-diagonal"

    def apply_linked_single_qubit_gate(self, gate: dict) -> None:
        q = gate["qargs"][0]
        p_root = self.current_assignment[q]
        diagonality = self.check_diag_gate(gate)
        if diagonality == "diagonal":
            self.apply_single_qubit_gate(gate)
        elif diagonality == "anti-diagonal":
            for p in range(self.num_partitions):
                if p != p_root:
                    # Pre-existing code path; groups[q][p] would KeyError if
                    # reached with the current groups structure - preserved as-is.
                    if self.current_assignment[q] not in self.qubit_manager.groups[q][p]:
                        continue
                    for linked_part in self.qubit_manager.groups[q]["linked_qubits"]:
                        comm_q = self.qubit_manager.groups[q]["linked_qubits"][linked_part]
                        if comm_q == self.qubit_manager.log_to_phys_idx.get(q):
                            continue
                        inner = XInstruction(qubit=comm_q, params=[], qubits=[comm_q])
                        self.builder.emit(inner)
            self.apply_single_qubit_gate(gate)
        else:
            raise ValueError(
                f"Gate {gate} is not diagonal or anti-diagonal and shouldn't be in group."
            )

    def process_group_gate(self, gate: dict, t: int) -> None:
        """Process a group gate: establish entanglement links, schedule sub-gates."""
        root_idx = gate["root"]
        start_time = gate["time"]
        p_root = self.partition_assignment[start_time][root_idx]
        sub_gates = gate["sub-gates"]
        if not sub_gates:
            return

        p_rec_set: set[int] = set()
        final_gates: dict[int, int] = {}
        for sub_gate in sub_gates[::-1]:
            if sub_gate["type"] == "two-qubit":
                final_t = sub_gate["time"]
                break

        final_p_root = int(self.partition_assignment[final_t][root_idx])

        p_root_set: set[int] = set()
        for time_step in range(start_time, final_t + 1):
            p_root_set.add(int(self.partition_assignment[time_step][root_idx]))

        if p_root_set != {p_root}:
            root_q = self.qubit_manager.log_to_phys_idx[root_idx]
            self.qubit_manager.release_data_qubit(p_root, root_q)
            root_comm = self.comm_manager.find_comm_idx(p_root)
            self.teleportation_manager.transfer_state(root_q, root_comm)
            self.qubit_manager.log_to_phys_idx[root_idx] = root_comm

        self.qubit_manager.groups[root_idx] = {}
        for sub_gate in sub_gates:
            if sub_gate["type"] == "two-qubit":
                q0, q1 = sub_gate["qargs"]
                time_step = sub_gate["time"]
                p_rec = int(self.partition_assignment[time_step][q1])
                p_rec_set.add(p_rec)
                final_gates[p_rec] = max(final_gates.get(p_rec, time_step), time_step)

        self.qubit_manager.groups[root_idx]["final_gates"] = final_gates
        self.qubit_manager.groups[root_idx]["init_time"] = start_time
        self.qubit_manager.groups[root_idx]["final_time"] = final_t
        self.qubit_manager.groups[root_idx]["final_p_root"] = final_p_root
        self.qubit_manager.groups[root_idx]["init_p_root"] = p_root
        self.qubit_manager.groups[root_idx]["p_rec_set"] = p_rec_set
        self.qubit_manager.groups[root_idx]["p_root_set"] = p_root_set

        linked_qubits = {p_root: self.qubit_manager.log_to_phys_idx[root_idx]}
        self.qubit_manager.groups[root_idx]["linked_qubits"] = linked_qubits

        target_partitions = list(p_rec_set.union(p_root_set) - {p_root})
        self.teleportation_manager.entangle_root_on_tree(
            root_idx, target_partitions, p_root, self.num_partitions, group_gate=True
        )

        for sub_gate in sub_gates:
            if sub_gate["type"] == "two-qubit":
                q0, q1 = sub_gate["qargs"]
                time_step = sub_gate["time"]
                p1 = int(self.partition_assignment[time_step][q1])
                new_gate = {
                    "type": "two-qubit-linked",
                    "name": sub_gate["name"],
                    "qargs": [q0, q1],
                    "params": sub_gate["params"],
                    "time": time_step,
                }
                if p1 == p_root:
                    if time_step == t:
                        self.apply_local_two_qubit_gate(sub_gate)
                    else:
                        self.layer_dict[time_step].append(new_gate)
                else:
                    if time_step == t:
                        self.apply_non_local_two_qubit_gate(sub_gate, p_root, p1)
                    else:
                        self.layer_dict[time_step].append(new_gate)

            elif sub_gate["type"] == "single-qubit":
                q = sub_gate["qargs"][0]
                time_step = sub_gate["time"]
                new_gate = {
                    "type": "single-qubit-linked",
                    "qargs": [q],
                    "params": sub_gate["params"],
                    "time": time_step,
                }
                if time_step == t:
                    self.apply_linked_single_qubit_gate(sub_gate)
                else:
                    self.layer_dict[time_step].append(sub_gate)

        if root_idx in self.qubit_manager.groups and final_t == t:
            self.teleportation_manager.close_group(root_idx)

    # ------------------------------------------------------------------
    # Main extraction entry point
    # ------------------------------------------------------------------

    def extract_partitioned_circuit(self) -> DistributedCircuit:
        for i, layer in sorted(self.layer_dict.items()):
            new_assignment_layer = self.partition_assignment[i]
            for q in range(self.num_qubits):
                if self.current_assignment[q] != new_assignment_layer[q]:
                    self.teleportation_manager.teleport_qubits(
                        self.current_assignment,
                        new_assignment_layer,
                        self.num_partitions,
                        self.num_qubits,
                    )
                    break

            self.current_assignment = new_assignment_layer
            self.partition_assignment[i] = new_assignment_layer

            for gate in layer:
                gtype = gate["type"]

                if gtype == "single-qubit":
                    self.apply_single_qubit_gate(gate)

                elif gtype == "two-qubit":
                    q0, q1 = gate["qargs"]
                    p0 = self.current_assignment[q0]
                    p1 = self.current_assignment[q1]
                    if p0 == p1:
                        self.apply_local_two_qubit_gate(gate)
                    else:
                        self.teleportation_manager.gate_teleport(q0, q1, gate, p0, p1)

                elif gtype == "group":
                    self.process_group_gate(gate, i)

                elif gtype == "two-qubit-linked":
                    q0, q1 = gate["qargs"]
                    p_root = self.qubit_manager.groups[q0]["init_p_root"]
                    p_rec = self.current_assignment[q1]
                    self.apply_non_local_two_qubit_gate(gate, p_root, p_rec)

        # Final measurements into the result register.
        for i in range(self.num_qubits):
            phys = self.qubit_manager.log_to_phys_idx[i]
            cbit = self.result_reg_base + i
            self.builder.emit(MeasureInstruction(qubit=phys, cbit=cbit, qubits=[phys]))

        flat_circuit = self.builder.build()
        return self._to_distributed_circuit(flat_circuit)

    # ------------------------------------------------------------------
    # Distributed circuit assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _node_from_reg_name(name: str) -> int | None:
        """Parse a register name like 'Q3_q' or 'C2_5' and return the QPU index."""
        if not name or name[0] not in {"Q", "C"}:
            return None
        digits: list[str] = []
        for ch in name[1:]:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if not digits:
            return None
        return int("".join(digits))

    def _to_distributed_circuit(self, bosonic: BosonicCircuit) -> DistributedCircuit:
        """Route a flat BosonicCircuit into a per-node DistributedCircuit.

        Each instruction is assigned to the per-node Circuit(s) whose QPU owns
        its qubits.  Instructions spanning multiple QPUs (EPR primitives) are
        wrapped as remote_* GateInstructions and shared by reference.
        Classical registers are shared across all per-node Circuits so
        classical-control conditions remain valid everywhere.
        """
        nodes = sorted(range(self.num_partitions))

        qubit_to_node: dict[int, int] = {}
        qubits_per_node: dict[int, list[int]] = {n: [] for n in nodes}
        per_node_qregs: dict[int, dict[str, BosonicRegister]] = {n: {} for n in nodes}

        for reg in bosonic.qregs.values():
            node = self._node_from_reg_name(reg.name)
            if node is None:
                node = 0
            if node not in qubits_per_node:
                qubits_per_node[node] = []
                per_node_qregs[node] = {}
            per_node_qregs[node][reg.name] = reg
            for offset in range(reg.size):
                global_idx = reg.base + offset
                qubit_to_node[global_idx] = node
                qubits_per_node[node].append(global_idx)

        shared_cregs: dict[str, BosonicRegister] = dict(bosonic.cregs)

        circuits: dict[int, BosonicCircuit] = {
            node: BosonicCircuit(
                qregs=per_node_qregs[node],
                cregs=shared_cregs,
                instructions=[],
            )
            for node in qubits_per_node
        }
        instruction_index: dict[int, int] = {}

        def involved_nodes(inst: BosonicInstruction) -> set[int]:
            inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
            qubits = list(getattr(inner, "qubits", []) or [])
            return {qubit_to_node[q] for q in qubits if q in qubit_to_node}

        def make_remote(inst: BosonicInstruction) -> GateInstruction:
            inner = inst.op if isinstance(inst, ConditionalInstruction) else inst
            base_name = str(
                getattr(inner, "name", getattr(inner, "kind", "gate"))
            ).lower()
            remote_name = (
                base_name if base_name.startswith("remote_") else f"remote_{base_name}"
            )
            params = [float(p) for p in (getattr(inner, "params", []) or [])]
            qubits = list(getattr(inner, "qubits", []) or [])
            return GateInstruction(
                name=remote_name, qubits=qubits, params=params, opaque=True
            )

        for order, inst in enumerate(bosonic.instructions):
            touched = involved_nodes(inst)
            if not touched:
                for node in circuits:
                    circuits[node].instructions.append(inst)
                instruction_index[id(inst)] = order
                continue
            if len(touched) == 1:
                node = next(iter(touched))
                circuits[node].instructions.append(inst)
                instruction_index[id(inst)] = order
                continue
            remote = make_remote(inst)
            for node in sorted(touched):
                if node in circuits:
                    circuits[node].instructions.append(remote)
            instruction_index[id(remote)] = order

        qubits_per_node = {n: q for n, q in qubits_per_node.items() if q}
        circuits = {n: c for n, c in circuits.items() if n in qubits_per_node}

        distributed = DistributedCircuit(
            qubits_per_node=qubits_per_node,
            circuits=circuits,
        )
        distributed._instruction_index = instruction_index
        return distributed
