"""Lower a DistributedCircuit to a Qiskit QuantumCircuit.

Each FanOut operation becomes a single named custom instruction whose sub-circuit
implements the full k-fold cat-entanglement tree.  Each FanIn (ending process)
and local state transfer also becomes a named custom instruction.  EPR generation
is itself a named sub-instruction inside each FanOut block.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Clbit, Qubit
from qiskit.circuit.classical import expr as qexpr

from disqco.circuit_extraction.distributed_circuit import (
    DistributedCircuit,
    FanIn,
    FanOut,
    ImmediateFanIn,
    JointFanIn,
    LinkedGate,
    LocalGate,
    StateTransfer,
)
from disqco.circuit_extraction.DQC_qubit_manager import (
    ClassicalBitManager,
    CommunicationQubitManager,
    DataQubitManager,
)
from disqco.circuit_extraction.circuit_extractor import reorder_registers_by_index

if TYPE_CHECKING:
    from disqco import QuantumNetwork


# ---------------------------------------------------------------------------
# Standalone sub-circuit builders (return Qiskit Instructions)
# ---------------------------------------------------------------------------

def _make_epr_gate() -> object:
    """EPR pair generator: H on qubit 0, CX(0→1).  Kept as a named primitive."""
    circ = QuantumCircuit(2, name="EPR")
    circ.h(0)
    circ.cx(0, 1)
    return circ.to_gate()


def _make_ending_process_instruction(label: str, compress_corrections: bool = True) -> object:
    """Ending process (fan-in) on 2 qubits + 1 classical bit.

    Qubit layout: [target_q, source_comm_q].
    Applies H + measure + reset on source_comm_q.
    If compress_corrections is False the Z correction on target_q is included
    inside the instruction; otherwise the caller applies it via if_test.
    """
    circ = QuantumCircuit(2, 1, name=label)
    circ.h(1)
    circ.measure(1, 0)
    circ.reset(1)
    if not compress_corrections:
        circ.z(0).c_if(0, 1)
    return circ.to_instruction(label=label)


def _make_local_transfer_instruction(label: str, compress_corrections: bool = True) -> object:
    """Transfer state from qubit 0 to qubit 1 within the same partition.

    Used to move state between a data qubit and a comm qubit slot.
    Applies CX(0→1), H(0), measure(0), reset(0).
    If compress_corrections is False the Z correction on qubit 1 is included
    inside the instruction; otherwise the caller applies it via if_test.
    """
    circ = QuantumCircuit(2, 1, name=label)
    circ.cx(0, 1)
    circ.h(0)
    circ.measure(0, 0)
    circ.reset(0)
    if not compress_corrections:
        circ.z(1).c_if(0, 1)
    return circ.to_instruction(label=label)


def _make_joint_fan_in_instruction(
    label: str,
    n_sources: int,
    compress_corrections: bool = True,
) -> object:
    """Build the joint fan-in as a named instruction.

    Qubit layout inside the sub-circuit:
      0        : target qubit (keep wire; Z corrections land here)
      i+1      : die comm qubit for source i  (H + measure + reset)

    Classical bits inside the sub-circuit (= outer phys_c[i]):
      i        : measurement outcome for source i

    If compress_corrections is True (default) Z corrections are omitted from
    the instruction; the caller applies a single if_test(XOR parity) Z on the
    outer circuit.  If False, sequential Z.c_if corrections are included inside
    the instruction (one per source measurement bit), matching the original
    single-FanIn behaviour.
    """
    circ = QuantumCircuit(1 + n_sources, n_sources, name=label)
    for i in range(n_sources):
        circ.h(i + 1)
        circ.measure(i + 1, i)
        circ.reset(i + 1)
        if not compress_corrections:
            circ.z(0).c_if(i, 1)
    return circ.to_instruction(label=label)


def _make_fan_out_instruction(
    root_q_idx: int,
    p_root: int,
    target_partitions: list[int],
    bfs_edges: list[tuple[int, int]],
    label: str,
    compress_corrections: bool = True,
) -> object:
    """Build the k-fold cat-entanglement (fan-out) as a named instruction.

    Qubit layout inside the sub-circuit:
      0               : root qubit
      2*i+1           : comm qubit at *parent* side of BFS edge i  (measured+reset)
      2*i+2           : comm qubit at *child*  side of BFS edge i  (holds cat state)

    Classical bits inside the sub-circuit (= outer phys_c[i]):
      i               : measurement outcome for BFS edge i

    If compress_corrections is True (default) X corrections are omitted from the
    instruction; the caller applies a single if_test(XOR) correction per target
    on the outer circuit.  If False, legacy c_if X corrections are included inside
    the instruction (one per accumulated bit, original behaviour).
    """
    n_edges = len(bfs_edges)
    if n_edges == 0:
        circ = QuantumCircuit(1, name=label)
        return circ.to_instruction(label=label)

    sub_qc = QuantumCircuit(1 + 2 * n_edges, n_edges, name=label)
    epr_gate = _make_epr_gate()

    # 1. Generate EPR pairs for every tree edge.
    for i in range(n_edges):
        sub_qc.append(epr_gate, [2 * i + 1, 2 * i + 2])

    # 2. BFS CX-measure-reset chain; track accumulated cbits when needed.
    node_in_q_local: dict[int, int] = {p_root: 0}
    node_cbits_local: dict[int, list[int]] = {p_root: []}
    correction_info: list[tuple[int, list[int]]] = []

    for i, (parent, child) in enumerate(bfs_edges):
        in_q        = node_in_q_local[parent]
        parent_comm = 2 * i + 1
        child_comm  = 2 * i + 2
        sub_qc.cx(in_q, parent_comm)
        sub_qc.measure(parent_comm, i)
        sub_qc.reset(parent_comm)
        node_in_q_local[child] = child_comm
        if not compress_corrections:
            child_cbits = node_cbits_local[parent] + [i]
            node_cbits_local[child] = child_cbits
            correction_info.append((child_comm, child_cbits))

    # 3. Legacy c_if corrections (only when compress_corrections=False).
    if not compress_corrections:
        for child_comm, cbits in correction_info:
            for cb in cbits:
                sub_qc.x(child_comm).c_if(cb, 1)

    return sub_qc.to_instruction(label=label)


# ---------------------------------------------------------------------------
# Main converter class
# ---------------------------------------------------------------------------

class DistributedCircuitToQiskit:
    """Lower a DistributedCircuit to a Qiskit QuantumCircuit.

    Each FanOut, FanIn, and state transfer operation becomes a named custom
    instruction block, with EPR generation as a sub-instruction within FanOut.

    Parameters
    ----------
    dc :
        The distributed circuit to lower.
    network :
        QPU network topology (provides qpu_sizes and comm_sizes).
    initial_assignment :
        1-D array/list of length num_qubits giving the partition index of each
        logical qubit at time 0 (i.e. ``partition_assignment[0]``).
    """

    def __init__(
        self,
        dc: DistributedCircuit,
        network: "QuantumNetwork",
        initial_assignment,
        compress_corrections: bool = False,
    ) -> None:
        self.dc = dc
        self.network = network
        self.compress_corrections = compress_corrections
        self.num_qubits = dc.num_qubits
        self.num_partitions = dc.num_partitions
        self.qpu_sizes = network.qpu_sizes
        self.comm_sizes = network.comm_sizes

        # Wrap the initial assignment so DataQubitManager can call [0][q].
        wrapped = [list(int(x) for x in initial_assignment)]

        # Build quantum and classical registers (mirrors PartitionedCircuitExtractor).
        self.partition_qregs = [
            QuantumRegister(sz, name=f"Q{i}_q")
            for i, sz in enumerate(self.qpu_sizes.values())
        ]
        self.comm_qregs = {
            i: [QuantumRegister(csz, name=f"C{i}_0")]
            for i, csz in enumerate(self.comm_sizes.values())
        }
        self.creg       = ClassicalRegister(self.num_qubits, name="cl")
        self.result_reg = ClassicalRegister(self.num_qubits, name="result")

        comm_regs_flat = [regs[0] for regs in self.comm_qregs.values()]
        self.qc = QuantumCircuit(
            *self.partition_qregs,
            *comm_regs_flat,
            self.creg,
            self.result_reg,
            name="DistributedCircuit",
        )

        # Resource managers.
        self.qubit_manager = DataQubitManager(
            self.partition_qregs, self.num_qubits, wrapped, self.qc
        )
        self.comm_manager  = CommunicationQubitManager(self.comm_qregs, self.qc)
        self.creg_manager  = ClassicalBitManager(self.qc, self.creg)

        # Per-group state for active fan-outs.
        # group_state[root_q] = {
        #   'init_p_root'  : int,
        #   'final_p_root' : int,
        #   'linked_qubits': {partition: physical_qubit},
        # }
        self.group_state: dict[int, dict] = {}
        # Deferred comm->data moves when destination partitions are temporarily full.
        # pending_settles[q_log] = (partition, comm_qubit)
        self.pending_settles: dict[int, tuple[int, Qubit]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> QuantumCircuit:
        """Process all events and return the concrete Qiskit circuit."""
        for event in self.dc.events:
            self._process_event(event)
            # Deferred local comm->data settles should only be retried after
            # ending-process phases, not after arbitrary events.
            if isinstance(event, (FanIn, JointFanIn, ImmediateFanIn)):
                self._retry_pending_settles(only_qubits={event.root_qubit})

        # One final retry pass before measurement.
        self._retry_pending_settles()

        # Final measurements.
        for i in range(self.num_qubits):
            self.qc.measure(self.qubit_manager.log_to_phys_idx[i], self.result_reg[i])

        self.qc = reorder_registers_by_index(self.qc)
        return self.qc

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _process_event(self, event) -> None:
        if isinstance(event, LocalGate):
            self._apply_local_gate(event)
        elif isinstance(event, StateTransfer):
            self._apply_state_transfer(event)
        elif isinstance(event, FanOut):
            self._apply_fan_out(event)
        elif isinstance(event, ImmediateFanIn):
            self._apply_immediate_fan_in(event)
        elif isinstance(event, LinkedGate):
            self._apply_linked_gate(event)
        elif isinstance(event, FanIn):
            self._apply_fan_in(event)
        elif isinstance(event, JointFanIn):
            self._apply_joint_fan_in(event)

    # ------------------------------------------------------------------
    # LocalGate
    # ------------------------------------------------------------------

    def _apply_local_gate(self, event: LocalGate) -> None:
        name   = event.gate_name
        params = event.params

        if len(event.qubits) == 1:
            q, _ = event.qubits[0]
            phys = self.qubit_manager.log_to_phys_idx[q]
            self._apply_single_qubit(name, params, phys)
        elif len(event.qubits) == 2:
            (q0, _), (q1, _) = event.qubits
            p0 = self.qubit_manager.log_to_phys_idx[q0]
            p1 = self.qubit_manager.log_to_phys_idx[q1]
            self._apply_two_qubit(name, params, p0, p1)

    def _apply_single_qubit(self, name: str, params: list, q: Qubit) -> None:
        if name in ('u', 'u3'):
            self.qc.u(params[0], params[1], params[2], q)
        elif name == 'h':  self.qc.h(q)
        elif name == 'x':  self.qc.x(q)
        elif name == 'y':  self.qc.y(q)
        elif name == 'z':  self.qc.z(q)
        elif name == 's':  self.qc.s(q)
        elif name == 'sdg': self.qc.sdg(q)
        elif name == 't':  self.qc.t(q)
        elif name == 'tdg': self.qc.tdg(q)
        elif name == 'rz': self.qc.rz(params[0], q)

    def _apply_two_qubit(self, name: str, params: list, q0: Qubit, q1: Qubit) -> None:
        if name == 'cx':    self.qc.cx(q0, q1)
        elif name == 'cz':  self.qc.cz(q0, q1)
        elif name == 'cp':  self.qc.cp(params[0], q0, q1)

    # ------------------------------------------------------------------
    # StateTransfer (non-group qubit move between partitions)
    # ------------------------------------------------------------------

    def _apply_state_transfer(self, event: StateTransfer) -> None:
        """Full state teleportation from source to destination partition."""
        q_log = event.qubit
        p_src = event.source_partition
        p_dst = event.target_partition

        # Build the communication tree (may involve intermediate hops).
        directed_tree = self._get_directed_tree(p_src, [p_dst])
        bfs_edges = list(nx.bfs_edges(directed_tree, source=p_src))

        # Allocate comm qubits for each tree edge.
        edges_to_comms = self._allocate_tree_comms(bfs_edges)

        # Build and append the named FanOut instruction.
        root_phys = self.qubit_manager.log_to_phys_idx[q_log]
        label = f"FanOut[q{q_log}:p{p_src}→{[p_dst]}]"
        fan_out_instr = _make_fan_out_instruction(
            q_log, p_src, [p_dst], bfs_edges, label, self.compress_corrections
        )
        phys_q, phys_c = self._fan_out_qubit_lists(root_phys, bfs_edges, edges_to_comms)
        self.qc.append(fan_out_instr, phys_q, phys_c)

        if self.compress_corrections:
            self._apply_fan_out_corrections(p_src, bfs_edges, edges_to_comms, phys_c)
        else:
            for cbit in phys_c:
                self.creg_manager.release_cbit(cbit)

        # Release parent-side comm qubits (measured+reset inside instruction).
        for p0, p1 in bfs_edges:
            comm_parent, _ = edges_to_comms[(p0, p1)]
            self.comm_manager.release_comm_qubit(p0, comm_parent)

        # The comm qubit at p_dst holds the state.
        comm_dst = self._node_comm_qubit(p_dst, bfs_edges, edges_to_comms)

        # Apply ending process: target=comm_dst, source=root_phys (data at src).
        cbit = self.creg_manager.allocate_cbit()
        label_fi = f"FanIn[q{q_log}:p{p_src}→p{p_dst}]"
        ending = _make_ending_process_instruction(label_fi, self.compress_corrections)
        self.qc.append(ending, [comm_dst, root_phys], [cbit])
        if self.compress_corrections:
            with self.qc.if_test(qexpr.lift(cbit)):
                self.qc.z(comm_dst)
        self.qubit_manager.release_data_qubit(p_src, root_phys)
        self.creg_manager.release_cbit(cbit)

        # Move comm_dst → new data qubit at p_dst.
        try:
            new_data = self.qubit_manager.allocate_data_qubit(p_dst)
            cbit2 = self.creg_manager.allocate_cbit()
            label_lt = f"LocalTransfer[q{q_log}:p{p_dst}]"
            lt_instr = _make_local_transfer_instruction(label_lt, self.compress_corrections)
            self.qc.append(lt_instr, [comm_dst, new_data], [cbit2])
            if self.compress_corrections:
                with self.qc.if_test(qexpr.lift(cbit2)):
                    self.qc.z(new_data)
            self.qubit_manager.assign_to_physical(p_dst, new_data, q_log)
            self.comm_manager.release_comm_qubit(p_dst, comm_dst)
            self.creg_manager.release_cbit(cbit2)
            self.pending_settles.pop(q_log, None)
        except Exception:
            # No free data slot — keep on comm qubit as a fallback.
            self._mark_pending_settle(q_log, p_dst, comm_dst)

    # ------------------------------------------------------------------
    # FanOut
    # ------------------------------------------------------------------

    def _apply_fan_out(self, event: FanOut) -> None:
        root_q   = event.root_qubit
        p_root   = event.root_partition
        final_p  = event.final_root_partition
        tree     = event.path_tree
        targets  = event.target_partitions

        if p_root in tree and tree.number_of_edges() > 0:
            bfs_edges = list(nx.bfs_edges(tree, source=p_root))
        else:
            bfs_edges = []

        # Nested case: root qubit moves to a different partition during the group.
        # Move the root from its data qubit to a comm qubit at p_root first.
        if final_p != p_root:
            data_q  = self.qubit_manager.log_to_phys_idx[root_q]
            root_comm = self.comm_manager.find_comm_idx(p_root)
            label_lt = f"LocalTransfer[q{root_q}:p{p_root}→comm]"
            lt_instr = _make_local_transfer_instruction(label_lt, self.compress_corrections)
            cbit = self.creg_manager.allocate_cbit()
            self.qc.append(lt_instr, [data_q, root_comm], [cbit])
            if self.compress_corrections:
                with self.qc.if_test(qexpr.lift(cbit)):
                    self.qc.z(root_comm)
            # Source data qubit is reset by local transfer; free the slot in bookkeeping.
            if data_q in self.qubit_manager.in_use_data[p_root]:
                self.qubit_manager.in_use_data[p_root].pop(data_q, None)
            if data_q not in self.qubit_manager.free_data[p_root]:
                self.qubit_manager.free_data[p_root].append(data_q)
            self.qubit_manager.log_to_phys_idx[root_q] = root_comm
            self.creg_manager.release_cbit(cbit)
            root_phys = root_comm
        else:
            root_phys = self.qubit_manager.log_to_phys_idx[root_q]

        if not bfs_edges:
            # Single-partition group — no fan-out needed.
            self.group_state[root_q] = {
                'init_p_root'  : p_root,
                'final_p_root' : final_p,
                'linked_qubits': {p_root: root_phys},
            }
            return

        # Allocate comm qubits for each tree edge.
        edges_to_comms = self._allocate_tree_comms(bfs_edges)

        # Build and append the named FanOut instruction.
        label = f"FanOut[q{root_q}:p{p_root}→{sorted(targets)}]"
        fan_out_instr = _make_fan_out_instruction(
            root_q, p_root, targets, bfs_edges, label, self.compress_corrections
        )
        phys_q, phys_c = self._fan_out_qubit_lists(root_phys, bfs_edges, edges_to_comms)
        self.qc.append(fan_out_instr, phys_q, phys_c)

        if self.compress_corrections:
            self._apply_fan_out_corrections(p_root, bfs_edges, edges_to_comms, phys_c)
        else:
            for cbit in phys_c:
                self.creg_manager.release_cbit(cbit)

        # Release parent-side comm qubits (measured+reset inside instruction).
        for p0, p1 in bfs_edges:
            comm_parent, _ = edges_to_comms[(p0, p1)]
            self.comm_manager.release_comm_qubit(p0, comm_parent)

        # Track linked qubits: root + every child node with its comm qubit.
        linked: dict[int, Qubit] = {p_root: root_phys}
        for p0, p1 in bfs_edges:
            _, comm_child = edges_to_comms[(p0, p1)]
            linked[p1] = comm_child

        # For tree nodes not in targets (intermediates), they will be handled by
        # ImmediateFanIn; their entries stay in linked until that event fires.
        self.group_state[root_q] = {
            'init_p_root'  : p_root,
            'final_p_root' : final_p,
            'linked_qubits': linked,
        }

    # ------------------------------------------------------------------
    # ImmediateFanIn
    # ------------------------------------------------------------------

    def _apply_immediate_fan_in(self, event: ImmediateFanIn) -> None:
        """Apply ending process for each intermediate (routing-only) partition."""
        root_q = event.root_qubit
        group  = self.group_state.get(root_q, {})
        linked = group.get('linked_qubits', {})
        init_p = group.get('init_p_root', event.root_qubit)

        for p_imm in event.intermediate_partitions:
            nearest = event.nearest_targets.get(p_imm, init_p)
            if nearest not in linked:
                nearest = init_p

            target_phys = linked[nearest]
            source_comm = linked.get(p_imm)
            if source_comm is None:
                continue

            label = f"ImmFanIn[q{root_q}:p{p_imm}→p{nearest}]"
            ending = _make_ending_process_instruction(label, self.compress_corrections)
            cbit = self.creg_manager.allocate_cbit()
            self.qc.append(ending, [target_phys, source_comm], [cbit])
            if self.compress_corrections:
                with self.qc.if_test(qexpr.lift(cbit)):
                    self.qc.z(target_phys)
            self.comm_manager.release_comm_qubit(p_imm, source_comm)
            self.creg_manager.release_cbit(cbit)
            del linked[p_imm]

    # ------------------------------------------------------------------
    # LinkedGate
    # ------------------------------------------------------------------

    def _apply_linked_gate(self, event: LinkedGate) -> None:
        """Apply a gate between root's linked copy and a local receiver qubit."""
        root_q = event.root_qubit
        tgt_q  = event.target_qubit
        p_tgt  = event.target_partition

        group  = self.group_state.get(root_q, {})
        linked = group.get('linked_qubits', {})

        # Find the partition that root and target share a comm qubit.
        if p_tgt in linked:
            root_phys = linked[p_tgt]
        else:
            root_phys = self.qubit_manager.log_to_phys_idx[root_q]

        tgt_phys = self.qubit_manager.log_to_phys_idx[tgt_q]
        self._apply_two_qubit(event.gate_name, event.params, root_phys, tgt_phys)

    # ------------------------------------------------------------------
    # FanIn
    # ------------------------------------------------------------------

    def _apply_fan_in(self, event: FanIn) -> None:
        """Apply ending process to fan in one partition's copy."""
        root_q  = event.root_qubit
        p_src   = event.source_partition
        p_tgt   = event.target_partition

        group  = self.group_state.get(root_q)
        if group is None:
            return

        linked    = group['linked_qubits']
        init_p    = group['init_p_root']
        final_p   = group['final_p_root']

        source_comm = linked.get(p_src)
        if source_comm is None:
            return

        # Determine the target physical qubit (where the state is absorbed).
        if p_tgt in linked:
            target_phys = linked[p_tgt]
        else:
            target_phys = self.qubit_manager.log_to_phys_idx.get(root_q)

        if p_src == p_tgt:
            # Root stays in the same partition — no ending process needed.
            # Just release the comm qubit if the root was on one.
            if source_comm != target_phys:
                # Nested: source_comm holds the state at final_p; transfer to data.
                self._settle_to_data(root_q, final_p, source_comm)
                del linked[p_src]
            return

        # Standard ending process: H+measure on source_comm, Z correction on target.
        label = f"FanIn[q{root_q}:p{p_src}→p{p_tgt}]"
        ending = _make_ending_process_instruction(label, self.compress_corrections)
        cbit = self.creg_manager.allocate_cbit()
        self.qc.append(ending, [target_phys, source_comm], [cbit])
        if self.compress_corrections:
            with self.qc.if_test(qexpr.lift(cbit)):
                self.qc.z(target_phys)
        self.comm_manager.release_comm_qubit(p_src, source_comm)
        self.creg_manager.release_cbit(cbit)
        del linked[p_src]

        # Nested root case: root ends up at final_p (different from init_p).
        # After this FanIn, the comm qubit at final_p holds the root state.
        # Settle it onto a data qubit.
        if p_src == init_p and final_p != init_p and p_tgt == final_p:
            target_comm = linked.get(final_p)
            if target_comm is not None:
                self._settle_to_data(root_q, final_p, target_comm)
                del linked[final_p]

        # Close the group once all linked partitions have been fanned in.
        if not linked or linked == {final_p: self.qubit_manager.log_to_phys_idx.get(root_q)}:
            self.group_state.pop(root_q, None)

    def _apply_joint_fan_in(self, event: JointFanIn) -> None:
        """Apply the joint fan-in as a single named instruction.

        All die comm qubits are collected, then a single JointFanIn instruction
        (H + measure + reset per source) is appended.  If compress_corrections
        is True a single if_test(XOR parity) Z correction is applied outside;
        otherwise sequential Z.c_if corrections are included inside.
        """
        root_q = event.root_qubit
        p_tgt  = event.target_partition

        group  = self.group_state.get(root_q)
        if group is None:
            return
        linked = group['linked_qubits']

        if p_tgt in linked:
            target_phys = linked[p_tgt]
        else:
            target_phys = self.qubit_manager.log_to_phys_idx.get(root_q)

        # Collect valid source comm qubits in order.
        source_comms: list[Qubit] = []
        valid_sources: list[int] = []
        for p_src in event.source_partitions:
            comm = linked.get(p_src)
            if comm is None:
                continue
            source_comms.append(comm)
            valid_sources.append(p_src)

        if not source_comms:
            return

        sources_str = valid_sources
        label = f"JointFanIn[q{root_q}:p{sources_str}→p{p_tgt}]"
        instr = _make_joint_fan_in_instruction(
            label, len(source_comms), self.compress_corrections
        )
        phys_c = [self.creg_manager.allocate_cbit() for _ in source_comms]
        self.qc.append(instr, [target_phys] + source_comms, phys_c)

        if self.compress_corrections:
            parity = qexpr.lift(phys_c[0])
            for cb in phys_c[1:]:
                parity = qexpr.bit_xor(parity, qexpr.lift(cb))
            with self.qc.if_test(parity):
                self.qc.z(target_phys)

        for cbit in phys_c:
            self.creg_manager.release_cbit(cbit)
        for p_src, comm in zip(valid_sources, source_comms):
            self.comm_manager.release_comm_qubit(p_src, comm)
            del linked[p_src]

        if not linked or linked == {p_tgt: self.qubit_manager.log_to_phys_idx.get(root_q)}:
            self.group_state.pop(root_q, None)

    def _settle_to_data(self, root_q: int, partition: int, comm_q: Qubit) -> None:
        """Move the root qubit's state from a comm qubit into a fresh data qubit."""
        try:
            new_data = self.qubit_manager.allocate_data_qubit(partition)
        except Exception:
            # No free slot — leave state on the comm qubit.
            self._mark_pending_settle(root_q, partition, comm_q)
            return
        label = f"LocalTransfer[q{root_q}:settle→p{partition}]"
        lt_instr = _make_local_transfer_instruction(label, self.compress_corrections)
        cbit = self.creg_manager.allocate_cbit()
        self.qc.append(lt_instr, [comm_q, new_data], [cbit])
        if self.compress_corrections:
            with self.qc.if_test(qexpr.lift(cbit)):
                self.qc.z(new_data)
        self.qubit_manager.assign_to_physical(partition, new_data, root_q)
        self.comm_manager.release_comm_qubit(partition, comm_q)
        self.creg_manager.release_cbit(cbit)
        self.pending_settles.pop(root_q, None)

    def _mark_pending_settle(self, q_log: int, partition: int, comm_q: Qubit) -> None:
        """Track a deferred local move from a communication qubit to data space."""
        self.qubit_manager.log_to_phys_idx[q_log] = comm_q
        self.pending_settles[q_log] = (partition, comm_q)

    def _retry_pending_settles(self, only_qubits: set[int] | None = None) -> None:
        """Retry deferred comm->data moves after events that may free data slots."""
        if not self.pending_settles:
            return
        for q_log, (partition, comm_q) in list(self.pending_settles.items()):
            if only_qubits is not None and q_log not in only_qubits:
                continue
            # Skip stale records that no longer match current placement.
            if self.qubit_manager.log_to_phys_idx.get(q_log) != comm_q:
                self.pending_settles.pop(q_log, None)
                continue
            self._settle_to_data(q_log, partition, comm_q)

    # ------------------------------------------------------------------
    # Tree and qubit-list helpers
    # ------------------------------------------------------------------

    def _get_directed_tree(self, p_root: int, target_partitions: list[int]) -> nx.DiGraph:
        """Return a BFS-directed tree from p_root to cover target_partitions."""
        undirected = self.network.get_full_tree(
            root_p=p_root, target_partitions=target_partitions
        )
        directed = nx.DiGraph()
        visited: set[int] = set()
        queue: deque[int] = deque([p_root])
        while queue:
            parent = queue.popleft()
            visited.add(parent)
            for child in undirected.neighbors(parent):
                if child not in visited:
                    directed.add_edge(parent, child)
                    queue.append(child)
        return directed

    def _allocate_tree_comms(
        self, bfs_edges: list[tuple[int, int]]
    ) -> dict[tuple[int, int], tuple[Qubit, Qubit]]:
        """Allocate comm qubit pairs for each BFS tree edge."""
        return {
            (p0, p1): (
                self.comm_manager.find_comm_idx(p0),
                self.comm_manager.find_comm_idx(p1),
            )
            for p0, p1 in bfs_edges
        }

    def _fan_out_qubit_lists(
        self,
        root_phys: Qubit,
        bfs_edges: list[tuple[int, int]],
        edges_to_comms: dict[tuple[int, int], tuple[Qubit, Qubit]],
    ) -> tuple[list[Qubit], list[Clbit]]:
        """Build [physical_qubits] and [classical_bits] for the FanOut instruction.

        Layout matches _make_fan_out_instruction:
          phys_q[0]      → root qubit
          phys_q[2*i+1]  → parent-side comm qubit for edge i
          phys_q[2*i+2]  → child-side  comm qubit for edge i
          phys_c[i]      → measurement outcome for edge i (XOR feed-forward)
        """
        phys_q: list[Qubit] = [root_phys]
        phys_c: list[Clbit] = []
        for p0, p1 in bfs_edges:
            comm_parent, comm_child = edges_to_comms[(p0, p1)]
            phys_q.extend([comm_parent, comm_child])
            phys_c.append(self.creg_manager.allocate_cbit())
        return phys_q, phys_c

    def _apply_fan_out_corrections(
        self,
        p_root: int,
        bfs_edges: list[tuple[int, int]],
        edges_to_comms: dict[tuple[int, int], tuple[Qubit, Qubit]],
        phys_c: list[Clbit],
    ) -> None:
        """Apply one X correction per target child using XOR parity of phys_c bits.

        phys_c[i] is the measurement outcome for bfs_edges[i].  A child at depth k
        needs X conditioned on the XOR of the k outcomes along its path from root.
        Cbits are released after corrections are emitted.
        """
        # Build the accumulated cbit list for each partition node.
        node_path_cbits: dict[int, list[Clbit]] = {p_root: []}
        for i, (p0, p1) in enumerate(bfs_edges):
            node_path_cbits[p1] = node_path_cbits[p0] + [phys_c[i]]

        for p0, p1 in bfs_edges:
            _, comm_child = edges_to_comms[(p0, p1)]
            cbits = node_path_cbits[p1]
            if not cbits:
                continue
            parity = qexpr.lift(cbits[0])
            for cb in cbits[1:]:
                parity = qexpr.bit_xor(parity, qexpr.lift(cb))
            with self.qc.if_test(parity):
                self.qc.x(comm_child)

        for cbit in phys_c:
            self.creg_manager.release_cbit(cbit)

    def _node_comm_qubit(
        self,
        node: int,
        bfs_edges: list[tuple[int, int]],
        edges_to_comms: dict[tuple[int, int], tuple[Qubit, Qubit]],
    ) -> Qubit:
        """Return the child-side comm qubit that corresponds to *node* in the tree."""
        for p0, p1 in bfs_edges:
            if p1 == node:
                return edges_to_comms[(p0, p1)][1]
        raise ValueError(f"Node {node} not found as a child in BFS edges.")
