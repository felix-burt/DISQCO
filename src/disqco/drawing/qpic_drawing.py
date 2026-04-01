"""Render a DistributedCircuit as a QPIC diagram.

Assumes qpic has lowercase helper macros:
  control target starting
  control target ending
where macros expand to STARTING/ENDING with desired decoration.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wire_name(qubit: int, partition: int) -> str:
    return f"q{qubit}p{partition}"


def _wire_label(qubit: int, partition: int) -> str:
    # qpic auto-wraps W labels in $...$, so no delimiters needed here.
    return f"q_{{{qubit}}}^{{({partition})}}"


def _comm_wire_name(partition: int, slot: int) -> str:
    return f"c{partition}_{slot}"


def _comm_wire_label(partition: int, slot: int) -> str:
    return f"c_{{{partition},{slot}}}"


def _epr_wire_name(partition: int, slot: int) -> str:
    """Transient EPR-source (parent-side) comm wire used in the lowered rendering."""
    return f"ep{partition}_{slot}"


def _cbit_wire_name(slot: int) -> str:
    """Classical measurement-output wire used in the lowered rendering."""
    return f"cl_{slot}"


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class DistributedCircuitQPIC:
    """Convert a :class:`~disqco.circuit_extraction.distributed_circuit.DistributedCircuit`
    to a QPIC text diagram compatible with the qpic_dev fork.

    Parameters
    ----------
    circuit:
        The distributed circuit to render.
    show_params:
        If True, gate parameter values are included in gate labels.
    """

    def __init__(
        self,
        circuit: DistributedCircuit,
        show_params: bool = True,
    ) -> None:
        self.circuit = circuit
        self.show_params = show_params
        self.initial_assignment = {
            q: int(p)
            for q, p in enumerate(circuit.initial_assignment or [])
        }
        self.data_wire_pairs = self._data_wire_pairs()
        self.comm_slots_per_partition = self._comm_slot_budget()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_qpic_string(self) -> str:
        """Return the complete QPIC diagram as a string."""
        lines: list[str] = []

        # --- Wire declarations ---

        for partition in range(self.circuit.num_partitions):
            ordered_data = self._ordered_data_wires(partition)
            for q, p in ordered_data:
                name = _wire_name(q, p)
                lines.append(f"{name} W {_wire_label(q, p)}")
            for slot in range(self.comm_slots_per_partition.get(partition, 0)):
                lines.append(f"{_comm_wire_name(partition, slot)} W")
            # Keep a visible vertical gap between QPU blocks.
            if partition < self.circuit.num_partitions - 1:
                lines.append(f"gap_p{partition} W type=o")

        lines.append("")

        # --- Events ---
        active_comm: dict[tuple[int, int], int] = {}
        self.comm_alias: dict[tuple[int, int], str] = {}
        free_comm: dict[int, list[int]] = {
            p: list(range(self.comm_slots_per_partition.get(p, 0)))
            for p in range(self.circuit.num_partitions)
        }
        for event in self.circuit.events:
            rendered = self._render_event(event, active_comm, free_comm)
            if rendered:
                lines.extend(rendered)

        return "\n".join(lines) + "\n"

    def to_qpic_file(self, path: str | Path) -> None:
        """Write the QPIC diagram to *path*."""
        Path(path).write_text(self.to_qpic_string())

    # ------------------------------------------------------------------
    # Per-event rendering
    # ------------------------------------------------------------------

    def _render_event(
        self,
        event,
        active_comm,
        free_comm,
    ) -> list[str]:
        if isinstance(event, FanOut):
            return self._render_fan_out(event, active_comm, free_comm)
        elif isinstance(event, ImmediateFanIn):
            return self._render_immediate_fan_in(event, active_comm, free_comm)
        elif isinstance(event, FanIn):
            return self._render_fan_in(event, active_comm, free_comm)
        elif isinstance(event, JointFanIn):
            return self._render_joint_fan_in(event, active_comm, free_comm)
        elif isinstance(event, StateTransfer):
            return self._render_state_transfer(event)
        elif isinstance(event, LocalGate):
            return self._render_local_gate(event, active_comm)
        elif isinstance(event, LinkedGate):
            return self._render_linked_gate(event, active_comm)
        return []

    # --- FanOut ---

    def _render_fan_out(self, event: FanOut, active_comm, free_comm) -> list[str]:
        """Fan-out: allocate communication wires in each remote partition.

        Each node in the BFS tree emits a single joint starting operation
        covering all its children: ``parent child1 child2 ... starting``.
        """
        comment = (
            f"% FanOut q{event.root_qubit} p{event.root_partition}"
            f" -> targets {event.target_partitions}"
            f" intermediates {event.intermediate_partitions}"
        )
        new_wires = event.target_partitions + event.intermediate_partitions
        if not new_wires:
            return [comment]

        root_name = self._resolve_root_wire(event.root_qubit, event.root_partition, active_comm)

        # Allocate comm slots for every target/intermediate partition first so
        # all wire names are available before building tree operations.
        partition_to_wire: dict[int, str] = {event.root_partition: root_name}
        for p in new_wires:
            slot = self._allocate_comm_slot(event.root_qubit, p, active_comm, free_comm)
            draw_wire = _comm_wire_name(p, slot)
            alias_wire = _wire_name(event.root_qubit, p)
            if (event.root_qubit, p) in self.data_wire_pairs:
                self.comm_alias[(event.root_qubit, p)] = alias_wire
                draw_wire = alias_wire
            partition_to_wire[p] = draw_wire

        # Build one starting op per root-to-leaf path in the tree.
        # For a linear chain p0→p1→p2 this gives one op: "root c1 c2 starting".
        # For a branching tree each branch gets its own op.
        ops = []
        if event.path_tree and event.path_tree.number_of_edges() > 0:
            leaves = [
                n for n in event.path_tree.nodes()
                if event.path_tree.out_degree(n) == 0
            ]
            for leaf in leaves:
                try:
                    path = nx.shortest_path(
                        event.path_tree, source=event.root_partition, target=leaf
                    )
                except nx.NetworkXNoPath:
                    continue
                wire_path = [partition_to_wire[p] for p in path]
                ops.append(" ".join(wire_path) + " starting")

        return [comment] + ops

    # --- ImmediateFanIn ---

    def _render_immediate_fan_in(
        self,
        event: ImmediateFanIn,
        active_comm,
        free_comm,
    ) -> list[str]:
        """Immediate fan-in of routing-only intermediates right after FanOut.

        Intermediates that share the same nearest target are grouped into a
        single joint ending operation: ``target die1 die2 ... ending``.
        """
        if not event.intermediate_partitions:
            return []
        comment = (
            f"% ImmediateFanIn q{event.root_qubit}"
            f" intermediates {event.intermediate_partitions}"
        )
        # Group dying wires by their keep (nearest-target) wire.
        keep_to_dies: dict[str, list[str]] = defaultdict(list)
        for p_imm in event.intermediate_partitions:
            nearest = event.nearest_targets.get(p_imm)
            if nearest is None:
                continue
            keep_wire = self._resolve_root_wire(event.root_qubit, nearest, active_comm)
            die_wire = self._resolve_root_wire(event.root_qubit, p_imm, active_comm)
            keep_to_dies[keep_wire].append(die_wire)
            self._release_comm_slot(event.root_qubit, p_imm, active_comm, free_comm)
        if not keep_to_dies:
            return []
        ops = [
            f"{keep} {' '.join(dies)} ending"
            for keep, dies in keep_to_dies.items()
        ]
        return [comment] + ops

    # --- FanIn ---

    def _render_fan_in(
        self,
        event: FanIn,
        active_comm,
        free_comm,
    ) -> list[str]:
        """Fan-in: source copy dies, target wire absorbs state."""
        comment = (
            f"% FanIn q{event.root_qubit}"
            f" p{event.source_partition}->p{event.target_partition}"
        )
        keep_wire = self._resolve_root_wire(event.root_qubit, event.target_partition, active_comm)
        die_wire = self._resolve_root_wire(event.root_qubit, event.source_partition, active_comm)
        if event.source_partition == event.target_partition:
            return [comment]

        line = self._ending_op(keep_wire, die_wire)
        self._release_comm_slot(
            event.root_qubit, event.source_partition, active_comm, free_comm
        )
        return [comment, line]

    # --- JointFanIn ---

    def _render_joint_fan_in(
        self,
        event: JointFanIn,
        active_comm,
        free_comm,
    ) -> list[str]:
        """Joint fan-in: all die wires end into the keep wire in one operation."""
        sources_str = event.source_partitions
        comment = (
            f"% JointFanIn q{event.root_qubit}"
            f" p{sources_str}->p{event.target_partition}"
        )
        keep_wire = self._resolve_root_wire(event.root_qubit, event.target_partition, active_comm)
        die_wires = []
        for p_src in event.source_partitions:
            die_wires.append(self._resolve_root_wire(event.root_qubit, p_src, active_comm))
            self._release_comm_slot(event.root_qubit, p_src, active_comm, free_comm)
        # Render as a chained ENDING with all die wires.
        line = f"{keep_wire} {' '.join(die_wires)} ending"
        return [comment, line]

    # --- StateTransfer ---

    def _render_state_transfer(self, event: StateTransfer) -> list[str]:
        """State transfer: qubit moves completely from source to target partition.

        Remote transfer remains a teleportation-style starting/ending pair.
        """
        src_name = _wire_name(event.qubit, event.source_partition)
        dst_name = _wire_name(event.qubit, event.target_partition)
        comment = (
            f"% StateTransfer q{event.qubit}"
            f" p{event.source_partition}->p{event.target_partition}"
        )
        return [
            comment,
            f"{src_name} {dst_name} starting",
            f"{dst_name} {src_name} ending",
        ]

    # --- LocalGate ---

    def _render_local_gate(self, event: LocalGate, active_comm) -> list[str]:
        if len(event.qubits) == 1:
            q, p = event.qubits[0]
            return [self._single_qubit_line(_wire_name(q, p), event.gate_name, event.params)]
        elif len(event.qubits) == 2:
            (q0, p0), (q1, p1) = event.qubits
            w0, w1 = _wire_name(q0, p0), _wire_name(q1, p1)
            return [self._two_qubit_line(w0, w1, event.gate_name, event.params)]
        else:
            wire_names = " ".join(_wire_name(q, p) for q, p in event.qubits)
            label = self._gate_label(event.gate_name, event.params)
            return [f"{wire_names} G {label}"]

    # --- LinkedGate ---

    def _render_linked_gate(self, event: LinkedGate, active_comm) -> list[str]:
        w_root = self._resolve_root_wire(
            event.root_qubit, event.root_partition, active_comm
        )
        w_tgt = _wire_name(event.target_qubit, event.target_partition)
        return [self._two_qubit_line(w_root, w_tgt, event.gate_name, event.params)]

    # ------------------------------------------------------------------
    # Wire classification
    # ------------------------------------------------------------------

    def _mid_circuit_wires(self) -> set[tuple[int, int]]:
        """Return wires that first appear mid-circuit via FanOut or StateTransfer."""
        mid: set[tuple[int, int]] = set()
        for event in self.circuit.events:
            if isinstance(event, FanOut):
                for p in event.target_partitions + event.intermediate_partitions:
                    mid.add((event.root_qubit, p))
            elif isinstance(event, StateTransfer):
                mid.add((event.qubit, event.target_partition))
        return mid

    def _data_wire_pairs(self) -> set[tuple[int, int]]:
        """Pairs rendered as data wires rather than communication wires."""
        pairs = set((q, p) for q, p in self.initial_assignment.items())
        for event in self.circuit.events:
            if isinstance(event, LocalGate):
                pairs.update((int(q), int(p)) for q, p in event.qubits)
            elif isinstance(event, StateTransfer):
                pairs.add((event.qubit, event.source_partition))
                pairs.add((event.qubit, event.target_partition))
        return pairs

    def _ordered_data_wires(self, partition: int) -> list[tuple[int, int]]:
        ordered = [
            wire
            for wire in self.circuit.wire_order
            if wire[1] == partition and wire in self.data_wire_pairs
        ]
        extras = sorted(
            wire for wire in self.data_wire_pairs
            if wire[1] == partition and wire not in ordered
        )
        return ordered + extras

    def _comm_slot_budget(self) -> dict[int, int]:
        """Maximum simultaneous fan-out copies per partition."""
        active: dict[tuple[int, int], bool] = {}
        counts = {p: 0 for p in range(self.circuit.num_partitions)}
        maxima = {p: 0 for p in range(self.circuit.num_partitions)}

        for event in self.circuit.events:
            if isinstance(event, FanOut):
                for p in event.target_partitions + event.intermediate_partitions:
                    key = (event.root_qubit, p)
                    if key in active:
                        continue
                    active[key] = True
                    counts[p] += 1
                    maxima[p] = max(maxima[p], counts[p])
            elif isinstance(event, ImmediateFanIn):
                for p in event.intermediate_partitions:
                    key = (event.root_qubit, p)
                    if key in active:
                        del active[key]
                        counts[p] -= 1
            elif isinstance(event, FanIn):
                key = (event.root_qubit, event.source_partition)
                if key in active:
                    del active[key]
                    counts[event.source_partition] -= 1
            elif isinstance(event, JointFanIn):
                for p in event.source_partitions:
                    key = (event.root_qubit, p)
                    if key in active:
                        del active[key]
                        counts[p] -= 1

        return maxima

    def _wire_ranks(self) -> dict[str, int]:
        ranks: dict[str, int] = {}
        index = 0
        for partition in range(self.circuit.num_partitions):
            for q, p in self._ordered_data_wires(partition):
                ranks[_wire_name(q, p)] = index
                index += 1
            for slot in range(self.comm_slots_per_partition.get(partition, 0)):
                ranks[_comm_wire_name(partition, slot)] = index
                index += 1
        return ranks

    def _allocate_comm_slot(self, root_qubit: int, partition: int, active_comm, free_comm) -> int:
        key = (root_qubit, partition)
        if key in active_comm:
            return active_comm[key]
        slots = free_comm[partition]
        if not slots:
            slot = self.comm_slots_per_partition.get(partition, 0)
            self.comm_slots_per_partition[partition] = slot + 1
        else:
            slot = slots.pop(0)
        active_comm[key] = slot
        return slot

    def _release_comm_slot(self, root_qubit: int, partition: int, active_comm, free_comm) -> None:
        key = (root_qubit, partition)
        slot = active_comm.pop(key, None)
        if slot is None:
            return
        self.comm_alias.pop(key, None)
        free_comm[partition].insert(0, slot)

    def _resolve_root_wire(self, root_qubit: int, partition: int, active_comm) -> str:
        alias = self.comm_alias.get((root_qubit, partition))
        if alias is not None:
            return alias
        slot = active_comm.get((root_qubit, partition))
        if slot is not None:
            return _comm_wire_name(partition, slot)
        return _wire_name(root_qubit, partition)

    def _ending_op(self, keep_wire: str, die_wire: str) -> str:
        """Render ending using qpic lowercase macro syntax."""
        return f"{keep_wire} {die_wire} ending"


    # ------------------------------------------------------------------
    # Gate-line builders
    # ------------------------------------------------------------------

    def _single_qubit_line(self, wire: str, name: str, params: list) -> str:
        lname = name.lower()
        upper = name.upper()
        if upper in ('H', 'X', 'Y', 'Z', 'S', 'T'):
            return f"{wire} {upper}"
        if lname in ('u', 'u3'):
            return f"{wire} G $U$"
        label = self._gate_label(name, params)
        return f"{wire} G {label}"

    def _two_qubit_line(self, w_ctrl: str, w_tgt: str, name: str, params: list) -> str:
        lname = name.lower()
        if lname == 'cx':
            return f"+{w_tgt} {w_ctrl}"       # CNOT: ⊕ on target
        elif lname == 'cz':
            return f"{w_ctrl} {w_tgt}"         # CZ: filled dot on both
        elif lname in ('cp', 'cphase'):
            return f"{w_ctrl} {w_tgt} cphase"
        label = self._gate_label(name, params)
        return f"{w_ctrl} {w_tgt} G {label}"

    def _gate_label(self, name: str, params: list) -> str:
        if not params or not self.show_params:
            return f"${name}$"
        param_str = ",".join(
            f"{p:.3g}" if isinstance(p, float) else str(p) for p in params
        )
        return f"${name}({param_str})$"

    # ------------------------------------------------------------------
    # Lowered rendering (physical-level qpic)
    # ------------------------------------------------------------------

    def _lowered_epr_budget(self) -> dict[int, int]:
        """Max simultaneous parent-side EPR wires needed per partition.

        Equals the max out-degree of each partition node across all FanOut
        path_trees (all EPR pairs for a single fan-out are generated in
        parallel before the CX-measure chain starts).
        """
        maxima: dict[int, int] = {p: 0 for p in range(self.circuit.num_partitions)}
        for event in self.circuit.events:
            if isinstance(event, FanOut) and event.path_tree:
                for node in event.path_tree.nodes():
                    out_deg = event.path_tree.out_degree(node)
                    maxima[node] = max(maxima.get(node, 0), out_deg)
        return maxima

    def to_qpic_string_lowered(self) -> str:
        """Return the lowered QPIC diagram as a string.

        Renders the physical-level circuit using native qpic cwire flow:
        - ``EPR`` custom instruction for entanglement generation
        - After ``wire M``, the qwire implicitly becomes a cwire
        - XOR on cwires: ``+cwire_dst cwire_src``
        - Classically-controlled gates: ``gate c:cwire`` (keep) or ``c:ocwire`` (terminate)
        - FanIn: H + M on die wire (→ cwire), Z correction on keep wire via ``c:o``
        """
        lines: list[str] = []

        epr_budget = self._lowered_epr_budget()

        # --- Wire declarations ---
        for partition in range(self.circuit.num_partitions):
            for q, p in self._ordered_data_wires(partition):
                lines.append(f"{_wire_name(q, p)} W {_wire_label(q, p)}")
            for slot in range(self.comm_slots_per_partition.get(partition, 0)):
                lines.append(f"{_comm_wire_name(partition, slot)} W")
            for slot in range(epr_budget.get(partition, 0)):
                lines.append(f"{_epr_wire_name(partition, slot)} W")
            if partition < self.circuit.num_partitions - 1:
                lines.append(f"gap_p{partition} W type=o")

        lines.append("")

        # --- Events ---
        self.comm_alias: dict[tuple[int, int], str] = {}
        active_comm: dict[tuple[int, int], int] = {}
        free_comm: dict[int, list[int]] = {
            p: list(range(self.comm_slots_per_partition.get(p, 0)))
            for p in range(self.circuit.num_partitions)
        }
        free_epr: dict[int, list[int]] = {
            p: list(range(epr_budget.get(p, 0)))
            for p in range(self.circuit.num_partitions)
        }

        for event in self.circuit.events:
            rendered = self._render_event_lowered(event, active_comm, free_comm, free_epr)
            if rendered:
                lines.extend(rendered)

        return "\n".join(lines) + "\n"

    def to_qpic_file_lowered(self, path: str | Path) -> None:
        """Write the lowered QPIC diagram to *path*."""
        Path(path).write_text(self.to_qpic_string_lowered())

    def _render_event_lowered(self, event, active_comm, free_comm, free_epr) -> list[str]:
        if isinstance(event, FanOut):
            return self._render_fan_out_lowered(event, active_comm, free_comm, free_epr)
        elif isinstance(event, ImmediateFanIn):
            return self._render_immediate_fan_in_lowered(event, active_comm, free_comm)
        elif isinstance(event, FanIn):
            return self._render_fan_in_lowered(event, active_comm, free_comm)
        elif isinstance(event, JointFanIn):
            return self._render_joint_fan_in_lowered(event, active_comm, free_comm)
        elif isinstance(event, StateTransfer):
            return self._render_state_transfer(event)
        elif isinstance(event, LocalGate):
            return self._render_local_gate(event, active_comm)
        elif isinstance(event, LinkedGate):
            return self._render_linked_gate(event, active_comm)
        return []

    def _render_fan_out_lowered(
        self,
        event: FanOut,
        active_comm,
        free_comm,
        free_epr,
    ) -> list[str]:
        """Render FanOut at the physical level.

        Structure mirrors the Qiskit lowering:
          Phase 1 — generate all EPR pairs along the BFS tree edges
          Phase 2 — CX-measure chain in BFS order; after ``M`` each ep wire
                    becomes a cwire carrying the measurement outcome
          Phase 3 — XOR-parity accumulation (``+ep_child ep_parent_accum``)
                    then classically-controlled X corrections (``X c:ep_wire``)

        ep wires are not explicitly terminated after corrections so they remain
        available as cwires for further XOR into deeper tree levels.
        """
        comment = (
            f"% FanOut (lowered) q{event.root_qubit} p{event.root_partition}"
            f" -> targets {event.target_partitions}"
            f" intermediates {event.intermediate_partitions}"
        )
        if not event.path_tree or event.path_tree.number_of_edges() == 0:
            return [comment]

        bfs_edges = list(nx.bfs_edges(event.path_tree, source=event.root_partition))

        root_wire = self._resolve_root_wire(
            event.root_qubit, event.root_partition, active_comm
        )

        # Allocate child-side comm wire for each target/intermediate partition.
        partition_to_comm_wire: dict[int, str] = {event.root_partition: root_wire}
        for p in event.target_partitions + event.intermediate_partitions:
            slot = self._allocate_comm_slot(event.root_qubit, p, active_comm, free_comm)
            draw_wire = _comm_wire_name(p, slot)
            alias_wire = _wire_name(event.root_qubit, p)
            if (event.root_qubit, p) in self.data_wire_pairs:
                self.comm_alias[(event.root_qubit, p)] = alias_wire
                draw_wire = alias_wire
            partition_to_comm_wire[p] = draw_wire

        # Allocate one parent-side EPR wire per BFS edge.
        edge_epr: dict[tuple[int, int], str] = {}
        edge_epr_slot: dict[tuple[int, int], int] = {}
        for p_parent, p_child in bfs_edges:
            epr_slot = free_epr[p_parent].pop(0)
            edge_epr[(p_parent, p_child)] = _epr_wire_name(p_parent, epr_slot)
            edge_epr_slot[(p_parent, p_child)] = epr_slot

        lines = [comment]

        # Phase 1: All EPR pairs.
        lines.append("% EPR pairs")
        for p_parent, p_child in bfs_edges:
            lines.append(
                f"{edge_epr[(p_parent, p_child)]} {partition_to_comm_wire[p_child]} epr"
            )

        # Phase 2: CX-measure chain.
        # node_in_wire tracks which wire holds the live copy at each partition node.
        lines.append("% CX-measure chain")
        node_in_wire: dict[int, str] = {event.root_partition: root_wire}
        for p_parent, p_child in bfs_edges:
            in_wire = node_in_wire[p_parent]
            epr_wire = edge_epr[(p_parent, p_child)]
            lines.append(f"+{epr_wire} {in_wire}")  # CNOT: copy → EPR parent side
            lines.append(f"{epr_wire} M")            # measure → ep wire becomes cwire
            node_in_wire[p_child] = partition_to_comm_wire[p_child]

        # Release EPR slots (each ep wire has been measured; its cwire will trail).
        for p_parent, p_child in bfs_edges:
            free_epr[p_parent].insert(0, edge_epr_slot[(p_parent, p_child)])

        # Phase 3: XOR-parity accumulation then corrections.
        # Classical XOR accumulations go first (before any quantum corrections).
        # accumulated_parity[p] = the ep cwire holding parity(root → p).
        # Root carries no parity (None sentinel).
        lines.append("% X corrections")
        accumulated_parity: dict[int, str | None] = {event.root_partition: None}

        # Phase 3a: purely classical XOR accumulations.
        for p_parent, p_child in bfs_edges:
            ep_wire = edge_epr[(p_parent, p_child)]
            parent_accum = accumulated_parity.get(p_parent)
            if parent_accum is not None:
                lines.append(f"+{ep_wire} {parent_accum}")
            accumulated_parity[p_child] = ep_wire

        # Phase 3b: quantum corrections — each ep cwire is terminated with :owire.
        for p_parent, p_child in bfs_edges:
            ep_wire = edge_epr[(p_parent, p_child)]
            child_wire = partition_to_comm_wire[p_child]
            lines.append(f"+{child_wire} {ep_wire}:owire")

        return lines

    def _render_fan_in_lowered(self, event: FanIn, active_comm, free_comm) -> list[str]:
        """Render FanIn at the physical level.

        H + M on die wire (die wire becomes cwire); Z correction on keep wire
        via ``c:o`` which simultaneously terminates the cwire.
        """
        comment = (
            f"% FanIn (lowered) q{event.root_qubit}"
            f" p{event.source_partition}->p{event.target_partition}"
        )
        keep_wire = self._resolve_root_wire(
            event.root_qubit, event.target_partition, active_comm
        )
        die_wire = self._resolve_root_wire(
            event.root_qubit, event.source_partition, active_comm
        )
        if event.source_partition == event.target_partition:
            return [comment]

        self._release_comm_slot(
            event.root_qubit, event.source_partition, active_comm, free_comm
        )
        return [
            comment,
            f"{die_wire} H",
            f"{die_wire} M",                          # die_wire becomes cwire
            f"{keep_wire} G $Z$ {die_wire}:owire",    # Z correction; terminate cwire
        ]

    def _render_immediate_fan_in_lowered(
        self,
        event: ImmediateFanIn,
        active_comm,
        free_comm,
    ) -> list[str]:
        """Render ImmediateFanIn at the physical level."""
        if not event.intermediate_partitions:
            return []
        comment = (
            f"% ImmediateFanIn (lowered) q{event.root_qubit}"
            f" intermediates {event.intermediate_partitions}"
        )
        lines = [comment]
        for p_imm in event.intermediate_partitions:
            nearest = event.nearest_targets.get(p_imm)
            if nearest is None:
                continue
            keep_wire = self._resolve_root_wire(event.root_qubit, nearest, active_comm)
            die_wire = self._resolve_root_wire(event.root_qubit, p_imm, active_comm)
            self._release_comm_slot(event.root_qubit, p_imm, active_comm, free_comm)
            lines.append(f"{die_wire} H")
            lines.append(f"{die_wire} M")
            lines.append(f"{keep_wire} G $Z$ {die_wire}:owire")
        return lines

    def _render_joint_fan_in_lowered(
        self,
        event: JointFanIn,
        active_comm,
        free_comm,
    ) -> list[str]:
        """Render JointFanIn at the physical level.

        H + M on each die wire (each becomes a cwire), XOR all outcomes into
        the first cwire, then a single Z correction on the keep wire.
        """
        comment = (
            f"% JointFanIn (lowered) q{event.root_qubit}"
            f" p{event.source_partitions}->p{event.target_partition}"
        )
        keep_wire = self._resolve_root_wire(
            event.root_qubit, event.target_partition, active_comm
        )
        die_wires = []
        for p_src in event.source_partitions:
            die_wires.append(self._resolve_root_wire(event.root_qubit, p_src, active_comm))
            self._release_comm_slot(event.root_qubit, p_src, active_comm, free_comm)

        lines = [comment]
        # H + measure each die wire.
        for dw in die_wires:
            lines.append(f"{dw} H")
            lines.append(f"{dw} M")
        # XOR all subsequent die cwires into the first.
        for dw in die_wires[1:]:
            lines.append(f"+{die_wires[0]} {dw}:owire")
        # Single Z correction conditioned on the accumulated parity.
        lines.append(f"{keep_wire} G $Z$ {die_wires[0]}:owire")
        return lines
