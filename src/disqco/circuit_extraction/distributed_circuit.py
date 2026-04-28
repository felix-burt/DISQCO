"""Intermediate distributed circuit representation.

Sits between the partitioned hypergraph and any concrete output format (Qiskit,
QASM, QPIC, ...).  Events capture fan-out / fan-in structure along network paths
without committing to EPR generation or LOCC details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from disqco import QuantumNetwork


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

@dataclass
class LocalGate:
    """A gate that acts entirely within a single partition."""
    gate_name: str
    params: list
    # Each entry is (logical_qubit_index, partition).
    qubits: list[tuple[int, int]]
    time: int


@dataclass
class StateTransfer:
    """Point-to-point move of a qubit between partitions (outside any group)."""
    qubit: int
    source_partition: int
    target_partition: int
    time: int


@dataclass
class FanOut:
    """Fan the root qubit state out along *path_tree* to target and intermediate nodes.

    This is a single combined operation.  The path_tree is a directed tree rooted at
    root_partition.  After this event:
      - root_partition still holds the original state
      - every node in target_partitions holds a linked copy
      - every node in intermediate_partitions holds a transient copy (will be fanned
        in immediately by the accompanying ImmediateFanIn event)

    final_root_partition: where the root qubit ends up after all FanIn events for this
        group complete.  Equals root_partition for non-nested groups; differs when the
        root qubit itself teleports to a new partition during the group (nested case).
    """
    root_qubit: int
    root_partition: int
    path_tree: nx.DiGraph
    target_partitions: list[int]
    intermediate_partitions: list[int]
    time: int
    final_root_partition: int = -1   # -1 → same as root_partition (non-nested)

    def __post_init__(self) -> None:
        if self.final_root_partition == -1:
            self.final_root_partition = self.root_partition


@dataclass
class ImmediateFanIn:
    """Fan in the intermediate (routing-only) nodes immediately after a FanOut.

    intermediate_partitions holds copies only used for routing; they should be
    disentangled right after the FanOut so only root and target copies remain live.
    nearest_targets maps each intermediate partition to the closest target or root
    partition in the path tree (used for the ending-process direction).
    """
    root_qubit: int
    intermediate_partitions: list[int]
    nearest_targets: dict[int, int]
    time: int


@dataclass
class LinkedGate:
    """A gate applied between a live root copy and a target qubit.

    root_partition denotes the partition of the root copy that participates in the
    gate. For teleported gates this is usually the same as target_partition.
    """
    gate_name: str
    params: list
    root_qubit: int
    root_partition: int
    target_qubit: int
    target_partition: int
    time: int


@dataclass
class FanIn:
    """Fan in one live copy of root_qubit when it is no longer needed.

    The copy at source_partition is disentangled; the state is absorbed into
    target_partition (which may equal the original root or a new location for
    nested teleportations).
    """
    root_qubit: int
    source_partition: int
    target_partition: int
    time: int


@dataclass
class JointFanIn:
    """Multiple fan-in sources merged into a single event.

    All copies listed in source_partitions are disentangled simultaneously and
    their measurement outcomes are XOR'd together, producing a single
    classically-controlled Z correction on the target partition rather than one
    per source.  Compatible FanIn events (same root_qubit and target_partition)
    can always be safely merged because by definition no further operations use
    a copy after its individual FanIn would have fired.
    """
    root_qubit: int
    source_partitions: list[int]
    target_partition: int
    time: int


# ---------------------------------------------------------------------------
# DistributedCircuit
# ---------------------------------------------------------------------------

DistributedCircuitEvent = (
    LocalGate | StateTransfer | FanOut | ImmediateFanIn | LinkedGate | FanIn | JointFanIn
)


class DistributedCircuit:
    """Abstract representation of a distributed quantum circuit.

    Stores an ordered sequence of events at the logical level: fan-outs,
    fan-ins, state transfers, and gates.  EPR generation and classical
    correction details are deliberately absent and left to the backend
    that lowers this to a concrete representation.
    """

    def __init__(
        self,
        num_qubits: int,
        num_partitions: int,
        network: "QuantumNetwork",
    ) -> None:
        self.num_qubits = num_qubits
        self.num_partitions = num_partitions
        self.network = network
        self.events: list[DistributedCircuitEvent] = []
        self.initial_assignment: list[int] | None = None
        self.wire_order: list[tuple[int, int]] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_event(self, event: DistributedCircuitEvent) -> None:
        self.events.append(event)

    def merge_fan_ins(self, adjacent_time_only: bool = False) -> None:
        """Post-process: merge compatible FanIn events into JointFanIn events.

        Only FanIn events that are adjacent in a root-qubit's own event timeline
        and share the same (root_qubit, target_partition) are merged. Self-loops
        (source == target) are excluded.

        If ``adjacent_time_only`` is True, then merged FanIn events must also be
        on consecutive integer time steps.

        "Adjacent in qubit timeline" means no other event touching that root
        qubit appears between the merged FanIn events. Events on other qubits are
        ignored for adjacency.
        """
        remove: set[int] = set()
        replacements: dict[int, JointFanIn] = {}

        root_qubits = {
            event.root_qubit
            for event in self.events
            if isinstance(event, (FanOut, ImmediateFanIn, FanIn, JointFanIn, LinkedGate))
        }

        for root_q in root_qubits:
            touches = [
                (idx, event)
                for idx, event in enumerate(self.events)
                if self._event_touches_qubit(event, root_q)
            ]

            pos = 0
            while pos < len(touches):
                idx, event = touches[pos]
                if not isinstance(event, FanIn) or event.source_partition == event.target_partition:
                    pos += 1
                    continue

                target_p = event.target_partition
                run: list[tuple[int, FanIn]] = [(idx, event)]
                pos += 1

                while pos < len(touches):
                    next_idx, next_event = touches[pos]
                    prev_time = run[-1][1].time
                    if (
                        isinstance(next_event, FanIn)
                        and next_event.source_partition != next_event.target_partition
                        and next_event.target_partition == target_p
                        and (
                            not adjacent_time_only
                            or next_event.time == prev_time + 1
                        )
                    ):
                        run.append((next_idx, next_event))
                        pos += 1
                        continue
                    break

                if len(run) >= 2:
                    sources = [ev.source_partition for _, ev in run]
                    last_idx, last_ev = run[-1]
                    replacements[last_idx] = JointFanIn(
                        root_qubit=root_q,
                        source_partitions=sources,
                        target_partition=target_p,
                        time=last_ev.time,
                    )
                    for rem_idx, _ in run[:-1]:
                        remove.add(rem_idx)

        if not replacements:
            return

        self.events = [
            replacements[i] if i in replacements else ev
            for i, ev in enumerate(self.events)
            if i not in remove
        ]

    @staticmethod
    def _event_touches_qubit(event: DistributedCircuitEvent, qubit: int) -> bool:
        if isinstance(event, LocalGate):
            return any(q == qubit for q, _ in event.qubits)
        if isinstance(event, StateTransfer):
            return event.qubit == qubit
        if isinstance(event, (FanOut, ImmediateFanIn, FanIn, JointFanIn)):
            return event.root_qubit == qubit
        if isinstance(event, LinkedGate):
            return event.root_qubit == qubit or event.target_qubit == qubit
        return False

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_events_by_type(self, event_type: type) -> list:
        return [e for e in self.events if isinstance(e, event_type)]

    def qubit_wire_ids(self) -> set[tuple[int, int]]:
        """Return all (qubit_index, partition) pairs that appear in any event."""
        wires: set[tuple[int, int]] = set()
        if self.initial_assignment is not None:
            wires.update((q, int(p)) for q, p in enumerate(self.initial_assignment))
        for event in self.events:
            if isinstance(event, LocalGate):
                wires.update(event.qubits)
            elif isinstance(event, StateTransfer):
                wires.add((event.qubit, event.source_partition))
                wires.add((event.qubit, event.target_partition))
            elif isinstance(event, FanOut):
                wires.add((event.root_qubit, event.root_partition))
                for p in event.target_partitions:
                    wires.add((event.root_qubit, p))
                for p in event.intermediate_partitions:
                    wires.add((event.root_qubit, p))
            elif isinstance(event, ImmediateFanIn):
                for p in event.intermediate_partitions:
                    wires.add((event.root_qubit, p))
            elif isinstance(event, LinkedGate):
                wires.add((event.root_qubit, event.root_partition))
                wires.add((event.target_qubit, event.target_partition))
            elif isinstance(event, FanIn):
                wires.add((event.root_qubit, event.source_partition))
                wires.add((event.root_qubit, event.target_partition))
            elif isinstance(event, JointFanIn):
                for p in event.source_partitions:
                    wires.add((event.root_qubit, p))
                wires.add((event.root_qubit, event.target_partition))
        return wires

    def set_layout(
        self,
        initial_assignment: list[int],
        wire_order: list[tuple[int, int]],
    ) -> None:
        self.initial_assignment = [int(p) for p in initial_assignment]
        self.wire_order = [(int(q), int(p)) for q, p in wire_order]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for event in self.events:
            name = type(event).__name__
            counts[name] = counts.get(name, 0) + 1
        lines = [f"DistributedCircuit ({self.num_qubits}q, {self.num_partitions}p):"]
        for name, count in sorted(counts.items()):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
