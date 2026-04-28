"""Convert a partitioned QuantumCircuitHyperGraph to a DistributedCircuit.

This module mirrors the traversal logic of PartitionedCircuitExtractor but
emits DistributedCircuit events rather than Qiskit instructions.  No qubit
allocation, EPR generation, or classical bits appear here.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

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

if TYPE_CHECKING:
    from disqco import QuantumCircuitHyperGraph, QuantumNetwork


class HypergraphToDistributed:
    """Build a DistributedCircuit from a partitioned hypergraph.

    Parameters
    ----------
    graph:
        The quantum circuit hypergraph (must already have layers populated).
    network:
        The QPU network topology.
    partition_assignment:
        2-D array of shape (depth, num_qubits) mapping each (time, qubit) to a
        partition index.
    """

    def __init__(
        self,
        graph: "QuantumCircuitHyperGraph",
        network: "QuantumNetwork",
        partition_assignment: np.ndarray,
    ) -> None:
        self.layer_dict: dict[int, list[dict]] = copy.deepcopy(graph.layers)
        self.layer_dict = self._remove_empty_groups()
        self.partition_assignment: list[list[int]] = partition_assignment.tolist()
        self.num_qubits: int = graph.num_qubits
        self.num_partitions: int = len(network.qpu_sizes)
        self.network = network

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(self) -> DistributedCircuit:
        dc = DistributedCircuit(self.num_qubits, self.num_partitions, self.network)
        dc.set_layout(
            initial_assignment=self.partition_assignment[0],
            wire_order=self._build_wire_order(),
        )

        current_assignment: list[int] = list(self.partition_assignment[0])

        # Tracks which root qubits are currently fanned out and the time step at
        # which their group ends (so we can skip state transfers for those qubits).
        active_roots: dict[int, int] = {}   # root_qubit -> final_t

        # Deferred events keyed by the time step at which they should be emitted.
        deferred: dict[int, list] = {}

        for i, layer in sorted(self.layer_dict.items()):
            new_assignment: list[int] = self.partition_assignment[i]

            # Expire groups whose final time step has passed.
            expired = [r for r, ft in active_roots.items() if ft < i]
            for r in expired:
                del active_roots[r]

            # Detect qubit assignment changes for qubits not currently fanned out.
            needs_transfer = any(
                current_assignment[q] != new_assignment[q]
                for q in range(self.num_qubits)
                if q not in active_roots
            )
            if needs_transfer:
                for q in range(self.num_qubits):
                    if q in active_roots:
                        continue
                    src = int(current_assignment[q])
                    dst = int(new_assignment[q])
                    if src != dst:
                        dc.add_event(StateTransfer(q, src, dst, i))

            current_assignment = list(new_assignment)

            # Emit deferred events for this time step.
            for event in deferred.pop(i, []):
                dc.add_event(event)

            for gate in layer:
                gtype = gate['type']

                if gtype == 'single-qubit':
                    q = gate['qargs'][0]
                    p = int(current_assignment[q])
                    dc.add_event(LocalGate(gate['name'], gate['params'],
                                           [(q, p)], i))

                elif gtype == 'two-qubit':
                    q0, q1 = gate['qargs']
                    p0 = int(current_assignment[q0])
                    p1 = int(current_assignment[q1])
                    if p0 == p1:
                        dc.add_event(LocalGate(gate['name'], gate['params'],
                                               [(q0, p0), (q1, p1)], i))
                    else:
                        self._process_gate_teleport(dc, gate, q0, q1, p0, p1, i)

                elif gtype == 'group':
                    self._process_group_gate(
                        dc, gate, i, active_roots, deferred
                    )

        dc.merge_fan_ins(adjacent_time_only=True)
        return dc

    # ------------------------------------------------------------------
    # Isolated non-local two-qubit gate (gate teleportation)
    # ------------------------------------------------------------------

    def _process_gate_teleport(
        self,
        dc: DistributedCircuit,
        gate: dict,
        q0: int,
        q1: int,
        p0: int,
        p1: int,
        t: int,
    ) -> None:
        directed_tree, intermediates, nearest_targets = (
            self._build_directed_tree_and_intermediates(p0, [p1])
        )
        dc.add_event(FanOut(q0, p0, directed_tree, [p1], intermediates, t))
        if intermediates:
            dc.add_event(ImmediateFanIn(q0, intermediates, nearest_targets, t))
        dc.add_event(LinkedGate(gate['name'], gate['params'], q0, p1, q1, p1, t))
        # Fan the root back in immediately — isolated gate teleportation ends here.
        dc.add_event(FanIn(q0, p1, p0, t))

    # ------------------------------------------------------------------
    # Group gate
    # ------------------------------------------------------------------

    def _process_group_gate(
        self,
        dc: DistributedCircuit,
        gate: dict,
        t: int,
        active_roots: dict[int, int],
        deferred: dict[int, list],
    ) -> None:
        root_idx: int = gate['root']
        start_time: int = gate['time']
        sub_gates: list[dict] = gate['sub-gates']

        if not sub_gates:
            return

        p_root = int(self.partition_assignment[start_time][root_idx])

        # Time of the last two-qubit sub-gate in the group.
        final_t = start_time
        for sg in reversed(sub_gates):
            if sg['type'] == 'two-qubit':
                final_t = sg['time']
                break

        final_p_root = int(self.partition_assignment[final_t][root_idx])

        # All partitions the root qubit visits during the group's time window.
        p_root_set: set[int] = {
            int(self.partition_assignment[ts][root_idx])
            for ts in range(start_time, final_t + 1)
        }

        # Receiver partitions and the last sub-gate time per partition.
        p_rec_set: set[int] = set()
        final_gates: dict[int, int] = {}
        for sg in sub_gates:
            if sg['type'] == 'two-qubit':
                q1 = sg['qargs'][1]
                ts: int = sg['time']
                p_rec = int(self.partition_assignment[ts][q1])
                p_rec_set.add(p_rec)
                prev = final_gates.get(p_rec, ts)
                final_gates[p_rec] = ts if ts > prev else prev

        # target_partitions: partitions that hold the root qubit's entangled state.
        # This includes receiver partitions AND any new root locations (nested case).
        target_partitions = list((p_rec_set | p_root_set) - {p_root})

        # Build the communication tree and classify intermediate nodes.
        if target_partitions:
            # Mark root as active so state transfers skip it while fan-out copies are live.
            active_roots[root_idx] = final_t
            directed_tree, intermediates, nearest_targets = (
                self._build_directed_tree_and_intermediates(p_root, target_partitions)
            )
        else:
            directed_tree = nx.DiGraph()
            intermediates = []
            nearest_targets = {}

        # Emit fan-out only if there is at least one remote target.
        if target_partitions:
            dc.add_event(FanOut(root_idx, p_root, directed_tree,
                                target_partitions, intermediates, start_time,
                                final_root_partition=final_p_root))
            if intermediates:
                dc.add_event(ImmediateFanIn(root_idx, intermediates,
                                            nearest_targets, start_time))

        # Schedule sub-gate events.
        for sg in sub_gates:
            sg_time = sg['time']
            if sg['type'] == 'two-qubit':
                _, q1_sg = sg['qargs']
                p_rec_sg = int(self.partition_assignment[sg_time][q1_sg])
                linked = LinkedGate(
                    sg['name'], sg['params'],
                    root_idx, p_rec_sg,
                    q1_sg, p_rec_sg,
                    sg_time,
                )
                _schedule(deferred, sg_time, t, dc, linked)

                # Schedule FanIn for this receiver partition at its final gate time.
                # Skip if receiver was never fanned out to (i.e. it equals p_root).
                if final_gates.get(p_rec_sg) == sg_time and p_rec_sg != p_root:
                    fan_in = FanIn(root_idx, p_rec_sg, final_p_root, sg_time)
                    _schedule(deferred, sg_time, t, dc, fan_in)

            elif sg['type'] == 'single-qubit':
                q_sg = sg['qargs'][0]
                p_sg = int(self.partition_assignment[sg_time][q_sg])
                local = LocalGate(sg['name'], sg['params'], [(q_sg, p_sg)], sg_time)
                _schedule(deferred, sg_time, t, dc, local)

        # Nested teleportation: root qubit moves to a different partition.
        # Emit a FanIn of the original root partition at the group's end time.
        if p_root_set != {p_root}:
            fan_in_root = FanIn(root_idx, p_root, final_p_root, final_t)
            _schedule(deferred, final_t, t, dc, fan_in_root)

    # ------------------------------------------------------------------
    # Network tree helpers
    # ------------------------------------------------------------------

    def _build_directed_tree_and_intermediates(
        self,
        p_root: int,
        target_partitions: list[int],
    ) -> tuple[nx.DiGraph, list[int], dict[int, int]]:
        """Return (directed_tree, intermediate_partitions, nearest_targets).

        directed_tree is a BFS-directed tree rooted at p_root spanning all
        target_partitions (plus any necessary routing nodes).
        intermediate_partitions are the routing-only (non-target) nodes.
        nearest_targets maps each intermediate to its closest target or root.
        """
        undirected_tree: nx.Graph = self.network.get_full_tree(
            root_p=p_root, target_partitions=target_partitions
        )

        # Orient edges away from root via BFS.
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

        # Classify nodes.
        target_set = set(target_partitions)
        all_nodes = set(directed_tree.nodes())
        intermediates = list(all_nodes - target_set - {p_root})

        # Find the nearest target/root for each intermediate node.
        nearest_targets: dict[int, int] = {}
        undirected = directed_tree.to_undirected()
        for aux in intermediates:
            best_len: int | None = None
            best_target: int | None = None
            for tgt in list(target_partitions) + [p_root]:
                try:
                    path = nx.shortest_path(undirected, source=aux, target=tgt)
                    if best_len is None or len(path) < best_len:
                        best_len = len(path)
                        best_target = tgt
                except nx.NetworkXNoPath:
                    continue
            if best_target is not None:
                nearest_targets[aux] = best_target

        return directed_tree, intermediates, nearest_targets

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _remove_empty_groups(self) -> dict[int, list[dict]]:
        """Mirror PartitionedCircuitExtractor.remove_empty_groups."""
        new_layers = copy.deepcopy(self.layer_dict)
        for i, layer in new_layers.items():
            for gate in layer[:]:
                if gate['type'] == 'group':
                    sub = gate['sub-gates']
                    if len(sub) == 1:
                        new_gate = sub.pop(0)
                        ts = new_gate.pop('time', i)
                        new_layers[ts].append(new_gate)
                        layer.remove(gate)
                    elif len(sub) == 0:
                        layer.remove(gate)
        return new_layers

    def _build_wire_order(self) -> list[tuple[int, int]]:
        """Build a static qpic wire order matching the partition layout.

        For each partition, keep initial data qubits in their hypergraph order and
        append preallocated communication/state-transfer wires that may be activated
        later in the circuit.
        """
        wires = set()
        initial_assignment = [int(p) for p in self.partition_assignment[0]]

        for q, p in enumerate(initial_assignment):
            wires.add((q, p))

        for layer_time, layer in sorted(self.layer_dict.items()):
            for gate in layer:
                gtype = gate['type']
                if gtype == 'single-qubit':
                    q = int(gate['qargs'][0])
                    wires.add((q, int(self.partition_assignment[layer_time][q])))
                elif gtype == 'two-qubit':
                    q0, q1 = (int(x) for x in gate['qargs'])
                    wires.add((q0, int(self.partition_assignment[layer_time][q0])))
                    wires.add((q1, int(self.partition_assignment[layer_time][q1])))
                elif gtype == 'group':
                    root = int(gate['root'])
                    for sg in gate['sub-gates']:
                        if sg['type'] == 'single-qubit':
                            q = int(sg['qargs'][0])
                            wires.add((q, int(self.partition_assignment[sg['time']][q])))
                        elif sg['type'] == 'two-qubit':
                            q1 = int(sg['qargs'][1])
                            wires.add((root, int(self.partition_assignment[sg['time']][q1])))
                            wires.add((q1, int(self.partition_assignment[sg['time']][q1])))

        ordered: list[tuple[int, int]] = []
        for p in range(self.num_partitions):
            data_wires = [
                (q, p)
                for q in range(self.num_qubits)
                if initial_assignment[q] == p and (q, p) in wires
            ]
            aux_wires = sorted(
                (q, p)
                for q in range(self.num_qubits)
                if initial_assignment[q] != p and (q, p) in wires
            )
            ordered.extend(data_wires)
            ordered.extend(aux_wires)

        return ordered


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _schedule(
    deferred: dict[int, list],
    event_time: int,
    current_time: int,
    dc: DistributedCircuit,
    event,
) -> None:
    """Emit *event* immediately if event_time == current_time, else defer it."""
    if event_time == current_time:
        dc.add_event(event)
    else:
        deferred.setdefault(event_time, []).append(event)
