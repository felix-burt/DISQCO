"""Targeted tests for the bosonic-native extraction path.

Covers:
  - Teleportation / correction instruction emission in bosonic form
  - Dynamic classical-bit allocation and reuse
  - Communication-qubit allocation and reuse
  - Multi-hop / tree-based entanglement
  - Extracted DistributedCircuit instruction-structure assertions
  - NativeCircuitBuilder register integrity
"""
from __future__ import annotations

import numpy as np
import pytest

from bosonic_converters import CircuitConverters
from bosonic_model import Circuit as BosonicCircuit, DistributedCircuit
from bosonic_model.instructions import (
    ConditionalInstruction,
    CxInstruction,
    GateInstruction,
    HInstruction,
    MeasureInstruction,
    ResetInstruction,
    XInstruction,
    ZInstruction,
)
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

from disqco import QuantumCircuitHyperGraph, QuantumNetwork, PartitionedCircuitExtractor
from disqco import set_initial_partition_assignment
from disqco.circuit_extraction.DQC_qubit_manager import (
    ClassicalBitManager,
    CommunicationQubitManager,
    DataQubitManager,
    NativeCircuitBuilder,
)
from disqco.circuit_extraction.circuit_extractor import TeleportationManager
from disqco.circuits.cp_fraction import cp_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_bosonic(num_qubits: int = 4, depth: int = 4) -> BosonicCircuit:
    qc = cp_fraction(num_qubits=num_qubits, depth=depth, fraction=0.5, seed=0)
    qc = transpile(qc, basis_gates=["u", "cp"])
    return CircuitConverters.from_qiskit(qc)


def _epr_count(dist: DistributedCircuit) -> int:
    return sum(dist.coupling_map().values())


# ---------------------------------------------------------------------------
# NativeCircuitBuilder unit tests
# ---------------------------------------------------------------------------


class TestNativeCircuitBuilder:
    def test_qubit_indices_are_contiguous(self):
        b = NativeCircuitBuilder()
        idx0 = b.add_qreg("Q0_q", 3, "Q", 0)
        idx1 = b.add_qreg("C0_0", 2, "C", 0)
        assert idx0 == [0, 1, 2]
        assert idx1 == [3, 4]

    def test_cbit_indices_are_contiguous(self):
        b = NativeCircuitBuilder()
        idx0 = b.add_creg("cl_global", 4)
        idx1 = b.add_creg("result", 2)
        assert idx0 == [0, 1, 2, 3]
        assert idx1 == [4, 5]

    def test_qubit_kind_and_partition_populated(self):
        b = NativeCircuitBuilder()
        b.add_qreg("Q0_q", 2, "Q", 0)
        b.add_qreg("C0_0", 2, "C", 0)
        b.add_qreg("Q1_q", 2, "Q", 1)
        assert b.qubit_kind[0] == "Q"
        assert b.qubit_kind[2] == "C"
        assert b.qubit_kind[4] == "Q"
        assert b.qubit_partition[0] == 0
        assert b.qubit_partition[2] == 0
        assert b.qubit_partition[4] == 1

    def test_build_produces_correct_register_names(self):
        b = NativeCircuitBuilder()
        b.add_qreg("Q0_q", 3, "Q", 0)
        b.add_creg("cl_global", 4)
        circuit = b.build()
        assert "Q0_q" in circuit.qregs
        assert "cl_global" in circuit.cregs
        assert circuit.qregs["Q0_q"].size == 3
        assert circuit.cregs["cl_global"].size == 4

    def test_emit_appends_instructions(self):
        b = NativeCircuitBuilder()
        b.add_qreg("Q0_q", 2, "Q", 0)
        b.emit(HInstruction(qubit=0, params=[], qubits=[0]))
        b.emit(CxInstruction(control=0, target=1, qubits=[0, 1]))
        circuit = b.build()
        assert len(circuit.instructions) == 2

    def test_dynamic_qreg_extends_qubit_space(self):
        b = NativeCircuitBuilder()
        b.add_qreg("Q0_q", 2, "Q", 0)
        assert b._next_qubit == 2
        b.add_qreg("C0_1", 1, "C", 0)  # dynamic extra comm reg
        assert b._next_qubit == 3
        assert 2 in b.qubit_kind
        assert b.qubit_kind[2] == "C"


# ---------------------------------------------------------------------------
# ClassicalBitManager unit tests
# ---------------------------------------------------------------------------


class TestClassicalBitManager:
    def _make_manager(self, size: int = 4):
        b = NativeCircuitBuilder()
        indices = b.add_creg("cl_global", size)
        return ClassicalBitManager(b, indices[0], size), b

    def test_allocate_returns_sequential_indices(self):
        mgr, _ = self._make_manager(4)
        c0 = mgr.allocate_cbit()
        c1 = mgr.allocate_cbit()
        assert c0 == 0
        assert c1 == 1

    def test_released_bit_reused_next(self):
        mgr, _ = self._make_manager(4)
        c0 = mgr.allocate_cbit()
        mgr.release_cbit(c0)
        c1 = mgr.allocate_cbit()
        assert c1 == c0

    def test_dynamic_growth_when_pool_empty(self):
        mgr, b = self._make_manager(2)
        mgr.allocate_cbit()
        mgr.allocate_cbit()
        # Pool is now empty; next allocation must grow.
        c2 = mgr.allocate_cbit()
        assert "cl_global_extra_0" in b.cregs
        assert b.cregs["cl_global_extra_0"].size == 1
        assert c2 == 2  # base of the new register

    def test_multiple_dynamic_growths(self):
        mgr, b = self._make_manager(1)
        mgr.allocate_cbit()
        for i in range(3):
            c = mgr.allocate_cbit()
            assert f"cl_global_extra_{i}" in b.cregs


# ---------------------------------------------------------------------------
# CommunicationQubitManager unit tests
# ---------------------------------------------------------------------------


class TestCommunicationQubitManager:
    def _make_manager(self, num_comm: int = 2):
        b = NativeCircuitBuilder()
        c_idx = b.add_qreg("C0_0", num_comm, "C", 0)
        comm_data = {0: c_idx}
        return CommunicationQubitManager(comm_data, b), b

    def test_find_comm_idx_returns_free_qubit(self):
        mgr, b = self._make_manager(2)
        q = mgr.find_comm_idx(0)
        assert q in b.qubit_kind
        assert b.qubit_kind[q] == "C"

    def test_released_qubit_returned_to_pool(self):
        mgr, _ = self._make_manager(2)
        q0 = mgr.find_comm_idx(0)
        mgr.release_comm_qubit(0, q0)
        # After release the qubit must be back in the free pool
        assert q0 in mgr.free_comm[0]
        # And allocatable again
        q1 = mgr.find_comm_idx(0)
        assert q1 in (0, 1)  # order within pool is an implementation detail

    def test_dynamic_comm_qubit_allocated_when_pool_empty(self):
        mgr, b = self._make_manager(1)
        mgr.find_comm_idx(0)  # exhaust pool
        q_new = mgr.find_comm_idx(0)
        assert "C0_1" in b.qregs
        assert b.qubit_kind[q_new] == "C"
        assert b.qubit_partition[q_new] == 0

    def test_in_use_tracking(self):
        mgr, _ = self._make_manager(2)
        q0 = mgr.find_comm_idx(0)
        assert q0 in mgr.in_use_comm[0]
        mgr.release_comm_qubit(0, q0)
        assert q0 not in mgr.in_use_comm[0]


# ---------------------------------------------------------------------------
# TeleportationManager emission tests
# ---------------------------------------------------------------------------


class TestTeleportationManagerEmission:
    """Verify that the low-level emit helpers produce the correct instruction sequence."""

    def _make_teleport_manager(self):
        b = NativeCircuitBuilder()
        q_idx = b.add_qreg("Q0_q", 4, "Q", 0)
        c_idx = b.add_qreg("C0_0", 4, "C", 0)
        cl_idx = b.add_creg("cl_global", 8)
        comm_data = {0: c_idx}
        part_data = {0: q_idx}
        assignment = [[0, 0, 0, 0]]
        qmgr = DataQubitManager(part_data, 4, assignment, b)
        cmgr = CommunicationQubitManager(comm_data, b)
        clmgr = ClassicalBitManager(b, cl_idx[0], 8)

        # Minimal stubs for network / hypergraph (not used in low-level emission)
        class _FakeHypergraph:
            pass

        class _FakeNetwork:
            qpu_sizes = [4]

        tmgr = TeleportationManager(b, _FakeHypergraph(), _FakeNetwork(), qmgr, cmgr, clmgr)
        return b, tmgr

    def test_emit_state_transfer_sequence(self):
        b, tmgr = self._make_teleport_manager()
        q1, q2, cbit = 0, 1, 8  # q indices, cbit from cl_global (base 8)
        start = len(b.instructions)
        tmgr._emit_state_transfer(q1, q2, cbit)
        seq = b.instructions[start:]
        assert len(seq) == 5
        assert isinstance(seq[0], CxInstruction)
        assert isinstance(seq[1], HInstruction)
        assert isinstance(seq[2], MeasureInstruction)
        assert isinstance(seq[3], ResetInstruction)
        assert isinstance(seq[4], ConditionalInstruction)
        # CX is q1 -> q2
        assert seq[0].control == q1 and seq[0].target == q2
        # Measure is q1
        assert seq[2].qubit == q1
        # Conditional wraps Z on q2
        assert isinstance(seq[4].op, ZInstruction)
        assert seq[4].op.qubit == q2
        assert seq[4].condition.cbit == cbit
        assert seq[4].condition.value is True

    def test_emit_starting_process_sequence(self):
        b, tmgr = self._make_teleport_manager()
        root_q, root_comm, rec_comm, cbit = 0, 4, 5, 8
        start = len(b.instructions)
        tmgr._emit_starting_process(root_q, root_comm, rec_comm, cbit)
        seq = b.instructions[start:]
        assert len(seq) == 5
        assert isinstance(seq[0], GateInstruction)
        assert seq[0].name == "remote_link_phi_plus"
        assert seq[0].qubits == [root_comm, rec_comm]
        assert isinstance(seq[1], CxInstruction)
        assert seq[1].control == root_q and seq[1].target == root_comm
        assert isinstance(seq[2], MeasureInstruction) and seq[2].qubit == root_comm
        assert isinstance(seq[3], ResetInstruction) and seq[3].qubit == root_comm
        assert isinstance(seq[4], ConditionalInstruction)
        assert isinstance(seq[4].op, XInstruction) and seq[4].op.qubit == rec_comm

    def test_emit_ending_process_sequence(self):
        b, tmgr = self._make_teleport_manager()
        target_q, link_comm, cbit = 0, 4, 8
        start = len(b.instructions)
        tmgr._emit_ending_process(target_q, link_comm, cbit)
        seq = b.instructions[start:]
        assert len(seq) == 4
        assert isinstance(seq[0], HInstruction) and seq[0].qubit == link_comm
        assert isinstance(seq[1], MeasureInstruction) and seq[1].qubit == link_comm
        assert isinstance(seq[2], ResetInstruction) and seq[2].qubit == link_comm
        assert isinstance(seq[3], ConditionalInstruction)
        assert isinstance(seq[3].op, ZInstruction) and seq[3].op.qubit == target_q

    def test_epr_gate_uses_opaque_named_instruction(self):
        b, tmgr = self._make_teleport_manager()
        # The EPR primitive should be a GateInstruction named remote_link_phi_plus
        comm0, comm1 = 4, 5
        tmgr._emit_starting_process(0, comm0, comm1, 8)
        epr = b.instructions[0]
        assert isinstance(epr, GateInstruction)
        assert epr.name == "remote_link_phi_plus"
        assert epr.opaque is True

    def test_conditional_instruction_has_qubits_set(self):
        """ConditionalInstruction.qubits must equal inner op qubits for routing."""
        b, tmgr = self._make_teleport_manager()
        tmgr._emit_state_transfer(0, 1, 8)
        cond = b.instructions[-1]
        assert isinstance(cond, ConditionalInstruction)
        assert cond.qubits == cond.op.qubits


# ---------------------------------------------------------------------------
# Data-qubit manager reset emission test
# ---------------------------------------------------------------------------


class TestDataQubitManagerResetEmission:
    def test_release_emits_reset(self):
        b = NativeCircuitBuilder()
        q_idx = b.add_qreg("Q0_q", 2, "Q", 0)
        part_data = {0: q_idx}
        assignment = [[0, 0]]
        mgr = DataQubitManager(part_data, 2, assignment, b)
        # Both qubits are in use after initial_placement; release one.
        phys0 = mgr.log_to_phys_idx[0]
        before = len(b.instructions)
        mgr.release_data_qubit(0, phys0)
        after_insts = b.instructions[before:]
        assert any(isinstance(i, ResetInstruction) and i.qubit == phys0 for i in after_insts)

    def test_released_qubit_returned_to_free_pool(self):
        b = NativeCircuitBuilder()
        q_idx = b.add_qreg("Q0_q", 3, "Q", 0)
        part_data = {0: q_idx}
        assignment = [[0, 0, 0]]
        mgr = DataQubitManager(part_data, 3, assignment, b)
        phys0 = mgr.log_to_phys_idx[0]
        mgr.release_data_qubit(0, phys0)
        assert phys0 in mgr.free_data[0]

    def test_log_to_phys_cleared_after_release(self):
        b = NativeCircuitBuilder()
        q_idx = b.add_qreg("Q0_q", 3, "Q", 0)
        part_data = {0: q_idx}
        assignment = [[0, 0, 0]]
        mgr = DataQubitManager(part_data, 3, assignment, b)
        phys0 = mgr.log_to_phys_idx[0]
        mgr.release_data_qubit(0, phys0)
        assert 0 not in mgr.log_to_phys_idx


# ---------------------------------------------------------------------------
# Extracted DistributedCircuit structure assertions
# ---------------------------------------------------------------------------


class TestExtractedCircuitStructure:
    @pytest.fixture
    def two_qpu_network(self):
        return QuantumNetwork.create([4, 4], "all_to_all")

    @pytest.fixture
    def cross_partition_circuit(self):
        """A 2-qubit circuit where qubits land in different partitions."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        return CircuitConverters.from_qiskit(qc)

    def test_each_node_has_result_creg(self, two_qpu_network):
        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        for circ in dist.circuits.values():
            assert "result" in circ.cregs
            assert circ.cregs["result"].size == 8

    def test_each_node_has_cl_global_creg(self, two_qpu_network):
        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        for circ in dist.circuits.values():
            assert any(name.startswith("cl_global") for name in circ.cregs)

    def test_data_qregs_match_qpu_sizes(self, two_qpu_network):
        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        for node, circ in dist.circuits.items():
            data_size = sum(
                reg.size for reg in circ.qregs.values() if reg.name.startswith("Q")
            )
            assert data_size == two_qpu_network.qpu_sizes[node]

    def test_measure_instructions_present_in_output(self, two_qpu_network):
        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        total_measures = 0
        for circ in dist.circuits.values():
            total_measures += sum(
                1 for inst in circ.instructions if isinstance(inst, MeasureInstruction)
            )
        # At least as many measurements as logical qubits (final result measures)
        assert total_measures >= 8

    def test_epr_instructions_are_remote_gate_instructions(self, two_qpu_network):
        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        for circ in dist.circuits.values():
            for inst in circ.instructions:
                if isinstance(inst, GateInstruction) and inst.name.startswith("remote_"):
                    assert inst.opaque is True

    def test_no_cross_partition_data_qubit_gates(self, two_qpu_network):
        """Data qubits from different QPUs must never appear together in one gate."""
        from disqco.circuit_extraction.verification import (
            check_no_cross_partition_instructions,
        )

        circuit = _simple_bosonic(8, 4)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, two_qpu_network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert check_no_cross_partition_instructions(dist, two_qpu_network.qpu_graph)

    def test_single_partition_produces_no_epr(self):
        circuit = _simple_bosonic(4, 4)
        network = QuantumNetwork({0: 6})
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert _epr_count(dist) == 0

    def test_optimized_fewer_epr_than_naive(self, two_qpu_network):
        from disqco.parti import FiducciaMattheyses

        circuit = _simple_bosonic(8, 8)
        hg = QuantumCircuitHyperGraph(circuit)
        init = set_initial_partition_assignment(hg, two_qpu_network)
        dist_init = PartitionedCircuitExtractor(
            graph=hg, network=two_qpu_network, partition_assignment=init
        ).extract_partitioned_circuit()

        part = FiducciaMattheyses(circuit, network=two_qpu_network)
        best = part.partition(num_passes=5)["best_assignment"]
        dist_opt = PartitionedCircuitExtractor(
            graph=QuantumCircuitHyperGraph(_simple_bosonic(8, 8)),
            network=two_qpu_network,
            partition_assignment=best,
        ).extract_partitioned_circuit()

        assert _epr_count(dist_opt) <= _epr_count(dist_init)


# ---------------------------------------------------------------------------
# Multi-hop / three-partition entanglement tests
# ---------------------------------------------------------------------------


class TestMultiHopEntanglement:
    def test_three_partition_extraction_produces_distributed_circuit(self):
        circuit = _simple_bosonic(6, 6)
        network = QuantumNetwork.create([2, 2, 2], "linear")
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert isinstance(dist, DistributedCircuit)
        assert sorted(dist.qubits_per_node.keys()) == [0, 1, 2]

    def test_three_partition_all_to_all_extraction(self):
        circuit = _simple_bosonic(6, 6)
        network = QuantumNetwork.create([2, 2, 2], "all_to_all")
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert isinstance(dist, DistributedCircuit)

    def test_four_partition_grid_extraction(self):
        circuit = _simple_bosonic(8, 4)
        network = QuantumNetwork.create([2, 2, 2, 2], "grid")
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert isinstance(dist, DistributedCircuit)
        assert sorted(dist.qubits_per_node.keys()) == [0, 1, 2, 3]

    def test_linear_three_partition_each_node_has_instructions(self):
        circuit = _simple_bosonic(6, 8)
        network = QuantumNetwork.create([2, 2, 2], "linear")
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        for node, circ in dist.circuits.items():
            assert len(circ.instructions) > 0, f"Node {node} has no instructions"


# ---------------------------------------------------------------------------
# Instruction-level qubit routing tests
# ---------------------------------------------------------------------------


class TestInstructionRouting:
    def _extract(self, num_qubits, depth, net):
        circuit = _simple_bosonic(num_qubits, depth)
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, net)
        return PartitionedCircuitExtractor(
            graph=hg, network=net, partition_assignment=assignment
        ).extract_partitioned_circuit()

    def test_all_measure_instruction_qubits_owned_by_node(self):
        net = QuantumNetwork.create([4, 4], "all_to_all")
        dist = self._extract(8, 4, net)
        for node, circ in dist.circuits.items():
            owned = set(dist.qubits_per_node[node])
            for inst in circ.instructions:
                if isinstance(inst, MeasureInstruction):
                    assert inst.qubit in owned, (
                        f"Node {node} has MeasureInstruction on qubit {inst.qubit} "
                        f"which is not in {owned}"
                    )

    def test_single_qubit_gate_qubits_owned_by_node(self):
        net = QuantumNetwork.create([4, 4], "all_to_all")
        dist = self._extract(8, 4, net)
        for node, circ in dist.circuits.items():
            owned = set(dist.qubits_per_node[node])
            for raw in circ.instructions:
                inst = raw.op if isinstance(raw, ConditionalInstruction) else raw
                if isinstance(inst, (HInstruction, XInstruction, ZInstruction)):
                    for q in inst.qubits:
                        assert q in owned, (
                            f"Node {node} has single-qubit gate on qubit {q} "
                            f"which is not in owned set {owned}"
                        )

    def test_reset_instructions_route_to_owning_node(self):
        net = QuantumNetwork.create([4, 4], "all_to_all")
        dist = self._extract(8, 4, net)
        for node, circ in dist.circuits.items():
            owned = set(dist.qubits_per_node[node])
            for inst in circ.instructions:
                if isinstance(inst, ResetInstruction):
                    assert inst.qubit in owned, (
                        f"Node {node} has ResetInstruction on qubit {inst.qubit} "
                        f"not in {owned}"
                    )

    def test_conditional_instruction_routes_by_inner_op_qubit(self):
        net = QuantumNetwork.create([4, 4], "all_to_all")
        dist = self._extract(8, 6, net)
        for node, circ in dist.circuits.items():
            owned = set(dist.qubits_per_node[node])
            for inst in circ.instructions:
                if isinstance(inst, ConditionalInstruction):
                    inner = inst.op
                    for q in inner.qubits:
                        assert q in owned, (
                            f"Node {node}: ConditionalInstruction inner op qubit {q} "
                            f"not in owned set {owned}"
                        )


# ---------------------------------------------------------------------------
# Classical bit allocation regression / stress test
# ---------------------------------------------------------------------------


class TestClassicalBitAllocation:
    def test_many_teleportations_do_not_exhaust_bits(self):
        """A circuit requiring many teleportations must not crash on cbit exhaustion."""
        circuit = _simple_bosonic(8, 12)
        network = QuantumNetwork.create([4, 4], "all_to_all")
        hg = QuantumCircuitHyperGraph(circuit)
        assignment = set_initial_partition_assignment(hg, network)
        # If cbit reuse fails, this will raise inside the extractor.
        dist = PartitionedCircuitExtractor(
            graph=hg, network=network, partition_assignment=assignment
        ).extract_partitioned_circuit()
        assert isinstance(dist, DistributedCircuit)

    def test_cbit_manager_roundtrip_invariant(self):
        """After allocate+release, the pool returns to its original state."""
        b = NativeCircuitBuilder()
        idx = b.add_creg("cl_global", 8)
        mgr = ClassicalBitManager(b, idx[0], 8)
        original_free = list(mgr.free_cbit)
        cbits = [mgr.allocate_cbit() for _ in range(4)]
        for c in cbits:
            mgr.release_cbit(c)
        assert set(mgr.free_cbit) == set(original_free)


# ---------------------------------------------------------------------------
# No Qiskit in extraction path
# ---------------------------------------------------------------------------


def test_no_qiskit_in_extractor_module():
    """circuit_extractor.py must not have any 'import qiskit' or 'from qiskit' lines."""
    import pathlib

    src = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "disqco"
        / "circuit_extraction"
        / "circuit_extractor.py"
    )
    for line in src.read_text().splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("from qiskit") or stripped.startswith("import qiskit")), (
            f"Qiskit import found in circuit_extractor.py: {line!r}"
        )


def test_no_qiskit_in_qubit_manager_module():
    """DQC_qubit_manager.py must not import any Qiskit types."""
    import pathlib

    src = (
        pathlib.Path(__file__).parent.parent
        / "src"
        / "disqco"
        / "circuit_extraction"
        / "DQC_qubit_manager.py"
    )
    for line in src.read_text().splitlines():
        stripped = line.strip()
        assert not (stripped.startswith("from qiskit") or stripped.startswith("import qiskit")), (
            f"Qiskit import found in DQC_qubit_manager.py: {line!r}"
        )
