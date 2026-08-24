"""
Test suite for DataQubitManager.swap_physical_slots.

Tests the bookkeeping-only method that records the exchange of contents
between two physical qubit slots within a partition. No gates are emitted
by the method under test; only the manager's internal maps change.
"""

from qiskit import QuantumRegister, QuantumCircuit
from disqco.circuit_extraction.DQC_qubit_manager import DataQubitManager


def make_manager(reg_size=3):
    """
    Build a minimal DataQubitManager: 2 partitions, one register of
    reg_size slots each, 4 logical qubits assigned two per partition.
    With reg_size=3 each partition has exactly one free slot.
    """
    partition_qregs = [
        QuantumRegister(reg_size, name="Q0_q"),
        QuantumRegister(reg_size, name="Q1_q"),
    ]
    qc = QuantumCircuit(*partition_qregs)
    partition_assignment = [[0, 0, 1, 1]]
    manager = DataQubitManager(partition_qregs, 4, partition_assignment, qc)
    return manager


def occupied_slots(manager, p):
    """Physical qubits currently holding a logical qubit in partition p."""
    return list(manager.in_use_data[p].keys())


def free_slot(manager, p):
    """The first free physical qubit in partition p."""
    return manager.free_data[p][0]


def assert_books_consistent(manager, p, reg_size=3):
    """The two mapping directions agree and no slot is double-counted."""
    for phys, log in manager.in_use_data[p].items():
        assert manager.log_to_phys_idx[log] is phys
    for qubit in manager.free_data[p]:
        assert qubit not in manager.in_use_data[p]
    assert len(manager.free_data[p]) + len(manager.in_use_data[p]) == reg_size


def test_swap_two_occupied_slots():
    """Swapping two occupied slots exchanges the logical qubits' positions."""
    manager = make_manager()
    slot_a, slot_b = occupied_slots(manager, 0)
    log_a = manager.in_use_data[0][slot_a]
    log_b = manager.in_use_data[0][slot_b]

    manager.swap_physical_slots(0, slot_a, slot_b)

    assert manager.log_to_phys_idx[log_a] is slot_b
    assert manager.log_to_phys_idx[log_b] is slot_a
    assert manager.in_use_data[0][slot_a] == log_b
    assert manager.in_use_data[0][slot_b] == log_a
    assert_books_consistent(manager, 0)
    assert_books_consistent(manager, 1)


def test_swap_occupied_into_free_slot():
    """Swapping an occupied slot with a free one moves the logical qubit."""
    manager = make_manager()
    occupied = occupied_slots(manager, 0)[0]
    free = free_slot(manager, 0)
    log_q = manager.in_use_data[0][occupied]

    manager.swap_physical_slots(0, occupied, free)

    assert manager.log_to_phys_idx[log_q] is free
    assert manager.in_use_data[0][free] == log_q
    assert occupied not in manager.in_use_data[0]
    assert occupied in manager.free_data[0]
    assert free not in manager.free_data[0]
    assert_books_consistent(manager, 0)
    assert_books_consistent(manager, 1)


def test_swap_free_into_occupied_slot():
    """Argument order must not matter: free slot first, occupied second."""
    manager = make_manager()
    occupied = occupied_slots(manager, 0)[0]
    free = free_slot(manager, 0)
    log_q = manager.in_use_data[0][occupied]

    manager.swap_physical_slots(0, free, occupied)

    assert manager.log_to_phys_idx[log_q] is free
    assert manager.in_use_data[0][free] == log_q
    assert occupied in manager.free_data[0]
    assert_books_consistent(manager, 0)
    assert_books_consistent(manager, 1)


def test_swap_two_free_slots_is_noop():
    """Swapping two free slots changes nothing in any structure."""
    manager = make_manager(reg_size=4)  # two free slots in partition 0
    free_a, free_b = manager.free_data[0][0], manager.free_data[0][1]
    in_use_before = dict(manager.in_use_data[0])
    free_before = list(manager.free_data[0])
    log_to_phys_before = dict(manager.log_to_phys_idx)

    manager.swap_physical_slots(0, free_a, free_b)

    assert manager.in_use_data[0] == in_use_before
    assert manager.free_data[0] == free_before
    assert manager.log_to_phys_idx == log_to_phys_before
    assert_books_consistent(manager, 0, reg_size=4)


def test_group_links_follow_swapped_state():
    """A gate-group reference to a swapped slot must follow the state.

    Regression test: group machinery stores physical slot references in
    groups[root]['linked_qubits']; routing SWAPs move states between slots,
    and stale references caused wrong-operand gates on dense circuits.
    """
    manager = make_manager()
    occupied = occupied_slots(manager, 0)[0]
    free = free_slot(manager, 0)
    manager.groups[0] = {"linked_qubits": {0: occupied}}

    manager.swap_physical_slots(0, occupied, free)

    assert manager.groups[0]["linked_qubits"][0] == free
    assert_books_consistent(manager, 0)


def test_group_links_follow_both_in_use_swap():
    """Both directions update when two occupied, group-linked slots swap."""
    manager = make_manager()
    slot_a, slot_b = occupied_slots(manager, 0)
    manager.groups[0] = {"linked_qubits": {0: slot_a}}
    manager.groups[1] = {"linked_qubits": {0: slot_b}}

    manager.swap_physical_slots(0, slot_a, slot_b)

    assert manager.groups[0]["linked_qubits"][0] == slot_b
    assert manager.groups[1]["linked_qubits"][0] == slot_a
    assert_books_consistent(manager, 0)


def test_group_links_unrelated_entries_untouched():
    """References to slots not involved in the swap must not change."""
    manager = make_manager()
    slot_a, slot_b = occupied_slots(manager, 0)
    bystander = occupied_slots(manager, 1)[0]
    manager.groups[2] = {"linked_qubits": {1: bystander}}
    manager.groups[3] = {}  # group mid-setup: no linked_qubits key yet

    manager.swap_physical_slots(0, slot_a, slot_b)

    assert manager.groups[2]["linked_qubits"][1] == bystander
    assert "linked_qubits" not in manager.groups[3]
    assert_books_consistent(manager, 0)
    assert_books_consistent(manager, 1)


def make_port_manager():
    """One QPU: data line 0-1-2 with comm node 3 attached to slot 2.
    Qubits 0,1 occupy slots 0,1; slot 2 free. Second QPU keeps the
    network valid."""
    import networkx as nx
    from disqco import QuantumNetwork
    topo = nx.path_graph(3)
    topo.add_edge(2, 3)  # comm qubit 0 = node 3, port at slot 2
    network = QuantumNetwork([3, 3], comm_sizes=[1, 1],
                             qpu_topologies={0: topo})
    partition_qregs = [
        QuantumRegister(3, name="Q0_q"),
        QuantumRegister(3, name="Q1_q"),
    ]
    qc = QuantumCircuit(*partition_qregs)
    manager = DataQubitManager(partition_qregs, 3, [[0, 0, 1]], qc,
                               network=network)
    return manager, qc


def test_route_hole_through_free_slot_emits_no_gate():
    """Hole at slot 2 (free, already adjacent to port) -> nothing happens;
    hole one step away through a free slot relabels without a gate."""
    manager, qc = make_port_manager()
    hole = manager.allocate_data_qubit(0)      # pops slot 2 (the free one)
    gates_before = len(qc.data)

    result = manager.route_hole_to_port(0, hole, 0)

    assert result is hole                      # slot 2 already adjacent to node 3
    assert len(qc.data) == gates_before        # no gates emitted
    assert result not in manager.free_data[0]  # still reserved
    assert result not in manager.in_use_data[0]
    # with one hole outstanding, ledgers cover reg_size - 1 slots
    assert len(manager.free_data[0]) + len(manager.in_use_data[0]) == 2
    for phys, log in manager.in_use_data[0].items():
        assert manager.log_to_phys_idx[log] is phys


def test_route_hole_through_occupied_slots():
    """Hole at slot 0 must reach the port at slot 2: two occupied hops.
    States shift back one slot each; hole arrives reserved and empty."""
    manager, qc = make_port_manager()
    reg = manager.partition_qregs[0]
    # occupy slot 2 as well so the hole must displace states: move qubit
    # from slot 0's occupant is easiest done by allocating slot 2 for a
    # third logical qubit... instead: free slot 0 by hand-crafting:
    # swap qubit at slot 0 into slot 2 (occupied<->free) via the manager.
    manager.swap_physical_slots(0, reg[0], reg[2])   # slot 0 now free
    hole = manager.allocate_data_qubit(0)            # free list holds only slot 0
    assert hole is reg[0]
    gates_before = len(qc.data)

    result = manager.route_hole_to_port(0, hole, 0)

    # path 0 -> 1 -> 2: both hops displace occupied states -> 2 swaps
    assert result is reg[2]
    assert len(qc.data) == gates_before + 2
    assert result not in manager.in_use_data[0]
    assert result not in manager.free_data[0]
    # displaced states: previous occupants of slots 1 and 2 now at 0 and 1
    assert manager.in_use_data[0][reg[0]] is not None
    assert manager.in_use_data[0][reg[1]] is not None
    for phys, log in manager.in_use_data[0].items():
        assert manager.log_to_phys_idx[log] is phys


def test_double_swap_restores_original_state():
    """Swapping the same pair twice returns the books to the initial state."""
    manager = make_manager()
    occupied = occupied_slots(manager, 0)[0]
    free = free_slot(manager, 0)
    in_use_before = dict(manager.in_use_data[0])
    free_before = set(manager.free_data[0])
    log_to_phys_before = dict(manager.log_to_phys_idx)

    manager.swap_physical_slots(0, occupied, free)
    manager.swap_physical_slots(0, occupied, free)

    assert manager.in_use_data[0] == in_use_before
    assert set(manager.free_data[0]) == free_before
    assert manager.log_to_phys_idx == log_to_phys_before
    assert_books_consistent(manager, 0)
    assert_books_consistent(manager, 1)
