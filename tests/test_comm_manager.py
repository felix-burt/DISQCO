"""
Test suite for CommunicationQubitManager capacity behaviour.

Without a comm-constrained topology the pool grows on demand (historical
behaviour: a new one-qubit register is spawned when the pool is empty).
When the network declares comm qubit positions for a QPU, capacity is
hard: exhausting the pool raises instead of spawning.
"""

import networkx as nx
import pytest
from qiskit import QuantumRegister, QuantumCircuit

from disqco import QuantumNetwork
from disqco.circuit_extraction.DQC_qubit_manager import CommunicationQubitManager


def make_comm_manager(network=None, comm_size=1):
    """One QPU with a single comm register of comm_size qubits."""
    reg = QuantumRegister(comm_size, name="C0_0")
    qc = QuantumCircuit(reg)
    return CommunicationQubitManager({0: [reg]}, qc, network=network)


def comm_inclusive_topology(n_data=4, n_comm=1):
    """Line of n_data slots with comm qubits chained off the last slot."""
    topo = nx.path_graph(n_data + n_comm)
    return topo


def test_unconstrained_pool_grows_without_network():
    """No network at all: exhausting the pool spawns a new register."""
    manager = make_comm_manager()
    first = manager.find_comm_idx(0)
    second = manager.find_comm_idx(0)

    assert first is not second
    assert len(manager.comm_qregs[0]) == 2
    assert second._register.name == "C0_1"


def test_unconstrained_pool_grows_with_untopologised_network():
    """A network with no topology for the QPU still allows growth."""
    network = QuantumNetwork([4], comm_sizes=[1])
    manager = make_comm_manager(network=network)
    manager.find_comm_idx(0)
    manager.find_comm_idx(0)
    assert len(manager.comm_qregs[0]) == 2


def test_data_only_topology_still_grows():
    """A data-only topology says nothing about comm positions -> growth allowed."""
    network = QuantumNetwork([4], comm_sizes=[1],
                             qpu_topologies={0: nx.path_graph(4)})
    assert network.comm_constrained(0) is False
    manager = make_comm_manager(network=network)
    manager.find_comm_idx(0)
    manager.find_comm_idx(0)
    assert len(manager.comm_qregs[0]) == 2


def test_constrained_pool_raises_when_exhausted():
    """Comm-constrained QPU: second allocation raises and spawns nothing."""
    network = QuantumNetwork([4], comm_sizes=[1],
                             qpu_topologies={0: comm_inclusive_topology()})
    assert network.comm_constrained(0) is True
    manager = make_comm_manager(network=network)

    manager.find_comm_idx(0)
    with pytest.raises(RuntimeError, match="no free communication qubit"):
        manager.find_comm_idx(0)

    assert len(manager.comm_qregs[0]) == 1


def test_constrained_pool_reuses_released_qubit():
    """Releasing a comm qubit makes it allocatable again without raising."""
    network = QuantumNetwork([4], comm_sizes=[1],
                             qpu_topologies={0: comm_inclusive_topology()})
    manager = make_comm_manager(network=network)

    comm = manager.find_comm_idx(0)
    manager.release_comm_qubit(0, comm)
    again = manager.find_comm_idx(0)

    assert again is comm
    assert len(manager.comm_qregs[0]) == 1


# ---------------------------------------------------------------------------
# Link-aware selection (comm_links)
# ---------------------------------------------------------------------------

def make_linked_manager(comm_links, num_qpus=2, comm_size=2):
    """All-to-all network of num_qpus QPUs, each with comm_size comm qubits,
    and a manager holding QPU 0's comm register."""
    network = QuantumNetwork([4] * num_qpus, comm_sizes=[comm_size] * num_qpus,
                             comm_links=comm_links)
    reg = QuantumRegister(comm_size, name="C0_0")
    qc = QuantumCircuit(reg)
    manager = CommunicationQubitManager({0: [reg]}, qc, network=network)
    return manager


def test_bound_link_selects_bound_qubit():
    """comm 1 of QPU 0 serves QPU 1 -> a request for QPU 1 gets index 1, not 0."""
    manager = make_linked_manager({0: {1: 1}})
    comm = manager.find_comm_idx(0, neighbor=1)
    assert manager.comm_index(0, comm) == 1


def test_unbound_neighbor_takes_any():
    """No binding to QPU 2 -> every comm qubit is eligible, pop-front wins."""
    manager = make_linked_manager({0: {1: 1}}, num_qpus=3)
    comm = manager.find_comm_idx(0, neighbor=2)
    assert manager.comm_index(0, comm) == 0


def test_no_bindings_declared_is_unchanged():
    """Without comm_links for QPU 0, neighbor= is ignored (pop-front)."""
    manager = make_linked_manager({})
    first = manager.find_comm_idx(0, neighbor=1)
    second = manager.find_comm_idx(0, neighbor=1)
    assert manager.comm_index(0, first) == 0
    assert manager.comm_index(0, second) == 1


def test_neighbor_none_is_unchanged():
    """Bindings declared but neighbor omitted -> old pop-front behaviour."""
    manager = make_linked_manager({0: {1: 1}})
    comm = manager.find_comm_idx(0)
    assert manager.comm_index(0, comm) == 0


def test_bound_link_exhausted_raises():
    """The only comm qubit bound to QPU 1 is busy -> raise, even though
    another (unbound) comm qubit is free."""
    manager = make_linked_manager({0: {0: 1}})
    manager.find_comm_idx(0, neighbor=1)          # takes comm 0
    assert len(manager.free_comm[0]) == 1         # comm 1 still free
    with pytest.raises(RuntimeError, match="serving"):
        manager.find_comm_idx(0, neighbor=1)


def test_bound_link_released_qubit_reusable():
    manager = make_linked_manager({0: {0: 1}})
    comm = manager.find_comm_idx(0, neighbor=1)
    manager.release_comm_qubit(0, comm)
    assert manager.find_comm_idx(0, neighbor=1) is comm


def test_comm_index_of_spawned_qubit_is_none():
    """Qubits from spawned registers have no declared index."""
    manager = make_comm_manager()                 # unconstrained, no network
    manager.find_comm_idx(0)
    spawned = manager.find_comm_idx(0)
    assert manager.comm_index(0, spawned) is None


# ---------------------------------------------------------------------------
# End-to-end: bindings honoured during real extraction
# ---------------------------------------------------------------------------

def _extract_with_bound_links(comm_size):
    from qiskit import transpile
    from disqco import (QuantumCircuitHyperGraph, PartitionedCircuitExtractor,
                        set_initial_partition_assignment)
    from disqco.circuits.cp_fraction import cp_fraction

    circuit = cp_fraction(num_qubits=8, depth=16, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=["u", "cp"])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    links = {0: {k: 1 for k in range(comm_size)},
             1: {k: 0 for k in range(comm_size)}}
    network = QuantumNetwork([5, 5], comm_sizes=[comm_size, comm_size],
                             comm_links=links)
    assignment = set_initial_partition_assignment(hypergraph, network,
                                                  round_robin=True)
    extractor = PartitionedCircuitExtractor(hypergraph, network, assignment)
    return extractor.extract_partitioned_circuit()


def test_extraction_with_sufficient_bound_comm_qubits_spawns_nothing():
    """With enough bound comm qubits the circuit extracts and no extra comm
    registers are spawned (bindings are honoured end to end)."""
    qc = _extract_with_bound_links(comm_size=4)
    comm_regs = [r.name for r in qc.qregs if r.name.startswith("C")]
    assert comm_regs == ["C0_0", "C1_0"]


def test_extraction_with_insufficient_bound_comm_qubits_raises():
    """Bound links are hard capacity: too few comm qubits for the circuit's
    peak simultaneous link demand raises instead of silently spawning."""
    with pytest.raises(RuntimeError, match="serving"):
        _extract_with_bound_links(comm_size=1)


def test_constrained_pool_allows_up_to_declared_size():
    """With comm_sizes=2, two allocations succeed and the third raises."""
    network = QuantumNetwork([4], comm_sizes=[2],
                             qpu_topologies={0: comm_inclusive_topology(n_comm=2)})
    manager = make_comm_manager(network=network, comm_size=2)

    a = manager.find_comm_idx(0)
    b = manager.find_comm_idx(0)
    assert a is not b
    with pytest.raises(RuntimeError):
        manager.find_comm_idx(0)
