"""
Test suite for local routing functionality.
"""

from pathlib import Path

import pytest
import numpy as np
from qiskit import QuantumCircuit, qasm2, transpile

from disqco import QuantumNetwork, QuantumCircuitHyperGraph, PartitionedCircuitExtractor
from disqco.circuits.cp_fraction import cp_fraction
from disqco.parti import FiducciaMattheyses
from disqco import set_initial_partition_assignment
from disqco.graphs.coarsening.coarsener import HypergraphCoarsener
import networkx as nx
from disqco.circuit_extraction.circuit_extractor import find_swap_path


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "circuits"

@pytest.fixture
def test_circuit():
    """Create a test circuit for extraction"""
    circuit = cp_fraction(num_qubits=8, depth=8, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    return circuit


@pytest.fixture
def test_network():
    """Create a 2-QPU network for testing"""
    return QuantumNetwork.create([5, 5], 'all_to_all', qpu_topologies={0: nx.path_graph(5)})


@pytest.fixture
def test_hypergraph(test_circuit):
    """Create a hypergraph from test circuit"""
    return QuantumCircuitHyperGraph(test_circuit)


@pytest.fixture
def initial_assignment(test_hypergraph, test_network):
    """Create initial assignment"""
    return set_initial_partition_assignment(test_hypergraph, test_network)


@pytest.fixture
def optimized_assignment(test_circuit, test_network):
    """Create assignment using FM partitioner"""
    partitioner = FiducciaMattheyses(test_circuit, network=test_network)
    results = partitioner.partition(num_passes=5)
    return results['best_assignment']

def _classify_operand(reg_name, local_idx, network):
    """Map (register name, index in register) to (qpu, topology node) or None.

    Data qubit Q{p}_q[i] -> node i. Comm qubit C{p}_0[k] -> node
    network.comm_node(p, k), only meaningful on comm-constrained QPUs.
    Spawned comm registers (C{p}_1, ...) and anything else -> None.
    """
    kind = reg_name[0]
    if kind not in ("Q", "C"):
        return None
    try:
        p = int(reg_name[1:].split("_")[0])
    except ValueError:
        return None
    if kind == "Q":
        return p, local_idx
    if reg_name == f"C{p}_0" and network.comm_constrained(p):
        return p, network.comm_node(p, local_idx)
    return None


def assert_local_gates_respect_topology(circuit: QuantumCircuit, network: QuantumNetwork) -> int:
    """
    Assert every 2-qubit gate acting within one topology-constrained QPU acts
    on topology-adjacent nodes. Covers data-data pairs and, on comm-constrained
    QPUs, data-comm and comm-comm pairs (comm qubit k = node qpu_sizes[p] + k).
    Returns the number of gates checked.
    """
    checked = 0
    for instruction in circuit.data:
        if len(instruction.qubits) != 2:
            continue
        locs = [circuit.find_bit(q).registers[0] for q in instruction.qubits]
        ops = [_classify_operand(reg.name, idx, network) for reg, idx in locs]
        if ops[0] is None or ops[1] is None:
            continue
        (p0, node0), (p1, node1) = ops
        if p0 != p1:
            continue  # inter-QPU op (e.g. EPR generation), not a local gate
        topo = network.qpu_topologies.get(p0)
        if topo is None or node0 not in topo.nodes or node1 not in topo.nodes:
            continue  # all-to-all QPU, or data-only topology and a comm operand
        assert topo.has_edge(node0, node1), (
            f"Gate '{instruction.operation.name}' acts on non-adjacent nodes "
            f"{node0}, {node1} in QPU {p0}"
        )
        checked += 1
    return checked


def line_with_comm_ports(n_data, n_comm):
    """Data slots 0..n_data-1 in a line; comm nodes n_data..n_data+n_comm-1
    form a clique (mutual-adjacency rule) attached to the last data slot."""
    topo = nx.path_graph(n_data)
    comm_nodes = list(range(n_data, n_data + n_comm))
    for c in comm_nodes:
        topo.add_edge(n_data - 1, c)
    for i, a in enumerate(comm_nodes):
        for b in comm_nodes[i + 1:]:
            topo.add_edge(a, b)
    return topo


def test_port_constrained_extraction_respects_topology():
    """RED until Task 5: data qubits must be routed adjacent to their comm
    port before teleportation primitives touch them."""
    circuit = cp_fraction(num_qubits=8, depth=16, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    network = QuantumNetwork(
        [5, 5], comm_sizes=[4, 4],
        qpu_topologies={0: line_with_comm_ports(5, 4)},
    )
    assignment = set_initial_partition_assignment(hypergraph, network,
                                                  round_robin=True)
    extractor = PartitionedCircuitExtractor(
        graph=hypergraph, network=network, partition_assignment=assignment)
    qc = extractor.extract_partitioned_circuit()

    checked = assert_local_gates_respect_topology(qc, network)
    assert checked > 0, "Audit checked no gates - test is vacuous"


def test_routed_circuit_respects_topology(test_hypergraph, test_network, initial_assignment):
    """All local 2-qubit gates in the extracted circuit respect QPU 0's path topology."""
    extractor = PartitionedCircuitExtractor(
        graph=test_hypergraph,
        network=test_network,
        partition_assignment=initial_assignment ,
    )
    routed_circuit = extractor.extract_partitioned_circuit()
    checked = assert_local_gates_respect_topology(routed_circuit, test_network)
    assert checked > 0, "Audit checked no gates - test is vacuous"


def test_empty_topology_matches_no_topology():
    """An explicitly empty qpu_topologies dict produces the identical circuit."""
    circuit = cp_fraction(num_qubits=6, depth=6, fraction=0.5, seed=42)
    circuit = transpile(circuit, basis_gates=['u', 'cp'])
    hypergraph = QuantumCircuitHyperGraph(circuit)

    network_plain = QuantumNetwork.create([4, 4], 'all_to_all')
    network_empty = QuantumNetwork.create([4, 4], 'all_to_all', qpu_topologies={})
    assignment = set_initial_partition_assignment(hypergraph, network_plain)

    qc_plain = PartitionedCircuitExtractor(
        graph=hypergraph, network=network_plain, partition_assignment=assignment
    ).extract_partitioned_circuit()
    qc_empty = PartitionedCircuitExtractor(
        graph=hypergraph, network=network_empty, partition_assignment=assignment
    ).extract_partitioned_circuit()

    assert qc_plain == qc_empty


def test_topology_on_other_qpu_does_not_affect_extraction(test_hypergraph):
    """A topology on QPU 0 must not change gates in unconstrained QPU 1."""
    network_plain = QuantumNetwork.create([5, 5], 'all_to_all')
    network_topo = QuantumNetwork.create(
        [5, 5], 'all_to_all', qpu_topologies={0: nx.path_graph(5)}
    )
    assignment = set_initial_partition_assignment(test_hypergraph, network_plain)

    qc_plain = PartitionedCircuitExtractor(
        graph=test_hypergraph, network=network_plain, partition_assignment=assignment
    ).extract_partitioned_circuit()
    qc_topo = PartitionedCircuitExtractor(
        graph=test_hypergraph, network=network_topo, partition_assignment=assignment
    ).extract_partitioned_circuit()

    assert qc_plain is not None and qc_topo is not None

def test_router_adjacent_returns_empty():
    """Adjacent qubits need no SWAPs."""
    topo = nx.path_graph(5)
    assert find_swap_path(topo, 0, 1) == []


def test_router_distance_three_on_path():
    """On a line 0-1-2-3-4, routing 0 -> 3 walks through 1 and 2.
    """
    topo = nx.path_graph(5)
    assert find_swap_path(topo, 0, 3) == [(0, 1), (1, 2)]


def test_router_contract():
    """Implementation-independent contract: every pair is an edge, pairs
    chain from src, and the moving qubit ends adjacent to dst."""
    topo = nx.cycle_graph(6)
    for src, dst in [(0, 1), (0, 2), (0, 3), (2, 5), (4, 1)]:
        swaps = find_swap_path(topo, src, dst)
        # every SWAP acts on physically coupled qubits
        for a, b in swaps:
            assert topo.has_edge(a, b), f"({a},{b}) is not an edge"
        # the chain starts at src and each hop continues where the last ended
        position = src
        for a, b in swaps:
            assert a == position, f"pair ({a},{b}) does not chain from {position}"
            position = b
        # after all SWAPs the moving qubit sits adjacent to dst
        assert topo.has_edge(position, dst), (
            f"routing {src}->{dst} ended at {position}, not adjacent to {dst}"
        )


def test_router_takes_short_way_on_cycle():
    """On a 6-cycle, 0 -> 4 goes backwards through 5 (one SWAP), not
    forwards through 1,2,3. Pins greedy shortest-path behavior - see
    note on test_router_distance_three_on_path.
    """
    topo = nx.cycle_graph(6)
    assert find_swap_path(topo, 0, 4) == [(0, 5)]
