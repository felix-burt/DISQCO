import networkx as nx

from bosonic_model import Circuit, DistributedCircuit, Register
from bosonic_model.instructions import GateInstruction

from disqco.circuit_extraction.verification import check_no_cross_partition_instructions


def _distributed_with_shared_remote(qpu_a: int, qpu_b: int, reg_prefix: str) -> DistributedCircuit:
    reg_a = Register(name=f"{reg_prefix}{qpu_a}_0" if reg_prefix == "C" else f"{reg_prefix}{qpu_a}_q", size=1, base=0)
    reg_b = Register(name=f"{reg_prefix}{qpu_b}_0" if reg_prefix == "C" else f"{reg_prefix}{qpu_b}_q", size=1, base=1)
    remote = GateInstruction(
        name="remote_test_link",
        qubits=[0, 1],
        params=[],
        opaque=True,
    )
    circ_a = Circuit(qregs={reg_a.name: reg_a}, cregs={}, instructions=[remote])
    circ_b = Circuit(qregs={reg_b.name: reg_b}, cregs={}, instructions=[remote])
    return DistributedCircuit(
        qubits_per_node={qpu_a: [0], qpu_b: [1]},
        circuits={qpu_a: circ_a, qpu_b: circ_b},
    )


def test_verifier_rejects_shared_remote_instruction_between_unconnected_qpus():
    distributed = _distributed_with_shared_remote(0, 1, "C")
    qpu_graph = nx.Graph()
    qpu_graph.add_nodes_from([0, 1])

    assert check_no_cross_partition_instructions(distributed, qpu_graph) is False


def test_verifier_accepts_shared_remote_instruction_between_connected_qpus():
    distributed = _distributed_with_shared_remote(0, 1, "C")
    qpu_graph = nx.Graph()
    qpu_graph.add_edge(0, 1)

    assert check_no_cross_partition_instructions(distributed, qpu_graph) is True
