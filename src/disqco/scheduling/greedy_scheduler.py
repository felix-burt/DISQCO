from collections import defaultdict
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.circuit import CommutationChecker

# Example duractions: {'u': 1, 'cp': 2, "swap": 6, 'measure': 5, 'EPR': 100}
cc = CommutationChecker()

def greedy_scheduler(grouped_circuit: QuantumCircuit, durations: dict):
    return

def create_grouped_graph(circuit, grouped_wire_ops):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(circuit.data)))

    for groups in grouped_wire_ops.values():
        for g_prev, g_next in zip(groups, groups[1:]):
            for a in g_prev:
                for b in g_next:
                    graph.add_edge(a, b)

    return graph

def group_commutative_gates(circuit: QuantumCircuit):
    wire_ops = operations_per_wire(circuit)
    grouped_wire_ops = defaultdict(list)

    for w in wire_ops:
        current_group = []
        for op_idx in wire_ops[w]:
            inst_a = circuit.data[op_idx]
            does_commute = True
            for cg_op_idx in current_group:
                inst_b = circuit.data[cg_op_idx]
                if not operation_commutes(inst_a, inst_b):
                    does_commute = False

            if does_commute:
                current_group.append(op_idx)
            else:
                grouped_wire_ops[w].append(current_group)
                current_group = [op_idx]

        if current_group:
            grouped_wire_ops[w].append(current_group)

    return grouped_wire_ops


def operations_per_wire(circuit: QuantumCircuit) -> dict:
    wire_ops = defaultdict(list)
    for i, inst in enumerate(circuit.data):
        for q in list (inst.qubits) + list(inst.clbits):
            wire_ops[q].append(i)
    return wire_ops

def operation_commutes(inst_a, inst_b):
    if inst_a.clbits or inst_b.clbits:
        return False
    if getattr(inst_a.operation, "condition", None) or getattr(inst_b.operation, "condition", None):
        return False
    return cc.commute(inst_a.operation, inst_a.qubits, inst_a.clbits, inst_b.operation, inst_b.qubits, inst_b.clbits)