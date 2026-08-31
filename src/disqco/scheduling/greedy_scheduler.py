import heapq
from collections import defaultdict
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import CommutationChecker
from disqco.scheduling.evaluator import wires_of, duration_of

cc = CommutationChecker()

# Example duractions: {'u': 1, 'cp': 2, "swap": 6, 'measure': 5, 'EPR': 100}
def greedy_scheduler(circuit: QuantumCircuit, durations: dict, priority="critical_path", include_clbits=True):
    total_quantum_runtime = 0
    schedule = []
    grouped = group_commutative_gates(circuit)
    grouped_graph = create_grouped_graph(circuit, grouped)
    weight = downstream_weights(grouped_graph, circuit, durations)

    if priority == "fifo":
        weight = {n: -n for n in grouped_graph.nodes}

    pred_count = {n: d for n, d in grouped_graph.in_degree()}
    ready = {n for n, d in pred_count.items() if d == 0}
    free_at = {}
    running = [] # heap of (finish, op_idx)
    clock = 0

    while ready or running:
        startable = sorted(
            (op for op in ready
            if all(free_at.get(w, 0) <= clock for w in wires_of(circuit.data[op], include_clbits))),
            key=lambda op: weight[op], reverse=True)

        for op in startable:
            wires = wires_of(circuit.data[op], include_clbits)
            if any(free_at.get(w, 0) > clock for w in wires):
                 continue
            finish = clock + duration_of(circuit.data[op].operation.name, durations)
            schedule.append({"op_idx": op, "name": circuit.data[op].operation.name, "qubits": list(circuit.data[op].qubits), "start": clock, "finish": finish})
            heapq.heappush(running, (finish, op))
            for w in wires:
                free_at[w] = finish
            ready.discard(op)
            total_quantum_runtime = max(total_quantum_runtime, finish)

        if running:
            clock = running[0][0]
            while running and running[0][0] == clock:
                _, done = heapq.heappop(running)
                for s in grouped_graph.successors(done):
                    pred_count[s] -= 1
                    if pred_count[s] == 0:
                        ready.add(s)

    assert len(schedule) == len(circuit.data), "dispatcher stalled"
    return (total_quantum_runtime, schedule)

def downstream_weights(graph, circuit, durations):
    weight = {}
    for n in reversed(list(nx.topological_sort(graph))):
        weight[n] = duration_of(circuit.data[n].operation.name, durations) + max((weight[s] for s in graph.successors(n)), default=0)
    return weight

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
        for w in wires_of(inst):
            wire_ops[w].append(i)
    return wire_ops

def operation_commutes(inst_a, inst_b):
    if inst_a.clbits or inst_b.clbits:
        return False
    if getattr(inst_a.operation, "condition", None) or getattr(inst_b.operation, "condition", None):
        return False
    return cc.commute(inst_a.operation, inst_a.qubits, inst_a.clbits, inst_b.operation, inst_b.qubits, inst_b.clbits)