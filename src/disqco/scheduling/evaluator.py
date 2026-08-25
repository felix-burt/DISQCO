from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

# Example duractions: {'u': 1, 'cp': 2, "swap": 6, 'measure': 5, 'EPR': 100}

def evaluate_quantum_runtime(circuit: QuantumCircuit, durations: dict):
    dag = circuit_to_dag(circuit)
    ready = {}
    schedule = []
    total_quantum_runtime = 0

    for node in dag.topological_op_nodes():
        if node.name == "barrier" or node.name == "delay":
            continue
        wires = list(node.qargs)
        start = max(ready.get(w, 0) for w in wires)
        finish = start + durations.get(node.name, 1)
        schedule.append({"name": node.name, "qubits": list(node.qargs), "start": start, "finish": finish})
        for w in wires:
            ready[w] = finish
        total_quantum_runtime = max(total_quantum_runtime, finish)

    return (total_quantum_runtime, schedule)