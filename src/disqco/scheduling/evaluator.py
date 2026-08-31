from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode

# Example duractions: {'u': 1, 'cp': 2, "swap": 6, 'measure': 5, 'EPR': 100}

def evaluate_quantum_runtime(circuit: QuantumCircuit, durations: dict, include_clbits=True):
    free_at = {}
    schedule = []
    total_quantum_runtime = 0

    for idx, inst in enumerate(circuit.data):
        wires = wires_of(inst, include_clbits)
        start = max((free_at.get(w, 0) for w in wires), default=0)
        finish = start + duration_of(inst.operation.name, durations)
        schedule.append({"op_idx": idx ,"name": inst.operation.name, "qubits": list(inst.qubits), "start": start, "finish": finish})
        for w in wires:
            free_at[w] = finish
        total_quantum_runtime = max(total_quantum_runtime, finish)

    return (total_quantum_runtime, schedule)

def emit_schedule(circuit, schedule):
    new = circuit.copy_empty_like()
    for entry in sorted(schedule, key=lambda e: (e["start"], e["op_idx"])):
        new.append(circuit.data[entry["op_idx"]])
    return new

def wires_of(inst, include_clbits=True):
    wires = list(inst.qubits)
    if include_clbits:
        wires += list(inst.clbits)
        condition = getattr(inst.operation, "condition", None)
        if condition is not None:
            target = condition[0]
            wires += list(target) if hasattr(target, "__iter__") else [target]

    return wires

def duration_of(name, durations):
    if name in ("barrier", "delay"):
        return 0
    return durations.get(name, 1)


