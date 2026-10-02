from qiskit import QuantumCircuit

from disqco.scheduling.evaluator import evaluate_quantum_runtime

def rehosted_qubits(inst, idx, session_of, state, epr_of):
    qubits = []
    for qubit in inst.qubits:
        session_id = session_of.get((idx, qubit))
        if session_id is None:
            # Data or wire no session, put back to where it should be
            qubits.append(qubit)
        elif session_id in state.moved and idx == epr_of[session_id]:
            # comm qubit epr tail moved to other: state.moved[session_Id] is original comm and state.host[session_id] is the new
            qubits.append(state.moved[session_id])
        else:
            # Communication qubit in a session
            qubits.append(state.host[session_id])
    return qubits

def schedule_to_circuit(circuit: QuantumCircuit, state, session_of, sessions=None):
    epr_of = [instrutions[0] for instrutions in sessions or []]
    events = [(state.start[idx], 0, idx, idx) for idx in range(len(circuit.data))]
    for session_id, comm in state.moved.items():
        ends = state.end[epr_of[session_id]]
        events.append((ends, 1, session_id, ("swap", comm, state.host[session_id])))
        events.append((ends, 2, session_id, ("reset", comm)))
    events.sort(key=lambda event: event[:3])

    new_circuit = circuit.copy_empty_like()
    for _, _, _, event in events:
        # insert swap and reset from buffered epr
        if isinstance(event, tuple):
            if event[0] == "swap":
                new_circuit.swap(event[1], event[2])
            else:
                new_circuit.reset(event[1])
            continue
        inst = circuit.data[event]
        new_circuit.append(inst.operation, rehosted_qubits(inst, event, session_of, state, epr_of), inst.clbits)
    return new_circuit

def best_of(original, candidates, durations, include_clbits=True):
    original_makespan, original_schedule = evaluate_quantum_runtime(original, durations, include_clbits)
    best = (original_makespan, original_schedule, original)
    for candidate in candidates:
        makespan, schedule = evaluate_quantum_runtime(candidate, durations, include_clbits)
        if makespan < best[0]:
            best = (makespan, schedule, candidate)
    return best