import re
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit

_DATA_REG = re.compile(r"^Q(\d+)_q$")
_COMM_REG = re.compile(r"^C(\d+)_(\d+)$")

def rename_classical_bits(circuit: QuantumCircuit):
    """Assign unique wire to each classical bit to remove false dependencies when scheduling the circuit."""
    clbit_collisions = {}
    for _, inst  in enumerate(circuit.data):
        for clbit in inst.clbits:
            if clbit in clbit_collisions:
                clbit_collisions[clbit] += 1
            else:
                clbit_collisions[clbit] = 0

    if any(count > 0 for count in clbit_collisions.values()):
        extra = ClassicalRegister(sum(clbit_collisions.values()), "schedule_extra")
        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, extra, name=circuit.name)
        new_circuit.global_phase = circuit.global_phase

        fresh = iter(extra)
        current = {}
        touched = set()
        for _, inst in enumerate(circuit.data):
            new_clbits = []
            for clbit in inst.clbits:
                if clbit in touched:
                    current[clbit] = next(fresh)
                touched.add(clbit)
                new_clbits.append(current.get(clbit, clbit))

            op = inst.operation.copy()
            condition = getattr(op, "condition", None)
            if condition is not None:
                touched.add(condition[0])
                op.condition = (current.get(condition[0], condition[0]), condition[1])
            new_circuit.append(op, inst.qubits, new_clbits)
        return new_circuit
    return circuit

def map_comm_qubits_to_qpus(circuit: QuantumCircuit):
    qpu_of = {}
    for register in circuit.qregs:
        match = _COMM_REG.match(register.name)
        if match is None:
            continue
        for comm_qubit in register:
            qpu_of[comm_qubit] = int(match.group(1))
    return qpu_of

def local_data_qubit(circuit: QuantumCircuit, inst, comm_qubit, qpu):
    others = [qubit for qubit in inst.qubits if qubit is not comm_qubit]
    if not others:
        return None
    register, index = circuit.find_bit(others[0]).registers[0]
    match = _DATA_REG.match(register.name)

    # Other end is a communication qubit
    if match is None:
        return None
    # Other end is a data qubit, but not local to the QPU of the communication qubit
    if int(match.group(1)) != qpu:
        return None

    return index

def partner_qpu(inst, comm_qubit, qpu_of):
    """Gives QPU of the other end of an EPR pair given one of the comm_qubits"""
    if inst.operation.name != "EPR":
        return None
    for qubit in inst.qubits:
        if qubit is not comm_qubit:
            return qpu_of.get(qubit, None)
    return None

def find_sessions(circuit: QuantumCircuit):
    qpu_of = map_comm_qubits_to_qpus(circuit)
    sessions, meta, session_of, open_session = [], [], {}, {}

    def start_session(comm_qubit: Qubit):
        session_id = len(sessions)
        sessions.append([])
        meta.append({
            "qubit": comm_qubit,
            "qpu": qpu_of[comm_qubit],
            "partners": set(),
            "data": set()
        })
        open_session[comm_qubit] = session_id
        return session_id

    for idx, inst in enumerate(circuit.data):
        for qubit in inst.qubits:
            if qubit not in qpu_of:
                continue    # ignore the data qubits

            if qubit not in open_session:
                start_session(qubit)
            session_id = open_session[qubit]

            sessions[session_id].append(idx)
            session_of[(idx, qubit)] = session_id

            partner = partner_qpu(inst, qubit, qpu_of)
            if partner is not None:
                meta[session_id]["partners"].add(partner)

            data_index = local_data_qubit(circuit, inst, qubit, qpu_of[qubit])
            if data_index is not None:
                meta[session_id]["data"].add(data_index)

            if inst.operation.name == "reset":
                del open_session[qubit]

    assert not open_session, f"{len(open_session)} open sessions at the end of the circuit"
    return sessions, session_of, meta
