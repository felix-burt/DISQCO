from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from disqco.scheduling.helper import DATA_REG, COMM_REG

def map_comm_qubits_to_qpus(circuit: QuantumCircuit):
    qpu_of = {}
    for register in circuit.qregs:
        match = COMM_REG.match(register.name)
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
    match = DATA_REG.match(register.name)

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

def sessions_opening_at(sessions):
    """{instruction index: [sessions opening at that instruction]}"""
    opening = {}
    for session_id, instructions in enumerate(sessions):
        opening.setdefault(instructions[0], []).append(session_id)
    return opening

def sessions_ending_at(sessions):
    """"Returns map of instruction index to list of sessions that finish at that instruction"""
    ending = {}
    for session_id, insts in enumerate(sessions):
        last_inst = insts[-1]
        if last_inst not in ending:
            ending[last_inst] = []
        ending[last_inst].append(session_id)
    return ending