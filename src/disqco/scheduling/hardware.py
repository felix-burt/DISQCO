from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.scheduling.helper import COMM_REG

def comm_index(circuit: QuantumCircuit, qubit):
    """The index of a communiation qubit given the circuit"""
    register, idx = circuit.find_bit(qubit).registers[0]
    match = COMM_REG.match(register.name)
    if match is None or match.group(2) != "0":
        return None
    return idx

def hosts_for_session(session_id, meta, pool, network: QuantumNetwork, circuit: QuantumCircuit):
    """Communication qubit hosts the given session can be hosted on"""
    qpu = meta[session_id]["qpu"]
    candidates = pool[qpu]
    if network is None or qpu not in network.comm_links:
        return list(candidates)

    partners = meta[session_id]["partners"]
    partner = next(iter(partners))

    allowed = network.comm_qubits_for_link(qpu, partner)
    elibible = []
    for qubit in candidates:
        idx = comm_index(circuit, qubit)
        if idx is None or idx in allowed:
            elibible.append(qubit)

    return elibible

def hosts_by_session(sessions, meta, pool, network, circuit):
    return [hosts_for_session(session_id, meta, pool, network, circuit)
            for session_id in range(len(sessions))]
