from qiskit import QuantumCircuit
from disqco.graphs.quantum_network import QuantumNetwork
from disqco.scheduling.helper import DATA_REG
from enum import IntEnum

class BufferEvent(IntEnum):
    SWAP_DONE = -1
    RESET_DONE = -2

def spare_slots_by_qpu(circuit: QuantumCircuit, network: QuantumNetwork | None=None):
    used = {qubit for inst in circuit.data for qubit in inst.qubits}
    spare = {}
    for register in circuit.qregs:
        match = DATA_REG.match(register.name)
        if match is None:
            continue
        qpu = int(match.group(1))
        slots = [qubit for qubit in register if qubit not in used]
        if network is not None and network.comm_constrained(qpu):
            topology = network.qpu_topologies[qpu]
            comm_nodes = range(network.qpu_sizes[qpu], network.qpu_sizes[qpu] + network.comm_sizes[qpu])
            slots = [qubit for qubit in slots
                     if all(topology.has_edge(circuit.find_bit(qubit).registers[0][1], node) for node in comm_nodes)]
        spare[qpu] = slots
    return spare

def epr_waiting_to_generate(state, qpu, meta, epr_of):
    for session_id, session_meta in enumerate(meta):
        if session_meta["qpu"] == qpu and session_id not in state.host and epr_of[session_id] in state.ready:
            return True
    return False

def maybe_move_tail(state, session_id, meta, spare, epr_of, swap_time, reset_time):
    qpu = meta[session_id]["qpu"]
    if not epr_waiting_to_generate(state, qpu, meta, epr_of):
        return
    free_slots = [slot for slot in spare.get(qpu, []) if slot not in state.held and state.is_free(slot)]
    if not free_slots:
        return
    slot = max(free_slots, key=lambda s: state.free_at.get(s, 0))
    comm = state.host[session_id]

    state.moved[session_id] = comm
    state.host[session_id] = slot
    state.held[slot] = session_id
    del state.held[comm]
    state.occupy(comm, state.clock + swap_time + reset_time)
    state.occupy(slot, state.clock + swap_time)
    state.push_running(BufferEvent.SWAP_DONE, state.clock + swap_time)
    state.push_running(BufferEvent.RESET_DONE, state.clock + swap_time + reset_time)


