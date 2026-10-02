import heapq
from qiskit import ClassicalRegister, QuantumCircuit
from disqco.scheduling.evaluator import duration_of, wires_of
from disqco.scheduling.schedule_to_circuit import best_of, schedule_to_circuit
from disqco.scheduling.sessions import sessions_ending_at, find_sessions, sessions_opening_at
from disqco.scheduling.helper import comm_qubits_of_qpu, predecessors_and_successors, priority_of_operation
from disqco.scheduling.hardware import hosts_by_session
from disqco.scheduling.buffer import BufferEvent, spare_slots_by_qpu, maybe_move_tail

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

class ScheduleState:
    def __init__(self, num_instructions, preds):
        self.clock = 0
        self.free_at = {}
        self.host  = {}
        self.held = {}
        self.start = [None] * num_instructions
        self.end = [None] * num_instructions
        self.indeg = [len(p) for p in preds]
        self.ready = {idx for idx, count in enumerate(self.indeg) if count == 0}
        self.running = []
        self.moved = {}

    def is_free(self, wire):
        return self.free_at.get(wire, 0) <= self.clock

    def occupy(self, wire, until):
        self.free_at[wire] = until

    def push_running(self, index, finish):
        heapq.heappush(self.running, (finish, index))

    def next_finish(self):
        return self.running[0][0]


def free_hosts(state, session_id, pool):
    """Gives list of comm qubits that are curretly free in the state"""
    return [qubit for qubit in pool[session_id]
            if qubit not in state.held and state.is_free(qubit)]

def claim_order_by_circuit(sessions, meta):
    """{qpu: [session_ids]} that relate to that qpu in order"""
    order = {}
    for session_id, _ in enumerate(sessions):
        qpu = meta[session_id]["qpu"]
        if qpu not in order:
            order[qpu] = []
        order[qpu].append(session_id)
    for qpu in order:
        order[qpu].sort(key=lambda session_id: sessions[session_id][0])
    return order

def claim_order_by_priority(sessions, meta, priority):
    """qpu: [session_ids] with longest path to the end first"""
    order = {}
    for session_id in sorted(range(len(sessions)), key=lambda s: -priority[sessions[s][0]]):
        order.setdefault(meta[session_id]["qpu"], []).append(session_id)
    return order

def earlier_sessions_on_same_qpu(order, meta, session_id):
    qpu = meta[session_id]["qpu"]
    position = order[qpu].index(session_id)
    return order[qpu][:position]

def all_earlier_sessions_have_claimed(state, order, meta, session_id):
    for earlier_session in earlier_sessions_on_same_qpu(order, meta, session_id):
        if earlier_session not in state.host:
            return False
    return True

def most_recently_freed(state, candidates):
    best_host = None
    best_freed_at = None
    for host in candidates:
        freed_at = state.free_at.get(host, 0)
        if best_freed_at is None or freed_at > best_freed_at:
            best_host = host
            best_freed_at = freed_at
    return best_host

def host_chooser(order, meta):
    def choose_host(state, session_id, candidates):
        if not candidates:
            return None
        if not all_earlier_sessions_have_claimed(state, order, meta, session_id):
            return None
        return most_recently_freed(state, candidates)
    return choose_host

def session_needing_a_host(state: ScheduleState, session_of, index, wire):
    session_id = session_of.get((index, wire))
    if session_id is None:
        return None
    if session_id in state.host:
        return None
    return session_id

def wire_is_avaliable(state, session_of, index, wire):
    session_id = session_of.get((index, wire))
    if session_id is None:
        return state.is_free(wire)
    if session_id in state.host:
        return state.is_free(state.host[session_id])
    return True

def claims_to_start(state: ScheduleState, index, circuit_wires, session_of, meta, pool, choose_host):
    claims = {}
    for wire in circuit_wires[index]:
        if not wire_is_avaliable(state, session_of, index, wire):
            return None

        session_id = session_needing_a_host(state, session_of, index, wire)
        if session_id is None:
            continue

        candidates = free_hosts(state, session_id, pool)
        already_promised = set(claims.values())
        candidates = [host for host in candidates if host not in already_promised]

        host = choose_host(state, session_id, candidates)
        if host is None:
            return None

        claims[session_id] = host

    return claims

def start_instruction(state, idx, claims, circuit, circuit_wires, session_of, durations):
    duration = duration_of(circuit.data[idx].operation.name, durations)
    finish = state.clock + duration

    for session_id, host in claims.items():
        state.host[session_id] = host
        state.held[host] = session_id

    state.start[idx] = state.clock
    state.end[idx] = finish

    for wire in circuit_wires[idx]:
        session_id = session_of.get((idx, wire))
        if session_id is not None:
            state.occupy(state.host[session_id], finish)
        else:
            state.occupy(wire, finish)

    state.push_running(idx, finish)
    state.ready.discard(idx)

def instruction_finished(state, idx, succs, sessions_closing_from_instruction, sessions_opening_from_instruction, meta, spare, epr_of, swap_time, reset_time):
    for successor in succs[idx]:
        state.indeg[successor] -= 1
        if state.indeg[successor] == 0:
            state.ready.add(successor)

    for session_id in sessions_closing_from_instruction.get(idx, []):    # if session ends here delete from state.host to free comm
        del state.held[state.host[session_id]]

    for session_id in sessions_opening_from_instruction.get(idx, []):   # if epr just finished generating (will run for 2 session ids)
        maybe_move_tail(state, session_id, meta, spare, epr_of, swap_time, reset_time)


def finish_instructions_at_clock(state, succs, sessions_closing_from_instruction, sessions_opening_from_instruction, meta, spare, epr_of, swap_time, reset_time):
    while state.running and state.next_finish() == state.clock:
        _, idx = heapq.heappop(state.running)
        if idx in BufferEvent:
            continue    # swap/reset sentinel: nothing to release
        instruction_finished(state, idx, succs, sessions_closing_from_instruction, sessions_opening_from_instruction,
                             meta, spare, epr_of, swap_time, reset_time)

def run(circuit, sessions, session_of, meta, preds, succs, priority, durations, pool, choose_host, spare=None):
    circuit_wires = [wires_of(instruction) for instruction in circuit.data]
    sessions_opening_from_instruction = sessions_opening_at(sessions)
    sessions_closing_from_instruction = sessions_ending_at(sessions)
    epr_of = [instructions[0] for instructions in sessions]
    spare = spare or {}
    swap_time, reset_time = duration_of("swap", durations), duration_of("reset", durations)
    state = ScheduleState(len(circuit.data), preds)

    while state.ready or state.running:
        started_anything = True
        while started_anything:
            started_anything = False
            for idx in sorted(state.ready, key=lambda i: -priority[i]):
                claims = claims_to_start(state, idx, circuit_wires, session_of, meta, pool, choose_host)
                if claims is None:
                    continue

                start_instruction(state, idx, claims, circuit, circuit_wires, session_of, durations)
                started_anything = True

        if state.running:
            state.clock = state.next_finish()
            finish_instructions_at_clock(state, succs, sessions_closing_from_instruction, sessions_opening_from_instruction, meta, spare, epr_of, swap_time, reset_time)
        elif state.ready:
            raise RuntimeError("stalled")

    return state

def run_without_stalling(*arguments):
    try:
        return run(*arguments)
    except RuntimeError:
        return None

def prepare(circuit, durations, network=None, include_clbits=True, commuting=False):
    renamed_circuit = rename_classical_bits(circuit)
    sessions, sessions_of, meta = find_sessions(renamed_circuit)
    preds, succs = predecessors_and_successors(renamed_circuit, sessions_of, include_clbits, commuting)
    pool = hosts_by_session(sessions, meta, comm_qubits_of_qpu(renamed_circuit), network, renamed_circuit)
    priority = priority_of_operation(renamed_circuit, succs, durations)
    return renamed_circuit, sessions, sessions_of, meta, preds, succs, pool, priority

def schedule(circuit: QuantumCircuit, durations, network=None, include_clbits=True, commuting=False, buffer=False):
    renamed_circuit, sessions, sessions_of, meta, preds, succs, pool, priority = prepare(circuit, durations, network, include_clbits, commuting)

    candidates = []
    for order in (claim_order_by_circuit(sessions, meta), claim_order_by_priority(sessions, meta, priority)):
        choose_host = host_chooser(order, meta)
        state = run_without_stalling(renamed_circuit, sessions, sessions_of, meta, preds, succs, priority, durations, pool, choose_host)
        if state is not None:
            candidates.append(schedule_to_circuit(renamed_circuit, state, sessions_of, sessions))
        if buffer:
            state = run_without_stalling(renamed_circuit, sessions, sessions_of, meta, preds, succs, priority, durations, pool, choose_host,
                                         spare_slots_by_qpu(renamed_circuit, network))
            if state is not None:
                candidates.append(schedule_to_circuit(renamed_circuit, state, sessions_of, sessions))

    return best_of(renamed_circuit, candidates, durations, include_clbits)

