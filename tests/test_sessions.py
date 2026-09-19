"""
Test suite for sessions (disqco.scheduling.sessions).

Two pieces, in dependency order.

`rename_classical_bits` fixes a measurement artefact of extraction. The extractor
recycles one classical bit for all of its cat-entanglement measurements, and a
classical bit is a wire, so measurements that have nothing to do with each other
look dependent. Renaming gives every write its own bit. It must remove those
false dependencies without changing what the circuit computes -- in particular
every conditioned correction has to keep reading the measurement it was paired
with, not a stale one.

`find_sessions` cuts each communication qubit's timeline into sessions: the run of
instructions from the EPR that creates its half of an entangled pair through to
the reset that releases it. A session is the unit the scheduler re-hosts, because
every instruction in it acts on the same physical qubit, so the cut has to be
exact.
"""

from functools import lru_cache

import networkx as nx
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

from disqco import (
    PartitionedCircuitExtractor,
    QuantumCircuitHyperGraph,
    QuantumNetwork,
    set_initial_partition_assignment,
)
from disqco.circuits.cp_fraction import cp_fraction
from disqco.scheduling.evaluator import evaluate_quantum_runtime
from disqco.scheduling.sessions import find_sessions, rename_classical_bits

DURATIONS = {"u": 1, "cp": 2, "cx": 2, "swap": 6, "measure": 5, "EPR": 100}


# --------------------------------------------------------------------------- #
# circuits to test against
# --------------------------------------------------------------------------- #

def ring_with_ports(n_data, n_comm):
    """A QPU whose comm qubits all hang off its last data qubit, as in the demo."""
    topology = nx.path_graph(n_data)
    comm_nodes = range(n_data, n_data + n_comm)
    for comm in comm_nodes:
        topology.add_edge(n_data - 1, comm)
    for a in comm_nodes:
        for b in comm_nodes:
            if a < b:
                topology.add_edge(a, b)
    return topology


@lru_cache(maxsize=None)
def extract(num_qubits=10, depth=20, comm=6, ported=False):
    """An extracted distributed circuit, and the network it was extracted for.

    Cached because extraction is the slow part and every test wants the same
    circuit. `ported` builds the demo's port-constrained network, where each
    QPU's comm qubits reach only its last data qubit.
    """
    circuit = transpile(cp_fraction(num_qubits=num_qubits, depth=depth,
                                    fraction=0.6, seed=7),
                        basis_gates=["u", "cp"])
    size = num_qubits // 2 + 1
    if ported:
        network = QuantumNetwork(
            {0: size, 1: size},
            qpu_topologies={0: ring_with_ports(size, comm),
                            1: ring_with_ports(size, comm)},
            comm_links={0: {k: 1 for k in range(comm)},
                        1: {k: 0 for k in range(comm)}},
            comm_sizes=[comm, comm])
    else:
        network = QuantumNetwork({0: size, 1: size}, comm_sizes=[comm, comm])
    hypergraph = QuantumCircuitHyperGraph(circuit, group_gates=True)
    assignment = set_initial_partition_assignment(hypergraph, network)
    extractor = PartitionedCircuitExtractor(
        hypergraph, network, partition_assignment=assignment)
    return extractor.extract_partitioned_circuit(), network


@lru_cache(maxsize=None)
def sessions_of(**kwargs):
    """(renamed circuit, sessions, session_of, meta) for one extracted circuit."""
    circuit, _ = extract(**kwargs)
    renamed = rename_classical_bits(circuit)
    return (renamed, *find_sessions(renamed))


def correction_circuit():
    """One classical bit, reused, with a correction conditioned on each write.

    This is the shape the extractor emits for cat-entanglement: measure a comm
    qubit, apply a correction conditioned on that bit, then reuse the bit for the
    next link. The outcome here is deterministic -- qubit 1 should be flipped
    exactly once, so `out` reads 1. If renaming lets the second correction read
    the first measurement's value, qubit 1 is flipped twice and `out` reads 0.
    """
    scratch = ClassicalRegister(1, "scratch")
    out = ClassicalRegister(1, "out")
    circuit = QuantumCircuit(QuantumRegister(2, "q"), scratch, out)
    circuit.x(0)                          # so the first measurement reads 1
    circuit.measure(0, scratch[0])
    circuit.x(1).c_if(scratch[0], 1)      # fires
    circuit.reset(0)
    circuit.measure(0, scratch[0])        # writes 0, recycling the bit
    circuit.x(1).c_if(scratch[0], 1)      # must NOT fire
    circuit.measure(1, out[0])
    return circuit, out[0]


def deterministic_bit(circuit, clbit, shots=256):
    """The value of one classical bit, for a circuit with a single outcome."""
    counts = AerSimulator().run(circuit, shots=shots,
                                seed_simulator=1).result().get_counts()
    assert len(counts) == 1, f"expected one outcome, got {counts}"
    bits = next(iter(counts)).replace(" ", "")[::-1]
    return bits[circuit.find_bit(clbit).index]


def written_bits(circuit):
    """How many times each classical bit is written."""
    counts = {}
    for instruction in circuit.data:
        for clbit in instruction.clbits:
            counts[clbit] = counts.get(clbit, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# rename_classical_bits
# --------------------------------------------------------------------------- #

def test_circuit_with_no_reuse_is_returned_unchanged():
    """Nothing to rename means nothing to rebuild: the same object comes back."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    assert rename_classical_bits(circuit) is circuit


def test_extractor_really_does_reuse_a_bit():
    """The premise of the whole function, asserted rather than assumed."""
    circuit, _ = extract()
    reused = [count for count in written_bits(circuit).values() if count > 1]
    assert reused, "extractor no longer reuses classical bits; renaming is moot"


def test_no_bit_is_written_twice_after_renaming():
    circuit, _ = extract()
    assert all(count == 1 for count in written_bits(rename_classical_bits(circuit)).values())


def test_renaming_preserves_the_instruction_sequence():
    """Renaming rewires; it must not reorder, add or drop anything. Every later
    stage indexes into this circuit, so instruction i has to stay instruction i."""
    circuit, _ = extract()
    renamed = rename_classical_bits(circuit)
    assert len(renamed.data) == len(circuit.data)
    assert ([instruction.operation.name for instruction in renamed.data]
            == [instruction.operation.name for instruction in circuit.data])
    assert renamed.global_phase == circuit.global_phase


def test_renaming_only_adds_classical_bits():
    circuit, _ = extract()
    renamed = rename_classical_bits(circuit)
    assert renamed.num_qubits == circuit.num_qubits
    assert renamed.num_clbits > circuit.num_clbits


def test_renaming_removes_false_dependencies():
    """The point of the exercise: a shared bit serialises unrelated measurements,
    so removing the sharing must shorten the as-soon-as-possible makespan."""
    circuit, _ = extract()
    before = evaluate_quantum_runtime(circuit, DURATIONS)[0]
    after = evaluate_quantum_runtime(rename_classical_bits(circuit), DURATIONS)[0]
    assert after < before


def test_shared_bit_is_what_causes_the_dependency():
    """Control for the test above: with classical wires ignored, renaming can
    make no difference at all. If it does, something other than clbits changed."""
    circuit, _ = extract()
    before = evaluate_quantum_runtime(circuit, DURATIONS, include_clbits=False)[0]
    after = evaluate_quantum_runtime(rename_classical_bits(circuit), DURATIONS,
                                     include_clbits=False)[0]
    assert after == before


def test_second_write_goes_to_a_fresh_bit():
    circuit, _ = correction_circuit()
    renamed = rename_classical_bits(circuit)
    first, second = [instruction for instruction in renamed.data
                     if instruction.operation.name == "measure"][:2]
    assert first.clbits[0] is not second.clbits[0]
    assert second.clbits[0] not in circuit.clbits, "should be from the new register"


def test_each_correction_reads_its_own_measurement():
    """The bug this guards against is silent: leave the conditions alone and the
    second correction reads the first measurement's result, so the circuit
    computes something else while still looking well-formed."""
    circuit, _ = correction_circuit()
    renamed = rename_classical_bits(circuit)
    measures = [instruction for instruction in renamed.data
                if instruction.operation.name == "measure"]
    corrections = [instruction for instruction in renamed.data
                   if instruction.operation.name == "x"
                   and getattr(instruction.operation, "condition", None) is not None]
    for measure, correction in zip(measures, corrections):
        assert correction.operation.condition[0] is measure.clbits[0]


def test_renaming_preserves_what_the_circuit_computes():
    """Deterministic end to end: qubit 1 is flipped once either way."""
    circuit, out = correction_circuit()
    assert deterministic_bit(circuit, out) == "1"
    assert deterministic_bit(rename_classical_bits(circuit), out) == "1"


# --------------------------------------------------------------------------- #
# find_sessions
# --------------------------------------------------------------------------- #

def test_two_sessions_per_entangled_pair():
    """An EPR touches one comm qubit on each QPU, so it opens two sessions."""
    circuit, sessions, _, _ = sessions_of()
    assert len(sessions) == 2 * circuit.count_ops()["EPR"]


def test_every_epr_instruction_joins_exactly_two_sessions():
    circuit, sessions, _, _ = sessions_of()
    for index, instruction in enumerate(circuit.data):
        if instruction.operation.name == "EPR":
            assert sum(1 for session in sessions if index in session) == 2


def test_sessions_run_from_an_epr_to_a_reset():
    """The definition, and the one thing a later stage cannot recover if wrong.
    A session does not end at its measurement: the comm qubit is still held."""
    circuit, sessions, _, _ = sessions_of()
    for session in sessions:
        assert circuit.data[session[0]].operation.name == "EPR"
        assert circuit.data[session[-1]].operation.name == "reset"


def test_instruction_indices_increase_within_a_session():
    _, sessions, _, _ = sessions_of()
    for session in sessions:
        assert session == sorted(session)
        assert len(session) == len(set(session)), "an index recorded twice"


def test_session_of_covers_every_instruction_of_every_session():
    """`session_of` is keyed by (index, qubit) because one EPR instruction sits in
    two sessions; the qubit is what distinguishes them."""
    _, sessions, session_of, meta = sessions_of()
    expected = {(index, meta[sid]["qubit"])
                for sid, session in enumerate(sessions) for index in session}
    assert set(session_of) == expected
    assert all(session_of[(index, meta[sid]["qubit"])] == sid
               for sid, session in enumerate(sessions) for index in session)


def test_every_session_talks_to_exactly_one_other_qpu():
    _, _, _, meta = sessions_of()
    for facts in meta:
        assert len(facts["partners"]) == 1
        assert facts["qpu"] not in facts["partners"]


def test_sessions_are_shared_between_both_qpus():
    """Each entangled pair puts one session on each side, so the split is even."""
    _, sessions, _, meta = sessions_of()
    per_qpu = {}
    for facts in meta:
        per_qpu[facts["qpu"]] = per_qpu.get(facts["qpu"], 0) + 1
    assert set(per_qpu) == {0, 1}
    assert per_qpu[0] == per_qpu[1] == len(sessions) // 2


def test_recorded_data_qubits_are_local_and_real():
    """A session's `data` set feeds the topology check later, so every index in it
    must name a data qubit of that session's own QPU."""
    circuit, _, _, meta = sessions_of()
    sizes = {0: 0, 1: 0}
    for register in circuit.qregs:
        if register.name.startswith("Q"):
            sizes[int(register.name[1:].split("_")[0])] = register.size
    for facts in meta:
        for index in facts["data"]:
            assert 0 <= index < sizes[facts["qpu"]]


def test_port_constrained_network_reaches_only_its_port_qubit():
    """On the demo's network every comm qubit hangs off the last data qubit, so a
    session can only ever touch that one. A good check that `data` is read from
    the circuit rather than guessed."""
    circuit, _, _, meta = sessions_of(ported=True)
    port = next(register.size for register in circuit.qregs
                if register.name == "Q0_q") - 1
    assert all(facts["data"] == {port} for facts in meta)


def test_overflow_registers_still_form_sessions():
    """When a QPU runs out of comm qubits the extractor invents C{p}_1, C{p}_2...
    Their work still has to be scheduled, so it still has to be found here. That
    the hardware lacks those qubits matters later, when hosts are handed out."""
    circuit, _ = extract(num_qubits=8, depth=12, comm=1)
    overflow = [register for register in circuit.qregs
                if register.name.startswith("C") and not register.name.endswith("_0")]
    assert overflow, "expected the extractor to invent registers at comm_sizes=1"

    renamed = rename_classical_bits(circuit)
    _, session_of, meta = find_sessions(renamed)
    overflow_qubits = {qubit for register in overflow for qubit in register}
    assert any(facts["qubit"] in overflow_qubits for facts in meta)
    assert all(qubit in circuit.qubits for _, qubit in session_of)


def test_no_session_is_left_open():
    """Every comm qubit the extractor uses is eventually reset. `find_sessions`
    asserts this itself; this test is what makes the assertion visible."""
    for kwargs in ({}, {"ported": True}, {"num_qubits": 8, "depth": 12, "comm": 1}):
        circuit, _ = extract(**kwargs)
        find_sessions(rename_classical_bits(circuit))
