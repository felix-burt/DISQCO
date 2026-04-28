"""Verification helpers for partitioned/distributed circuits.

`check_no_cross_partition_instructions` validates that the per-node circuits in a
DistributedCircuit only contain locally-feasible gates: any 2-qubit instruction must
either touch one QPU's data + one of its own comm qubits, or two comm qubits whose
QPUs are connected in the network's qpu_graph (this is the EPR / remote_link channel).

`run_sampler` and friends are kept for backwards compatibility with notebooks that
sample from a Qiskit QuantumCircuit (the round-trip path: DistributedCircuit ->
as_monolithic_circuit -> CircuitConverters.to_qiskit -> run_sampler).
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from bosonic_model import DistributedCircuit
from bosonic_model.instructions import (
    BarrierInstruction,
    ConditionalInstruction,
    GateInstruction,
    Instruction,
    MeasureInstruction,
    ResetInstruction,
)


_REG_RE = re.compile(r"^([QC])(\d+)")


def _parse_reg_name(name: str) -> tuple[str, int] | None:
    m = _REG_RE.match(name or "")
    if m is None:
        return None
    return m.group(1), int(m.group(2))


def check_no_cross_partition_instructions(distributed: DistributedCircuit, qpu_graph) -> bool:
    """Validate that every two-qubit instruction in a DistributedCircuit is locally feasible.

    Allowed:
    - Both qubits on the same QPU (Q{i}_q-Q{i}_q, Q{i}_q-C{i}_*, C{i}_*-C{i}_*).
    - Cross-QPU 2-qubit gates between two C{i}_* and C{j}_* registers when (i,j) is an edge
      in `qpu_graph` (the EPR / remote_link channel).

    Disallowed:
    - Two-qubit gates between data registers Q{i}_q and Q{j}_q with i != j.
    - Cross-QPU C-C gates between unconnected QPUs.
    """
    valid = True
    global_qubit_kind: dict[int, tuple[str, int]] = {}
    for circ in distributed.circuits.values():
        for reg in circ.qregs.values():
            parsed = _parse_reg_name(reg.name)
            if parsed is None:
                continue
            kind, idx = parsed
            for offset in range(reg.size):
                global_qubit_kind[reg.base + offset] = (kind, idx)

    for node, circ in distributed.circuits.items():
        for raw in circ.instructions:
            inst = raw.op if isinstance(raw, ConditionalInstruction) else raw
            if isinstance(inst, (BarrierInstruction, MeasureInstruction, ResetInstruction)):
                continue
            qubits = list(getattr(inst, "qubits", []) or [])
            if len(qubits) != 2:
                continue
            kinds = [global_qubit_kind.get(q) for q in qubits]
            if any(k is None for k in kinds):
                continue
            (k0, r0), (k1, r1) = kinds  # type: ignore[misc]
            name = str(getattr(inst, "name", "")).lower()
            if k0 == "Q" and k1 == "Q" and r0 != r1:
                print(f"[node {node}] Invalid {name} between data registers Q{r0} and Q{r1} on qubits {qubits}")
                valid = False
            elif k0 == "C" and k1 == "C" and r0 != r1:
                if not (qpu_graph.has_edge(r0, r1) or qpu_graph.has_edge(r1, r0)):
                    print(f"[node {node}] Invalid {name} between comm registers C{r0} and C{r1} (not connected) on qubits {qubits}")
                    valid = False
    return valid


def run_sampler(circuit, shots=4096):
    """Run a Qiskit circuit through the AER sampler. Use for circuits produced by
    `CircuitConverters.to_qiskit(distributed.as_monolithic_circuit())` for example."""
    from qiskit import transpile
    from qiskit_aer.primitives import SamplerV2
    sampler = SamplerV2()
    num_qubits = circuit.num_qubits
    dec_circuit = circuit.copy()
    # Decompose any opaque named gates (EPR/remote_link_*) into their unitary definitions.
    dec_circuit = dec_circuit.decompose()
    if num_qubits <= 13:
        job = sampler.run([dec_circuit], shots=shots)
        job_result = job.result()
        data = job_result[0].data
    else:
        print("Too many qubits")
        data = None
    return data


def plot(data, labels=False):
    from qiskit.visualization import plot_histogram
    if data is None:
        print("No data to plot")
        return
    if 'result' in data:
        info = data['result']
    elif 'meas' in data:
        info = data['meas']
    elif 'measure' in data:
        info = data['measure']
    else:
        print("No data to plot")
        return

    counts_base = info.get_counts()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    plot_histogram(counts_base, bar_labels=False, ax=ax)
    if not labels:
        ax.set_xticks([])


def get_fidelity(data1, data2, shots):
    if data1 is None or data2 is None:
        print("No data to compare")
        return None
    if 'result' in data1:
        info1 = data1['result']
    else:
        info1 = data1['meas']

    if 'result' in data2:
        info2 = data2['result']
    else:
        info2 = data2['meas']

    counts1 = info1.get_counts()
    counts2 = info2.get_counts()
    for key in counts1:
        digits = len(key)
        break
    norm = 0
    max_string = '1' * digits
    integer = int(max_string, 2)
    for i in range(integer + 1):
        binary = bin(i)
        binary = binary[2:]
        binary = '0' * (digits - len(binary)) + binary
        if binary in counts1:
            counts1_val = counts1[binary] / shots
        else:
            counts1_val = 0
        if binary in counts2:
            counts2_val = counts2[binary] / shots
        else:
            counts2_val = 0
        norm += np.abs(counts1_val - counts2_val)
    return 1 - norm**2


# Backwards-compat alias for the previous Qiskit-circuit validator.
def check_no_cross_partition_gates(distributed: DistributedCircuit, qpu_graph) -> bool:
    return check_no_cross_partition_instructions(distributed, qpu_graph)
