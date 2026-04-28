"""Convert a bosonic_model.Circuit into the layer-dict format DISQCO consumes.

The output shape mirrors the legacy `circuit_to_gate_layers` + `layer_list_to_dict`
pipeline that previously consumed Qiskit circuits, so the rest of DISQCO
(`QuantumCircuitHyperGraph`, `group_distributable_packets_*`, etc.) can be reused
unchanged.

Layer dict shape:
    dict[int, list[gate_dict]]

gate_dict keys:
    type: 'single-qubit' | 'two-qubit' | 'measure'
    name: str (lowercase gate name)
    qargs: list[int] (global flat qubit indices)
    qregs: list[str] (register name per qarg, for traceability)
    params: list[float]
    cbit: int (optional - measurement target or classical-control bit)
    classical_control_bit: int (optional)
    classical_control_register: str (optional)
    classical_control_val: int (optional)
"""
from __future__ import annotations

from typing import Optional

from bosonic_model import Circuit
from bosonic_model.instructions import (
    BarrierInstruction,
    ClassicalInstruction,
    ConditionalInstruction,
    Instruction,
    MeasureInstruction,
    ResetInstruction,
)


def _build_qubit_to_reg(circuit: Circuit) -> dict[int, str]:
    """Map global qubit index -> owning register name."""
    out: dict[int, str] = {}
    for reg in circuit.qregs.values():
        for offset in range(reg.size):
            out[reg.base + offset] = reg.name
    return out


def find_max_interactions(qpu_info) -> int:
    """Maximum number of two-qubit pairs that can fit in a layer given QPU sizes."""
    if isinstance(qpu_info, dict):
        sizes = list(qpu_info.values())
    else:
        sizes = list(qpu_info)
    pairs = 0
    for s in sizes:
        if s % 2 == 1:
            pairs += (s - 1) // 2
        else:
            pairs += s // 2
    return pairs


def _gate_qubits(inst: Instruction) -> list[int]:
    return list(getattr(inst, "qubits", []) or [])


def _make_gate_dict(
    inst: Instruction,
    qubit_to_reg: dict[int, str],
    classical_control_bit: Optional[int] = None,
) -> dict:
    if isinstance(inst, MeasureInstruction):
        return {
            "type": "measure",
            "name": "measure",
            "qargs": [inst.qubit],
            "qregs": [qubit_to_reg.get(inst.qubit, "")],
            "params": [],
            "cbit": inst.cbit,
        }

    if isinstance(inst, ResetInstruction):
        return {
            "type": "single-qubit",
            "name": "reset",
            "qargs": [inst.qubit],
            "qregs": [qubit_to_reg.get(inst.qubit, "")],
            "params": [],
        }

    qargs = _gate_qubits(inst)
    name = str(getattr(inst, "name", getattr(inst, "kind", "gate"))).lower()
    params = [float(p) for p in (getattr(inst, "params", []) or [])]
    qregs = [qubit_to_reg.get(q, "") for q in qargs]

    if len(qargs) == 1:
        gate_type = "single-qubit"
    elif len(qargs) == 2:
        gate_type = "two-qubit"
    else:
        raise NotImplementedError(
            f"DISQCO hypergraph does not support gates on {len(qargs)} qubits "
            f"(got {name!r} on qubits {qargs}). Decompose to 1q/2q gates first."
        )

    gate_dict = {
        "type": gate_type,
        "name": name,
        "qargs": qargs,
        "qregs": qregs,
        "params": params,
    }
    if classical_control_bit is not None:
        gate_dict["classical_control_bit"] = classical_control_bit
        gate_dict["cbit"] = classical_control_bit
    return gate_dict


def bosonic_to_layer_dict(circuit: Circuit, qpu_sizes=None) -> dict[int, list[dict]]:
    """ASAP-schedule a bosonic Circuit into the legacy DISQCO layer-dict format.

    Each instruction lands at the earliest layer where all of its qubits are free.
    If `qpu_sizes` is provided, two-qubit pairs per layer are capped at the maximum
    feasible across the network (as in the legacy `find_max_interactions` heuristic).

    Barriers are skipped. Conditional instructions are unwrapped into the underlying
    gate with `classical_control_bit` set. Measurements record their `cbit` target.
    Resets are emitted as single-qubit ops named "reset".
    """
    qubit_to_reg = _build_qubit_to_reg(circuit)
    num_qubits = circuit.qubits()
    qubit_last = [-1] * num_qubits
    cbit_last: dict[int, int] = {}
    layers: dict[int, list[dict]] = {}
    pair_count: dict[int, int] = {}
    max_pairs = find_max_interactions(qpu_sizes) if qpu_sizes is not None else None

    for raw_inst in circuit.instructions:
        ccbit: Optional[int] = None
        inst = raw_inst
        if isinstance(raw_inst, ConditionalInstruction):
            ccbit = raw_inst.condition.cbit
            inst = raw_inst.op
            if isinstance(inst, ConditionalInstruction):
                raise NotImplementedError("Nested ConditionalInstruction not supported")

        if isinstance(inst, BarrierInstruction):
            continue
        if isinstance(inst, ClassicalInstruction):
            raise ValueError(f"Classical instruction not supported by DISQCO: {inst.name}")

        qubits = _gate_qubits(inst)
        # Special-case 0-qubit ops (shouldn't occur for real gates) - skip.
        if not qubits and not isinstance(inst, (MeasureInstruction, ResetInstruction)):
            continue

        if isinstance(inst, MeasureInstruction):
            qubits = [inst.qubit]
        elif isinstance(inst, ResetInstruction):
            qubits = [inst.qubit]

        t = max((qubit_last[q] + 1) for q in qubits)
        if ccbit is not None:
            t = max(t, cbit_last.get(ccbit, -1) + 1)
        gate_dict = _make_gate_dict(inst, qubit_to_reg, classical_control_bit=ccbit)

        if max_pairs is not None and gate_dict["type"] == "two-qubit":
            while pair_count.get(t, 0) >= max_pairs:
                t += 1

        layers.setdefault(t, []).append(gate_dict)
        if gate_dict["type"] == "two-qubit":
            pair_count[t] = pair_count.get(t, 0) + 1
        for q in qubits:
            qubit_last[q] = t
        if isinstance(inst, MeasureInstruction):
            cbit_last[inst.cbit] = t

    if not layers:
        return {0: []}

    # Compact to consecutive layer indices.
    out: dict[int, list[dict]] = {}
    for new_idx, old_idx in enumerate(sorted(layers.keys())):
        out[new_idx] = layers[old_idx]
    return out


def basis_gates_from_circuit(circuit: Circuit) -> list[str]:
    """Collect the set of gate names appearing in the circuit (excluding meta ops)."""
    names: set[str] = set()
    for raw in circuit.instructions:
        inst = raw.op if isinstance(raw, ConditionalInstruction) else raw
        if isinstance(inst, (BarrierInstruction, ClassicalInstruction)):
            continue
        if isinstance(inst, MeasureInstruction):
            names.add("measure")
            continue
        if isinstance(inst, ResetInstruction):
            names.add("reset")
            continue
        names.add(str(getattr(inst, "name", getattr(inst, "kind", "gate"))).lower())
    return sorted(names)
