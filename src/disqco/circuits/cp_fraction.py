import random
import numpy as np
from qiskit import QuantumCircuit
import math as mt

def cz_fraction(
    num_qubits: int,
    depth: int,
    fraction: float,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Fixed-depth random circuit using CZ and Hadamard gates.

    At each layer each qubit independently becomes either a CZ target (with
    probability ``fraction``) or receives a Hadamard gate. CZ targets are
    paired randomly; an odd qubit is left unmatched and skipped. Based on the
    circuit family from Sundaram et al. 2021.

    Args:
        num_qubits: Number of qubits in the circuit.
        depth: Number of layers.
        fraction: Probability that a qubit participates in a CZ gate at each
            layer (vs receiving a single-qubit H gate).
        seed: Optional random seed for reproducibility.

    Returns:
        A Qiskit ``QuantumCircuit`` with the generated gates.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    circuit = QuantumCircuit(num_qubits)
    for l in range(depth):
        indeces = []
        for w in range(num_qubits):
            rand = random.random()
            if rand > fraction:
                circuit.h(w)
            else:
                indeces.append(w)
        indeces_shuffled = np.random.permutation(indeces)
        if (len(indeces_shuffled) % 2) != 0:
            indeces_shuffled = indeces_shuffled[:-1]
        pairs = indeces_shuffled.reshape(-1, 2)
        for pair in pairs:
            circuit.cz(pair[0],pair[1])
    return circuit

def cp_fraction(
    num_qubits: int,
    depth: int,
    fraction: float,
    seed: int | None = None,
) -> QuantumCircuit:
    """
    Fixed-depth random circuit using CP (controlled-phase) and U gates.

    Generalises :func:`cz_fraction` by replacing CZ with CP gates (random
    phase) and H gates with general U3 gates (random angles).

    Args:
        num_qubits: Number of qubits in the circuit.
        depth: Number of layers.
        fraction: Probability that a qubit participates in a CP gate at each
            layer (vs receiving a random single-qubit U gate).
        seed: Optional random seed for reproducibility.

    Returns:
        A Qiskit ``QuantumCircuit`` with the generated gates.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    circuit = QuantumCircuit(num_qubits)
    for l in range(depth):
        indeces = []
        for w in range(num_qubits):
            rand = random.random()
            if rand > fraction:
                theta = random.uniform(0,2 * mt.pi)
                phi = random.uniform(0,2 * mt.pi)
                lam = random.uniform(0,2 * mt.pi)
                circuit.u(theta,phi,lam,w)
            else:
                indeces.append(w)
        indeces_shuffled = np.random.permutation(indeces)
        if (len(indeces_shuffled) % 2) != 0:
            indeces_shuffled = indeces_shuffled[:-1]
        pairs = indeces_shuffled.reshape(-1, 2)
        for pair in pairs:
            phase = random.uniform(0,2 * mt.pi)
            circuit.cp(phase,pair[0],pair[1])
    return circuit
    