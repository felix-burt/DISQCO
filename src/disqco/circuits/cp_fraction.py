import random
import numpy as np
from qiskit import QuantumCircuit
import math as mt

def cz_fraction(num_qubits,depth,fraction, seed=None):
    "Fixed depth random circuit using CZ and Hadamard gates. From Sundaram et al. 2021."
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

def cp_fraction(num_qubits,depth,fraction, seed=None):
    "Generalized version of the previous function, using CPhase gates with a random phase and U gates with random parameters."
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
    