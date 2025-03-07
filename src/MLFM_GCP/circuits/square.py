import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import random_unitary

def build_square_circuit(n, seed=None) -> QuantumCircuit:
    # Optional: Set a seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Initialize an n-qubit circuit
    circuit = QuantumCircuit(n, n)  # n qubits, n classical bits for measurement
    
    # For each layer from 1 to n (worst-case depth n)
    for layer in range(n):
        # 1. Randomly shuffle qubits
        shuffled_qubits = list(range(n))
        random.shuffle(shuffled_qubits)
        
        # 2. Pair them up
        pairs = []
        for i in range(0, n - (n % 2), 2):
            pairs.append((shuffled_qubits[i], shuffled_qubits[i+1]))
        
        # 3. For each pair, generate a random 2-qubit SU(4) gate and apply
        for (q0, q1) in pairs:
            # Generate a 4x4 Haar-random unitary
            U_2q = random_unitary(4)
            
            # Convert this 4x4 unitary into a Qiskit gate
            # circuit.unitary() can take a full matrix and the qubit indices
            circuit.unitary(U_2q, [q0, q1], label=f"U_{layer}")
    
    # Measure all qubits in the computational basis
    circuit.measure(range(n), range(n))
    
    return circuit