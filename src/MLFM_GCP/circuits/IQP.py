import random
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from math import pi
from typing import Optional

def generate_random_binomial_graph(
    n: int,
    p: float = 0.5,
    seed: Optional[int] = None,
    max_tries: int = 1000
) -> nx.Graph:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    for _ in range(max_tries):
        # Generate an Erdos-Renyi G(n, p) graph
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        
        # Check connectivity
        if nx.is_connected(G):
            # Check degree constraints
            if all(d < 4 for _, d in G.degree()):
                return G
    
    raise RuntimeError(
        f"Could not find a connected G(n, p) with max degree < 4 in {max_tries} tries."
    )

def build_IQP(n: int, seed: Optional[int] = None) -> QuantumCircuit:
    # Step 1: Initialize an n-qubit circuit
    qc = QuantumCircuit(n, n)
    
    # Step 2: Apply H to each qubit
    qc.h(range(n))
    
    # Step 3: Generate (and post-select) the random binomial graph G(n, p=0.5)
    G = generate_random_binomial_graph(n, p=0.5, seed=seed)
    
    # Step 4: For each edge in G, apply CZ
    for (i, j) in G.edges():
        qc.cz(i, j)
    
    # Step 5: For each qubit, apply RZ(α_i) with α_i in [0, 2π]
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    for qubit_idx in range(n):
        alpha = 2 * pi * random.random()
        qc.rz(alpha, qubit_idx)
    
    # Step 6: Apply H to each qubit again
    qc.h(range(n))
    
    # Step 7: Measure all qubits
    qc.measure(range(n), range(n))
    
    return qc
