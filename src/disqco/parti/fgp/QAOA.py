import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
from qiskit import transpile
import random
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz


def random_graph(n,p):
    "Function to create a random graph with edge probability p."
    graph = rx.PyGraph()
    graph.add_nodes_from(np.arange(0, n, 1))
    for m in range(n):
        for k in range(m):
            if random.random() < p:
                graph.add_edge(m,k,1.0)
    return graph

def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Encode the MaxCut problem as a list of Pauli strings.
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::-1], weight))

    return pauli_list

def QAOA_random(num_qubits,prob,reps):
    "Function to create a random QAOA circuit for solving max-cut on input graph."
    graph = random_graph(num_qubits,prob)
    max_cut_paulis = build_max_cut_paulis(graph)
    cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
    ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
    param_values = np.random.rand(len(ansatz.parameters))
    param_dict = dict(zip(ansatz.parameters, param_values))
    circuit = ansatz.assign_parameters(param_dict) 
    return circuit
