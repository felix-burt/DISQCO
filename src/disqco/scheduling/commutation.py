from qiskit import QuantumCircuit
from qiskit.circuit import CommutationChecker

_checker = CommutationChecker()

def operation_commutes(inst_a, inst_b):
    """Returns true if two instructions commute and false otherwise"""
    if inst_a.clbits or inst_b.clbits:
        return False
    if getattr(inst_a.operation, "condition", None) or getattr(inst_b.operation, "condition", None):
        return False
    return _checker.commute(inst_a.operation, inst_a.qubits, inst_a.clbits, 
                            inst_b.operation, inst_b.qubits, inst_b.clbits)

def split_instruction_indicies_to_commuting_groups(circuit: QuantumCircuit, indices):
    """[1, 2, 3, 4] -> [[1, 2], [3], [4]] if 1, 2 commutes, nothing else does"""
    groups = []
    current = []
    for index in indices:
        if current and all(operation_commutes(circuit.data[index], circuit.data[other] ) for other in current):
            current.append(index)
        else:
            if current:
                groups.append(current)
            current = [index]
    if current:
        groups.append(current)
    return groups
