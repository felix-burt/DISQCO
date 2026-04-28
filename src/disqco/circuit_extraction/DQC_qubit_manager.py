"""Qubit and classical-bit managers for bosonic-native circuit extraction.

All qubit and cbit references are plain ``int`` global indices; there are no
Qiskit ``Qubit``/``Clbit`` objects anywhere in this module.

The ``NativeCircuitBuilder`` collects bosonic registers and instructions and
produces a flat ``bosonic_model.Circuit`` at the end of extraction.  The
per-partition managers (data qubits, communication qubits, classical bits) all
emit to a single shared builder instance.
"""
from __future__ import annotations

import copy

from bosonic_model import Circuit as BosonicCircuit
from bosonic_model import Register as BosonicRegister
from bosonic_model.instructions import ResetInstruction


# -------------------------------------------------------------------
# NativeCircuitBuilder
# -------------------------------------------------------------------

class NativeCircuitBuilder:
    """Accumulates bosonic registers and instructions for a flat BosonicCircuit.

    Qubit / cbit allocations return plain ``int`` global indices.

    ``qubit_kind``      maps qubit_idx -> 'Q' (data) or 'C' (comm).
    ``qubit_partition`` maps qubit_idx -> QPU partition index (int).
    """

    def __init__(self) -> None:
        self.qregs: dict[str, BosonicRegister] = {}
        self.cregs: dict[str, BosonicRegister] = {}
        self.instructions: list = []
        self._next_qubit: int = 0
        self._next_cbit: int = 0
        self.qubit_kind: dict[int, str] = {}
        self.qubit_partition: dict[int, int] = {}

    def add_qreg(self, name: str, size: int, kind: str, partition: int) -> list[int]:
        """Register *size* qubits; return their global indices as a list."""
        base = self._next_qubit
        reg = BosonicRegister(name=name, size=size, base=base)
        self.qregs[name] = reg
        self._next_qubit += size
        indices = list(range(base, base + size))
        for i in indices:
            self.qubit_kind[i] = kind
            self.qubit_partition[i] = partition
        return indices

    def add_creg(self, name: str, size: int) -> list[int]:
        """Register *size* classical bits; return their global indices as a list."""
        base = self._next_cbit
        reg = BosonicRegister(name=name, size=size, base=base)
        self.cregs[name] = reg
        self._next_cbit += size
        return list(range(base, base + size))

    def emit(self, inst) -> None:
        self.instructions.append(inst)

    def build(self) -> BosonicCircuit:
        return BosonicCircuit(
            qregs=self.qregs,
            cregs=self.cregs,
            instructions=self.instructions,
        )


# -------------------------------------------------------------------
# CommunicationQubitManager
# -------------------------------------------------------------------

class CommunicationQubitManager:
    """Manages communication qubits on a per-partition basis.

    Allocates comm qubits for entanglement tasks and returns them to the pool
    when the task is done.  New registers are created dynamically if the
    initial pool is exhausted.
    """

    def __init__(
        self,
        comm_data: dict[int, list[int]],
        builder: NativeCircuitBuilder,
    ) -> None:
        self.builder = builder
        self.free_comm: dict[int, list[int]] = {}
        self.in_use_comm: dict[int, set[int]] = {}
        self._reg_counts: dict[int, int] = {}

        for p, qubits in comm_data.items():
            self.free_comm[p] = list(qubits)
            self.in_use_comm[p] = set()
            self._reg_counts[p] = 1

    def find_comm_idx(self, p: int) -> int:
        """Allocate a free communication qubit in partition *p*."""
        if self.free_comm[p]:
            q = self.free_comm[p].pop(0)
        else:
            num_regs = self._reg_counts[p]
            new_indices = self.builder.add_qreg(f"C{p}_{num_regs}", 1, "C", p)
            self._reg_counts[p] += 1
            q = new_indices[0]
        self.in_use_comm[p].add(q)
        return q

    def release_comm_qubit(self, p: int, q: int) -> None:
        """Return a comm qubit to the free pool."""
        if q in self.in_use_comm[p]:
            self.in_use_comm[p].remove(q)
            self.free_comm[p].append(q)

    def get_status(self, p: int) -> tuple[list, list]:
        """Return (in_use, free) lists for partition *p*."""
        return list(self.in_use_comm.get(p, set())), self.free_comm.get(p, [])


# -------------------------------------------------------------------
# ClassicalBitManager
# -------------------------------------------------------------------

class ClassicalBitManager:
    """Manages classical bits; allocates from a reuse pool, grows dynamically."""

    def __init__(
        self,
        builder: NativeCircuitBuilder,
        creg_base: int,
        creg_size: int,
    ) -> None:
        self.builder = builder
        self.free_cbit: list[int] = list(range(creg_base, creg_base + creg_size))
        self.in_use_cbit: dict[int, bool] = {}
        self._extra_count: int = 0

    def allocate_cbit(self) -> int:
        """Allocate a classical bit, growing the register pool if needed."""
        if not self.free_cbit:
            idx = self._extra_count
            new_indices = self.builder.add_creg(f"cl_global_extra_{idx}", 1)
            self._extra_count += 1
            self.free_cbit.extend(new_indices)
        cbit = self.free_cbit.pop(0)
        self.in_use_cbit[cbit] = True
        return cbit

    def release_cbit(self, cbit: int) -> None:
        """Release a classical bit back to the pool."""
        if cbit in self.in_use_cbit:
            del self.in_use_cbit[cbit]
            self.free_cbit.insert(0, cbit)


# -------------------------------------------------------------------
# DataQubitManager
# -------------------------------------------------------------------

class DataQubitManager:
    """Manages data qubits; tracks the logical -> physical qubit mapping.

    Physical qubit indices are plain ``int`` global indices (as registered in
    the ``NativeCircuitBuilder``).
    """

    def __init__(
        self,
        partition_data: dict[int, list[int]],
        num_qubits_log: int,
        partition_assignment: list[list],
        builder: NativeCircuitBuilder,
    ) -> None:
        self.builder = builder
        self.num_qubits_log = num_qubits_log
        self.in_use_data: dict[int, dict[int, int]] = {}   # p -> {phys: log}
        self.free_data: dict[int, list[int]] = {}           # p -> [phys, ...]
        self.partition_assignment = partition_assignment
        self.log_to_phys_idx: dict[int, int] = {}
        self.num_partitions = len(partition_data)
        self.groups: dict = {}
        self.queue: dict = {}
        self.active_roots: dict = {}
        self.active_receivers: dict = {}
        self.relocated_receivers: dict = {}

        for p, qubits in partition_data.items():
            self.free_data[p] = list(qubits)
            self.in_use_data[p] = {}

        self.initial_placement(partition_assignment)
        self.inital_qubit_placement = copy.deepcopy(self.log_to_phys_idx)

    def initial_placement(self, partition_assignment: list[list]) -> None:
        """Place each logical qubit in the partition given by assignment[0]."""
        for q in range(self.num_qubits_log):
            part0 = partition_assignment[0][q]
            qubit0 = self.allocate_data_qubit(part0)
            self.assign_to_physical(part0, qubit0, q)

    def allocate_data_qubit(self, p: int) -> int:
        """Pop and return a free physical data qubit index in partition *p*."""
        return self.free_data[p].pop(0)

    def assign_to_physical(self, part: int, qubit_phys: int, qubit_log: int) -> None:
        """Bind a logical qubit to a physical slot."""
        self.log_to_phys_idx[qubit_log] = qubit_phys
        self.in_use_data[part][qubit_phys] = qubit_log

    def release_data_qubit(self, p: int, qubit: int) -> None:
        """Free a physical data qubit, emitting a Reset and returning it to the pool."""
        if qubit in self.in_use_data[p]:
            log_qubit = self.in_use_data[p].pop(qubit)
            del self.log_to_phys_idx[log_qubit]
            self.builder.emit(ResetInstruction(qubit=qubit, qubits=[qubit]))
            self.free_data[p].append(qubit)
        if qubit not in self.free_data[p]:
            self.free_data[p].append(qubit)
