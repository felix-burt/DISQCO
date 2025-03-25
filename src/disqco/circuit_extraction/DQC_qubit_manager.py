import logging
import copy
import numpy as np
import networkx as nx

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit import Qubit, Clbit
from disqco.graphs.GCP_hypergraph import QuantumCircuitHyperGraph

# -------------------------------------------------------------------
# Set up a logger
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to logging.INFO or higher to reduce verbosity

# If you want logs printed to console:
console_handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.disabled = True

# -------------------------------------------------------------------
# CommunicationQubitManager
# -------------------------------------------------------------------
class CommunicationQubitManager:
    """
    Manages communication qubits on a per-partition basis. Allocates communication qubits for tasks 
    requiring entanglement and releases them when done.
    """
    def __init__(self, comm_qregs: dict, qc: QuantumCircuit):
        self.qc = qc  # Store copy of the QuantumCircuit
        self.comm_qregs = comm_qregs  # Store the QuantumRegisters for communication qubits
        self.free_comm = {}  # Store free communication qubits for each partition
        self.in_use_comm = {}  # Store in-use communication qubits for each partition
        self.linked_qubits = {}  # Store comm qubits linked to root qubits for gate teleportation

        self.initilize_communication_qubits()

    def initilize_communication_qubits(self) -> None:
        """
        Set all communication qubits to free.
        """
        for p, reg_list in self.comm_qregs.items():
            self.free_comm[p] = []
            self.in_use_comm[p] = set()
            for reg in reg_list:
                for qubit in reg:
                    self.free_comm[p].append(qubit)
            logger.debug(f"[initilize_communication_qubits] Partition {p} -> free_comm: {self.free_comm[p]}")

    def find_comm_idx(self, p: int) -> Qubit:
        """
        Allocate a free communication qubit in partition p.
        """
        logger.debug(f"[find_comm_idx] Requesting comm qubit in partition {p}")
        free_comm_p = self.free_comm[p]
        if free_comm_p:
            comm_qubit = free_comm_p.pop(0)
            logger.debug(f"[find_comm_idx] Found existing free comm qubit {comm_qubit} in partition {p}")
        else:
            # Create a new communication qubit by adding a new register
            num_regs_p = len(self.comm_qregs[p])
            new_reg = QuantumRegister(1, name=f"C{p}_{num_regs_p}")
            self.comm_qregs[p].append(new_reg)
            self.qc.add_register(new_reg)
            comm_qubit = new_reg[0]
            logger.debug(f"[find_comm_idx] No free comm qubits in partition {p}; created new comm qubit {comm_qubit}")

        self.in_use_comm[p].add(comm_qubit)
        logger.info(f"[find_comm_idx] ALLOCATE comm_qubit {comm_qubit} in partition {p}")
        logger.debug(f"[find_comm_idx] free_comm[{p}] now {self.free_comm[p]}")
        logger.debug(f"[find_comm_idx] in_use_comm[{p}] now {self.in_use_comm[p]}")
        return comm_qubit

    def allocate_comm_qubits(self, root_q, p_root, p_set_rec):
        """
        Allocate communication qubits for multi-gate teleportation.
        Allocate one communication qubit in p_root, plus one in each partition in p_set_rec.
        Link them to root_q as needed.
        """
        logger.debug(f"[allocate_comm_qubits] root_q={root_q}, p_root={p_root}, p_set_rec={p_set_rec}")
        if not p_set_rec:
            return None, {}

        comm_root = self.find_comm_idx(p_root)
        comm_rec_dict = {}

        for p_rec in p_set_rec:
            comm_rec = self.find_comm_idx(p_rec)
            comm_rec_dict[p_rec] = comm_rec
            # Link comm_rec to root_q
            self.linked_qubits[comm_rec] = root_q
            logger.debug(f"[allocate_comm_qubits] Linked {comm_rec} to root_q {root_q}")

        return comm_root, comm_rec_dict

    def release_comm_qubit(self, p: int, comm_qubit: Qubit) -> None:
        """
        Resets the qubit and returns it to the free pool in partition p.
        """
        if comm_qubit in self.in_use_comm[p]:
            self.in_use_comm[p].remove(comm_qubit)
            self.free_comm[p].append(comm_qubit)
            logger.info(f"[release_comm_qubit] RELEASED comm_qubit {comm_qubit} in partition {p}")
        else:
            logger.warning(f"[release_comm_qubit] Tried to release comm_qubit {comm_qubit} not found in in_use_comm[{p}]")

        logger.debug(f"[release_comm_qubit] free_comm[{p}] = {self.free_comm[p]}")
        logger.debug(f"[release_comm_qubit] in_use_comm[{p}] = {self.in_use_comm[p]}")

    def get_status(self, p: int) -> tuple[list, list]:
        """
        Return a tuple (in_use, free) for partition p.
        """
        return self.in_use_comm.get(p, []), self.free_comm.get(p, [])

# -------------------------------------------------------------------
# ClassicalBitManager
# -------------------------------------------------------------------
class ClassicalBitManager:
    """
    Manages classical bits, allocating from a pool and releasing after use.
    """
    def __init__(self, qc: QuantumCircuit, creg: ClassicalRegister):
        self.qc = qc          # Store copy of the QuantumCircuit
        self.creg = creg      # Store the ClassicalRegister for classical bits
        self.free_cbit = []   # Store free classical bits
        self.in_use_cbit = {} # Store in-use classical bits

        self.initilize_classical_bits()

    def initilize_classical_bits(self) -> None:
        """
        Mark all classical bits as free.
        """
        for cbit in self.creg:
            self.free_cbit.append(cbit)
        logger.debug(f"[initilize_classical_bits] Initialized free_cbit with {len(self.free_cbit)} bits.")

    def allocate_cbit(self) -> Clbit:
        """
        Allocate a classical bit for a measurement operation.
        """
        if len(self.free_cbit) == 0:
            # Add a new classical register of size 1
            idx = len(self.creg)
            new_creg = ClassicalRegister(1, name=f"cl_{idx}")
            self.qc.add_register(new_creg)
            self.creg = new_creg
            self.free_cbit.append(new_creg[0])
            logger.debug(f"[allocate_cbit] No free classical bits, created new creg {new_creg}")

        cbit = self.free_cbit.pop(0)
        self.in_use_cbit[cbit] = True
        logger.info(f"[allocate_cbit] ALLOCATED cbit {cbit}")
        return cbit

    def release_cbit(self, cbit: Clbit) -> None:
        """
        Release a classical bit after use.
        """
        if cbit in self.in_use_cbit:
            del self.in_use_cbit[cbit]
            self.free_cbit.insert(0, cbit)
            logger.info(f"[release_cbit] RELEASED cbit {cbit}")
        else:
            logger.warning(f"[release_cbit] Tried to release cbit={cbit} which was not in in_use_cbit")

        logger.debug(f"[release_cbit] free_cbit now {self.free_cbit}")

# -------------------------------------------------------------------
# DataQubitManager
# -------------------------------------------------------------------
class DataQubitManager:
    """
    Manages data qubits for teleportation of quantum states. Allocates and releases data qubits as needed,
    tracking which slots are free and which logical qubits are mapped to which slots.
    """
    def __init__(
        self,
        partition_qregs: list[QuantumRegister],
        num_qubits_log: int,
        partition_assignment: list[list],
        qc: QuantumCircuit
    ):
        self.qc = qc
        self.partition_qregs = partition_qregs
        self.num_qubits_log = num_qubits_log
        self.in_use_data = {}
        self.free_data = {}
        self.partition_assignment = partition_assignment
        self.log_to_phys_idx = {}
        self.num_partitions = len(partition_qregs)
        self.linked_comm_qubits = {i : {} for i in range(self.num_qubits_log)}
        self.num_data_qubits_per_partition = []
        self.active_roots = {}
        self.queue = {}
        self.groups = {}
        self.active_receivers = {}
        self.relocated_receivers = {}

        self.initialise_data_qubits()
        self.initial_placement(partition_assignment)

    def initialise_data_qubits(self) -> None:
        """
        Initialize the free_data and in_use_data dictionaries.
        """
        logger.debug("[initialise_data_qubits] Initializing data qubits in each partition.")
        for p in range(self.num_partitions):
            reg = self.partition_qregs[p]
            num_qubits_p = len(reg)
            self.free_data[p] = [qubit for qubit in reg]
            self.in_use_data[p] = {}
            self.num_data_qubits_per_partition.append(num_qubits_p)

            logger.debug(f"  Partition {p}: free_data -> {self.free_data[p]}")

    def initial_placement(self, partition_assignment: list[list]) -> None:
        """
        At t=0, place each logical qubit in the partition specified by partition_assignment[0].
        """
        logger.debug("[initial_placement] Placing each logical qubit based on partition_assignment[0].")
        for q in range(self.num_qubits_log):
            part0 = partition_assignment[0][q]
            qubit0 = self.allocate_data_qubit(part0)
            self.assign_to_physical(part0, qubit0, q)
            logger.debug(f"  Logical qubit {q} -> {qubit0} in partition {part0}")

    def allocate_data_qubit(self, p: int) -> Qubit:
        """
        Allocate a free data qubit slot in partition p.
        """
        # if not self.free_data[p]:
        #     logger.warning(f"[allocate_data_qubit] No free data qubits in partition {p}; adding new QRegister.")
        #     # Create a new data qubit in partition p
        #     idx = len(self.partition_qregs[p])
        #     new_reg = QuantumRegister(1, name=f"part{p}_data_{idx}")
        #     self.partition_qregs[p].append(new_reg)
        #     self.qc.add_register(new_reg)
        #     new_qubit = new_reg[0]
        #     self.free_data[p].append(new_qubit)

        qubit = self.free_data[p].pop(0)
        logger.info(f"[allocate_data_qubit] ALLOCATED data_qubit {qubit} in partition {p}")
        return qubit

    def assign_to_physical(self, part: int, qubit_phys: Qubit, qubit_log: int):
        """
        Assign a logical qubit to a physical qubit slot in a partition.
        """
        logger.debug(f"[assign_to_physical] Assigning logical qubit {qubit_log} to {qubit_phys} in partition {part}")
        self.log_to_phys_idx[qubit_log] = qubit_phys
        self.in_use_data[part][qubit_phys] = qubit_log

    def release_data_qubit(self, p: int, qubit: Qubit) -> None:
        """
        Release a data qubit, clearing any state. 
        Note: Qiskit doesn't have a direct 'free' notion, so we reset or reuse.
        """
        logger.debug(f"[release_data_qubit] Releasing data qubit {qubit} from partition {p}")
        if qubit in self.in_use_data[p]:
            log_qubit = self.in_use_data[p].pop(qubit)
            del self.log_to_phys_idx[log_qubit]
            self.qc.reset(qubit)
            self.free_data[p].append(qubit)
            logger.info(f"[release_data_qubit] RELEASED data_qubit {qubit} from partition {p}")
        else:
            logger.warning(f"[release_data_qubit] Qubit {qubit} was not in use in partition {p}.")
        """
        Release a data qubit after the state has been teleported to another partition.
        """
        if qubit in self.in_use_data[p]:
            del self.in_use_data[p][qubit] # Remove the logical qubit from the in_use_data dictionary
        if qubit not in self.free_data[p]:
            self.free_data[p].append(qubit) # Add the slot to the free_data list