# Multi-level Quantum Circuit Partitioning

This repository provides a framework for partitioning quantum circuits into sub-circuits connected using only shared entanglement. Shared entanglement is used to facilitate quantum state teleportation, gate teleporation and multi-gate teleporation. A hypergraph approach is used to model the optimisation, while a multi-level variation of the Fiduccia-Mattheyses partitioning heuristic is adapted to the problem. 

This optimisation is integrated into the Qiskit SDK, and converts single-QPU quantum circuits into partitioned sub-circuits with minimal entanglement requirements.

