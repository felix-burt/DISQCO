# DisQCO: Distributed Quantum Circuit Optimisation

## About

This repository provides tools for optimising distributed quantum circuits as described in [A Multilevel Framework for Partitioning Quantum Circuits](https://arxiv.org/abs/2503.19082), integrated with IBM Qiskit.

---

## Quantum circuit partitioning

The primary function is a partitioning tool, which uses a temporally extended hypergraph framework to model the problem of optimally choosing possible qubit and gate teleportations between QPUs. The backbone of this is based on the Fiduccia-Mattheyses heuristic for hypergraph partitioning, though the objective is designed spceifically for the problem. An overview is given in the [walkthrough notebook](demos/walkthrough.ipynb)

## Multilevel partitioning

For larger circuits, a multi-level partitioner is available, as inspired by tools such as [METIS](https://github.com/KarypisLab/METIS) and [KaHyPar](https://github.com/kahypar). 

This uses a *temporal coarsener* to produce a sequence of coarsened versions of the orignal graph. The FM algorithm is used to partition over increasing levels of granularity. The coarseners are described and compared in the [multilevel demo](demos/Multilevel_FM_demo.ipynb).

## Circuit extraction

A circuit extraction tool is also included which is integrated with IBM qiskit, through which we can extract a circuit from our partitioned hypergraph which splits qubits over multiple registers and handles all cross-register communication using shared entanglement and LOCC. QPUs are implemented as separate registers of a joint quantum circuit, where each QPU has a data qubit register and a communication qubit register. A joint classical register is shared among all. This can be tested in the [circuit extractor notebook](demos/circuit_extraction_demo.ipynb).

## Large-scale, heterogeneous networks

Coming soon

## Intra-QPU compilation and virtual DQC

Coming soon
