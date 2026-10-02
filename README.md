# DisQCO: Distributed Quantum Circuit Optimisation

## About

This repository provides tools for optimising distributed quantum circuits as described in [A Multilevel Framework for Partitioning Quantum Circuits](https://arxiv.org/abs/2503.19082), integrated with IBM Qiskit.

This now includes support for general and large scale quantum networks, the primary topic of [Entanglement-Efficient Distribution of Quantum Circuits over Large-Scale Quantum Networks](https://arxiv.org/abs/2507.16036).

---

## Quantum circuit partitioning

The primary function is a partitioning tool, which uses a temporally extended hypergraph framework to model the problem of optimally choosing possible qubit and gate teleportations between QPUs. The backbone of this is based on the Fiduccia-Mattheyses heuristic for hypergraph partitioning, though the objective is designed spceifically for the problem. An overview is given in the [introduction notebook](demos/disqco_introduction.ipynb).

## Multilevel partitioning

For larger circuits, a multi-level partitioner is available, as inspired by tools such as [METIS](https://github.com/KarypisLab/METIS) and [KaHyPar](https://github.com/kahypar). 

This uses a *temporal coarsener* to produce a sequence of coarsened versions of the orignal graph. The FM algorithm is used to partition over increasing levels of granularity. The coarseners are described and compared in the [multilevel demo](demos/Multilevel_FM_demo.ipynb).

## General networks

Now support is included for general network topologies. The partitioning is automatically adjusted to be aware of the network topology, and optimises for the axuiliary EPR pairs in long-range links. Different topologies are compared in [heterogeneous networks demo](demos/heterogeneous_partitioning_demo.ipynb)

## Network coarsening

For larger networks, network coarsening can be employed, which permits hierarchical partitioning over different sub-regions of the network. A full walkthrough is available [here](demos/net_coarsening_walkthrough.ipynb), as well as an easy-access [demo](demos/net_coarsened_partitioning.ipynb).

## Circuit extraction

A circuit extraction tool is also included which is integrated with IBM qiskit, through which we can extract a circuit from our partitioned hypergraph which splits qubits over multiple registers and handles all cross-register communication using shared entanglement and LOCC. QPUs are implemented as separate registers of a joint quantum circuit, where each QPU has a data qubit register and a communication qubit register. A joint classical register is shared among all. This can be tested in the [circuit extractor notebook](demos/circuit_extraction_demo.ipynb). 

Circuit extraction is compatible with general networks, and produces an output which only requests EPR pairs on directly connected network nodes. This is demonstrated in [this notebook](demos/circuit_extraction_heterogeneous.ipynb).
## Intra-QPU compilation and virtual DQC

Coming soon

## Installation

While this repository is very much a work in progress, the current version can be installed by cloning the repository.

```
git clone https://github.com/felix-burt/DISQCO.git
cd DISQCO
```

> [!NOTE]
> DISQCO depends on `qiskit-aer`, which does not provide wheels for Python versions beyond 3.12. Make sure that the version of Python you use is <3.13. (`uv sync` will handle this automatically)

### Using `pip`

DISQCO can be installed using the standard Python package manager, which installs the packages listed in [`setup.py`](./setup.py).

```
pip install .
```

### Using `uv`

Alternatively, DISQCO can be installed using [`uv`](https://docs.astral.sh/uv/), which reads the project information in [`pyproject.toml`](./pyproject.toml) to not only install dependencies but also automatically create a virtual environment.

```
# install dependencies and initialize venv
uv sync 

# add DISQCO to venv and make it executable
uv pip install -e .

# run package tests
uv run pytest
```

If you are using DISQCO within another project, you can add it as a dependency directly from this repo:

```
uv add "disqco @ git+https://github.com/felix-burt/DISQCO.git"
```
