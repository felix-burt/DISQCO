"""Tests for DistributedCircuit event post-processing rules."""

import numpy as np
from typing import Any, cast

from disqco import QuantumNetwork
from disqco.circuit_extraction import HypergraphToDistributed
from disqco.circuit_extraction.distributed_circuit import (
    DistributedCircuit,
    FanIn,
    FanOut,
    JointFanIn,
    LocalGate,
)


def _make_circuit() -> DistributedCircuit:
    network = QuantumNetwork.create([2, 2], 'all_to_all')
    return DistributedCircuit(num_qubits=2, num_partitions=2, network=network)


def test_merge_fan_ins_requires_adjacency_in_qubit_timeline():
    """Do not merge when another event touching the same qubit is in between."""
    dc = _make_circuit()

    dc.add_event(FanIn(root_qubit=0, source_partition=1, target_partition=0, time=1))
    dc.add_event(LocalGate(gate_name='u', params=[0.1, 0.2, 0.3], qubits=[(0, 0)], time=2))
    dc.add_event(FanIn(root_qubit=0, source_partition=1, target_partition=0, time=3))

    dc.merge_fan_ins()

    fan_ins = [e for e in dc.events if isinstance(e, FanIn)]
    joint = [e for e in dc.events if isinstance(e, JointFanIn)]

    assert len(fan_ins) == 2
    assert len(joint) == 0


def test_merge_fan_ins_allows_unrelated_interleaving():
    """Merge when FanIns are adjacent for that qubit, even with unrelated events."""
    dc = _make_circuit()

    dc.add_event(FanIn(root_qubit=0, source_partition=1, target_partition=0, time=1))
    dc.add_event(LocalGate(gate_name='u', params=[0.4, 0.5, 0.6], qubits=[(1, 1)], time=2))
    dc.add_event(FanIn(root_qubit=0, source_partition=1, target_partition=0, time=3))

    dc.merge_fan_ins()

    fan_ins = [e for e in dc.events if isinstance(e, FanIn)]
    joint = [e for e in dc.events if isinstance(e, JointFanIn)]

    assert len(fan_ins) == 0
    assert len(joint) == 1
    assert joint[0].root_qubit == 0
    assert joint[0].target_partition == 0
    assert len(joint[0].source_partitions) == 2


def test_local_only_group_does_not_create_fanout_event():
    """Fully local grouped hyperedges should not emit empty FanOut events."""

    class _MockGraph:
        def __init__(self):
            self.num_qubits = 2
            self.layers = {
                0: [
                    {
                        'type': 'group',
                        'root': 0,
                        'time': 0,
                        'sub-gates': [
                            {
                                'type': 'two-qubit',
                                'name': 'cp',
                                'params': [0.2],
                                'qargs': [0, 1],
                                'time': 0,
                            }
                        ],
                    }
                ]
            }

    graph = _MockGraph()
    network = QuantumNetwork.create([2], 'all_to_all')
    assignment = np.array([[0, 0]])

    dc = HypergraphToDistributed(cast(Any, graph), network, partition_assignment=assignment).build()
    fan_outs = [event for event in dc.events if isinstance(event, FanOut)]

    assert fan_outs == []
