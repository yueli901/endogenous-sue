"""The five-node witness network of Example 1.

A network small enough to enumerate exhaustively and large enough to carry the phenomenon: at moderate
dispersion the strict map on it has no fixed point at all, so the equilibrium is necessarily a mixture.
It is the paper's existence counterexample and the acceptance test for the tied-face solver.

Nodes are numbered 1..5 and named O, D, A, B, C, with O and D the two zones. Demand runs from O to D.
"""
from __future__ import annotations

import numpy as np

from .network import Topology

TAIL = np.array([1, 1, 1, 3, 4, 5, 3, 4, 4, 5, 5, 3])
HEAD = np.array([3, 4, 5, 2, 2, 2, 4, 3, 5, 4, 3, 5])
FFT = np.array([1.0, 1, 1, 1, 1, 1, .5, .5, .5, .5, .5, .5])
CAPACITY = np.array([1e6, 1e6, 1e6, 700.0, 1000.0, 1300.0, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6])
B = np.full(12, 0.15)
POWER = np.full(12, 4.0)

NODE_NAMES = {1: "O", 2: "D", 3: "A", 4: "B", 5: "C"}
ARC_LABELS = [f"{NODE_NAMES[t]}->{NODE_NAMES[h]}" for t, h in zip(TAIL, HEAD)]

DESTINATION = 1
DEMAND = 5000.0


def network() -> dict:
    return dict(tail=TAIL.copy(), head=HEAD.copy(), fft=FFT.copy(), capacity=CAPACITY.copy(),
                b=B.copy(), power=POWER.copy(), n_nodes=5, n_zones=2)


def demand() -> np.ndarray:
    od = np.zeros((2, 2))
    od[0, DESTINATION] = DEMAND
    return od


def topology() -> Topology:
    return Topology.build(TAIL - 1, HEAD - 1, FFT, 5, 2)
