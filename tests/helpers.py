"""Shared test helpers.

The benchmark networks are not part of the repository, so on a fresh checkout the tests that need them
cannot run. They skip with an actionable message rather than failing: a wall of tracebacks on a first
clone says the package is broken when in fact one setup step has not been taken.
"""
from __future__ import annotations

from unittest import SkipTest

import numpy as np

from endogenous_sue.config import DATA_ROOT


def require_networks():
    """Skip the calling module unless the benchmark networks are present."""
    if not DATA_ROOT.is_dir():
        raise SkipTest(f"benchmark networks not found at {DATA_ROOT}. "
                       f"Run reproduction/tools/fetch_networks.py first.")


def transit_network():
    """A network whose cheapest route from the origin crosses a foreign centroid.

    Nodes zero, one and two are zones; three and four are through nodes. The first-through-node header
    forbids transit, so the two-link route across zone one is illegal and the demand must take the longer
    legal route. Used wherever the transit rule itself is under test.
    """
    net = dict(tail=np.array([1, 2, 1, 4, 5]), head=np.array([2, 3, 4, 5, 3]),
               fft=np.array([2.0, 2.0, 3.0, 3.0, 3.0]), capacity=np.full(5, 1e6),
               b=np.full(5, 0.15), power=np.full(5, 4.0),
               n_nodes=5, n_zones=3, first_thru=4)
    od = np.zeros((3, 3))
    od[0, 2] = 10.0
    return net, od


def two_destination_witness():
    """The witness with a second destination that no tie touches.

    The witness carries the phenomenon but has one destination, so it cannot exercise any clause about
    how a mixture accounts for destinations it never split. Its nodes are renumbered to free a zone --
    O, D stay 1 and 2, the new zone E is 3, and A, B, C become 4, 5, 6 -- and one link runs O to E with
    no alternative. The tied structure for D is the witness's, untouched: E's link shares no capacity
    with it, so the flows on the original twelve links are the witness's own.

    Destination index 1 is therefore split across a mixture and destination index 2 carries its whole
    flow on its base support, appearing in no vertex entry at all.
    """
    from endogenous_sue import witness

    remap = {1: 1, 2: 2, 3: 4, 4: 5, 5: 6}
    tail = np.array([remap[t] for t in witness.TAIL] + [1])
    head = np.array([remap[h] for h in witness.HEAD] + [3])
    net = dict(tail=tail, head=head,
               fft=np.concatenate([witness.FFT, [2.0]]),
               capacity=np.concatenate([witness.CAPACITY, [1e6]]),
               b=np.full(13, 0.15), power=np.full(13, 4.0),
               n_nodes=6, n_zones=3, first_thru=4)
    od = np.zeros((3, 3))
    od[0, 1] = witness.DEMAND
    od[0, 2] = 100.0
    return net, od
