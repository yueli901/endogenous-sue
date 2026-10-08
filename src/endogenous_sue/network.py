"""Per-network arrays and the fixed adjacency structure derived from them.

A network is carried as a plain dictionary of arrays, the form :func:`endogenous_sue.tntp.read_network`
returns. :class:`NetworkArrays` is the typed view of that dictionary in solver conventions, and
:class:`Topology` is the adjacency structure built once per network and shared across destinations and
across outer iterations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["NetworkArrays", "Topology", "topology_of", "unpack"]


@dataclass(frozen=True)
class NetworkArrays:
    """Link attributes in solver conventions: nodes zero-indexed, costs float64.

    The stored dictionary uses one-indexed nodes, as the TNTP files do. Converting at every entry point
    is how two conventions drift apart, so it is done once, here.
    """

    tail: np.ndarray
    head: np.ndarray
    fft: np.ndarray
    cap: np.ndarray
    b: np.ndarray
    power: np.ndarray
    n_nodes: int
    n_zones: int
    first_thru: int | None

    @classmethod
    def from_dict(cls, net: dict) -> NetworkArrays:
        return cls(
            tail=net["tail"].astype(np.int64) - 1,
            head=net["head"].astype(np.int64) - 1,
            fft=net["fft"].astype(np.float64),
            cap=net["capacity"].astype(np.float64),
            b=net["b"].astype(np.float64),
            power=net["power"].astype(np.float64),
            n_nodes=int(net["n_nodes"]),
            n_zones=int(net["n_zones"]),
            first_thru=net.get("first_thru"),
        )

    @property
    def n_links(self) -> int:
        return int(self.tail.shape[0])

    def topology(self) -> Topology:
        return Topology.build(self.tail, self.head, self.fft,
                              self.n_nodes, self.n_zones, self.first_thru)

    def cost(self, x: np.ndarray) -> np.ndarray:
        from .costs import bpr_cost
        return bpr_cost(x, self.fft, self.cap, self.b, self.power)

    def cost_prime(self, x: np.ndarray) -> np.ndarray:
        from .costs import bpr_cost_prime
        return bpr_cost_prime(x, self.fft, self.cap, self.b, self.power)


@dataclass(frozen=True)
class Topology:
    """Adjacency arrays shared across destinations and outer iterations.

    ``out_ptr`` and ``out_links`` are a CSR grouping of link indices by tail node, which is what reduces
    the per-destination value and loading passes to a single ordered sweep over the links.

    Two flags record how the network treats zone centroids, and they answer different questions.

    ``zero_dep_conn`` says the network has at least one zero-cost departure connector. Such a link leaves
    the cost to a destination unchanged, so strict decrease never admits it and flow could not leave its
    zone; departure connectors are therefore forced into every support. The test is whether *any*
    departure connector is zero-cost rather than all of them, because a network can carry thousands of
    zero-cost connectors alongside one of positive cost, and requiring the maximum to vanish would
    disable the rule and trap every trip.

    ``no_thru`` says flow may not transit a centroid other than its own destination. It is stated by the
    data, in the TNTP first-through-node header, and is read from there rather than inferred from
    ``zero_dep_conn``: the two disagree on eight of the seventeen corpus networks, and on seven of those
    ``no_thru`` holds where the inference says otherwise, which is the direction that strands demand.
    """

    tail: np.ndarray
    head: np.ndarray
    n_nodes: int
    n_zones: int
    out_ptr: np.ndarray
    out_links: np.ndarray
    zero_dep_conn: bool
    no_thru: bool

    @staticmethod
    def build(tail: np.ndarray, head: np.ndarray, fft: np.ndarray,
              n_nodes: int, n_zones: int, first_thru: int | None = None) -> Topology:
        order = np.argsort(tail, kind="stable").astype(np.int64)
        out_ptr = np.zeros(n_nodes + 1, dtype=np.int64)
        out_ptr[1:] = np.cumsum(np.bincount(tail, minlength=n_nodes))
        departure = tail < n_zones
        zero_dep_conn = bool(departure.any() and (fft[departure] <= 0.0).any())
        no_thru = bool(first_thru > 1) if first_thru is not None else zero_dep_conn
        return Topology(tail.astype(np.int64), head.astype(np.int64), n_nodes, n_zones,
                        out_ptr, order, zero_dep_conn, no_thru)

    @property
    def n_links(self) -> int:
        return int(self.tail.shape[0])


def topology_of(net: dict) -> Topology:
    """Build the topology directly from a stored network dictionary."""
    return NetworkArrays.from_dict(net).topology()


def unpack(net: dict):
    """Solver-convention arrays as ``(tail, head, fft, cap, b, power, n_nodes, n_zones)``."""
    arrays = NetworkArrays.from_dict(net)
    return (arrays.tail, arrays.head, arrays.fft, arrays.cap, arrays.b, arrays.power,
            arrays.n_nodes, arrays.n_zones)
