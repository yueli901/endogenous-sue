"""Cost-to-destination potentials, by reverse Dijkstra from every zone centroid.

``potentials(topology, cost)[n, d]`` is the least ``cost``-distance from node ``n`` to destination zone
``d``. These potentials define the efficient support (:mod:`endogenous_sue.subnetwork`), and computing
them dominates the cost of an outer iteration at scale.

The transit restriction is read from the topology and cannot be supplied separately. It has to agree with
the rule :func:`endogenous_sue.subnetwork.active_for_dest` applies, because a potential built over routes
the support then deletes prices journeys that cannot be made: an origin whose cheapest route ran through a
forbidden centroid is left with no admissible link and injects nothing, while the flow residual stays at
machine precision. Taking the restriction from the same object the support takes it from is what makes the
two agree by construction rather than by convention.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra

from ._compat import HAVE_NUMBA, njit, prange
from .config import PARALLEL_MIN_WORK_POTENTIALS
from .network import Topology

__all__ = ["potentials"]


def potentials(topo: Topology, cost: np.ndarray) -> np.ndarray:
    """Cost-to-destination potentials for every zone, shaped ``(n_nodes, n_zones)``."""
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    if HAVE_NUMBA:
        return _potentials_compiled(topo, cost)
    if topo.no_thru:
        return _potentials_restricted(topo, cost)
    reversed_graph = csr_matrix((cost, (topo.head, topo.tail)), shape=(topo.n_nodes, topo.n_nodes))
    distance = scipy_dijkstra(reversed_graph, directed=True, indices=np.arange(topo.n_zones))
    return np.ascontiguousarray(distance.T)


@njit(nogil=True)
def _reverse_csr(tail, head, n_nodes):
    """Reversed-graph CSR holding link indices, so one build serves every cost vector on a network."""
    n_links = tail.shape[0]
    pointer = np.zeros(n_nodes + 1, dtype=np.int64)
    for e in range(n_links):
        pointer[head[e] + 1] += 1
    for u in range(n_nodes):
        pointer[u + 1] += pointer[u]
    cursor = pointer[:n_nodes].copy()
    index = np.empty(n_links, dtype=np.int64)
    for e in range(n_links):
        position = cursor[head[e]]
        index[position] = e
        cursor[head[e]] = position + 1
    return pointer, index


@njit(nogil=True)
def _dijkstra_one(pointer, index, tail, cost, n_nodes, d, n_zones, restrict, heap_capacity):
    """Reverse Dijkstra from ``d``, optionally refusing to expand out of any other centroid.

    Expanding out of node ``u`` in the reversed graph uses exactly the links arriving at ``u`` in the
    original, so a guard on expansion implements the whole transit restriction and no graph is rebuilt.
    The destination's own arrival links are expanded because ``u == d`` is exempt.
    """
    distance = np.full(n_nodes, np.inf)
    settled = np.zeros(n_nodes, dtype=np.bool_)
    heap_node = np.empty(heap_capacity, dtype=np.int64)
    heap_key = np.empty(heap_capacity, dtype=np.float64)
    distance[d] = 0.0
    heap_node[0] = d
    heap_key[0] = 0.0
    size = 1
    while size > 0:
        u = heap_node[0]
        key = heap_key[0]
        size -= 1
        heap_node[0] = heap_node[size]
        heap_key[0] = heap_key[size]
        i = 0
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            if left < size and heap_key[left] < heap_key[smallest]:
                smallest = left
            if right < size and heap_key[right] < heap_key[smallest]:
                smallest = right
            if smallest == i:
                break
            node_swap = heap_node[i]; key_swap = heap_key[i]
            heap_node[i] = heap_node[smallest]; heap_key[i] = heap_key[smallest]
            heap_node[smallest] = node_swap; heap_key[smallest] = key_swap
            i = smallest
        if settled[u]:
            continue
        settled[u] = True
        if restrict and u < n_zones and u != d:
            continue
        for p in range(pointer[u], pointer[u + 1]):
            e = index[p]
            v = tail[e]
            candidate = key + cost[e]
            if candidate < distance[v]:
                distance[v] = candidate
                j = size
                heap_node[j] = v
                heap_key[j] = candidate
                size += 1
                while j > 0:
                    parent = (j - 1) // 2
                    if heap_key[parent] <= heap_key[j]:
                        break
                    node_swap = heap_node[parent]; key_swap = heap_key[parent]
                    heap_node[parent] = heap_node[j]; heap_key[parent] = heap_key[j]
                    heap_node[j] = node_swap; heap_key[j] = key_swap
                    j = parent
    return distance


@njit(parallel=True)
def _dijkstra_all_parallel(pointer, index, tail, cost, n_nodes, n_zones, restrict, heap_capacity):
    out = np.empty((n_zones, n_nodes))
    for d in prange(n_zones):
        out[d] = _dijkstra_one(pointer, index, tail, cost, n_nodes, d, n_zones, restrict,
                               heap_capacity)
    return out


@njit(nogil=True)
def _dijkstra_all_serial(pointer, index, tail, cost, n_nodes, n_zones, restrict, heap_capacity):
    out = np.empty((n_zones, n_nodes))
    for d in range(n_zones):
        out[d] = _dijkstra_one(pointer, index, tail, cost, n_nodes, d, n_zones, restrict,
                               heap_capacity)
    return out


def _potentials_compiled(topo: Topology, cost: np.ndarray) -> np.ndarray:
    """Destination-parallel above the measured crossover, serial below.

    Each destination's search is independent and the only shared array is the write-disjoint output, so
    both branches call the same per-destination kernel and the dispatch changes the running time and not
    the answer. Heap capacity ``n_links + 2`` is a bound rather than a guess: under lazy deletion a node
    is expanded at most once, so pushes are at most one per link plus the source.
    """
    tail = np.ascontiguousarray(topo.tail, dtype=np.int64)
    head = np.ascontiguousarray(topo.head, dtype=np.int64)
    pointer, index = _reverse_csr(tail, head, topo.n_nodes)
    kernel = (_dijkstra_all_parallel
              if topo.n_zones * topo.n_links >= PARALLEL_MIN_WORK_POTENTIALS
              else _dijkstra_all_serial)
    out = kernel(pointer, index, tail, cost, topo.n_nodes, topo.n_zones, topo.no_thru,
                 topo.n_links + 2)
    return np.ascontiguousarray(out.T)


def _collapse_parallel_links(rows, cols, data, n_nodes):
    """Keep the cheapest of any links joining the same ordered pair of nodes.

    Building a sparse matrix from coordinates sums duplicate entries, which would turn two links between
    one pair of nodes into a single link costing their total. Shortest paths want the minimum.
    """
    key = rows.astype(np.int64) * n_nodes + cols
    order = np.lexsort((data, key))
    sorted_key = key[order]
    first = np.empty(sorted_key.shape[0], dtype=bool)
    first[0] = True
    np.not_equal(sorted_key[1:], sorted_key[:-1], out=first[1:])
    keep = order[first]
    return rows[keep], cols[keep], data[keep]


def _potentials_restricted(topo: Topology, cost: np.ndarray) -> np.ndarray:
    """Restricted search without numba, one Dijkstra per destination.

    Deleting the centroid-arrival links leaves every centroid with no outgoing link in the reversed
    graph, so the destination's own arrival links have to be restored for each destination in turn. The
    compiled path restricts by a guard on node expansion and rebuilds nothing, which is why it is
    preferred whenever numba is present.
    """
    keep = topo.head >= topo.n_zones
    kept_rows, kept_cols, kept_data = topo.head[keep], topo.tail[keep], cost[keep]
    arrivals: dict[int, list[int]] = {}
    for e in np.flatnonzero(~keep):
        arrivals.setdefault(int(topo.head[e]), []).append(int(e))
    distance = np.full((topo.n_nodes, topo.n_zones), np.inf)
    for d in range(topo.n_zones):
        extra = arrivals.get(d)
        if extra:
            extra = np.asarray(extra, dtype=np.int64)
            rows = np.concatenate((kept_rows, topo.head[extra]))
            cols = np.concatenate((kept_cols, topo.tail[extra]))
            data = np.concatenate((kept_data, cost[extra]))
        else:
            rows, cols, data = kept_rows, kept_cols, kept_data
        rows, cols, data = _collapse_parallel_links(rows, cols, data, topo.n_nodes)
        reversed_graph = csr_matrix((data, (rows, cols)), shape=(topo.n_nodes, topo.n_nodes))
        distance[:, d] = scipy_dijkstra(reversed_graph, directed=True, indices=d)
    return np.ascontiguousarray(distance)
