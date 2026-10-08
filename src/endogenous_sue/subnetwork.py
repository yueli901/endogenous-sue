"""The efficient support: which links a destination's flow may use, and in what order.

A link is *efficient* for destination ``d`` when the cost to ``d`` strictly decreases along it. The
resulting subgraph is acyclic, which is what reduces both the recursive-logit value function and the
network loading to a single ordered sweep over the links.

Because cost depends on flow, the support depends on flow. That is the endogeneity the method addresses,
and it is why the support is recomputed rather than fixed in advance.
"""
from __future__ import annotations

import numpy as np

from ._compat import njit
from .network import Topology

__all__ = ["active_for_dest", "kahn_order", "cycle_links", "isolation_margin",
           "exclusion_margin", "path_excess_margin"]


def active_for_dest(topo: Topology, potential: np.ndarray, d: int) -> np.ndarray:
    """Boolean efficient-link mask for destination ``d`` given one column of node potentials.

    The rule is strict decrease of the potential along the link. A zero-cost link leaves the potential
    unchanged and so is never efficient under that rule, which for centroid connectors would trap all
    flow. The three centroid clauses below exist for that reason, and they answer three separate
    questions. Bundling any two of them loses demand.

    *May flow leave a zone?* Yes wherever a departure connector is zero-cost, since such a link has no
    potential gap and strict decrease can never admit it. Applied when ``zero_dep_conn`` holds.

    *May flow arrive at* ``d``? Always. A link into the destination is not transit, and where the arrival
    connector is zero-cost it is inadmissible under strict decrease for the same reason. This clause is
    unconditional, and in particular is not nested inside the transit clause: a network without a transit
    restriction but with zero-cost arrival connectors would otherwise admit no link into any destination
    at all, and every link still carrying flow would still be efficient, so the certificate would pass on
    an assignment that delivers almost nothing.

    *May flow transit a foreign centroid?* Answered by the data, through the TNTP first-through-node
    header. Applied when ``no_thru`` holds, and gated identically in
    :func:`endogenous_sue.shortestpath.potentials`, which reads it from this same topology.

    Acyclicity survives all three. A centroid other than ``d`` has no admissible incoming link when the
    transit clause applies; when it does not, its incoming links are admitted only by strict decrease, so
    no cycle can close through it. The destination absorbs and has no admissible outgoing link.

    The potential column need not be an exact shortest-path potential: the two-phase solver supplies
    frozen potentials, for which the same argument holds unchanged.
    """
    tail, head, n_zones = topo.tail, topo.head, topo.n_zones
    at_tail, at_head = potential[tail], potential[head]
    active = at_head < at_tail
    if topo.zero_dep_conn:
        active[tail < n_zones] = True
    if topo.no_thru:
        active[head < n_zones] = False
    active[head == d] = True
    active &= np.isfinite(at_tail) & np.isfinite(at_head)
    active[tail == d] = False
    return active


@njit(nogil=True)
def _active_for_dest_kernel(tail, head, potential, d, n_zones, zero_dep, no_thru, n_links):
    """Compiled counterpart of :func:`active_for_dest`. The two must stay identical."""
    active = np.zeros(n_links, dtype=np.bool_)
    for e in range(n_links):
        t = tail[e]; h = head[e]
        at_tail = potential[t]; at_head = potential[h]
        admit = at_head < at_tail
        if zero_dep and t < n_zones:
            admit = True
        if no_thru and h < n_zones:
            admit = False
        if h == d:
            admit = True
        if not (np.isfinite(at_tail) and np.isfinite(at_head)):
            admit = False
        if t == d:
            admit = False
        active[e] = admit
    return active


def kahn_order(topo: Topology, active: np.ndarray) -> np.ndarray:
    """Topological order of the active subgraph, tails before heads.

    Raises :class:`RuntimeError` if the active links contain a directed cycle, which cannot happen while
    through-link free-flow costs are strictly positive.
    """
    order, emitted = _kahn_order_python(topo, active)
    if emitted != topo.n_nodes:
        raise RuntimeError(f"cycle in active subgraph: ordered {emitted}/{topo.n_nodes} nodes")
    return order


def _kahn_order_python(topo: Topology, active: np.ndarray):
    head = topo.head
    n = topo.n_nodes
    indegree = np.bincount(head[active], minlength=n).astype(np.int64)
    out_ptr, out_links = topo.out_ptr, topo.out_links
    order = np.empty(n, dtype=np.int64)
    stack = np.flatnonzero(indegree == 0).astype(np.int64).tolist()
    emitted = 0
    while stack:
        u = stack.pop()
        order[emitted] = u
        emitted += 1
        for idx in range(out_ptr[u], out_ptr[u + 1]):
            e = out_links[idx]
            if active[e]:
                v = head[e]
                indegree[v] -= 1
                if indegree[v] == 0:
                    stack.append(int(v))
    return order, emitted


@njit(nogil=True)
def _kahn_order_kernel(out_ptr, out_links, head, active, n_nodes):
    indegree = np.zeros(n_nodes, dtype=np.int64)
    for e in range(active.shape[0]):
        if active[e]:
            indegree[head[e]] += 1
    order = np.empty(n_nodes, dtype=np.int64)
    stack = np.empty(n_nodes, dtype=np.int64)
    top = 0
    for u in range(n_nodes):
        if indegree[u] == 0:
            stack[top] = u
            top += 1
    emitted = 0
    while top > 0:
        top -= 1
        u = stack[top]
        order[emitted] = u
        emitted += 1
        for idx in range(out_ptr[u], out_ptr[u + 1]):
            e = out_links[idx]
            if active[e]:
                v = head[e]
                indegree[v] -= 1
                if indegree[v] == 0:
                    stack[top] = v
                    top += 1
    return order, emitted


def cycle_links(topo: Topology, active: np.ndarray) -> np.ndarray:
    """Active links lying on, or reachable from, a directed cycle of the active subgraph.

    Kahn's algorithm emits every node that neither lies on a cycle nor is fed by one, so the nodes it
    fails to emit are exactly those a cycle passes through or reaches. An active link with both endpoints
    unemitted is a candidate; that superset is what a repair needs, since it only requires somewhere to
    cut. Empty when the mask is acyclic.
    """
    n = topo.n_nodes
    indegree = np.bincount(topo.head[active], minlength=n).astype(np.int64)
    out_ptr, out_links = topo.out_ptr, topo.out_links
    emitted = np.zeros(n, dtype=bool)
    stack = np.flatnonzero(indegree == 0).astype(np.int64).tolist()
    while stack:
        u = stack.pop()
        emitted[u] = True
        for idx in range(out_ptr[u], out_ptr[u + 1]):
            e = out_links[idx]
            if active[e]:
                v = topo.head[e]
                indegree[v] -= 1
                if indegree[v] == 0:
                    stack.append(int(v))
    if emitted.all():
        return np.empty(0, dtype=np.int64)
    stuck = ~emitted
    return np.flatnonzero(active & stuck[topo.tail] & stuck[topo.head])


def isolation_margin(topo: Topology, potential: np.ndarray) -> float:
    """Smallest potential gap over the links whose efficiency can still change.

    This minimises over every eligible link, which is what the exactness bound is stated with; it is not
    the exclusion margin, which minimises over excluded links only.

    Three classes of link are ineligible, and none of the exclusions is a numerical tolerance. A link
    arriving at a centroid other than ``d`` lies outside the efficient support *when the transit rule
    applies*, so its gap has nothing to change; where transit is permitted such a link is an ordinary
    member of the support and is eligible. A link leaving ``d`` is in no support, because the destination
    absorbs. And a forced connector is in every admissible support, so the sign of its gap decides
    nothing: a zero-cost connector has gap exactly zero and, if counted, would drive this minimum to zero
    on every network that has one.

    The exclusions are gated on the same flags :func:`active_for_dest` gates them on. Applying the
    transit exclusion unconditionally makes the margin vacuous on a network whose nodes are all
    centroids, because every link then arrives at some centroid and none is eligible.
    """
    tail, head = topo.tail, topo.head
    best = np.inf
    for d in range(topo.n_zones):
        column = potential[:, d]
        eligible = _eligible_links(topo, column, d)
        if not eligible.any():
            continue
        gap = np.abs(column[tail][eligible] - column[head][eligible])
        if gap.size:
            best = min(best, float(gap.min()))
    return best


def _eligible_links(topo: Topology, column: np.ndarray, d: int) -> np.ndarray:
    """Links whose efficiency toward ``d`` is decided by the potential gap rather than by a rule.

    Forced connectors and links touching the destination are excluded, and a foreign centroid's incoming
    links only where the transit rule applies. This is the same gating :func:`active_for_dest` uses, and
    the two must agree: a margin minimised over links the support decides by rule is not a margin.
    """
    tail, head, n_zones = topo.tail, topo.head, topo.n_zones
    eligible = np.isfinite(column[tail]) & np.isfinite(column[head])
    if topo.no_thru:
        eligible &= ~((head < n_zones) & (head != d))
    if topo.zero_dep_conn:
        eligible &= ~((tail < n_zones) & (tail != d))
    eligible &= (tail != d) & (head != d)
    return eligible


def _reached_by_flow(topo: Topology, active: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Nodes the destination's flow can occupy: origins carrying demand, and the support below them."""
    reached = np.zeros(topo.n_nodes, dtype=bool)
    out_ptr, out_links, head = topo.out_ptr, topo.out_links, topo.head
    stack = [int(o) for o in origins]
    for o in stack:
        reached[o] = True
    while stack:
        u = stack.pop()
        for idx in range(out_ptr[u], out_ptr[u + 1]):
            e = out_links[idx]
            if active[e]:
                v = int(head[e])
                if not reached[v]:
                    reached[v] = True
                    stack.append(v)
    return reached


def exclusion_margin(topo: Topology, potential: np.ndarray, od: np.ndarray,
                     tolerance: float = 0.0) -> float:
    """Smallest excess cost over the links the support excludes.

    This is the margin the full-graph approximation bound is stated with. It minimises over every excluded
    link, whether or not any flow stands at its tail, which is what makes it smaller than the path-excess
    margin and what makes the bound it carries a bound.

    Returns infinity where the support excludes nothing, which is a network on which the comparison the
    bound describes has no content.
    """
    tail, head = topo.tail, topo.head
    best = np.inf
    for d in range(topo.n_zones):
        if od[:, d].sum() <= 0.0:
            continue
        column = potential[:, d]
        finite = np.isfinite(column[tail]) & np.isfinite(column[head])
        gap = np.zeros(topo.n_links)
        np.subtract(column[tail], column[head], out=gap, where=finite)
        excluded = _eligible_links(topo, column, d) & (gap < -tolerance)
        if excluded.any():
            best = min(best, float(np.abs(gap[excluded]).min()))
    return best


def path_excess_margin(topo: Topology, potential: np.ndarray, od: np.ndarray,
                       tolerance: float = 0.0) -> float:
    """Smallest excess cost of a link excluded at a node the destination's flow reaches.

    This is the diagnostic margin the experiments report. It differs from the exclusion margin, which
    minimises over every excluded link whether or not any flow ever stands at its tail, and from the
    isolation margin, which also admits tied links and is therefore zero wherever a tie exists. Because it
    minimises over a subset of the excluded links, it is at least the exclusion margin, and the theory is
    never stated with it.

    Returns infinity when no excluded link stands at a reached node, which is the case on a network whose
    support admits everything the flow can see.
    """
    tail, head = topo.tail, topo.head
    best = np.inf
    for d in range(topo.n_zones):
        origins = np.flatnonzero(od[:, d] > 0.0)
        if origins.size == 0:
            continue
        column = potential[:, d]
        finite = np.isfinite(column[tail]) & np.isfinite(column[head])
        # Subtracting where the potential is infinite is a difference of infinities, so the gap is
        # computed only where both ends are reachable and left at zero elsewhere.
        gap = np.zeros(topo.n_links)
        np.subtract(column[tail], column[head], out=gap, where=finite)
        excluded = _eligible_links(topo, column, d) & (gap < -tolerance)
        if not excluded.any():
            continue
        reached = _reached_by_flow(topo, active_for_dest(topo, column, d), origins)
        standing = excluded & reached[tail]
        if standing.any():
            best = min(best, float(np.abs(gap[standing]).min()))
    return best
