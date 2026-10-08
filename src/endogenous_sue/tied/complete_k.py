"""The unconditional rung: simplicial search over the whole flow space.

This is the branch the guarantee rests on. Everything else in this subpackage is an accelerator: the face
searches look at a *face*, and that is the right face only if the working set was chosen well and only if
the atom family can represent the answer. This routine assumes neither.

The label is the loading of the strict efficient support at the vertex's own cost. That support is acyclic
at every cost, since the gaps around a directed cycle are strictly positive and telescope to zero, so the
label is always defined, always assigns the whole demand, and always lies in the equilibrium
correspondence. No mixture is formed, no atom family is indexed, and the hardness of separating the
achievable-marginal set does not arise. There is no monotonicity condition, no regularity condition, no
bracketing, no enumeration of supports and no exceptional set.

The label is used as a *label* and never as an iteration. It is the naive map, discontinuous exactly at
the ties, and as an iteration it cannot converge. Complementary pivoting returns a convex combination of
the labels at the vertices of one small simplex, and a convex combination of loadings over supports
admissible at nearby costs is a point of the correspondence, so the pivoting is driven directly and its
completely labelled simplex read off; the two approximation clauses are then evaluated at the returned
flow.

The search is in as many dimensions as there are links, and no polynomial bound on the pivot count is
claimed or exists. This rung is not the method of choice and is not expected to run on a city network. It
is what the guarantee descends to, and it is here so that the guarantee is a property of the code and not
only of the argument.
"""
from __future__ import annotations

import time

import numpy as np

from ..config import DEFAULTS, WEIGHT_EPSILON, Settings
from ..costs import bpr_cost
from ..loading import absorbing, load_destination
from ..shortestpath import potentials
from ..subnetwork import active_for_dest, kahn_order
from .simplicial import walk_one_level

__all__ = ["naive_label", "solve_complete_core"]


def naive_label(topo, od, mu, arrays, demanded, x):
    """The loading of the strict efficient support at ``x``'s own cost, with the supports it used."""
    fft, cap, b, power, _, _ = arrays
    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    potential = potentials(topo, cost)
    y = np.zeros(topo.n_links)
    masks = {}
    for d in demanded:
        mask = absorbing(topo, active_for_dest(topo, potential[:, d], d), d)
        masks[d] = mask
        y += load_destination(topo, od, mu, cost, d, mask, kahn_order(topo, mask))
    return y, masks, cost, potential


def _clauses(topo, od, mu, arrays, demanded, x, masks_seen, weights):
    """The two approximation clauses, evaluated at the cost the returned point induces.

    The first is the mixture residual: the supports come from the simplex vertices and the cost from the
    returned point. The second is the worst violation of the support inclusion by any support carrying
    weight, measured in gap units at that same cost.
    """
    fft, cap, b, power, _, _ = arrays
    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    potential = potentials(topo, cost)
    tail, head = topo.tail, topo.head
    mixture = np.zeros(topo.n_links)
    worst = 0.0
    for weight, masks in zip(weights, masks_seen):
        if weight <= WEIGHT_EPSILON:
            continue
        for d in demanded:
            mask = masks[d]
            mixture += weight * load_destination(topo, od, mu, cost, d, mask, kahn_order(topo, mask))
            gap = potential[tail, d] - potential[head, d]
            strict = np.isfinite(gap) & (gap > 0.0)
            if strict.any():
                worst = max(worst, float(np.max(np.where(strict & ~mask, gap, 0.0))))
            negative = np.isfinite(gap) & (gap < 0.0) & mask
            if negative.any():
                worst = max(worst, float(np.max(np.where(negative, -gap, 0.0))))
    return float(np.max(np.abs(x - mixture))), worst


def _mixture(demanded, masks_seen, weights):
    """One ``(destination, support, mass)`` entry per weighted vertex."""
    out = []
    for weight, masks in zip(weights, masks_seen):
        if weight <= WEIGHT_EPSILON:
            continue
        for d in demanded:
            out.append((int(d), masks[d], float(weight)))
    return out



def solve_complete_core(topo, arrays, od, mu, demanded, settings: Settings = DEFAULTS,
                        grid=2, max_levels=10, x_start=None, jitter=1e-12, progress=None):
    """The same search on an already-built topology, so a ladder can descend to it.

    It terminates at every mesh by the combinatorial argument: the triangulation is finite, each
    completely labelled facet is shared by at most two simplices, and the lexicographic rule makes the
    successor unique, so the path is simple and must end. The mesh is halved until both approximation
    clauses fall below the target, which terminates because a sufficient mesh exists for every target.
    """
    n_links = topo.n_links
    demand = float(od.sum())
    if demand <= 0.0:
        raise ValueError("the network carries no demand")

    counter = [0]
    started = time.perf_counter()

    def psi(u):
        return naive_label(topo, od, mu, arrays, demanded, demand * np.asarray(u, dtype=float))[0] / demand

    centre = (np.clip(np.asarray(x_start, dtype=float) / demand, 0.0, 1.0)
              if x_start is not None else np.full(n_links, 0.5))
    trail, best = [], None

    for level in range(max_levels):
        mesh = int(grid * (2 ** level))
        point, pivots, status, certificate = walk_one_level(
            psi, n_links, mesh, centre, settings.max_pivots, counter, jitter=jitter)
        rung = dict(level=level, mesh=1.0 / mesh, pivots=pivots, status=status)
        if certificate is None:
            trail.append(rung)
            if progress is not None:
                progress(f"level {level}/{max_levels} mesh=1/{mesh} status={status} pivots={pivots}")
            continue

        points, weights = certificate
        x = demand * (weights @ points)
        masks_seen = [naive_label(topo, od, mu, arrays, demanded, demand * p)[1] for p in points]
        mixture_residual, inclusion_defect = _clauses(topo, od, mu, arrays, demanded, x,
                                                      masks_seen, weights)
        reached = max(mixture_residual, inclusion_defect)
        rung.update(mixture_residual=mixture_residual, inclusion_defect=inclusion_defect,
                    reached=reached)
        trail.append(rung)
        if progress is not None:
            progress(f"level {level}/{max_levels} mesh=1/{mesh} pivots={pivots} "
                     f"mixture={mixture_residual:.3e} inclusion={inclusion_defect:.3e}")

        if best is None or reached < best[1]:
            best = (x, reached, level, mixture_residual, inclusion_defect,
                    _mixture(demanded, masks_seen, weights))
        if reached <= settings.pivot_epsilon:
            return x, dict(status="certified", reached=reached,
                           mixture_residual=mixture_residual, inclusion_defect=inclusion_defect,
                           level=level, mesh=1.0 / mesh, evaluations=counter[0],
                           mixture=_mixture(demanded, masks_seen, weights), trail=trail,
                           wall_seconds=time.perf_counter() - started)
        # Restart the search centred on the answer just found.
        centre = np.clip(x / demand, 0.0, 1.0)

    if best is None:
        return None, dict(status="no labelled simplex", evaluations=counter[0], trail=trail,
                          wall_seconds=time.perf_counter() - started)
    x, reached, level, mixture_residual, inclusion_defect, mixture = best
    return x, dict(status="target not reached", reached=reached,
                   mixture_residual=mixture_residual, inclusion_defect=inclusion_defect,
                   level=level, evaluations=counter[0], mixture=mixture, trail=trail,
                   wall_seconds=time.perf_counter() - started)
