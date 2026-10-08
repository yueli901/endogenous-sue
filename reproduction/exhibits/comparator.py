"""The full-graph recursive-logit reference: the model without an efficient-support restriction.

Used only as a comparator. It is defined on the cyclic graph, so its value function exists only while the
spectral radius of the exponential weight matrix stays below one, and it is evaluated by iteration rather
than by an ordered sweep. That is the point of the comparison: where the radius reaches one there is
nothing to compare against.

The centroid rule here is the one the method itself uses, taken from the same topology, so the two models
are compared under one convention.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

from endogenous_sue.config import VALUE_ITERATIONS
from endogenous_sue.network import NetworkArrays

__all__ = ["spectral_radius", "value_residuals", "loading_operator", "solve_msa"]



def spectral_radius(tail, head, cost, mu, n_nodes, max_iterations: int = 3000) -> float:
    """Largest eigenvalue in modulus of the matrix summing the exponential weights over node pairs."""
    with np.errstate(over="ignore", under="ignore"):
        weight = np.exp(-mu * cost)
    matrix = coo_matrix((weight, (tail, head)), shape=(n_nodes, n_nodes)).tocsr()
    try:
        return float(np.abs(eigs(matrix, k=1, which="LM", return_eigenvectors=False,
                                 maxiter=max_iterations, tol=1e-6)[0]))
    except Exception:
        # Power iteration, for the matrices the sparse eigensolver declines. It converges to the same
        # quantity and is only slower.
        vector = np.random.default_rng(0).random(n_nodes)
        vector /= np.linalg.norm(vector)
        radius = 0.0
        for _ in range(2000):
            product = matrix @ vector
            norm = np.linalg.norm(product)
            if norm < 1e-300 or not np.isfinite(norm):
                return float("inf") if not np.isfinite(norm) else 0.0
            vector = product / norm
            radius = norm
        return float(radius)


def _value_iteration(tail, head, cost, mu, n_nodes, n_zones, iterations, track=False):
    value = np.zeros((n_nodes, n_zones))
    link_cost = (-mu * cost)[:, None]
    residuals = []
    for _ in range(iterations):
        contribution = link_cost + value[head, :]
        largest = np.full((n_nodes, n_zones), -np.inf)
        np.maximum.at(largest, tail, contribution)
        finite = np.isfinite(largest[tail, :])
        shift = np.where(finite, largest[tail, :], 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            terms = np.where(finite, np.exp(np.clip(contribution - shift, -500, 50)), 0.0)
        total = np.zeros((n_nodes, n_zones))
        np.add.at(total, tail, terms)
        following = np.where(np.isfinite(largest), largest + np.log(np.maximum(total, 1e-300)), 0.0)
        following[np.arange(n_zones), np.arange(n_zones)] = 0.0
        if track:
            delta = (following - value)[np.isfinite(following - value)]
            change = float(np.max(np.abs(delta))) if delta.size else float("inf")
            residuals.append(change if np.isfinite(change) else 1e300)
        value = following
        if track and not np.all(np.isfinite(value)):
            residuals.append(1e300)
            break
    return value, residuals


def value_residuals(tail, head, cost, mu, n_nodes, n_zones, iterations=200) -> list[float]:
    """Per-iteration sup-norm change of the value iteration.

    Above a unit spectral radius the iteration does not overflow: the value grows without bound at an
    asymptotically constant rate, so this settles at the logarithm of the radius instead of decaying.
    Divergence is therefore visible as a residual that never reaches zero.
    """
    return _value_iteration(tail, head, cost, mu, n_nodes, n_zones, iterations, track=True)[1]


def _propagate(tail, head, choice, od, n_nodes, n_zones, n_links, iterations, block_arrival):
    """Push demand forward along the choice probabilities, destination by destination."""
    y = np.zeros(n_links)
    for d in range(n_zones):
        column = od[:, d]
        if column.sum() <= 0.0:
            continue
        at_node = np.zeros(n_nodes)
        at_node[:n_zones] = column
        for _ in range(iterations):
            if at_node.sum() <= 1e-10:
                break
            at_node[d] = 0.0
            flow = at_node[tail] * choice[:, d]
            if block_arrival:
                flow = np.where((head < n_zones) & (head != d), 0.0, flow)
            y += flow
            at_node = np.zeros(n_nodes)
            np.add.at(at_node, head, flow)
    return y


def loading_operator(net: dict, od: np.ndarray, mu: float, iterations: int = VALUE_ITERATIONS):
    """Return the one-pass full-graph loading at the cost of a flow, and the radius at that flow."""
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()

    def radius_at(x):
        try:
            return spectral_radius(arrays.tail, arrays.head, arrays.cost(np.maximum(x, 0.0)), mu,
                                   arrays.n_nodes)
        except Exception:
            return float("nan")

    def load(x):
        cost = arrays.cost(np.maximum(x, 0.0))
        value, _ = _value_iteration(arrays.tail, arrays.head, cost, mu, arrays.n_nodes,
                                    arrays.n_zones, iterations)
        with np.errstate(over="ignore", invalid="ignore"):
            choice = np.exp(np.clip(-mu * cost[:, None] + value[arrays.head, :]
                                    - value[arrays.tail, :], -500, 50))
        return _propagate(arrays.tail, arrays.head, choice, od, arrays.n_nodes, arrays.n_zones,
                          arrays.n_links, iterations, topo.zero_dep_conn)

    return load, radius_at


def solve_msa(net: dict, od: np.ndarray, mu: float, value_iterations: int = 300,
              outer: int = 800, tolerance: float = 1e-5):
    """Full-graph equilibrium by averaging over the value iteration.

    Returns the flow, a status of ``converged``, ``limit`` or ``overflow``, and the wall time. The
    tolerance is matched to the rate of the averaging, and numerical blow-up aborts rather than being
    reported as a converged but exploded solution.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    ceiling = float(od.sum()) * 10.0 + 1.0
    failed = (np.full(arrays.n_links, np.nan), "overflow")

    started = time.perf_counter()
    x = np.zeros(arrays.n_links)
    for iteration in range(1, outer + 1):
        cost = arrays.cost(x)
        value, _ = _value_iteration(arrays.tail, arrays.head, cost, mu, arrays.n_nodes,
                                    arrays.n_zones, value_iterations)
        if not np.all(np.isfinite(value)):
            return (*failed, time.perf_counter() - started)
        with np.errstate(over="ignore", invalid="ignore"):
            choice = np.exp(np.clip(-mu * cost[:, None] + value[arrays.head, :]
                                    - value[arrays.tail, :], -500, 50))
        if not np.all(np.isfinite(choice)):
            return (*failed, time.perf_counter() - started)
        y = _propagate(arrays.tail, arrays.head, choice, od, arrays.n_nodes, arrays.n_zones,
                       arrays.n_links, value_iterations, topo.zero_dep_conn)
        if (not np.all(np.isfinite(y))) or float(np.max(np.abs(y))) > ceiling:
            return (*failed, time.perf_counter() - started)
        gap = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
        x = x + (1.0 / iteration) * (y - x)
        if iteration > 5 and gap < tolerance:
            return x, "converged", time.perf_counter() - started
    return x, "limit", time.perf_counter() - started
