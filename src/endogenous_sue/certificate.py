"""The equilibrium certificate: whether a returned flow satisfies the definition.

What is certified is not that the solver did the right thing but that the flow it returned is an
equilibrium. Three clauses have to hold together at the flow's own realised cost.

*Admissibility.* Every support carrying weight lies between the strict and the closed efficient sets of
that cost. This is the clause a solver residual can be small without satisfying, because the residual is
measured against the support the solver held rather than the one the cost induces.

*Representation.* The flow is a convex mixture of the loadings on those supports. Where a single support
carries all the weight and equals the strict efficient set, the flow is a strict equilibrium; otherwise it
is a relaxed one.

*Conservation.* The flow carries the demand. Admissibility does not imply it: on a network with zero-cost
links the strict efficient set admits a link only where the potential strictly decreases, so a zero-cost
link has gap exactly zero and is excluded, and demand that can move only through one is dropped while
every other clause still passes.

A verdict is returned for every flow. Nothing is reported as an equilibrium without one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import DEFAULTS, Settings
from .loading import absorbing, load_destination
from .network import NetworkArrays, Topology
from .shortestpath import potentials
from .subnetwork import active_for_dest, kahn_order

__all__ = ["Certificate", "conservation_error", "admissibility_violations", "violating_pairs",
           "supports_from_potential",
           "reload_on_own_support", "certify"]


@dataclass
class Certificate:
    """The outcome of testing one flow against the definition."""

    certified: bool
    kind: str
    reason: str = ""
    violations: int = 0
    worst_gap: float = 0.0
    worst_violating_flow: float = 0.0
    mixture_residual: float = 0.0
    conservation_max: float = 0.0
    conservation_l1: float = 0.0
    conservation_relative: float = 0.0
    detail: dict = field(default_factory=dict)


def conservation_error(net: dict, od: np.ndarray, x: np.ndarray) -> tuple[float, float, float]:
    """Does the flow carry the demand? Returns the largest and total node errors, and total demand.

    For an assignment carrying the full demand the divergence of aggregate link flow at each node equals
    the demand leaving it less the demand arriving, and vanishes at non-zone nodes. Exact, ``O(E)``, and
    needing no per-destination decomposition.
    """
    arrays = NetworkArrays.from_dict(net)
    divergence = np.zeros(arrays.n_nodes)
    np.add.at(divergence, arrays.tail, x)
    np.add.at(divergence, arrays.head, -x)
    expected = np.zeros(arrays.n_nodes)
    expected[:arrays.n_zones] = od.sum(axis=1) - od.sum(axis=0)
    error = np.abs(divergence - expected)
    return float(error.max()), float(error.sum()), float(od.sum())


def supports_from_potential(topo: Topology, od: np.ndarray,
                            potential: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """The ``(destination, support)`` pairs a potential table induces, over demanded destinations."""
    return [(d, active_for_dest(topo, potential[:, d], d))
            for d in range(topo.n_zones) if od[:, d].sum() > 0.0]


def admissibility_violations(topo: Topology, supports, potential: np.ndarray,
                             x: np.ndarray | None = None,
                             tolerance: float = 0.0) -> tuple[int, float, float]:
    """Violations of the inclusion between the strict and closed efficient sets, with their magnitude.

    ``supports`` is a sequence of ``(destination, mask)`` pairs; ``potential`` is the table the flow's own
    cost induces. A support is admissible exactly when every link on which it differs from the strict set
    is tied, so a violating link has a size: the amount by which its gap misses zero, and the flow it
    carries. Both matter, since a link violating by ``1e-12`` and carrying nothing is a different object
    from one violating by a tenth and carrying a tenth of the demand.

    The test is the model's own. The efficient sets are defined inside the link set the transit rule
    admits, and departure connectors are admissible by assumption, so their gap does not decide their
    membership. Applying plain admissibility instead would judge the model's own support rule against a
    different model, and every network with zero-cost connectors would fail by construction. The two
    clauses are separately gated because the link set follows the transit flag and the connector rule
    follows the zero-cost-connector flag, and one flag cannot answer both.

    ``tolerance`` widens the tied band. At a mixed equilibrium the contested links sit exactly on the tie
    boundary, where strict membership is decided by floating-point noise; widening the band makes this
    test judge the uncontested links, which is what it is for.
    """
    violations, worst_gap, worst_flow = 0, 0.0, 0.0
    for d, mask in supports:
        bad, gap = _inadmissible(topo, d, mask, potential, tolerance)
        if bad.any():
            violations += int(bad.sum())
            worst_gap = max(worst_gap, float(np.abs(gap[bad]).max()))
            if x is not None:
                worst_flow = max(worst_flow, float(x[bad].max()))
    return violations, worst_gap, worst_flow


def _inadmissible(topo: Topology, d: int, mask: np.ndarray, potential: np.ndarray,
                  tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Which links of one destination's support the potential at a flow's own cost refuses, and by how
    much. The single definition of the rule: everything that tests admissibility reads it from here, so
    a check made before a face is accepted cannot drift from the check that grades the result."""
    tail, head, n_zones = topo.tail, topo.head, topo.n_zones
    column = potential[:, d]
    at_tail, at_head = column[tail], column[head]
    finite = np.isfinite(at_tail) & np.isfinite(at_head)
    # Subtracting only where both ends are reachable: an unreachable node holds an infinity, and
    # subtracting one from another is not a number rather than a gap.
    gap = np.zeros(at_tail.shape)
    np.subtract(at_tail, at_head, out=gap, where=finite)
    admissible = (~((head < n_zones) & (head != d)) if topo.no_thru
                  else np.ones(topo.n_links, dtype=bool))
    # The forced links: those the model puts in every admissible support whatever their gap. Flow may
    # always arrive at its destination, and may always leave a zone where the departure connector is
    # zero-cost. They belong in the strict set as well as the closed one, or a support that drops one
    # is not a violation of anything and the checker never sees demand being stranded.
    forced = finite & (tail != d) & ((head == d)
                                     | (topo.zero_dep_conn & (tail < n_zones) & admissible))
    strict = (finite & (gap > tolerance) & (tail != d) & admissible) | forced
    closed = (finite & (gap >= -tolerance) & (tail != d) & admissible) | forced
    return (strict & ~mask) | (mask & ~closed & finite), gap


def violating_pairs(topo: Topology, supports, potential: np.ndarray,
                    tolerance: float = 0.0) -> list[tuple[int, int, float]]:
    """The distinct ``(destination, link, gap)`` triples :func:`admissibility_violations` counts.

    It counts incidences, once per mixture component, so one wrongly-included link shared by four
    components is four violations of one pair. A caller that means to reopen the offending pairs needs
    them named and deduplicated instead, which is what this returns: Anaheim at mu=0.5 reports eleven
    violations that are five pairs.
    """
    found: dict[tuple[int, int], float] = {}
    for d, mask in supports:
        bad, gap = _inadmissible(topo, int(d), mask, potential, tolerance)
        for link in np.flatnonzero(bad):
            key = (int(d), int(link))
            if abs(float(gap[link])) > abs(found.get(key, 0.0)):
                found[key] = float(gap[link])
    return [(d, link, found[(d, link)]) for d, link in sorted(found)]


def reload_on_own_support(net: dict, od: np.ndarray, mu: float, x: np.ndarray) -> np.ndarray:
    """Load once on the strict efficient support that ``x``'s own cost induces.

    A strict equilibrium reproduces itself this way, so the distance between ``x`` and what this returns
    is the strict-reload defect. A relaxed equilibrium is a mixture over several supports and is not
    expected to equal the loading on any single one, so a non-zero defect there is not a failure.

    Evaluated on the flow alone, taking nothing from the solver that produced it.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    cost = arrays.cost(np.maximum(x, 0.0))
    potential = potentials(topo, cost)
    y = np.zeros(arrays.n_links)
    for d, mask in supports_from_potential(topo, od, potential):
        y += load_destination(topo, od, mu, cost, d, mask, kahn_order(topo, mask))
    return y


def certify(net: dict, od: np.ndarray, mu: float, x: np.ndarray,
            mixture: list[tuple[int, np.ndarray, float]] | None = None,
            settings: Settings = DEFAULTS) -> Certificate:
    """Test a flow against the definition and return the verdict.

    ``mixture`` is a list of ``(destination, support, weight)``. When it is omitted the flow is tested as
    a strict equilibrium against the supports its own cost induces, each carrying full weight.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    cost = arrays.cost(np.maximum(x, 0.0))
    potential = potentials(topo, cost)

    # Without a mixture the flow can only be tested for strictness, against the supports its own cost
    # induces. A relaxed equilibrium is a mixture over several supports and is not expected to equal the
    # loading on any single one, so a large defect here is a measurement and not a failure: it says the
    # flow is not strict, and a relaxed verdict needs the mixture the tied solver returns.
    strict_test = mixture is None
    if strict_test:
        supports = supports_from_potential(topo, od, potential)
        mixture = [(d, mask, 1.0) for d, mask in supports]
    else:
        supports = [(d, mask) for d, mask, weight in mixture if weight > 0.0]

    violations, worst_gap, worst_flow = admissibility_violations(
        topo, supports, potential, x, tolerance=settings.residual_tolerance)

    reconstructed = np.zeros(arrays.n_links)
    for d, mask, weight in mixture:
        if weight <= 0.0:
            continue
        active = absorbing(topo, mask, d)
        reconstructed += weight * load_destination(topo, od, mu, cost, d, active,
                                                   kahn_order(topo, active))
    scale = max(float(np.abs(x).max()), 1e-30)
    mixture_residual = float(np.max(np.abs(x - reconstructed)) / scale)

    largest, total_error, demand = conservation_error(net, od, x)
    relative = total_error / max(demand, 1e-30)
    conserves = largest <= settings.conservation_tolerance * max(demand, 1.0)

    reasons = []
    if violations:
        reasons.append(f"{violations} support inclusion violation(s), worst gap {worst_gap:.3e}")
    if mixture_residual > settings.residual_tolerance:
        reasons.append("the flow is not the loading on the strict support of its own cost, defect "
                       f"{mixture_residual:.3e}" if strict_test else
                       f"the mixture does not reproduce the flow, residual {mixture_residual:.3e}")
    if not conserves:
        reasons.append(f"demand not conserved, worst node error {largest:.3e} of {demand:.3e}")

    if not reasons:
        kind = "strict" if len(mixture) == 1 or strict_test else "relaxed"
    elif strict_test and not violations and conserves:
        kind = "not strict"
        reasons.append("a relaxed verdict requires the mixture of supports carrying weight")
    else:
        kind = "not certified"

    return Certificate(
        certified=not reasons,
        kind=kind,
        reason="; ".join(reasons),
        violations=violations,
        worst_gap=worst_gap,
        worst_violating_flow=worst_flow,
        mixture_residual=mixture_residual,
        conservation_max=largest,
        conservation_l1=total_error,
        conservation_relative=relative,
        detail=dict(n_supports=len(mixture), demand=demand),
    )
