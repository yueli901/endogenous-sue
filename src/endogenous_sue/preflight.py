"""What a run checks about its inputs before it spends anything on them.

A sweep cell can run for hours. If the network it read is the wrong one, or its capacities are missing, or
its demand cannot reach its destinations, the run still produces numbers and they are still wrong. These
checks are cheap, they run first, and each prints the value it saw beside the threshold it is judged
against, so a run started on bad input can be stopped in its first seconds rather than diagnosed from its
output.

Two kinds of check. *Inputs* -- the file exists, is non-empty, parses to the size it declares, and carries
the demand it should. *Degeneracies* -- the elements the model has no answer for, each reported with what
the solver will actually do with it, because a zero-capacity link is not an error to be raised but a cost
that no longer increases with flow, and the reader needs to know how many there are.

The existence condition of the method itself is also checked here, before the solve and again after it:
strictly increasing link cost is what makes the equilibrium unique on a frozen support, and it fails
exactly where the representability clamp binds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (
    FLOAT64_EXPONENT_HEADROOM,
    MIN_CAPACITY,
    REACHABILITY_MAX_ZONES,
    REACHABLE_RATIO,
    UNIT_CAPACITY_EXPONENT,
    UNIT_CAPACITY_LIMIT,
    UNREACHABLE_LIMIT,
    ZERO_CAPACITY_LIMIT,
)
from .network import NetworkArrays

__all__ = ["NetworkReport", "inspect_network", "cost_is_increasing", "report_lines"]


@dataclass
class NetworkReport:
    """What one network looks like, and what about it the solver cannot answer for."""

    name: str
    n_nodes: int
    n_zones: int
    n_links: int
    demand: float
    demand_pairs: int
    zero_capacity: int
    unit_capacity: int
    zero_free_flow: int
    self_loops: int
    isolated_nodes: int
    unreachable_demand: float | None
    unreachable_pairs: int | None
    max_power: float
    clamp_ratio: float
    warnings: list = field(default_factory=list)

    @property
    def unreachable_share(self) -> float | None:
        """The share of demand no route can carry, or ``None`` where it was not measured."""
        if self.unreachable_demand is None:
            return None
        return self.unreachable_demand / max(self.demand, 1e-30)



def cost_is_increasing(arrays: NetworkArrays, x: np.ndarray) -> tuple[int, int, float]:
    """Where the cost has stopped increasing with flow, split by cause, and its smallest positive slope.

    The equilibrium on a frozen support is unique because the cost is strictly increasing. Wherever the
    derivative is zero that argument is gone, so this is the method's own validity condition, measured on
    a returned flow and recorded rather than enforced.

    Two causes, and they are not interchangeable. A link with zero free-flow time has a cost that is
    identically zero, so its derivative is zero at every flow: that is the zero-cost connector convention
    the model already admits, and it is a property of the benchmark data rather than a numerical failure.
    A link with positive free-flow time whose derivative has nonetheless reached zero is the
    representability clamp binding, or the load ratio underflowing beneath it, and that is a real loss of
    the uniqueness argument on a link the model means to be strictly increasing.

    Reported separately because a single count conflates them. On Chicago-Sketch at mu=1, 772 of the 784
    flat links carry flow and every one of them is a zero-free-flow-time connector touching a centroid;
    Berlin-Tiergarten at mu=1 is 173 of 173 the same way; Anaheim, Sioux Falls and Eastern Massachusetts
    have none at all, and none of them has a zero-free-flow-time link.

    Returns ``(flat_by_convention, flat_despite_positive_cost, smallest_positive_derivative)``.
    """
    derivative = arrays.cost_prime(np.maximum(x, 0.0))
    flat = derivative <= 0.0
    carrying = flat & (np.asarray(x) > 0.0)
    by_convention = int(np.count_nonzero(carrying & (arrays.fft <= 0.0)))
    despite = int(np.count_nonzero(carrying & (arrays.fft > 0.0)))
    positive = derivative[derivative > 0.0]
    return by_convention, despite, float(positive.min()) if positive.size else 0.0


def inspect_network(name: str, net: dict, od: np.ndarray,
                    potential: np.ndarray | None = None,
                    measure_reachability: bool | None = None) -> NetworkReport:
    """Describe one network and count what is degenerate in it.

    ``potential`` is the free-flow potential table when the caller already has it; it decides which
    origin-destination pairs no route can serve. Computed here when it is not supplied, unless the
    network is larger than ``REACHABILITY_MAX_ZONES`` and no caller asked for it: that measurement is one
    shortest-path pass per zone, which is the cost of a solver iteration, and a check meant to run in the
    first seconds must not silently cost minutes. Where it is skipped the fields say so rather than
    reading zero.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    if measure_reachability is None:
        measure_reachability = potential is not None or arrays.n_zones <= REACHABILITY_MAX_ZONES
    if measure_reachability and potential is None:
        from .shortestpath import potentials
        potential = potentials(topo, arrays.fft)

    # The trips file can declare fewer zones than the network does. Indexing past the demand matrix
    # would raise here, in a diagnostic, on a network the corpus does not even include; it is reported
    # instead, because a demand matrix that does not cover the zones is exactly what this is for.
    served = min(arrays.n_zones, od.shape[1])
    lost, pairs = (0.0, 0) if measure_reachability else (None, None)
    for d in range(served if measure_reachability else 0):
        if od[:, d].sum() <= 0.0:
            continue
        column = potential[:, d]
        for o in np.flatnonzero(od[:, d] > 0.0):
            if o >= arrays.n_nodes:
                continue
            if not np.isfinite(column[o]):
                lost += float(od[o, d])
                pairs += 1

    touched = np.zeros(arrays.n_nodes, dtype=bool)
    touched[arrays.tail] = True
    touched[arrays.head] = True

    # The load ratio at which ``ratio ** power`` stops being representable, for the steepest link. Above
    # it the cost is clamped and its derivative is zero.
    max_power = float(arrays.power.max()) if arrays.n_links else 0.0
    clamp_ratio = float(10.0 ** (FLOAT64_EXPONENT_HEADROOM / max_power)) if max_power > 0 else np.inf

    report = NetworkReport(
        name=name, n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
        demand=float(od.sum()), demand_pairs=int(np.count_nonzero(od > 0.0)),
        zero_capacity=int(np.count_nonzero(arrays.cap <= MIN_CAPACITY)),
        unit_capacity=int(np.count_nonzero(arrays.cap == 1.0)),
        zero_free_flow=int(np.count_nonzero(arrays.fft <= 0.0)),
        self_loops=int(np.count_nonzero(arrays.tail == arrays.head)),
        isolated_nodes=int(np.count_nonzero(~touched)),
        unreachable_demand=lost, unreachable_pairs=pairs,
        max_power=max_power, clamp_ratio=clamp_ratio)

    if report.demand <= 0.0:
        report.warnings.append("no demand: every cell on this network assigns nothing")
    if report.unreachable_share is not None and report.unreachable_share > UNREACHABLE_LIMIT:
        report.warnings.append(
            f"{report.unreachable_share:.3%} of demand reaches no destination at free flow, above the "
            f"{UNREACHABLE_LIMIT:.3%} limit; every total reported for it is short by that much")
    if report.zero_capacity > ZERO_CAPACITY_LIMIT * report.n_links:
        report.warnings.append(
            f"{report.zero_capacity} of {report.n_links} links have capacity at or below "
            f"{MIN_CAPACITY:g}, above the {ZERO_CAPACITY_LIMIT:.0%} limit; the load ratio is not a load "
            f"ratio and the cost is not the model's cost")
    if report.self_loops:
        report.warnings.append(f"{report.self_loops} self-loop(s); these can never be efficient and "
                               f"carry no flow, but they inflate the link count the tables report")
    if served < arrays.n_zones:
        report.warnings.append(
            f"the demand matrix covers {served} of the {arrays.n_zones} zones the network declares, so "
            f"the file's node indices and its trips do not describe the same network")
    if (report.unit_capacity > UNIT_CAPACITY_LIMIT * report.n_links
            and report.max_power > UNIT_CAPACITY_EXPONENT):
        report.warnings.append(
            f"{report.unit_capacity} of {report.n_links} links carry capacity exactly one and the "
            f"steepest exponent is {report.max_power:.2f}: this file folds capacity into the BPR "
            f"coefficient, so the cost is an unnormalised flow raised to that power rather than a load "
            f"ratio. This is the corpus exclusion criterion")
    if report.clamp_ratio < REACHABLE_RATIO:
        report.warnings.append(
            f"the steepest exponent is {report.max_power:.2f}, so the representability clamp binds at a "
            f"load ratio of {report.clamp_ratio:.3g}, below the {REACHABLE_RATIO:.0g} an ordinary link "
            f"reaches. Above the clamp the cost stops increasing and uniqueness is lost")
    return report


def report_lines(report: NetworkReport) -> list[str]:
    """The pre-flight block for one network, values beside the thresholds that judge them."""
    lines = [
        f"  {report.name}: N={report.n_nodes} Z={report.n_zones} E={report.n_links} "
        f"demand={report.demand:,.0f} over {report.demand_pairs} pairs",
        (f"    unreachable demand {report.unreachable_demand:,.1f} "
         f"({report.unreachable_share:.3%}, limit {UNREACHABLE_LIMIT:.3%}) over "
         f"{report.unreachable_pairs} pair(s)"
         if report.unreachable_demand is not None else
         f"    unreachable demand n.r. (not measured above {REACHABILITY_MAX_ZONES} zones)"),
        f"    degenerate: {report.zero_capacity} zero-capacity link(s) "
        f"(limit {ZERO_CAPACITY_LIMIT:.0%} of {report.n_links}), "
        f"{report.unit_capacity} unit-capacity (limit {UNIT_CAPACITY_LIMIT:.0%}), "
        f"{report.zero_free_flow} zero-cost, {report.self_loops} self-loop(s), "
        f"{report.isolated_nodes} isolated node(s)",
        f"    cost: steepest exponent {report.max_power:.2f}, clamp binds at load ratio "
        f"{report.clamp_ratio:.3g} (limit {REACHABLE_RATIO:.0g})",
    ]
    lines += [f"    WARNING: {message}" for message in report.warnings]
    return lines
