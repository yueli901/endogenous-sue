"""How large the equilibrium set is, and when a computed flow is provably the only one.

Two deterministic tie-breaks are applied to the cost that *defines* the support, leaving the cost flow is
loaded at untouched, so each selects a different admissible support at a tie without moving the
equilibrium away from one. The distance between the two solutions is a lower bound on the diameter of the
set.

Where the tie set is empty the isolation radius applies, and a flow whose cost lies within it is the only
equilibrium nearby. The constant in that radius is combinatorial: it counts links on a cost-minimal path,
and the second equilibrium being compared against is by construction not computed, so the only admissible
constant is the a priori bound on path length. The measured longest cost-minimal path is recorded beside
it as the interesting quantity, and the radius it would give is recorded as not certified.

The comparison is made in one space. The radius is an absolute distance in cost, so the diameter is
measured the same way; the relative flow-space figure is recorded alongside it.

Two margins are recorded and they are not interchangeable. The isolation margin decides local uniqueness
and is zero wherever a tie exists. The path-excess margin is the diagnostic the tables report: the
smallest excess cost of a link excluded at a node the flow actually reaches, which minimises over a subset
of the excluded links and is therefore never smaller than the exclusion margin the theory is stated with.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.config import COST_MINIMAL_TOLERANCE, DeadlineReached
from endogenous_sue.equilibrium import polish, solve_phase1
from endogenous_sue.network import NetworkArrays
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import isolation_margin, path_excess_margin
from reproduction.exhibits.records import Exhibit


def tiebreak_pair(net, od, mu, settings, checkpoint=None, deadline=None, progress=None):
    """The two solutions selected by opposite deterministic tie-breaks.

    Each arm keeps its own checkpoint: they are two different solves of the same cell, and sharing one
    file would resume each from the other's state.
    """
    arrays = NetworkArrays.from_dict(net)
    solutions = []
    for arm, sign in (("plus", +1.0), ("minus", -1.0)):
        if progress is not None:
            progress(f"tie-break {arm}")
        state = None if checkpoint is None else checkpoint(arm)
        phase1 = solve_phase1(net, od, mu, settings, tiebreak=arrays.fft, tiebreak_sign=sign,
                              checkpoint=state, deadline=deadline, progress=progress)
        if not phase1.converged:
            raise DeadlineReached(f"tie-break {arm}: {phase1.stopped_on}")
        polished = polish(net, od, mu, phase1.x, phase1.potential, settings,
                          checkpoint=state, deadline=deadline, progress=progress)
        if not polished.converged:
            raise DeadlineReached(f"tie-break {arm}: phase 2 {polished.method}")
        solutions.append(polished.x)
    return solutions[0], solutions[1]


def longest_cost_minimal_path(arrays, od, cost, potential) -> int:
    """Most links on a cost-minimal path, over destinations that carry demand.

    A link is cost-minimal toward a destination when its reduced cost vanishes. Those links form a
    directed acyclic graph, because a positive cost makes the potential strictly decrease along one, so
    the longest path is a topological recursion rather than a search.
    """
    longest = 0
    for d in range(arrays.n_zones):
        if od[:, d].sum() <= 0.0:
            continue
        column = potential[:, d]
        finite = np.isfinite(column)
        usable = finite[arrays.tail] & finite[arrays.head]
        reduced = np.full(arrays.n_links, np.inf)
        reduced[usable] = (cost[usable] + column[arrays.head][usable] - column[arrays.tail][usable])
        tight = np.abs(reduced) <= COST_MINIMAL_TOLERANCE
        if not tight.any():
            continue
        hops = np.zeros(arrays.n_nodes, dtype=np.int64)
        order = np.argsort(np.where(finite, column, np.inf), kind="stable")
        rank = np.empty(arrays.n_nodes, dtype=np.int64)
        rank[order] = np.arange(arrays.n_nodes)
        links = np.flatnonzero(tight)
        for link in links[np.argsort(rank[arrays.tail[links]], kind="stable")]:
            hops[arrays.tail[link]] = max(hops[arrays.tail[link]], hops[arrays.head[link]] + 1)
        carrying = np.flatnonzero(od[:, d] > 0.0)
        if carrying.size:
            longest = max(longest, int(hops[carrying].max()))
    return longest


def main(argv=None):
    exhibit = Exhibit("solution_set", __doc__.splitlines()[0])
    exhibit.parse(argv)
    exhibit.preflight()
    settings = exhibit.settings
    written = 0

    for name in exhibit.networks:
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        print(f"\n{name}", flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            report = exhibit.progress_for(name, mu)
            try:
                plus, minus = tiebreak_pair(
                    net, od, mu, settings,
                    checkpoint=lambda arm, name=name, mu=mu: exhibit.checkpoint_for(name, mu, arm),
                    deadline=exhibit.deadline, progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, str(reached))
                continue
            cost_diameter = float(np.abs(arrays.cost(plus) - arrays.cost(minus)).max())
            flow_diameter = float(np.linalg.norm(plus - minus) / max(np.linalg.norm(plus), 1e-30))

            cost = arrays.cost(plus)
            potential = potentials(topo, cost)
            margin = isolation_margin(topo, potential)
            excess = path_excess_margin(topo, potential, od)
            hops = longest_cost_minimal_path(arrays, od, cost, potential)

            bound = arrays.n_nodes - 1
            radius = margin / (2.0 * bound) if np.isfinite(margin) else None
            radius_measured = (margin / (2.0 * hops)) if (np.isfinite(margin) and hops) else None

            exhibit.emit(dict(
                exhibit="solution_set", network=name, mu=mu,
                n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                isolation_margin=None if not np.isfinite(margin) else float(margin),
                path_excess_margin=None if not np.isfinite(excess) else float(excess),
                path_bound=bound, radius=radius,
                longest_cost_minimal_path=hops, radius_measured_path=radius_measured,
                diameter_cost=cost_diameter, diameter_flow_relative=flow_diameter,
                unique=(bool(radius > cost_diameter) if radius is not None else None)))
            exhibit.clear_checkpoint(name, mu)
            written += 1
            print(f"  mu={mu:<5} margin={margin:<12.4g} radius={radius} "
                  f"cost diameter={cost_diameter:.4g} flow diameter={flow_diameter:.4g}", flush=True)

    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
