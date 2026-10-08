"""What the congestion-adaptive support buys over the free-flow one.

Both models are solved on the same network at the same dispersion, differing only in whether the support
is rebuilt as the flow changes. Reported per cell: how many destination-link pairs the support moves, the
share of the support that is, and the relative difference between the two flows.

The two solves use the same iteration budget and the same tolerances, so the flow difference is a property
of the support rule and not of how hard each was solved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.config import DeadlineReached
from endogenous_sue.equilibrium import SupportRule, solve
from endogenous_sue.network import NetworkArrays
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import active_for_dest
from reproduction.exhibits.records import Exhibit


def support_of(topo, od, potential):
    """The whole family of per-destination supports, as one boolean array."""
    demanded = [d for d in range(topo.n_zones) if od[:, d].sum() > 0.0]
    return np.array([active_for_dest(topo, potential[:, d], d) for d in demanded])


def main(argv=None):
    exhibit = Exhibit("support_comparison", __doc__.splitlines()[0])
    exhibit.parse(argv)
    exhibit.preflight()
    written = 0

    for name in exhibit.networks:
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        free_flow_support = support_of(topo, od, potentials(topo, arrays.fft))
        print(f"\n{name}", flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            report = exhibit.progress_for(name, mu)
            try:
                arms = {}
                for arm, rule in (("adaptive", SupportRule.DAMPED), ("fixed", SupportRule.FIXED)):
                    report(f"{arm} support rule")
                    arms[arm] = solve(net, od, mu, exhibit.settings, rule=rule,
                                      checkpoint=exhibit.checkpoint_for(name, mu, arm),
                                      deadline=exhibit.deadline, progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, str(reached))
                continue
            adaptive, fixed = arms["adaptive"], arms["fixed"]
            if not (adaptive.converged and fixed.converged):
                exhibit.to_be_continued(
                    name, mu, f"adaptive {adaptive.stopped_on}, fixed {fixed.stopped_on}")
                continue
            adaptive_support = support_of(topo, od, adaptive.phase1.potential)

            moved = int(np.count_nonzero(adaptive_support != free_flow_support))
            in_support = int(np.count_nonzero(free_flow_support))
            difference = float(np.linalg.norm(adaptive.x - fixed.x)
                               / max(np.linalg.norm(fixed.x), 1e-30))

            exhibit.emit(dict(
                exhibit="support_comparison", network=name, mu=mu,
                n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                support_moved=moved, support_size=in_support,
                moved_share=moved / max(in_support, 1),
                flow_difference=difference,
                adaptive_residual=float(adaptive.residual), fixed_residual=float(fixed.residual),
                adaptive_converged=bool(adaptive.converged), fixed_converged=bool(fixed.converged),
                adaptive_stopped_on=adaptive.stopped_on, fixed_stopped_on=fixed.stopped_on))
            exhibit.clear_checkpoint(name, mu)
            written += 1
            print(f"  mu={mu:<5} moved={moved:<7} of {in_support:<7} "
                  f"({moved / max(in_support, 1):.3%})  flow difference={difference:.3e}", flush=True)

    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
