"""The corpus sweep: one equilibrium per network and dispersion, with the clauses that judge it.

Produces the corpus table and the headline count of solved combinations. A combination counts as solved
only when the frozen-support residual is below tolerance *and* the flow conserves demand: the residual is
measured against the support the solver held, so on its own it can sit at machine precision on a flow that
never delivered several per cent of its trips.

Every record carries the phase-one stopping criterion and the settings the run used, so a record that
stopped on the iteration limit can be told from one that converged.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.certificate import conservation_error, reload_on_own_support
from endogenous_sue.config import BUDGET_SENSITIVE, SOLVED_BELOW
from endogenous_sue.equilibrium import solve
from endogenous_sue.network import NetworkArrays
from endogenous_sue.preflight import (
    cost_is_increasing,
    inspect_network,
)
from reproduction.exhibits.records import Exhibit, allocated_cores


def main(argv=None):
    exhibit = Exhibit("corpus_sweep", __doc__.splitlines()[0])
    exhibit.parse(argv)
    exhibit.preflight()
    written = 0

    for name in exhibit.networks:
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        # The same inspection the pre-flight printed, so what a record says about unreachable demand and
        # what the reader was shown before the run cannot come from two implementations.
        inspected = inspect_network(name, net, od)
        demand = inspected.demand
        print(f"\n{name}  nodes={arrays.n_nodes} zones={arrays.n_zones} links={arrays.n_links} "
              f"demand={demand:,.0f}", flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            started = time.perf_counter()
            equilibrium = solve(net, od, mu, exhibit.settings,
                                checkpoint=exhibit.checkpoint_for(name, mu),
                                deadline=exhibit.cell_deadline(),
                                progress=exhibit.progress_for(name, mu))
            seconds = time.perf_counter() - started
            if not equilibrium.converged:
                # A network the method is not expected to settle on, which has reached the end of a
                # budget declared for it, has produced its result: it did not settle. Recording nothing
                # would leave it indistinguishable from a cell nobody ran, and the reader cannot tell
                # those apart. Any other network, or any stop that is not the declared budget, is still
                # unfinished and resumes from its checkpoint.
                budgeted = (name in BUDGET_SENSITIVE
                            and exhibit.args.cell_seconds is not None
                            and equilibrium.stopped_on == "time budget")
                if not budgeted:
                    exhibit.to_be_continued(name, mu, equilibrium.stopped_on)
                    continue
                worst, total, _ = conservation_error(net, od, equilibrium.x)
                support_changes = [value for value in equilibrium.phase1.support_changes
                                   if value is not None]
                exhibit.unresolved(
                    name, mu, "support set did not settle",
                    n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                    demand=demand, unreachable_demand=inspected.unreachable_demand,
                    unreachable_pairs=inspected.unreachable_pairs,
                    # Not measured rather than zero: phase two never ran, so there is no residual
                    # against a frozen support and no strict-reload defect to report.
                    residual=None, solved=False, polish_method="not reached",
                    iterations=equilibrium.phase1.iterations,
                    resumed_from=equilibrium.phase1.resumed_from,
                    final_gap=(equilibrium.phase1.gap_history[-1]
                               if equilibrium.phase1.gap_history else None),
                    conservation_max=worst, conservation_l1=total, conserves=None,
                    strict_reload_defect=None,
                    support_changes_final=(support_changes[-1] if support_changes else None),
                    support_measured=bool(support_changes),
                    budget_seconds=float(exhibit.args.cell_seconds),
                    seconds=seconds, phase_seconds=equilibrium.phase1.seconds,
                    cores=allocated_cores())
                exhibit.clear_checkpoint(name, mu)
                written += 1
                continue

            worst, total, _ = conservation_error(net, od, equilibrium.x)
            conserves = worst <= exhibit.settings.conservation_tolerance * max(demand, 1.0)
            # The model's own validity condition, recorded and never enforced. The equilibrium on a
            # frozen support is unique because the cost is strictly increasing; where the
            # representability clamp binds the derivative is exactly zero and that argument is gone.
            # The manuscript invokes the assumption, so every reported network should show whether it
            # actually held on the flow it reports rather than leaving the reader to assume it did.
            # A diagnostic and not a certificate clause: it changes no verdict.
            flat_by_convention, flat_despite_cost, smallest_derivative = cost_is_increasing(
                arrays, equilibrium.x)
            reloaded = reload_on_own_support(net, od, mu, equilibrium.x)
            scale = max(float(np.abs(equilibrium.x).max()), 1e-30)
            strict_defect = float(np.max(np.abs(reloaded - equilibrium.x)) / scale)

            support_changes = [value for value in equilibrium.phase1.support_changes
                               if value is not None]
            exhibit.emit(dict(
                exhibit="corpus_sweep", network=name, mu=mu,
                n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                demand=demand, unreachable_demand=inspected.unreachable_demand,
                unreachable_pairs=inspected.unreachable_pairs,
                residual=float(equilibrium.residual),
                solved=bool(equilibrium.residual < SOLVED_BELOW and conserves),
                converged=bool(equilibrium.converged), stopped_on=equilibrium.stopped_on,
                polish_method=equilibrium.polish_method,
                iterations=equilibrium.phase1.iterations,
                resumed_from=equilibrium.phase1.resumed_from,
                conservation_max=worst, conservation_l1=total, conserves=bool(conserves),
                flat_cost_connectors=flat_by_convention,
                flat_cost_despite_positive_fft=flat_despite_cost,
                smallest_cost_derivative=smallest_derivative,
                strict_reload_defect=strict_defect,
                support_changes_final=(support_changes[-1] if support_changes else None),
                support_measured=bool(support_changes),
                seconds=seconds, phase_seconds=equilibrium.phase1.seconds,
                cores=allocated_cores()))
            exhibit.clear_checkpoint(name, mu)
            written += 1
            print(f"  mu={mu:<5} {equilibrium.stopped_on:<16} iters={equilibrium.phase1.iterations:<6} "
                  f"residual={equilibrium.residual:.2e} conservation={worst:.2e} "
                  f"{'conserves' if conserves else 'SHORT'} {seconds:.1f}s", flush=True)

    print(f"\ntotal wall time {time.perf_counter() - exhibit.started:.1f}s "
          f"on {allocated_cores()} cores", flush=True)
    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
