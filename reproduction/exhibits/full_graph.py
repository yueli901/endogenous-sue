"""The full-graph comparator: where it is well posed, and how close it is where it is.

Three measurements per cell. The spectral radius of the exponential weight matrix at free flow and at the
returned flow, which decides whether the full-graph value function exists at all. The difference between
the two equilibria where both are defined, against the dispersion parameter, which is what the accuracy
bound predicts. And a recovery check: the eDSP model is solved at a known dispersion and the value that
best explains the resulting flow is recovered, which tests how identifiable the dispersion is from the
generated flows.

Where the radius reaches one the comparator has no value function, and the cell is recorded as ill-posed
rather than as a large difference.

The dispersion grid is the corpus one plus the extra points the accuracy fit needs, both declared in the
configuration. The recovery check runs on its own declared cells rather than behind a flag.
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
from endogenous_sue.config import (
    BUDGET_SENSITIVE,
    RECOVERY_DISPERSIONS,
    RECOVERY_NETWORKS,
    DeadlineReached,
)
from endogenous_sue.equilibrium import solve
from endogenous_sue.network import NetworkArrays
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import exclusion_margin
from reproduction.exhibits.comparator import solve_msa, spectral_radius, value_residuals
from reproduction.exhibits.records import Exhibit, allocated_cores


def recover_dispersion(net, od, mu_true, settings, low=0.5, high=2.0, tolerance=1e-5,
                       max_solves=40, deadline=None, progress=None):
    """Recover the dispersion parameter from the flow it produced, by golden-section search.

    The flow is generated at ``mu_true`` and the search looks for the value whose own equilibrium flow is
    closest to it. A grid would report only the resolution of the grid, so the interval is narrowed
    instead. The number of solves and the wall time are recorded because they are part of what the
    exhibit reports: whether the parameter is identifiable from flows, and at what cost.
    """
    started = time.perf_counter()
    solves = [1]

    def equilibrium_at(value):
        """One solve, refusing to hand back a flow the solver did not converge to."""
        result = solve(net, od, value, settings, deadline=deadline, progress=progress)
        if not result.converged:
            raise DeadlineReached(f"recovery at mu={value:.6f}: {result.stopped_on}")
        return result.x

    target = equilibrium_at(mu_true)

    def distance(candidate):
        solves[0] += 1
        if progress is not None:
            progress(f"recovery solve {solves[0]} at mu={candidate:.6f}")
        x = equilibrium_at(candidate)
        return float(np.linalg.norm(x - target) / max(np.linalg.norm(target), 1e-30))

    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    left, right = mu_true * low, mu_true * high
    inner_low, inner_high = right - ratio * (right - left), left + ratio * (right - left)
    value_low, value_high = distance(inner_low), distance(inner_high)
    while right - left > tolerance * mu_true and solves[0] < max_solves:
        if value_low < value_high:
            right, inner_high, value_high = inner_high, inner_low, value_low
            inner_low = right - ratio * (right - left)
            value_low = distance(inner_low)
        else:
            left, inner_low, value_low = inner_low, inner_high, value_high
            inner_high = left + ratio * (right - left)
            value_high = distance(inner_high)
    recovered = 0.5 * (left + right)
    return recovered, abs(recovered - mu_true), solves[0], time.perf_counter() - started


def main(argv=None):
    exhibit = Exhibit("full_graph", __doc__.splitlines()[0])
    exhibit.parse(argv)
    exhibit.preflight()
    written = 0

    for name in exhibit.networks:
        net, od = corpus.load(net_name := name)
        arrays = NetworkArrays.from_dict(net)
        print(f"\n{name}", flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            cell_deadline = exhibit.cell_deadline()
            free_flow_radius = spectral_radius(arrays.tail, arrays.head, arrays.fft, mu,
                                               arrays.n_nodes)
            report = exhibit.progress_for(name, mu)
            try:
                equilibrium = solve(net, od, mu, exhibit.settings,
                                    checkpoint=exhibit.checkpoint_for(name, mu),
                                    deadline=cell_deadline, progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, str(reached))
                continue
            if not equilibrium.converged:
                # A declared budget-sensitive network that reached the end of a declared budget has its
                # answer: the support did not settle. Recording nothing leaves it indistinguishable from
                # a cell nobody ran. The free-flow radius is measured at the free-flow cost and so is
                # available regardless; the equilibrium radius and everything downstream of it are not,
                # and are absent rather than zero.
                budgeted = (net_name in BUDGET_SENSITIVE
                            and exhibit.args.cell_seconds is not None
                            and equilibrium.stopped_on == "time budget")
                if not budgeted:
                    exhibit.to_be_continued(name, mu, equilibrium.stopped_on)
                    continue
                exhibit.unresolved(
                    net_name, mu, "support set did not settle",
                    n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                    radius_free_flow=free_flow_radius, radius_equilibrium=None,
                    well_posed=None, residual=None,
                    iterations=equilibrium.phase1.iterations,
                    final_gap=(equilibrium.phase1.gap_history[-1]
                               if equilibrium.phase1.gap_history else None),
                    margin=None, budget_seconds=float(exhibit.args.cell_seconds),
                    cores=allocated_cores())
                exhibit.clear_checkpoint(name, mu)
                written += 1
                continue
            equilibrium_radius = spectral_radius(arrays.tail, arrays.head,
                                                 arrays.cost(equilibrium.x), mu, arrays.n_nodes)
            well_posed = bool(np.isfinite(equilibrium_radius) and equilibrium_radius < 1.0)

            # The margin the accuracy bound is stated with, at this cell's own equilibrium cost. The
            # certificate sweep records it too, but only on the corpus grid; the accuracy comparison runs
            # dispersions that grid does not have, and the sentence about the bound is about those.
            margin = exclusion_margin(arrays.topology(),
                                      potentials(arrays.topology(), arrays.cost(equilibrium.x)), od)
            record = dict(exhibit="full_graph", network=net_name, mu=mu,
                          n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                          radius_free_flow=free_flow_radius, radius_equilibrium=equilibrium_radius,
                          well_posed=well_posed, residual=float(equilibrium.residual),
                          exclusion_margin=None if not np.isfinite(margin) else float(margin))

            if well_posed:
                x_full, status, seconds = solve_msa(net, od, mu)
                record.update(full_graph_status=status, full_graph_seconds=seconds)
                if status != "overflow":
                    difference = float(np.linalg.norm(x_full - equilibrium.x)
                                       / max(np.linalg.norm(equilibrium.x), 1e-30))
                    record["flow_difference"] = difference
                else:
                    record["flow_difference"] = None
            else:
                tail_residual = value_residuals(arrays.tail, arrays.head,
                                                arrays.cost(equilibrium.x), mu,
                                                arrays.n_nodes, arrays.n_zones)[-1]
                record.update(full_graph_status="ill-posed", flow_difference=None,
                              value_residual_tail=float(tail_residual))

            # Recovery runs on the cells declared for it, not on a command-line flag: a table whose rows
            # depend on whether someone remembered to pass an argument is a table with no definition.
            if (name, mu) in {(network, value) for network in RECOVERY_NETWORKS
                              for value in RECOVERY_DISPERSIONS} and well_posed:
                try:
                    recovered, error, solves, seconds = recover_dispersion(
                        net, od, mu, exhibit.settings, deadline=cell_deadline, progress=report)
                except DeadlineReached as reached:
                    exhibit.to_be_continued(name, mu, str(reached))
                    continue
                record.update(recovered_mu=recovered, recovery_error=error,
                              recovery_solves=solves, recovery_seconds=seconds)

            exhibit.emit(record)
            exhibit.clear_checkpoint(name, mu)
            written += 1
            print(f"  mu={mu:<5} radius(free flow)={free_flow_radius:.4f} "
                  f"radius(equilibrium)={equilibrium_radius:.4f} "
                  f"{'well posed' if well_posed else 'ILL POSED'} "
                  f"difference={record.get('flow_difference')}", flush=True)

    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
