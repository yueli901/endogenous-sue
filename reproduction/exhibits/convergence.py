"""How the support settles, and what it costs.

Two exhibits from one pass. The first contrasts the support rules: rebuilding from the instantaneous cost
leaves the support flipping on the tie boundary indefinitely, while rebuilding from the averaged cost
freezes it in finite time. The second times the two phases on an otherwise idle machine.

Timing is refused above a load threshold. A time measured under load is not the quantity the table claims,
and a machine's load average lags, so the check is repeated after a pause and both samples must pass.
Every record carries the core count, because a per-core second and a wall-clock second are different
quantities and the table reports the second of these.

The a posteriori certificate is evaluated and timed on the same cells, because what the certificate costs
to *check* is a separate quantity from what an equilibrium costs to *reach*, and only the second of them
is the solve time reported beside it.

What is checked here is strictness, and only that. The flow tested is this exhibit's own timed solve on a
frozen support, and no mixture is supplied, so the verdict recorded is whether that flow is the loading
on the strict support of its own cost. A tied face cannot pass such a test and is not expected to: a cell
recorded as not strict here is not a failed relaxed equilibrium, and this exhibit says nothing either way
about whether a relaxed one exists for it. The certificate sweep answers that, on its own flows, with the
mixtures those flows came with -- mixtures which belong to different computed candidates and must never
be attached to these. The fields are named for what they hold: ``strictly_certified``,
``strict_certificate_kind`` and ``strict_certify_seconds``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.certificate import certify
from endogenous_sue.config import DeadlineReached
from endogenous_sue.equilibrium import SupportRule, polish, solve_phase1
from endogenous_sue.network import NetworkArrays
from reproduction.exhibits.records import Exhibit, allocated_cores


def load_average() -> float:
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return 0.0


def settled(threshold: float, samples: int = 2, pause: float = 60.0) -> bool:
    """Whether the machine is quiet, on consecutive samples.

    One sample is not enough: the load average is itself an average, so it lags, and a run starting the
    moment a heavy job ends sees a figure that describes the machine a minute ago.
    """
    for index in range(samples):
        if load_average() > threshold:
            return False
        if index + 1 < samples:
            time.sleep(pause)
    return load_average() <= threshold


def main(argv=None):
    exhibit = Exhibit("convergence", __doc__.splitlines()[0])
    exhibit.add_argument("--max-load", type=float, default=0.5,
                         help="refuse to time above this load average")
    exhibit.add_argument("--force", action="store_true",
                         help="time anyway, recording that the machine was loaded")
    exhibit.add_argument("--oscillation-iterations", type=int, default=300,
                         help="iterations of the support-rule contrast")
    exhibit.parse(argv)

    quiet = settled(exhibit.args.max_load)
    loaded = load_average()
    if not quiet and not exhibit.args.force:
        print(f"refusing to time: load average {loaded:.2f} exceeds {exhibit.args.max_load}. "
              f"The reported times claim an otherwise idle machine, and a time taken under load is not "
              f"that quantity. Re-run when idle, or pass --force and expect the records to say so.",
              flush=True)
        return 1

    exhibit.preflight()
    cores = allocated_cores()
    written = 0
    settings = exhibit.settings

    for name in exhibit.networks:
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        print(f"\n{name}  links={arrays.n_links}", flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            # Its own budget, so one cell that will not settle does not take the wall and
            # leave the remaining dispersions untried and then reported as missing.
            cell_deadline = exhibit.cell_deadline()
            report = exhibit.progress_for(name, mu)
            contrast = {}
            for rule in (SupportRule.INSTANTANEOUS, SupportRule.DAMPED):
                capped = type(settings)(**{**settings.as_record(),
                                           "max_phase1_iterations":
                                               exhibit.args.oscillation_iterations})
                phase1 = solve_phase1(net, od, mu, capped, rule=rule, exact_stop=False,
                                      progress=report)
                measured = [value for value in phase1.support_changes if value is not None]
                contrast[rule.value] = dict(
                    gap_history=[float(value) for value in phase1.gap_history],
                    support_changes=measured or None,
                    support_measured=bool(measured),
                    settled_at=(phase1.iterations if phase1.converged else None))

            # No checkpoint here, and the budget abandons the cell rather than resuming it: a time
            # measured across a stop and a restart is not the quantity this table reports.
            started = time.perf_counter()
            try:
                phase1 = solve_phase1(net, od, mu, settings, deadline=cell_deadline,
                                      progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, f"timing abandoned: {reached}")
                continue
            if not phase1.converged:
                exhibit.to_be_continued(name, mu, f"timing abandoned: {phase1.stopped_on}")
                continue
            phase1_seconds = time.perf_counter() - started
            started = time.perf_counter()
            # No checkpoint here, unlike every other exhibit: this one is timing the phases, and a phase
            # resumed from a previous run's state would report the seconds of the remainder as the cost
            # of the whole. The deadline still applies, and a cell that hits it has no timing to report.
            try:
                polished = polish(net, od, mu, phase1.x, phase1.potential, settings,
                                  deadline=cell_deadline, progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, f"timing abandoned: {reached}")
                continue
            if not polished.converged:
                exhibit.to_be_continued(name, mu, f"timing abandoned: phase 2 {polished.method}")
                continue
            phase2_seconds = time.perf_counter() - started
            # No mixture is supplied, and none should be. This exhibit times the *check*, and the
            # flow it checks is the one its own timed solve produced on a frozen support: a candidate
            # of its own, not the certificate sweep's, whose mixtures belong to different computed
            # flows and must not be attached to these. Without a mixture the certificate can only test
            # strictness, so what is recorded here is strict certification of the timed fixed-support
            # flow and nothing wider. A cell that fails it is not a failed relaxed equilibrium; it is a
            # flow that is not the loading on a single support, which for a tied face is expected.
            started = time.perf_counter()
            certificate = certify(net, od, mu, polished.x, settings=settings)
            strict_certify_seconds = time.perf_counter() - started

            exhibit.emit(dict(
                exhibit="convergence", network=name, mu=mu,
                n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                phase1_seconds=phase1_seconds, phase2_seconds=phase2_seconds,
                phase1_iterations=phase1.iterations, phase1_stopped_on=phase1.stopped_on,
                phase1_support_measured=bool([v for v in phase1.support_changes if v is not None]),
                residual=float(polished.residual), polish_method=polished.method,
                polish_converged=bool(polished.converged),
                strict_certify_seconds=strict_certify_seconds,
                strictly_certified=bool(certificate.certified),
                strict_certificate_kind=certificate.kind,
                phase_seconds=phase1.seconds,
                cores=cores, load_average=loaded, forced=bool(exhibit.args.force),
                contrast=contrast))
            written += 1
            print(f"  mu={mu:<5} phase 1 {phase1_seconds:8.2f}s ({phase1.iterations} iterations, "
                  f"{phase1.stopped_on})  phase 2 {phase2_seconds:6.2f}s  "
                  f"residual={polished.residual:.2e}",
                  flush=True)

    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
