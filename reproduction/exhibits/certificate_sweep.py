"""The certificate sweep: a certified equilibrium per cell, or a stated refusal.

The inclusion scheme runs first. Where it settles on one support the flow is tested for strictness; where
several supports survive in its trailing window the ties are unresolved, the equilibrium is a mixture, and
the tied solver is asked for one and for the supports carrying weight.

Every cell ends with the a posteriori certificate evaluated on the flow alone, at the cost that flow
induces, taking nothing from the solver that produced it. The verdict is certified or not certified, and
a cell that is not certified records why.

The returned flows are deposited beside the records so their bytes can be checked against the digests in
the result rows.

One declared cell also has the local dimension of its equilibrium manifold measured, which costs one extra
blend per tied group and is affordable only there.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from dataclasses import replace

from endogenous_sue import corpus
from endogenous_sue.certificate import certify
from endogenous_sue.config import (
    DIRECT_FACE_CELLS,
    DIRECT_FACE_ENVELOPE_WINDOWS,
    FACE_CELL,
    TIE_TOLERANCE,
    DeadlineReached,
)
from endogenous_sue.equilibrium import polish, solve_inclusion
from endogenous_sue.loading import absorbing
from endogenous_sue.network import (
    NetworkArrays,
    Topology,
    unpack,
)
from endogenous_sue.preflight import cost_is_increasing
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import active_for_dest, exclusion_margin
from endogenous_sue.tied import contested_pairs, solve_face_admissible
from endogenous_sue.tied.faces import solve_face_ladder
from reproduction.exhibits.records import RESULTS, Exhibit, allocated_cores, peak_rss_mb

FLOWS = RESULTS / "flows"

#: Mass below which a destination's vertices are taken to carry the whole of its flow.
MIXTURE_WEIGHT_EPSILON = 1e-12


def tie_inventory(topo, od, potential, cost, tie_tolerance):
    """Tied and excluded pairs at one cost, with the margins the theory is stated with."""
    tied_pairs, tied_links = 0, set()
    excluded, smallest_tie_cost = 0, np.inf
    tail, head, n_zones = topo.tail, topo.head, topo.n_zones
    for d in range(n_zones):
        if od[:, d].sum() <= 0.0:
            continue
        column = potential[:, d]
        finite = np.isfinite(column[tail]) & np.isfinite(column[head])
        gap = np.zeros(topo.n_links)
        np.subtract(column[tail], column[head], out=gap, where=finite)
        eligible = finite & ~((head < n_zones) & (head != d)) & (tail != d) & (head != d)
        if topo.zero_dep_conn:
            eligible &= ~((tail < n_zones) & (tail != d))
        tied = eligible & (np.abs(gap) <= tie_tolerance)
        away = eligible & (gap < -tie_tolerance)
        tied_pairs += int(tied.sum())
        tied_links.update(np.flatnonzero(tied).tolist())
        excluded += int(away.sum())
        if tied.any():
            smallest_tie_cost = min(smallest_tie_cost, float(cost[tied].min()))
    # The margin itself comes from the package, not from a second implementation here: it is the
    # quantity the approximation bound is stated with and it is reported by two exhibits.
    margin = exclusion_margin(topo, potential, od, tie_tolerance)
    return dict(n_tied_pairs=tied_pairs, n_tied_links=len(tied_links), n_excluded=excluded,
                exclusion_margin=None if not np.isfinite(margin) else float(margin),
                smallest_tie_cost=None if not np.isfinite(smallest_tie_cost) else smallest_tie_cost)


def _how_it_stopped(settled: bool, envelope_settled: bool) -> str:
    """How the inclusion scheme stopped, for the line a reader scans.

    A cell that stopped on its envelope has not had its support settle, and saying only that would read
    as a failure of the run rather than as the finding it is: the support identity kept moving and the
    object the face is built from did not.
    """
    if settled:
        return ""
    if envelope_settled:
        return "(envelope settled, support identity did not) "
    return "(support set did not settle) "


def orientation_changed(topo, od, arrays, inclusion, x):
    """Destination-link pairs where the support the flow was polished on is not the support it induces.

    The inclusion scheme stops when no new support appears across its window, which is a statement about
    the iterates it saw. It is not a guarantee that polishing on the support it settled leaves that
    support valid at the polished flow, and `polish` moves the flow away from the inclusion iterate. On
    SiouxFalls at mu=1 and Chicago-Sketch at mu=10 one near-balanced two-way edge -- 9 to 15 and 554 to
    624 -- is oriented one way by the inclusion potential and the other by the polished flow's own cost.
    Those cells record a frozen-support residual of 5.21e-16 and 4.25e-12 and fail the independent strict
    reload at 3.55e-10 and 1.96e-10, because the destinations whose support depends on that orientation
    carry about 1e-06 of flow across an edge carrying 1e+04.

    Returns the contested pairs and the base each is stated against, which is the intersection of the two
    supports: what both agree on is the base, and what they disagree on is the face.
    """
    potential = potentials(topo, arrays.cost(np.maximum(x, 0.0)))
    contested, base = [], {}
    for d in range(arrays.n_zones):
        if od[:, d].sum() <= 0.0:
            continue
        frozen = active_for_dest(topo, inclusion.potential[:, d], d)
        induced = active_for_dest(topo, potential[:, d], d)
        base[d] = frozen & induced
        for link in np.flatnonzero(frozen ^ induced):
            contested.append((d, int(link)))
    return contested, base


def direct_face(net, od, mu, x0, contested, base_masks, settings,
                use_complete=False, progress=None, deadline=None, checkpoint=None):
    """Solve one declared cell's face on the ladder, bypassing the accelerators and the pair walk.

    Returns the flow and a record shaped like the tied solver's, so everything downstream -- the
    certificate, the mixture reconstruction, the emitted row -- is the same code on either route. The
    route is named in the record rather than inferred, because a reader comparing two certificate rows
    has to be able to see that they were not reached the same way.

    ``use_complete`` is false by default, and that is a restriction rather than a detail. The last rung
    searches the whole flow space and carries the guarantee; the nested family is a hypothesis, and a face
    whose ties form an unavoidable ring is not resolvable within it -- the five-node witness is exactly
    such a face and reports the ring rather than a point. The cells declared for this route are ones where
    the complete rung is not affordable, so they are answered under the nested hypothesis or not at all.

    ``added`` is computed against the raw base mask and not the absorbing one the ladder searches over,
    because the caller rebuilds each support as ``base_masks[d]`` plus those links. Taken against the
    absorbing base the two would disagree wherever absorption added a link, and the mixture certified
    would not be the mixture the ladder solved.
    """
    tail, head, fft, cap, b, power, n_nodes, n_zones = unpack(net)
    arrays = (fft, cap, b, power, n_nodes, n_zones)
    topo = Topology.build(tail, head, fft, n_nodes, n_zones, net.get("first_thru"))
    demanded = [d for d in range(n_zones) if od[:, d].sum() > 0.0]
    base = {d: absorbing(topo, base_masks[d], d) for d in demanded}
    tied = [(int(d), int(link)) for d, link in contested]

    x, _, report = solve_face_ladder(
        topo, arrays, od, mu, demanded, base, tied, x0, settings=settings,
        tie_tolerance=TIE_TOLERANCE, use_complete=use_complete, progress=progress,
        deadline=deadline, checkpoint=_face_checkpoint(checkpoint))

    record = dict(report)
    record["phase"] = "ladder"
    record["route"] = "face ladder, entered directly"
    mixture = report.get("mixture") or []
    record["vertex_mixture"] = [
        [int(d), np.flatnonzero(np.asarray(mask, dtype=bool) & ~base_masks[int(d)]).tolist(),
         float(value)]
        for d, mask, value in mixture]
    record.pop("mixture", None)
    record.pop("gaps", None)
    return x, record


def _sibling_checkpoint(checkpoint):
    """Where the post-polish face keeps its state, beside the cell's own rather than in it."""
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}_postpolish{path.suffix}")


def _face_checkpoint(checkpoint):
    """Where the ladder's own state lives, beside the cell's.

    The rung and each of its refinements name files beneath this one. Sharing the cell's own path would
    have the inclusion scheme and the search overwrite each other.
    """
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}_face{path.suffix}")


def main(argv=None, exhibit_name: str = "certificate_sweep"):
    """``exhibit_name`` selects the declaration and the sink and nothing else.

    The held-out run is this function, unchanged, over cells declared elsewhere. A separate
    implementation would measure a separate implementation, which is not what a held-out set is for.
    """
    exhibit = Exhibit(exhibit_name, __doc__.splitlines()[0])
    exhibit.add_argument("--keep-flows", action=argparse.BooleanOptionalAction, default=True,
                         help="deposit the certified flow arrays (default: true)")
    exhibit.parse(argv)
    exhibit.preflight()
    FLOWS.mkdir(parents=True, exist_ok=True)
    settings = exhibit.settings
    written = 0

    for name in exhibit.networks:
        try:
            net, od = corpus.load(name)
        except (ValueError, KeyError, OSError) as exc:
            # A network the loader refuses is a result of this sweep, not an absence from it. Recorded
            # once per declared cell, because a cell simply missing from the sink cannot be told apart
            # from one nobody ran, and the held-out set exists precisely to report what the method does
            # on networks it was not developed against. Munich is refused this way: its network and
            # trips files declare different zone counts.
            print(f"\n{name}  REFUSED: {exc}", flush=True)
            for mu in exhibit.dispersions_for(name):
                exhibit.unresolved(name, mu, f"the network could not be read: {exc}")
            continue
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        print(f"\n{name}  nodes={arrays.n_nodes} zones={arrays.n_zones} links={arrays.n_links}",
              flush=True)

        for mu in exhibit.dispersions_for(name):
            if exhibit.out_of_time():
                exhibit.to_be_continued(name, mu)
                continue
            started = time.perf_counter()
            report = exhibit.progress_for(name, mu)
            checkpoint = exhibit.checkpoint_for(name, mu)
            report("inclusion scheme starting")
            # A declared cell stops on its envelope as well as on support recurrence. The face solver
            # consumes the window's intersection and union and never a support identity, so on a cell
            # whose supports keep arriving the recurrence test waits for something the face does not
            # need: Chicago-Sketch at mu=1 has produced 441,722 of them and its envelope did not move
            # over eight thousand iterations. The rule is empirical and is recorded as what was observed.
            cell_settings = (replace(settings, envelope_stable_windows=DIRECT_FACE_ENVELOPE_WINDOWS)
                             if (name, mu) in DIRECT_FACE_CELLS else settings)
            try:
                inclusion = solve_inclusion(net, od, mu, cell_settings, checkpoint=checkpoint,
                                            deadline=exhibit.deadline, progress=report)
            except DeadlineReached as reached:
                exhibit.to_be_continued(name, mu, f"inclusion: {reached}")
                continue
            if not inclusion.converged and inclusion.stopped_on == "time budget":
                # What the cell reached, not merely that it stopped. An evaluation run has to be able to
                # say how far a non-settling cell got, and on the large held-out networks that is the
                # whole result: Berlin-Center reached 29,000 iterations with its window permanently full
                # and its move still falling, which is a different statement from having made no
                # progress. Under --terminal these fields land in an unresolved row; otherwise the cell
                # stays absent and resumes.
                exhibit.to_be_continued(
                    name, mu, "inclusion: time budget",
                    iterations=inclusion.iterations,
                    supports_seen=inclusion.n_supports_seen,
                    supports_in_window=len(inclusion.supports),
                    envelope_windows_held=inclusion.envelope_windows_held,
                    envelope_stable_from=inclusion.envelope_stable_from,
                    seconds=time.perf_counter() - started,
                    peak_rss_mb=peak_rss_mb(),
                    n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links)
                continue
            report(f"inclusion done in {inclusion.iterations} iterations, "
                   f"{inclusion.n_supports_seen} distinct supports seen, "
                   f"{len(inclusion.supports)} in the window")
            # Only where that rule actually fired. The watcher accumulates whenever it is enabled, so
            # an envelope is present after the first complete window whatever eventually stopped the
            # scheme; building the face from it on a cell that stopped on recurrence or on the wall would
            # hand the solver a window the scheme never accepted.
            if inclusion.stopped_on == "envelope settled" and inclusion.envelope_base is not None:
                # What the scheme stopped on, rather than a trailing window that has moved since.
                n_links = arrays.n_links
                base_masks = {d: np.unpackbits(bits)[:n_links].astype(bool)
                              for d, bits in inclusion.envelope_base.items()}
                unions = {d: np.unpackbits(bits)[:n_links].astype(bool)
                          for d, bits in inclusion.envelope_union.items()}
                contested = [(int(d), int(link)) for d in sorted(base_masks)
                             for link in np.flatnonzero(unions[d] & ~base_masks[d])]
                report(f"envelope held {inclusion.envelope_windows_held} window(s) from iteration "
                       f"{inclusion.envelope_stable_from}; {len(contested)} contested pair(s)")
            else:
                contested, base_masks = contested_pairs(inclusion.masks())

            if not contested:
                report("no contested links; polishing on the settled support")
                polished = polish(net, od, mu, inclusion.x, inclusion.potential,
                                  settings, checkpoint=checkpoint,
                                  deadline=exhibit.deadline, progress=report)
                x, residual, method = polished.x, polished.residual, polished.method
                if not polished.converged:
                    exhibit.to_be_continued(name, mu, f"phase 2 above tolerance ({method})")
                    continue
                route, tied_record = "uncontested", {}
                # The support survived the handoff, or it did not. A changed orientation is a reason to
                # attempt face resolution and is not a verdict by itself: what decides the row is the
                # independent mixture certificate below, exactly as on any other route.
                moved, moved_base = orientation_changed(topo, od, arrays, inclusion, x)
                if moved:
                    report(f"the polished flow induces a different support on {len(moved)} pair(s); "
                           f"resolving them as a face")
                    try:
                        x, _, tied_record = solve_face_admissible(
                            net, od, mu, polished.x, moved, moved_base, settings,
                            tie_tolerance=TIE_TOLERANCE, progress=report,
                            deadline=exhibit.deadline,
                            checkpoint=_sibling_checkpoint(checkpoint))
                    except DeadlineReached as reached:
                        exhibit.to_be_continued(name, mu, f"post-polish face: {reached}")
                        continue
                    except (ValueError, RuntimeError) as exc:
                        report(f"post-polish face did not solve: {type(exc).__name__}: {exc}")
                        x, tied_record = polished.x, {}
                    else:
                        contested, base_masks = moved, moved_base
                        residual = float(tied_record.get("flow_residual", residual))
                        method = tied_record.get("phase", method)
                    route = "post-polish face fallback"
                    tied_record = dict(tied_record, post_polish_fallback=True,
                                       post_polish_pairs=len(moved))
            elif (name, mu) in DIRECT_FACE_CELLS:
                report(f"{len(contested)} contested pairs; entering the ladder directly")
                try:
                    x, tied_record = direct_face(net, od, mu, inclusion.x, contested, base_masks,
                                                 settings, progress=report,
                                                 deadline=exhibit.deadline, checkpoint=checkpoint)
                except DeadlineReached as reached:
                    exhibit.to_be_continued(name, mu, f"tied face: {reached}")
                    continue
                if not tied_record.get("certified_face"):
                    exhibit.to_be_continued(name, mu, f"face ladder: {tied_record.get('status')}")
                    continue
                residual = float(tied_record["flow_residual"])
                method = tied_record["phase"]
                route = tied_record["route"]
            else:
                report(f"{len(contested)} contested pairs; solving the tied face")
                try:
                    x, _, tied_record = solve_face_admissible(
                        net, od, mu, inclusion.x, contested, base_masks, settings,
                        tie_tolerance=TIE_TOLERANCE, progress=report,
                        deadline=exhibit.deadline, checkpoint=checkpoint,
                        measure_dimension=((name, mu) == FACE_CELL))
                except DeadlineReached as reached:
                    exhibit.to_be_continued(name, mu, f"tied face: {reached}")
                    continue

                # The walk retires contested pairs to a bound, and a pair pinned to a bound is out of
                # the face before the ladder searches it. Where the correct weight is strictly between
                # the bounds, no amount of reopening recovers it: reopening chooses a side, and on
                # Anaheim at mu=0.5 the gap on the offending pairs simply changed sign each time the
                # side changed, -4.886e-04 carried against +7.417e-03 excluded. Entering the ladder
                # directly hands it every contested pair and never pins one, and the same cell then
                # certifies relaxed at a mixture residual of 3.24e-14 over 141 supports.
                #
                # Escalated on a condition the run measures rather than on a list of cells: the face
                # was solved, and the flow it returned still carries pairs its own cost refuses.
                if tied_record.get("inadmissible_pairs"):
                    refused = tied_record["inadmissible_pairs"]
                    report(f"the solved face still carries {len(refused)} pair(s) its own cost "
                           f"refuses; re-entering the ladder directly, without the pair walk")
                    try:
                        direct_x, direct_record = direct_face(
                            net, od, mu, inclusion.x, contested, base_masks, settings,
                            progress=report, deadline=exhibit.deadline,
                            checkpoint=_sibling_checkpoint(checkpoint))
                    except DeadlineReached as reached:
                        exhibit.to_be_continued(name, mu, f"direct face: {reached}")
                        continue
                    except (ValueError, RuntimeError) as exc:
                        report(f"the direct face did not solve: {type(exc).__name__}: {exc}")
                    else:
                        if direct_record.get("certified_face"):
                            x, tied_record = direct_x, dict(
                                direct_record, escalated_from_walk=True,
                                walk_refused_pairs=refused)

                residual = float(tied_record["flow_residual"])
                method = tied_record["phase"]
                route = tied_record.get("route", tied_record["phase"])

            cost = arrays.cost(np.maximum(x, 0.0))
            potential = potentials(topo, cost)
            # Recorded, never enforced. See the note in reproduction/exhibits/corpus_sweep.py.
            flat_by_convention, flat_despite_cost, smallest_derivative = cost_is_increasing(arrays, x)
            if contested and tied_record.get("vertex_mixture"):
                mixture = []
                carried: dict = {}
                # The base each support is stated against comes from the solve, not from here. The
                # accelerating routes search against a copy of the base with links pinned in and out and
                # pass that copy down, so rebuilding on the inclusion scheme's base drops the links
                # pinned in and restores the ones pinned out. Every cell solved through the pattern route
                # failed its independent certificate that way, at a mixture residual of 1e-2 on a flow
                # whose own residual was 1e-15.
                listed = tied_record.get("support_base") or {}
                working = {}
                for d in base_masks:
                    links = listed.get(str(d), listed.get(d))
                    if links is None:
                        working[d] = base_masks[d].copy()
                        continue
                    mask = np.zeros(arrays.n_links, dtype=bool)
                    if len(links):
                        mask[np.asarray(links, dtype=int)] = True
                    working[d] = mask
                for d, added, weight in tied_record["vertex_mixture"]:
                    mask = working[int(d)].copy()
                    for link in added:
                        mask[int(link)] = True
                    mixture.append((int(d), mask, float(weight)))
                    carried[int(d)] = carried.get(int(d), 0.0) + float(weight)
                # The mass a destination's listed vertices do not carry sits on its base support, which
                # every admissible resolution contains. The accelerating routes return only the vertices
                # that add links and hold that remainder separately, so rebuilding from the listed
                # vertices alone loads less flow than the solve produced: ten of the fourteen cells this
                # sweep called uncertified were that, not a failed certificate. SiouxFalls at mu=0.5 read
                # a mixture residual of 7.965e-01 and reads 6.352e-16 with the remainder restored.
                # Taken as a deficit rather than from the recorded base weight because the ladder route
                # lists its base vertices among the others, where a recorded weight would double-count.
                # Over every destination the solve recorded a base mass for, not only those its
                # mixture names. A destination the route pinned but did not split carries its whole flow
                # on the base the solve loaded and appears in no vertex entry at all; taking the deficit
                # only over the named ones leaves it to the fall-through below, which rebuilds it on the
                # strict support of the final cost and discards the pinning. Four of Berlin-Tiergarten
                # mu=1's eight contested destinations were exactly that, which is why recording the
                # pinned base correctly changed nothing until this loop was widened.
                #
                # The deficit rather than the recorded mass, because the ladder lists its base vertices
                # among the others: there the deficit is zero and a recorded mass would double-count.
                for d in sorted(int(key) for key in (tied_record.get("base_weight") or {})):
                    if d not in working:
                        continue
                    deficit = 1.0 - carried.get(d, 0.0)
                    if deficit > MIXTURE_WEIGHT_EPSILON:
                        mixture.append((d, working[d].copy(), deficit))
                for d in range(arrays.n_zones):
                    if od[:, d].sum() > 0.0 and d not in {entry[0] for entry in mixture}:
                        mixture.append((d, active_for_dest(topo, potential[:, d], d), 1.0))
            else:
                mixture = None

            certificate = certify(net, od, mu, x, mixture, settings)
            # Why the scheme stopped, not merely that it did. An envelope stop also sets ``converged``,
            # so reading settledness off that flag would record a cell whose support recurrence was never
            # observed as one whose support settled. They are different claims, and the envelope rule
            # exists precisely because a cell can satisfy the second without ever satisfying the first.
            settled = inclusion.stopped_on == "support set settled"
            envelope_settled = inclusion.stopped_on == "envelope settled"
            inventory = tie_inventory(topo, od, potential, cost, TIE_TOLERANCE)

            deposited = None
            flow_digest = None
            flow_bytes = None
            if exhibit.args.keep_flows:
                deposited = FLOWS / f"{name}_mu{mu:g}.npz"
                np.savez_compressed(deposited, x=x)
                blob = deposited.read_bytes()
                flow_digest = hashlib.sha256(blob).hexdigest()[:16]
                flow_bytes = len(blob)

            exhibit.emit(dict(
                exhibit="certificate_sweep", network=name, mu=mu,
                n_nodes=arrays.n_nodes, n_zones=arrays.n_zones, n_links=arrays.n_links,
                n_contested=len(contested), route=route, method=method,
                inclusion_iterations=inclusion.iterations,
                inclusion_supports=len(inclusion.supports),
                inclusion_supports_seen=inclusion.n_supports_seen,
                inclusion_stopped_on=inclusion.stopped_on,
                inclusion_resumed_from=inclusion.resumed_from,
                flow_residual=residual,
                certified=bool(certificate.certified), kind=certificate.kind,
                inclusion_settled=settled,
                inclusion_envelope_settled=envelope_settled,
                envelope_windows_held=inclusion.envelope_windows_held,
                envelope_stable_from=inclusion.envelope_stable_from,
                envelope_last_change=inclusion.envelope_last_change,
                budget_limited=bool(tied_record.get("budget_limited")),
                flat_cost_connectors=flat_by_convention,
                flat_cost_despite_positive_fft=flat_despite_cost,
                smallest_cost_derivative=smallest_derivative,
                face_reopen_rounds=tied_record.get("reopen_rounds"),
                face_reopened_pairs=tied_record.get("reopened_pairs"),
                escalated_from_walk=bool(tied_record.get("escalated_from_walk")),
                reason=certificate.reason, violations=certificate.violations,
                worst_gap=certificate.worst_gap,
                mixture_residual=certificate.mixture_residual,
                conservation_max=certificate.conservation_max,
                conservation_relative=certificate.conservation_relative,
                tied=tied_record, flow_file=(deposited.name if deposited else None),
                flow_bytes=flow_bytes, flow_sha256=flow_digest,
                seconds=time.perf_counter() - started, cores=allocated_cores(), **inventory))
            exhibit.clear_checkpoint(name, mu)
            written += 1
            print(f"  mu={mu:<5} contested={len(contested):<6} {certificate.kind:<12} "
                  f"certified={bool(certificate.certified)} "
                  f"{_how_it_stopped(settled, envelope_settled)}residual={residual:.2e} "
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            if certificate.reason:
                print(f"        {certificate.reason}", flush=True)

    print(f"\ntotal wall time {time.perf_counter() - exhibit.started:.1f}s "
          f"on {allocated_cores()} cores", flush=True)
    return exhibit.done(written)


if __name__ == "__main__":
    raise SystemExit(main())
