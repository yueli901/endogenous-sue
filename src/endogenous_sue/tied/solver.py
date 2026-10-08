"""From a contested set to a certified relaxed equilibrium, or to a plain refusal.

The weights a face solve returns pin the flow but not themselves: duplicated tie rows leave the system
rank-deficient, so what comes back is an affine representation of the flow over the vertex supports. Where
that representation is not already convex, the convex one is recovered afterwards by a fit over the subset
family at the final cost, which is cheap because the flow is already exact.

A result is certified when every surviving tie residual is below tolerance, a convex per-destination
mixture over admissible supports reproduces the flow at the final cost, every support carrying weight is
admissible there, and the blended fixed point is solved. The verdict is certified or not certified, and
nothing between the two is reported.

The search proceeds from cheap and specific to expensive and general. A single contested link resolved by
bracketing is the proven route and is tried first. Small tie patterns are enumerated next. The joint
system comes after those, and whatever it returns is measured against the guaranteed ladder, which is
adopted whenever it does better by its own standard.
"""
from __future__ import annotations

import time
from hashlib import blake2b
from itertools import combinations, product
from pathlib import Path

import numpy as np

from ..certificate import admissibility_violations, conservation_error, violating_pairs
from ..config import (
    ACTIVE_TOLERANCE,
    DEFAULTS,
    FACE_REOPEN_ROUNDS,
    PATTERN_MAX_BASE_FLIPS,
    PATTERN_MAX_GROUPS,
    PATTERN_MAX_LINKS,
    PATTERN_MAX_ORIENTATION_EDGES,
    PATTERN_MAX_SOLVES,
    SIMPLEX_SLACK,
    TIE_TOLERANCE,
    DeadlineReached,
    Settings,
)
from ..costs import bpr_cost
from ..loading import absorbing, load_destination
from ..network import Topology, unpack
from ..shortestpath import potentials
from ..subnetwork import kahn_order
from .contested import orientation_family
from .faces import convex_certificate, solve_by_bisection, solve_face_ladder, solve_square_system

__all__ = ["solve_tied", "solve_face_admissible"]



def _sibling(checkpoint, suffix):
    """A checkpoint beside another, named for the stage it belongs to.

    One cell can have several stages in flight -- the walk, and the ladder rung inside it -- and they must
    not overwrite each other. Built as a proper sibling path so the files are named
    ``<cell>_<stage>.npz`` and the sweep's own cleanup glob still finds them.
    """
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.npz'}")


def _save_walk(checkpoint, x, tied, weights, base, outer) -> None:
    """Write the active-set walk's own state, atomically."""
    if checkpoint is None:
        return
    from ..equilibrium import save_checkpoint

    pinned = sorted(weights.items())
    save_checkpoint(
        checkpoint, phase=4, x=x, outer=np.int64(outer),
        tied=np.array([[d, link] for d, link in tied], dtype=np.int64).reshape(-1, 2),
        pinned=np.array([[d, link] for (d, link), _ in pinned], dtype=np.int64).reshape(-1, 2),
        pinned_value=np.array([value for _, value in pinned], dtype=float),
        base_dest=np.array(sorted(base), dtype=np.int64),
        base_mask=np.array([base[d] for d in sorted(base)], dtype=bool))


def walk_counters(contested, tied, weights, rounds) -> dict:
    """What the active-set walk did, in the terms Proposition S7's bound is stated in.

    Each quantity is defined here and nowhere else, because the bound is a count of events and a count is
    only comparable to a bound when both sides mean the same thing.

    ``live_initial``   contested pairs the walk started from, the size of the face it was asked about.
    ``pivots``         times a pair left the unit interval and was pinned to a bound.
    ``crossings``      boundary contacts: every pinning, plus every un-pinning of a pair later found on
                       the wrong side of its own gap at the solved cost. This is the event the bound
                       counts, and it is bounded by the destination-link pairs because neither record is
                       ever removed.
    ``working_set``    pairs still carrying a free weight when the walk ended.
    ``interior``       of those, the ones whose weight is strictly inside the unit interval, which is the
                       dimension of the mixture actually being reported.
    """
    pivots = sum(int(entry.get("retired", 0)) for entry in rounds)
    un_retired = sum(int(entry.get("n_un_retired", 0)) for entry in rounds)
    interior = sum(1 for (d, link) in tied
                   if ACTIVE_TOLERANCE < weights.get((d, link), 0.0) < 1.0 - ACTIVE_TOLERANCE)
    return dict(live_initial=len(contested), pivots=pivots, un_retirements=un_retired,
                crossings=pivots + un_retired, working_set=len(tied), interior=interior,
                rounds=len(rounds))


def _load_walk(checkpoint):
    """Read the walk state back, or ``None`` where there is none for this phase."""
    if checkpoint is None:
        return None
    from ..equilibrium import load_checkpoint

    stored = load_checkpoint(checkpoint)
    if stored is None or int(stored.get("phase", 1)) != 4:
        return None
    tied = [(int(d), int(link)) for d, link in stored["tied"]]
    weights = {(int(d), int(link)): float(value)
               for (d, link), value in zip(stored["pinned"], stored["pinned_value"])}
    base = {int(d): mask.copy() for d, mask in zip(stored["base_dest"], stored["base_mask"])}
    return stored["x"], tied, weights, base, int(stored["outer"])


class _Budget:
    """How much the pattern search may spend, counted in solves and never in seconds.

    The search decides which branch produces the certificate, so a wall-clock bound on it would make the
    certificate depend on how fast the machine is. The reproduction guarantee is stated over the network,
    the demand, the dispersion and the settings, and a stopwatch is none of those. ``deadline`` is the
    run's own wall-clock budget, which is a different thing: it stops the run, and a run it stops records
    that it was stopped rather than an answer.
    """

    def __init__(self, max_solves=PATTERN_MAX_SOLVES, deadline=None):
        self.solves = 0
        self.max_solves = max_solves
        self.deadline = deadline

    def available(self) -> bool:
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise DeadlineReached(f"pattern search, {self.solves} solve(s) tried")
        return self.solves < self.max_solves

    def spend(self):
        self.solves += 1


def _bracket_single_links(net, od, mu, x0, groups, base_masks, tie_tolerance, settings, solve,
                          deadline=None, progress=None):
    """Try each contested link alone, resolved by bracketing, accepting the first that certifies.

    The contested set taken from the late supports overestimates the true tie pattern: on the five-node
    witness six links flip during the transient while the equilibrium ties exactly one. Bracketing is the
    proven route, needing only continuity and a sign change, so each candidate link is tried alone with
    all of its destinations tied together and everything else pinned to the base support.
    """
    topo = Topology.build(*unpack(net)[:3], *unpack(net)[6:], net.get("first_thru"))
    candidates = sorted(groups.items(), key=lambda item: -len(item[1]))[:4]
    for index, (link, destinations) in enumerate(candidates, start=1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached(f"bracketing, {index - 1} of {len(candidates)} link(s) tried")
        if progress is not None:
            progress(f"bracketing link {link} ({index} of {len(candidates)}), "
                     f"{len(destinations)} destination(s)")
        pinned = {d: absorbing(topo, base_masks[d], d) for d in base_masks}
        flow, weight, record = solve_by_bisection(net, od, mu, x0, destinations, link, pinned,
                                                  tie_tolerance=tie_tolerance, deadline=deadline)
        if flow is None:
            continue
        pairs = [(d, link) for d in destinations]
        # The budget travels into the sub-solve. It is the run's, not this search's, so a sub-solve
        # entered a second before it expires must not run to its own limit. No checkpoint: the state
        # that matters is the outer walk's, and this call is a probe.
        x, weights, report = solve(net, od, mu, flow, pairs, base_masks, settings=settings,
                                   tie_tolerance=tie_tolerance, single_pass=True,
                                   deadline=deadline, progress=progress)
        if report.get("verdict") == "CERTIFIED":
            report["route"] = (f"bracketed link {link} at weight {record['weight']:.6f} in "
                               f"{record['evaluations']} evaluations")
            return x, weights, report
    return None


def _search_patterns(net, od, mu, x0, contested, groups, base_masks, tie_tolerance, settings,
                     solve, topo, tail, head, arrays, deadline=None, progress=None):
    """Enumerate small tie patterns, accepting the first that certifies.

    Three enumerations in increasing generality. A single candidate link, retrying after flipping any
    base link the certificate reports on the wrong side of its own gap; the orientations of the two-way
    contested pairs, which is the structurally right space because contested links arrive in opposite
    pairs with gaps of equal magnitude and opposite sign; and finally patterns of two or three links,
    which is what the five-node witness needs, where the certifying pattern is a directed ring and
    neither a single link nor the joint system certifies.
    """
    fft, cap, b, power, _, _ = arrays
    budget = _Budget(deadline=deadline)
    every_pair = list(contested)
    if progress is not None:
        progress(f"pattern search over {len(groups)} contested link(s), at most "
                 f"{budget.max_solves} solves")

    candidates = sorted(groups.items(), key=lambda item: -len(item[1]))[:PATTERN_MAX_LINKS]
    for link, destinations in candidates:
        if not budget.available():
            break
        pairs = [(d, link) for d in destinations]
        pinned = {d: base_masks[d].copy() for d in base_masks}
        promoted: list = []
        flipped_once: set = set()
        for _ in range(PATTERN_MAX_BASE_FLIPS):
            if not budget.available():
                break
            budget.spend()
            try:
                x, weights, report = solve(net, od, mu, x0, pairs + promoted, pinned,
                                           settings=settings, tie_tolerance=tie_tolerance,
                                           single_pass=True, deadline=deadline, progress=progress)
            except RuntimeError:
                break
            if report.get("verdict") == "CERTIFIED":
                report["route"] = (f"single link {link} of {len(groups)} contested, "
                                   f"{len(promoted)} promoted")
                return x, weights, report

            # A candidate's failure is usually not the candidate's fault: the other contested links are
            # pinned to the base side, and the certificate reports which of them sit on the wrong side of
            # their own gap. Flip those and retry; a link that flips back and forth is genuinely tied and
            # is promoted into this candidate's tie set.
            cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
            potential = potentials(topo, cost)
            flips = []
            for other_d, other_link in every_pair:
                if other_link == link or (other_d, other_link) in promoted:
                    continue
                gap = float(potential[tail[other_link], other_d] - potential[head[other_link], other_d])
                inside = bool(pinned[other_d][other_link])
                if (not inside and gap > tie_tolerance) or (inside and gap < -tie_tolerance):
                    flips.append((other_d, other_link, inside))
            if not flips:
                break
            for other_d, other_link, inside in flips:
                key = (other_d, other_link)
                if key in flipped_once:
                    promoted.append(key)
                elif not inside:
                    flipped_once.add(key)
                    trial = pinned[other_d].copy()
                    trial[other_link] = True
                    try:
                        kahn_order(topo, trial)
                        pinned[other_d] = trial
                    except RuntimeError:
                        promoted.append(key)
                else:
                    flipped_once.add(key)
                    pinned[other_d][other_link] = False
            promoted = list(dict.fromkeys(promoted))

    edges: dict = {}
    for link in groups:
        key = (min(int(tail[link]), int(head[link])), max(int(tail[link]), int(head[link])))
        edges.setdefault(key, []).append(int(link))
    two_way = [members for members in edges.values() if len(members) == 2]
    if 1 <= len(two_way) <= PATTERN_MAX_ORIENTATION_EDGES:
        for choice in product(*two_way):
            if not budget.available():
                break
            budget.spend()
            pairs = [(d, link) for link in choice for d in groups[link]]
            try:
                x, weights, report = solve(net, od, mu, x0, pairs, base_masks, settings=settings,
                                           tie_tolerance=tie_tolerance, single_pass=True,
                                           deadline=deadline, progress=progress)
            except DeadlineReached:
                # Re-raised rather than swallowed with the rest: a probe that ran out of the run's time
                # is not a probe that failed, and retrying it spends the time twice.
                raise
            except Exception:
                continue
            if report.get("verdict") == "CERTIFIED":
                report["route"] = f"orientation {list(choice)} of {len(two_way)} two-way pairs"
                return x, weights, report

    links = sorted(groups)[:PATTERN_MAX_LINKS + 2]
    for size in (2, 3):
        if not budget.available() or len(links) < size:
            break
        for pattern in combinations(links, size):
            if not budget.available():
                break
            budget.spend()
            pairs = [(d, link) for link in pattern for d in groups[link]]
            try:
                x, weights, report = solve(net, od, mu, x0, pairs, base_masks, settings=settings,
                                           tie_tolerance=tie_tolerance, single_pass=True,
                                           deadline=deadline, progress=progress)
            except DeadlineReached:
                # Re-raised rather than swallowed with the rest: a probe that ran out of the run's time
                # is not a probe that failed, and retrying it spends the time twice.
                raise
            except Exception:
                continue
            if report.get("verdict") == "CERTIFIED":
                report["route"] = f"pattern of {size} links {list(pattern)} of {len(links)}"
                return x, weights, report
    return None


def _representation_residual(topo, od, mu, cost, x, base, vertex_weights, base_weight):
    """How far the decomposition this run will record is from the flow it claims to describe.

    This is the check that was missing. ``representation["residual"]`` carried ``flow_residual`` -- how
    well the flow solves its own fixed point, which says nothing about whether the recorded supports and
    masses reproduce it -- so a row could be written CERTIFIED with a decomposition that loads to
    something else entirely. Three cells did: every face solved through the pattern route recorded a
    mixture that the independent test rebuilt at a residual of 1e-2 on a flow whose own residual was
    1e-15, and nothing in between looked.

    Loaded exactly as :func:`endogenous_sue.certificate.certify` loads it, so the number here and the
    number the a posteriori test produces are the same quantity measured twice rather than two
    quantities that happen to share a name.
    """
    reconstructed = np.zeros(topo.n_links)
    carried: dict = {}
    for d, added, value in vertex_weights:
        carried[int(d)] = carried.get(int(d), 0.0) + float(value)
        if float(value) <= ACTIVE_TOLERANCE:
            continue
        mask = base[int(d)].copy()
        for link in added:
            mask[int(link)] = True
        reconstructed += float(value) * load_destination(topo, od, mu, cost, int(d), mask,
                                                         kahn_order(topo, mask))
    # The deficit, not the recorded base mass. The ladder lists its base vertices among the others and
    # defines `base_weight` as the sum of exactly those, so adding it again loaded them twice: Anaheim at
    # mu=0.5 reported an internal representation residual of 0.3488 where the independent reconstruction
    # of the same record gives 8.43e-14. Where an accelerating route holds the base mass separately the
    # deficit is that mass, so one expression serves both.
    for d in sorted(base_weight):
        deficit = 1.0 - carried.get(int(d), 0.0)
        if deficit > ACTIVE_TOLERANCE:
            mask = base[int(d)]
            reconstructed += deficit * load_destination(topo, od, mu, cost, int(d), mask,
                                                        kahn_order(topo, mask))
    return float(np.max(np.abs(x - reconstructed)) / max(float(np.abs(x).max()), 1e-30))


def solve_tied(net: dict, od: np.ndarray, mu: float, x0: np.ndarray, contested: list,
               base_masks: dict, settings: Settings = DEFAULTS,
               tie_tolerance: float = TIE_TOLERANCE, tolerance: float = 1e-11,
               max_iterations: int = 300, single_pass: bool = False,
               warm_weights: dict | None = None, progress=None, deadline=None,
               checkpoint=None, measure_dimension: bool = False):
    """Solve the tied system and certify it. Returns the flow, the pair weights and a record.

    ``warm_weights`` optionally supplies starting marginals per contested pair, as returned by an earlier
    call. A caller growing the tie set one pair at a time would otherwise rediscover weights it already
    knew. This changes the starting point of the solve and never the system solved, so the certificate is
    unaffected.

    ``deadline`` and ``checkpoint`` make the outer walk resumable across a scheduler's wall-clock limit.
    The state written is the walk's own: the flow, which pairs are still live, what the retired ones were
    pinned to, and the base supports those pinnings have mutated. A resumed run re-enters the walk where
    it left rather than at the first round. A rung of the ladder is resumable too, through its own file
    beneath this one: each of its refinements writes the simplicial search's level, mesh and best point,
    so a face whose rung outlasts the budget continues rather than restarting.
    """
    tail, head, fft, cap, b, power, n_nodes, n_zones = unpack(net)
    arrays = (fft, cap, b, power, n_nodes, n_zones)
    topo = Topology.build(tail, head, fft, n_nodes, n_zones, net.get("first_thru"))
    demanded = [d for d in range(n_zones) if od[:, d].sum() > 0.0]

    if not single_pass:
        groups: dict = {}
        for d, link in contested:
            groups.setdefault(int(link), []).append(int(d))
        found = _bracket_single_links(net, od, mu, x0, groups, base_masks, tie_tolerance,
                                      settings, solve_tied, deadline=deadline, progress=progress)
        if found is not None:
            return found
        if 1 < len(groups) <= PATTERN_MAX_GROUPS:
            found = _search_patterns(net, od, mu, x0, contested, groups, base_masks, tie_tolerance,
                                     settings, solve_tied, topo, tail, head, arrays,
                                     deadline=deadline, progress=progress)
            if found is not None:
                return found

    base = {d: absorbing(topo, base_masks[d], d) for d in demanded}
    tied = [(int(d), int(link)) for d, link in contested]
    weights: dict = {}
    x = np.asarray(x0, dtype=float).copy()
    resumed_round = None
    stored = _load_walk(checkpoint)
    if stored is not None:
        x, tied, weights, base, resumed_round = stored
        if progress is not None:
            progress(f"resumed the walk at outer round {resumed_round}: {len(tied)} live pair(s), "
                     f"{len(weights)} already pinned")
    w = np.full(len(tied), 0.5)
    scale = max(1.0, float(np.abs(x).max()))
    started = time.perf_counter()
    rounds: list = []
    solution = None

    # Single-link vertices, one per contested pair, with an active-set outer loop. At most one pair is
    # retired per round, the worst excursion, so a single ill-conditioned corner cannot wipe the tie set;
    # and a retired pair is checked against its own gap at the solved cost, so a pair retired to the wrong
    # side is un-retired, at most once each, which keeps termination.
    from ..equilibrium import PeriodicCheckpoint, over_memory_ceiling

    un_retired: set = set()
    periodic = PeriodicCheckpoint(checkpoint, settings.checkpoint_seconds)
    for outer in range((resumed_round or 0) + 1, 4):
        for round_index in range(1, len(contested) + 2):
            if periodic.due():
                _save_walk(checkpoint, x, tied, weights, base, outer - 1)
                periodic.last = time.perf_counter()
            if over_memory_ceiling(settings):
                _save_walk(checkpoint, x, tied, weights, base, outer - 1)
                raise DeadlineReached(f"memory ceiling at outer round {outer}, "
                                      f"{len(tied)} pair(s) still live")
            if deadline is not None and time.perf_counter() >= deadline:
                _save_walk(checkpoint, x, tied, weights, base, outer - 1)
                raise DeadlineReached(f"outer round {outer}, {len(tied)} pair(s) still live")
            if progress is not None:
                progress(f"walk: outer {outer}, round {round_index}, {len(tied)} live pair(s), "
                         f"{len(weights)} pinned, {time.perf_counter() - started:.0f}s elapsed")
            vertices, cyclic = [], []
            for d, link in tied:
                mask = base[d].copy()
                mask[link] = True
                try:
                    kahn_order(topo, mask)
                except RuntimeError:
                    cyclic.append((d, link))
                    continue
                vertices.append((d, (link,), mask))
            if cyclic:
                keep = [i for i, pair in enumerate(tied) if pair not in cyclic]
                for pair in cyclic:
                    weights[pair] = 0.0
                w = w[keep]
                tied = [tied[i] for i in keep]
                if not tied:
                    break
                continue

            try:
                x, w, solution = solve_square_system(topo, arrays, od, mu, demanded, base, vertices,
                                                     tied, x, w, scale, tolerance, max_iterations)
            except (ValueError, RuntimeError):
                if len(tied) <= 1:
                    raise
                d, link = tied[-1]
                weights[(d, link)] = 0.0
                tied = tied[:-1]
                w = w[:-1]
                continue

            worst_index, worst_excursion = -1, 0.0
            for i in range(len(tied)):
                excursion = (-w[i]) if w[i] < 0.0 else (w[i] - 1.0 if w[i] > 1.0 else 0.0)
                if excursion > worst_excursion:
                    worst_index, worst_excursion = i, excursion
            if worst_index < 0:
                rounds.append(dict(phase="single link", outer=outer, round=round_index,
                                   n_tied=len(tied), converged=bool(solution.success),
                                   residual=float(np.max(np.abs(solution.fun))), retired=0))
                break

            d, link = tied[worst_index]
            value = 0.0 if w[worst_index] < 0.0 else 1.0
            if value == 1.0:
                base[d][link] = True
            weights[(d, link)] = value
            rounds.append(dict(phase="single link", outer=outer, round=round_index, n_tied=len(tied),
                               converged=bool(solution.success),
                               residual=float(np.max(np.abs(solution.fun))), retired=1,
                               retired_pair=[d, link, float(w[worst_index]), value]))
            keep = [i for i in range(len(tied)) if i != worst_index]
            tied = [tied[i] for i in keep]
            w = w[keep]

        cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
        potential = potentials(topo, cost)
        wrongly_pinned = []
        for (d, link), value in list(weights.items()):
            if (d, link) in tied or (d, link) in un_retired:
                continue
            gap = float(potential[tail[link], d] - potential[head[link], d])
            if (value == 1.0 and gap < -tie_tolerance) or (value == 0.0 and gap > tie_tolerance):
                wrongly_pinned.append((d, link, value))
        if not wrongly_pinned:
            break
        for d, link, value in wrongly_pinned:
            if value == 1.0:
                base[d][link] = False
            del weights[(d, link)]
            un_retired.add((d, link))
            tied.append((d, link))
        w = np.concatenate([w, np.full(len(wrongly_pinned), 0.5)])
        rounds.append(dict(phase="un-retire", outer=outer, n_un_retired=len(wrongly_pinned)))

    vertices = [(d, (link,), None) for d, link in tied]
    for i, (d, link) in enumerate(tied):
        weights[(d, link)] = float(w[i])
    base_weight = {d: 1.0 - sum(weights[(other, link)] for other, link in tied if other == d)
                   for d in demanded}
    phase = "single link"

    # A negative implied base weight means the equilibrium lies outside the single-link simplex, which is
    # what chained ties produce. Re-solve on the orientation family, where the system stays square.
    if any(value < -SIMPLEX_SLACK for value in base_weight.values()):
        vertices, square = [], True
        for d in demanded:
            links = [link for other, link in tied if other == d]
            if not links:
                continue
            if base_weight[d] >= -SIMPLEX_SLACK:
                for link in links:
                    mask = base[d].copy()
                    mask[link] = True
                    vertices.append((d, (link,), mask))
            else:
                family = orientation_family(topo, base[d], links)
                if len(family) != len(links):
                    square = False
                    break
                vertices += [(d, added, mask) for added, mask in family]
        if square:
            start = np.array([1.0 / sum(1 for other, _, _ in vertices if other == d)
                              for d, _, _ in vertices])
            if warm_weights:
                for i, (d, added, _) in enumerate(vertices):
                    if len(added) == 1 and (d, int(added[0])) in warm_weights:
                        start[i] = float(np.clip(warm_weights[(d, int(added[0]))], 1e-6, 1 - 1e-6))
            x, w, solution = solve_square_system(topo, arrays, od, mu, demanded, base, vertices,
                                                 tied, x, start, scale, tolerance, max_iterations)
            rounds.append(dict(phase="orientation", n_tied=len(tied),
                               converged=bool(solution.success),
                               residual=float(np.max(np.abs(solution.fun)))))
            for d, link in tied:
                weights[(d, link)] = float(sum(w[i] for i, (other, added, _) in enumerate(vertices)
                                               if other == d and link in added))
            base_weight = {d: 1.0 - sum(w[i] for i, (other, _, _) in enumerate(vertices)
                                        if other == d) for d in demanded}
            phase = "orientation"

    # Everything above is an accelerator and may end anywhere, because a matrix-free solve on the face has
    # no convergence theorem: it can stop with gaps above tolerance or with weights off the simplex. When
    # it does, the same face is re-solved by the ladder, whose bottom rung terminates with no hypothesis.
    # The accelerator is never trusted on its own word: its answer is measured first and kept only if the
    # ladder cannot beat it.
    ladder = None
    if solution is not None and tied:
        cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
        potential = potentials(topo, cost)
        worst_gap = max(abs(float(potential[tail[link], d] - potential[head[link], d]))
                        for d, link in tied)
        def on_simplex(value):
            return -ACTIVE_TOLERANCE <= value <= 1.0 + ACTIVE_TOLERANCE

        off_simplex = (any(not on_simplex(weights[(d, link)]) for d, link in tied)
                       or any(value < -ACTIVE_TOLERANCE for value in base_weight.values()))
        if worst_gap >= tie_tolerance or off_simplex or not bool(solution.success):
            try:
                ladder_x, ladder_w, ladder = solve_face_ladder(
                    topo, arrays, od, mu, demanded, base, tied, x, settings=settings,
                    tie_tolerance=tie_tolerance, progress=progress, deadline=deadline,
                    measure_dimension=measure_dimension,
                    checkpoint=_sibling(checkpoint, "face"))
            except (ValueError, RuntimeError) as exc:
                ladder = dict(status=f"failed: {type(exc).__name__}: {str(exc)[:60]}",
                              complementarity=float("inf"), certified_face=False)
            # Both answers are judged by the same defect: distance from satisfying the face conditions.
            # For the accelerator that is its worst tie gap; for the ladder it is the full
            # complementarity residual, which does not excuse a gap pointing the wrong way at a bound.
            # The comparison is meaningful only when the accelerator's answer is a mixture: with weights
            # off the simplex, a small gap is achieved by a signed combination no convex mixture
            # realises, which is a number rather than a defect. A face the ladder certified on its own
            # standard is not up for comparison at all.
            if ladder.get("certified_face") or ladder["complementarity"] < (
                    float("inf") if off_simplex else worst_gap):
                x, w, phase = ladder_x, ladder_w, "ladder"
                vertices = [(d, (link,), None) for d, link in tied]
                for i, (d, link) in enumerate(tied):
                    weights[(d, link)] = float(ladder_w[i])
                base_weight = {d: 1.0 - sum(float(ladder_w[i]) for i, (other, _) in enumerate(tied)
                                            if other == d) for d in demanded}
                rounds.append(dict(phase="ladder", n_tied=len(tied),
                                   converged=bool(ladder.get("certified_face")),
                                   residual=float(ladder["complementarity"])))

    # The active set is not the face. The loop above retires a pair whenever its weight leaves the unit
    # interval, so it can end having retired every pair, on a face still many pairs wide. The tie gap then
    # maxes over an empty list and reports zero beside a flow with admissibility violations, and the flow
    # residual is tiny because the flow solves the pinned system exactly: it solves the wrong system
    # exactly. The ladder is therefore re-offered the face it was asked about, under the unmutated base.
    if solution is not None and not tied and contested:
        cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
        potential = potentials(topo, cost)
        wrongly_pinned = [
            (d, link) for (d, link), value in weights.items()
            if (value >= 1.0 - SIMPLEX_SLACK
                and float(potential[tail[link], d] - potential[head[link], d]) < -tie_tolerance)
            or (value <= SIMPLEX_SLACK
                and float(potential[tail[link], d] - potential[head[link], d]) > tie_tolerance)]
        if wrongly_pinned:
            full = [(int(d), int(link)) for d, link in contested]
            full_base = {d: absorbing(topo, base_masks[d], d) for d in demanded}
            try:
                ladder_x, ladder_w, ladder = solve_face_ladder(
                    topo, arrays, od, mu, demanded, full_base, full, x0, settings=settings,
                    tie_tolerance=tie_tolerance, progress=progress, deadline=deadline,
                    measure_dimension=measure_dimension,
                    checkpoint=_sibling(checkpoint, "face"))
            except (ValueError, RuntimeError) as exc:
                ladder = dict(status=f"failed: {type(exc).__name__}: {str(exc)[:80]}",
                              complementarity=float("inf"), certified_face=False)
            # Adopted only on the ladder's own standard. If it does not certify, the accelerator's answer
            # stands exactly as before and the record now says why.
            if ladder.get("certified_face"):
                x, w, phase, tied, base = ladder_x, ladder_w, "ladder", full, full_base
                weights = {(d, link): float(ladder_w[i]) for i, (d, link) in enumerate(full)}
                vertices = [(d, (link,), None) for d, link in full]
                base_weight = {d: 1.0 - sum(float(ladder_w[i]) for i, (other, _) in enumerate(full)
                                            if other == d) for d in demanded}
                rounds.append(dict(phase="ladder re-admit", n_tied=len(full), converged=True,
                                   residual=float(ladder["complementarity"])))

    if solution is None:
        return x, weights, dict(
            phase="degenerate", n_contested=len(contested), n_tied=0, rounds=rounds,
            walk=walk_counters(contested, [], weights, rounds),
            verdict="NOT CERTIFIED",
            reason="every contested pair was shed: cyclic vertices or singular solves",
            wall_seconds=time.perf_counter() - started)

    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    potential = potentials(topo, cost)
    tie_gap = max((abs(float(potential[tail[link], d] - potential[head[link], d]))
                   for d, link in tied), default=0.0)
    flow_residual = (float(ladder["flow_residual"]) if phase == "ladder" and ladder
                     else float(np.linalg.norm(solution.fun[:topo.n_links] * scale)
                                / max(np.linalg.norm(x), 1e-30)))

    vertex_weights = ([[d, list(added), float(w[i])] for i, (d, added, _) in enumerate(vertices)]
                      if phase in ("orientation", "ladder")
                      else [[d, [link], float(weights[(d, link)])] for d, link in tied])

    # The orientation and ladder families name only the destinations their own search resolved, and an
    # accelerator that answered on a handful of links leaves the rest of the tied set out of the record
    # entirely. Every cell solved through `pattern of N links` failed its independent certificate for
    # that reason: the sweep rebuilt those destinations on the strict support of the final cost, which is
    # not what the solve pinned them to, and read mixture residuals of 1e-2 on flows whose own residual
    # was 1e-15. Their resolutions are known -- they are in ``weights`` -- so they are recorded.
    #
    # Only where a destination's tied links are each pinned to one side. A destination left genuinely
    # split by a route that did not describe it is not representable as one mask, and rounding it would
    # write down a mixture the solve never loaded. That case is left out and the representation check
    # below is what refuses the row.
    named = {int(entry[0]) for entry in vertex_weights}
    unresolved_destinations = []
    for d in sorted({int(d) for d, _ in tied} - named):
        pinned = [(link, float(weights[(d, link)])) for other, link in tied if other == d]
        if any(ACTIVE_TOLERANCE < value < 1.0 - ACTIVE_TOLERANCE for _, value in pinned):
            unresolved_destinations.append(d)
            continue
        added = [link for link, value in pinned if value >= 1.0 - ACTIVE_TOLERANCE]
        vertex_weights.append([d, added, 1.0])
        base_weight[d] = 0.0

    ladder_mixture = ladder.get("mixture") if (phase == "ladder" and ladder) else None
    # Weights within a solver tolerance of the simplex are on the simplex. A raw negative entry of size
    # 1e-14 must not divert judgment to the fit, whose family differs from the orientation family and
    # which would then report a meaningless residual.
    directly_convex = (all(-ACTIVE_TOLERANCE <= value <= 1.0 + ACTIVE_TOLERANCE
                           for _, _, value in vertex_weights)
                       and all(value >= -ACTIVE_TOLERANCE for value in base_weight.values()))

    if ladder_mixture:
        # The ladder loaded these supports at these masses, so that mixture is what is certified.
        # Reconstructing one from the marginals would place a cube point on the single-link simplex,
        # where it need not lie.
        weighted = [(d, mask) for d, mask, value in ladder_mixture if value > ACTIVE_TOLERANCE]
        vertex_weights = [[int(d), np.flatnonzero(mask & ~base[d]).tolist(), float(value)]
                          for d, mask, value in ladder_mixture]
        base_weight = {d: float(sum(value for other, mask, value in ladder_mixture
                                    if other == d and not (mask & ~base[d]).any()))
                       for d in demanded}
        for d, link in tied:
            weights[(d, link)] = float(sum(value for other, mask, value in ladder_mixture
                                           if other == d and mask[link]))
        representation = dict(
            representation=f"ladder {ladder.get('family', 'unknown')}", feasible=True,
            residual=_representation_residual(topo, od, mu, cost, x, base,
                                              vertex_weights, base_weight))
    elif directly_convex:
        vertex_weights = [[d, added, min(max(value, 0.0), 1.0)] for d, added, value in vertex_weights]
        base_weight = {d: max(value, 0.0) for d, value in base_weight.items()}
        weighted = [(d, base[d]) for d in demanded if base_weight[d] > ACTIVE_TOLERANCE]
        for d, added, value in vertex_weights:
            if value > ACTIVE_TOLERANCE:
                mask = base[d].copy()
                for link in added:
                    mask[link] = True
                weighted.append((d, mask))
        representation = dict(
            representation="direct", feasible=True,
            residual=_representation_residual(topo, od, mu, cost, x, base,
                                              vertex_weights, base_weight))
    else:
        fitted = convex_certificate(topo, od, mu, cost, x, demanded, base, tied)
        weighted = fitted.pop("weighted", [])
        representation = dict(representation="fitted", feasible=fitted["feasible"],
                              residual=fitted["residual"], status=fitted["status"])
        if fitted.get("masks"):
            vertex_weights = fitted["masks"]
            for d, link in tied:
                weights[(d, link)] = float(sum(value for other, added, value in vertex_weights
                                               if other == d and link in added))
            base_weight = {d: float(sum(value for other, added, value in vertex_weights
                                        if other == d and not added)) for d in demanded}

    # The base each recorded support is stated against, absolutely, for the destinations the mixture
    # names. A delta cannot be used here: the accelerating routes search against a copy of the base with
    # links pinned in and out and pass that copy down, so a delta recorded by an outer search is relative
    # to a base its winning solve may never have seen, and a delta recorded by an inner one is relative
    # to a base the caller does not hold. Absolute composes through any depth of recursion, which is the
    # only property that makes this safe. All three cells solved through the pattern route failed their
    # independent certificate because the sweep rebuilt their supports on the wrong base.
    mixture_destinations = ({int(entry[0]) for entry in vertex_weights}
                            | {int(d) for d, value in base_weight.items() if value > ACTIVE_TOLERANCE})
    support_base = {int(d): np.flatnonzero(base[int(d)]).tolist()
                    for d in sorted(mixture_destinations) if int(d) in base}

    violations, worst_gap, worst_flow = admissibility_violations(topo, weighted, potential, x,
                                                                 tolerance=tie_tolerance)
    # The same violations, named and deduplicated, so a caller can reopen exactly the pairs that failed
    # rather than re-deriving them from a count. A pair here is a link the recorded mixture carries that
    # the returned flow's own cost excludes: the face was identified wrongly, and widening a tolerance or
    # spending more iterations on the same face cannot repair it.
    inadmissible = violating_pairs(topo, weighted, potential, tolerance=tie_tolerance)
    largest, total_error, demand = conservation_error(net, od, x)
    conserves = largest <= settings.conservation_tolerance * max(demand, 1.0)

    # A face the budget stopped is never certified, whatever its residual: the search returned the best
    # point it had reached, not a point it had shown to be a fixed one.
    budget_limited = bool(ladder and ladder.get("stopped_on") == "time budget")
    # A destination the route left split and did not describe cannot be written down as one mask, and
    # rounding it would record a mixture the solve never loaded. The row says so rather than guessing.
    if unresolved_destinations:
        representation["feasible"] = False
        representation["unresolved_destinations"] = unresolved_destinations

    certified = (not budget_limited
                 and tie_gap < tie_tolerance and violations == 0
                 and flow_residual < settings.residual_tolerance
                 and representation["feasible"]
                 and representation["residual"] < settings.residual_tolerance and conserves)

    reasons = []
    if tie_gap >= tie_tolerance:
        reasons.append(f"tie gap {tie_gap:.3e}")
    if violations:
        reasons.append(f"{violations} support inclusion violation(s)")
    if not representation["feasible"] or representation["residual"] >= settings.residual_tolerance:
        reasons.append("no convex representation of the flow")
    if not conserves:
        reasons.append(f"demand not conserved, worst node error {largest:.3e}")

    if budget_limited:
        reasons.append("the face solve stopped on its time budget")

    record = dict(phase=phase, n_contested=len(contested), n_tied=len(tied), rounds=rounds,
                  budget_limited=budget_limited,
                  walk=walk_counters(contested, tied, weights, rounds),
                  n_interior=(ladder or {}).get("n_interior"),
                  gap_rank=(ladder or {}).get("gap_rank"),
                  manifold_dimension=(ladder or {}).get("manifold_dimension"),
                  flow_residual=flow_residual, tie_gap=float(tie_gap),
                  weights=[[d, link, float(value)] for (d, link), value in sorted(weights.items())],
                  base_weight={str(d): float(value) for d, value in base_weight.items()},
                  vertex_mixture=vertex_weights, support_base=support_base,
                  violations=violations, worst_gap=worst_gap,
                  inadmissible_pairs=[[d, link, gap] for d, link, gap in inadmissible],
                  worst_violating_flow=worst_flow,
                  conservation_max=largest, conservation_l1=total_error,
                  verdict="CERTIFIED" if certified else "NOT CERTIFIED",
                  reason="; ".join(reasons),
                  wall_seconds=time.perf_counter() - started, **representation)
    if ladder is not None:
        record["ladder"] = {key: value for key, value in ladder.items() if key != "mixture"}
    return x, weights, record


def solve_face_admissible(net: dict, od: np.ndarray, mu: float, x0: np.ndarray, contested: list,
                          base_masks: dict, settings: Settings = DEFAULTS,
                          tie_tolerance: float = TIE_TOLERANCE,
                          rounds: int = FACE_REOPEN_ROUNDS, progress=None, checkpoint=None,
                          **kwargs):
    """Solve the tied face, reopening every pair the solved flow's own cost refuses.

    A face is a hypothesis about which destination-link pairs are tied. Two sources feed it and neither
    is rechecked once the face is solved: the outer walk pins contested pairs to a bound, and the
    inclusion scheme's frozen support supplies the base for everything it never contested. Where the
    potential at the returned flow's cost excludes a link the recorded mixture carries, the hypothesis
    was wrong. That is not a shortfall of iterations and not a tolerance that wants widening: the face
    itself has to change, so the offending pairs are named, reopened, and the face is solved again.

    Reopening a pair is two edits, and one without the other does nothing. The pair joins the tie set,
    and the link is cleared from that destination's base mask. Leaving it in the base would make the
    tie a blend of the base against the base with a link it already has, whose two vertices are the same
    support and whose weight therefore cannot move anything. Each attempt also searches through its own
    checkpoint, because resuming the previous one would restore the very pinning being reopened.

    Returns whatever the last attempt returned. A face still refusing pairs after ``rounds`` is reported
    with them rather than repaired: the caller's ladder descends to a rung that forms no face at all.
    """
    pairs = sorted({(int(d), int(link)) for d, link in contested})
    masks = {d: np.asarray(mask).copy() for d, mask in base_masks.items()}
    reopened: list = []
    x = weights = record = None
    for attempt in range(rounds + 1):
        # Named for the pair set, not the attempt number. `solve_tied` restores a stored walk without
        # checking that its tie set is the one being asked for, so a file named `reopen1` would be
        # loaded by any first reopen -- including one that reopened different pairs on an earlier
        # submission of the same cell, which would silently solve the previous face.
        step = checkpoint if attempt == 0 else _sibling(
            checkpoint, "reopen" + blake2b(repr(sorted(reopened)).encode(), digest_size=5).hexdigest())
        x, weights, record = solve_tied(net, od, mu, x0, list(pairs), masks, settings,
                                        tie_tolerance=tie_tolerance, progress=progress,
                                        checkpoint=step, **kwargs)
        refused = [(int(d), int(link), float(gap))
                   for d, link, gap in (record.get("inadmissible_pairs") or [])]
        fresh = [entry for entry in refused if (entry[0], entry[1]) not in reopened]
        if not refused or not fresh or attempt == rounds:
            break
        # The admissibility rule refuses a pair in one of three ways and each wants a different repair.
        # Which one applies is read off the gap against the tie tolerance the model already declares, so
        # nothing here is a tuned choice.
        #
        #   gap below -tolerance  the support carries a link the cost strictly excludes. Take it out of
        #                         the base and out of the tie set. Leaving it tied is what failed on
        #                         Anaheim's pair (29, 454): the walk pinned the reopened weight straight
        #                         back to 1.0, which puts the link in every support for that destination
        #                         and folds it back into the base, so the tie could not express the fix.
        #   gap above +tolerance  the link is strictly efficient and the support omits it. It belongs in.
        #   gap within tolerance  genuinely tied, so make it a contested pair and let the face decide.
        excluded, tied_again = [], []
        for d, link, gap in fresh:
            if gap < -tie_tolerance:
                if d in masks and masks[d][link]:
                    masks[d] = masks[d].copy()
                    masks[d][link] = False
                excluded.append((d, link))
            elif gap > tie_tolerance:
                if d in masks and not masks[d][link]:
                    masks[d] = masks[d].copy()
                    masks[d][link] = True
                excluded.append((d, link))
            else:
                tied_again.append((d, link))
        reopened.extend((d, link) for d, link, _ in fresh)
        pairs = sorted((set(pairs) | set(tied_again)) - set(excluded))
        if progress is not None:
            progress(f"reopening {len(fresh)} pair(s) the final cost refuses "
                     f"(round {attempt + 1} of {rounds}): {len(excluded)} forced out of or into the "
                     f"base, {len(tied_again)} made contested; "
                     f"{[(d, link) for d, link, _ in fresh][:6]}"
                     + ("..." if len(fresh) > 6 else ""))
    return x, weights, dict(record, reopen_rounds=attempt,
                            reopened_pairs=[[d, link] for d, link in reopened])
