"""Solving for equilibrium when the efficient support is endogenous.

The difficulty is the support, not the flow. Rebuilding the support from the instantaneous congested cost
makes the map discontinuous, and the support then oscillates indefinitely on the tie boundary. The
two-phase scheme separates the two problems.

Phase one builds the support from a running average of the congested cost while loading at the current
cost. Averaging the cost that *defines* the support stabilises the efficiency boundary, so the support
stops changing in finite time whenever the equilibrium is off the tie boundary. It converges at the rate
of the averaging, which leaves the flow residual near ``1e-3``.

Phase two holds that frozen support fixed and solves the now-smooth fixed point to machine precision.
Running it while the support still moves would sharpen the oscillation rather than damp it, so the order
matters.

Neither phase stops on an iteration count. Phase one stops when the flow update falls below the flow
tolerance, or earlier when the exactness bound holds at the current cost and certifies that no efficiency
gap can still change sign. The iteration limits in :class:`endogenous_sue.config.Settings` mark a run
unconverged; they never define an answer.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from hashlib import blake2b
from itertools import count
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import DEFAULTS, PARALLEL_MIN_WORK_LOADING, DeadlineReached, Settings
from .loading import LoadingReport, load, load_parallel
from .network import NetworkArrays
from .shortestpath import potentials
from .subnetwork import active_for_dest, isolation_margin

__all__ = ["SupportRule", "Phase1Result", "Equilibrium", "InclusionResult", "PolishResult",
           "DeadlineReached", "PeriodicCheckpoint", "over_memory_ceiling", "save_checkpoint",
           "load_checkpoint", "solve_phase1", "solve_inclusion", "polish", "solve"]


class SupportRule(str, Enum):
    """How the efficient support is rebuilt between iterations.

    ``DAMPED`` is the method. The other two are the comparisons it is measured against and are not
    alternatives to it: ``FIXED`` never rebuilds, which is the exogenous-support model, and
    ``INSTANTANEOUS`` rebuilds from the current cost at every step, which is the scheme whose support
    keeps flipping on the tie boundary.
    """

    DAMPED = "damped"
    FIXED = "fixed"
    INSTANTANEOUS = "instantaneous"


@dataclass
class Phase1Result:
    """Outcome of phase one, with the diagnostics the convergence claims are read from."""

    x: np.ndarray
    potential: np.ndarray
    converged: bool
    stopped_on: str
    iterations: int
    gap_history: list = field(default_factory=list)
    support_changes: list = field(default_factory=list)
    flipped_flow_share: list = field(default_factory=list)
    seconds: dict = field(default_factory=dict)
    wall_seconds: float = 0.0
    resumed_from: int | None = None


@dataclass
class Equilibrium:
    """A returned assignment, with what is needed to judge it.

    ``converged`` refers to phase one reaching a stopping criterion. A run that instead reached its
    iteration limit is reported here as unconverged, and is not a result.
    """

    x: np.ndarray
    residual: float
    converged: bool
    stopped_on: str
    polish_method: str
    phase1: Phase1Result
    settings: Settings


@dataclass
class PolishResult:
    """The outcome of the frozen-support solve.

    ``converged`` is the residual actually reaching the tolerance, and not the method having been asked
    to try. The Krylov solve can exhaust its iterations, and the averaging fallback has no convergence
    test of its own, so a flow returned by either can sit above tolerance; recording the method alone
    leaves that indistinguishable from a solve that succeeded.
    """

    x: np.ndarray
    residual: float
    method: str
    converged: bool


class PeriodicCheckpoint:
    """Writes a checkpoint at most every ``interval`` seconds, overwriting it in place.

    The run is expected to be killed by the scheduler rather than to stop itself, so what matters is that
    the state on disk is never more than one interval old and is never half-written. The write is atomic,
    so a job killed during one leaves the previous state readable.
    """

    def __init__(self, path, interval: float):
        self.path = path
        self.interval = interval
        self.last = perf_counter()

    def due(self) -> bool:
        return (self.path is not None and self.interval > 0.0
                and perf_counter() - self.last >= self.interval)

    def write(self, **state) -> None:
        if self.path is None:
            return
        save_checkpoint(self.path, **state)
        self.last = perf_counter()


def _resident_mb() -> float:
    """This process's high-water memory, for the progress line.

    Reported beside the iteration because a loop that accumulates without bound looks exactly like a loop
    that is working until the machine stops it, and the only difference visible from outside is this
    number climbing.
    """
    try:
        import resource
    except ImportError:                       # pragma: no cover - Windows
        return 0.0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(peak) * scale / (1024.0 * 1024.0)


def over_memory_ceiling(settings: Settings) -> bool:
    """Whether this process has passed the ceiling its settings allow."""
    ceiling = settings.memory_ceiling_mb
    return ceiling is not None and _resident_mb() > ceiling


def save_checkpoint(path, **state) -> None:
    """Write resumable state atomically, so a job killed mid-write leaves the previous one intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    # Written through a handle rather than by name: given a name, the archive writer appends its own
    # suffix, and the rename would then look for a file that does not exist.
    with open(temporary, "wb") as handle:
        np.savez(handle, **state)
    os.replace(temporary, path)


def load_checkpoint(path):
    """Resumable state previously written, or ``None`` where there is none to resume from."""
    path = Path(path)
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


class _EnvelopeWatch:
    """Whether the window's envelope has held still across consecutive complete windows.

    The inclusion scheme's own test asks whether one *support* recurs. The face solver never reads a
    support identity: :func:`endogenous_sue.tied.contested_pairs` consumes the per-destination
    intersection and union of the window and nothing else. On a large cell those two questions have
    different answers -- Chicago-Sketch at mu=1 produced 441,722 distinct supports in a million
    iterations while its intersection and union did not move across eight thousand -- so a scheme waiting
    on recurrence there waits forever for something the face does not need.

    Windows are complete and consecutive, and only a full one is compared. What is reported is what was
    observed: the range held, when it last changed, and how many windows have passed since. That is an
    empirical condition for attempting the face solve and is never a proof that the envelope stays fixed.
    """

    def __init__(self, window: int, required: int):
        self.window = int(window)
        self.required = int(required)
        self.base: dict = {}
        self.union: dict = {}
        self.previous: tuple | None = None
        self.first = None
        self.held = 0
        self.stable_from: int | None = None
        self.last_change: int | None = None

    @staticmethod
    def _print(base: dict, union: dict) -> tuple:
        hasher = blake2b(digest_size=16)
        for d in sorted(base):
            hasher.update(np.uint32(d).tobytes())
            hasher.update(base[d].tobytes())
        for d in sorted(union):
            hasher.update(union[d].tobytes())
        return hasher.digest()

    def see(self, iteration: int, packed: dict) -> None:
        if self.first is None:
            self.first = iteration
        for d, bits in packed.items():
            if d in self.base:
                self.base[d] &= bits
                self.union[d] |= bits
            else:
                self.base[d], self.union[d] = bits.copy(), bits.copy()
        if iteration - self.first + 1 < self.window:
            return
        fingerprint = self._print(self.base, self.union)
        if self.previous is not None and fingerprint == self.previous:
            self.held += 1
            if self.stable_from is None:
                self.stable_from = self.first
        else:
            if self.previous is not None:
                self.last_change = iteration
            self.held = 0
            self.stable_from = None
        # The envelope of the window that has just closed is kept, so a cell stopped here hands the face
        # solver the object it was stopped on. The next window starts from nothing: derived afresh, or
        # the accumulation would only ever grow and could not disagree with itself.
        self.previous = fingerprint
        self.settled_base = {d: bits.copy() for d, bits in self.base.items()}
        self.settled_union = {d: bits.copy() for d, bits in self.union.items()}
        self.base, self.union, self.first = {}, {}, None


def _use_parallel(net: dict) -> bool:
    return int(net["n_zones"]) * len(net["tail"]) > PARALLEL_MIN_WORK_LOADING


def solve_phase1(net: dict, od: np.ndarray, mu: float, settings: Settings = DEFAULTS,
                 rule: SupportRule = SupportRule.DAMPED, parallel: bool | None = None,
                 tiebreak: np.ndarray | None = None, tiebreak_sign: float = 0.0,
                 exact_stop: bool = True, checkpoint=None,
                 deadline: float | None = None, progress=None) -> Phase1Result:
    """Average the flow while the support settles. Returns the flow and the frozen support.

    A positive ``potential_dead_band`` freezes a node's cost-to-destination inside a band of that
    half-width relative to the median free-flow cost, so a potential is updated only when it moves by
    more than the band. Near-tied links then stop flipping, and the degenerate case, where the exclusion
    margin is zero and the support genuinely never settles, freezes onto one admissible selection at an
    error of the same order. The support is always the strict efficient set of the frozen potentials,
    hence always acyclic.

    ``tiebreak`` tilts the cost that *defines* the support and leaves the cost flow is loaded at alone,
    so it selects among supports at a tie without moving the equilibrium away from one. Two runs with
    opposite ``tiebreak_sign`` therefore bracket the solution set.

    Under :attr:`SupportRule.FIXED` the support cannot move, so phase one has nothing to settle and
    stops immediately; the frozen-support solve is what produces that comparator's flow.

    ``exact_stop`` adds the exactness test. Once the bound holds at the current cost, no efficiency gap
    can change sign, the support is already final, and phase two will polish on it. The test needs
    potentials and a loading at the *current* cost rather than at the averaged cost the support is built
    from, so it costs one extra shortest-path pass and one extra loading per rebuild. It cannot fire
    where the isolation margin is zero, which is every tied cell; there the flow tolerance is the stop.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    if parallel is None:
        parallel = _use_parallel(net)

    positive_fft = arrays.fft[arrays.fft > 0]
    band = settings.potential_dead_band * (float(np.median(positive_fft)) if positive_fft.size else 1.0)
    tilt = 0.0 if tiebreak is None else tiebreak_sign * settings.tiebreak_scale

    x = np.zeros(arrays.n_links)
    averaged_cost = arrays.fft.copy()
    potential = None
    settled_rebuilds = 0
    first_iteration = 1
    periodic = PeriodicCheckpoint(checkpoint, settings.checkpoint_seconds)
    resumed = load_checkpoint(checkpoint) if checkpoint is not None else None
    if resumed is not None and int(resumed.get("phase", 1)) != 1:
        resumed = None                       # the checkpoint belongs to phase two; that phase resumes it
    if resumed is not None:
        x = resumed["x"]
        averaged_cost = resumed["averaged_cost"]
        potential = resumed["potential"]
        settled_rebuilds = int(resumed["settled_rebuilds"])
        first_iteration = int(resumed["iteration"]) + 1
        if progress is not None:
            progress(f"resumed phase 1 at iteration {first_iteration}, "
                     f"{settled_rebuilds} stable rebuild(s) behind it")

    result = Phase1Result(x=x, potential=np.empty(0), converged=False,
                          stopped_on="iteration limit", iterations=first_iteration - 1,
                          resumed_from=first_iteration - 1 if resumed is not None else None)
    periodic = PeriodicCheckpoint(checkpoint, settings.checkpoint_seconds)
    cache: dict = {}
    seconds = dict(shortest_path=0.0, support=0.0, order=0.0, value=0.0, flow=0.0)
    started = perf_counter()
    last_progress = started

    # No iteration count by default: the run is bounded by its wall clock, and its state is checkpointed
    # so a cell too long for one job continues in the next.
    limit = settings.max_phase1_iterations
    for iteration in (count(first_iteration) if limit is None
                      else range(first_iteration, limit + 1)):
        cost = arrays.cost(x)
        step = 1.0 / (iteration ** settings.step_power)
        averaged_cost += step * (cost - averaged_cost)

        if rule is SupportRule.FIXED:
            rebuild = potential is None
            defining_cost = arrays.fft
        elif rule is SupportRule.INSTANTANEOUS:
            rebuild = True
            defining_cost = cost
        else:
            rebuild = potential is None or (iteration - 1) % settings.support_rebuild_interval == 0
            defining_cost = averaged_cost if tiebreak is None else averaged_cost + tilt * tiebreak

        if rebuild:
            mark = perf_counter()
            fresh = potentials(topo, defining_cost)
            if potential is None or band <= 0.0 or rule is not SupportRule.DAMPED:
                updated = fresh
            else:
                with np.errstate(invalid="ignore"):
                    # A difference of infinities at an unreachable node is not a number, compares false,
                    # and correctly leaves the stored potential in place.
                    moved = np.abs(fresh - potential) > band
                updated = np.where(moved, fresh, potential)
            stable = potential is not None and np.array_equal(updated, potential)
            settled_rebuilds = settled_rebuilds + 1 if stable else 0
            potential = updated
        if rule is SupportRule.FIXED:
            # The support is the free-flow one and is never rebuilt, so it is settled from the first
            # iteration. Counting rebuilds would never reach the threshold and the run would stop on its
            # iteration limit with a flow the frozen-support solve then polishes anyway; this says so
            # instead of reporting a comparator arm as unconverged on every cell.
            settled_rebuilds += 1
            seconds["shortest_path"] += perf_counter() - mark

        if parallel:
            y = load_parallel(topo, od, mu, cost, potential)
            result.support_changes.append(None)
            result.flipped_flow_share.append(None)
        else:
            report = LoadingReport()
            y = load(topo, od, mu, cost, potential, cache=cache,
                     reuse_support=not rebuild, report=report)
            result.support_changes.append(report.support_changes)
            result.flipped_flow_share.append(report.flipped_flow_share)
            for key, value in report.seconds.items():
                seconds[key] += value

        gap = float(np.linalg.norm(y - x) / max(np.linalg.norm(x), 1e-30))
        result.gap_history.append(gap)
        result.iterations = iteration

        if (settled_rebuilds >= settings.support_stable_rebuilds
                and rule is not SupportRule.INSTANTANEOUS):
            x = y
            result.converged, result.stopped_on = True, "support settled"
            break

        if iteration > 1 and gap < settings.flow_tolerance:
            x = y
            result.converged, result.stopped_on = True, "flow tolerance"
            break

        if exact_stop and rebuild and iteration > 1 and rule is SupportRule.DAMPED:
            current = potentials(topo, cost)
            margin = isolation_margin(topo, current)
            here = load(topo, od, mu, cost, current)
            distance = float(np.linalg.norm(here - x))
            # The modulus must bound the cost derivative over every flow the remaining error can reach,
            # not only at the current iterate. The BPR derivative increases with flow, so evaluating it
            # at the far corner of the ball that contains both iterates is an upper bound; taking it at
            # ``x`` alone under-states it and lets the test fire before the theorem licenses it.
            bound = (2.0 * (arrays.n_nodes - 1)
                     * float(np.max(arrays.cost_prime(x + distance))) * distance)
            if np.isfinite(margin) and bound < margin:
                x = y
                result.converged, result.stopped_on = True, "exactness bound"
                break

        x = x + step * (y - x)
        # The time budget is checked here rather than left to the scheduler. A job killed at its
        # wall-clock limit records nothing about the cell it was in; stopping voluntarily writes the
        # state once, names the reason, and lets the next run continue from this iteration.
        now = perf_counter()
        if progress is not None and now - last_progress >= settings.progress_seconds:
            progress(f"phase 1 iteration {iteration}, gap {gap:.3e}, support "
                     f"{'settled' if settled_rebuilds else 'moving'}, "
                     f"{now - started:.0f}s elapsed")
            last_progress = now
        if periodic.due():
            periodic.write(phase=1, x=x, averaged_cost=averaged_cost, potential=potential,
                           settled_rebuilds=np.int64(settled_rebuilds), iteration=np.int64(iteration))
        if deadline is not None and now >= deadline:
            result.stopped_on = "time budget"
            result.iterations = iteration
            break

    if potential is None:
        potential = potentials(topo, arrays.fft)
    if checkpoint is not None and not result.converged:
        # One checkpoint per cell, written only where the run stops short of an answer. Nothing is
        # written periodically: the state is dominated by the potential table, which is nearly a
        # gigabyte on the largest network, and a cell that finishes needs no state kept at all.
        save_checkpoint(checkpoint, phase=1, x=x, averaged_cost=averaged_cost, potential=potential,
                        settled_rebuilds=settled_rebuilds, iteration=result.iterations)
    result.x = x
    result.potential = potential
    result.seconds = seconds
    result.wall_seconds = perf_counter() - started
    return result


@dataclass
class InclusionResult:
    """Outcome of the inclusion scheme: a flow, and the supports still competing at the end."""

    x: np.ndarray
    supports: list
    potential: np.ndarray
    converged: bool
    stopped_on: str
    iterations: int
    final_move: float
    wall_seconds: float
    n_links: int = 0
    resumed_from: int | None = None
    #: The envelope the scheme stopped on, packed per destination, where the envelope rule
    #: fired. The face solver consumes the intersection and the union and nothing else, so
    #: a cell stopped this way hands over exactly what it was stopped on rather than a
    #: trailing window that may have moved since.
    envelope_base: dict | None = None
    envelope_union: dict | None = None
    envelope_stable_from: int | None = None
    envelope_last_change: int | None = None
    envelope_windows_held: int = 0
    n_supports_seen: int = 0

    def masks(self):
        """Unpack the window's supports one at a time, as ``{destination: boolean mask}``.

        Packed while stored and unpacked while read, because the window holds a few hundred of them and a
        consumer only ever needs one at a time. On the largest cell of the certificate tier the packed
        window is tens of megabytes where the potential tables it replaced were most of a gigabyte.
        """
        for support in self.supports:
            yield {d: np.unpackbits(packed, count=self.n_links).astype(bool)
                   for d, packed in support.items()}


def solve_inclusion(net: dict, od: np.ndarray, mu: float, settings: Settings = DEFAULTS,
                    checkpoint=None, deadline: float | None = None,
                    progress=None, observer=None) -> InclusionResult:
    """Average the flow while rebuilding the support at the current cost.

    The iterate is ``x <- x + (y - x) / k`` with ``y`` the loading on the strict efficient support of the
    current cost, so ``y`` lies in the equilibrium correspondence at ``x`` by construction and the scheme
    is a stochastic-approximation of the associated differential inclusion. Neither the cost averaging nor
    the dead band of :func:`solve_phase1` is applied here, because both would break that correspondence.

    Distinct supports seen in the trailing window are returned. One support means it settled; several mean
    the ties are unresolved and the equilibrium is a mixture, and their intersection is the base support
    every resolution contains.

    The scheme stops when a whole window passes without a support it has not seen before. That is the
    criterion the window already defines, and it is reachable in finite time. The flow tolerance is kept
    as a secondary test, but it is not the criterion: harmonic averaging converges at rate ``1 / k`` and
    will not reach a tolerance of ``1e-9`` on a congested network, so a scheme stopping only on it stops
    on its iteration limit, which is a budget and not an answer.

    ``observer`` is called with ``(iteration, packed, move)`` after each support is formed, where
    ``packed`` maps a destination to its packed mask. It exists so the loop can be instrumented from
    outside without a second copy of it living in a diagnostic script; nothing in the scheme reads it.

    The window holds packed supports rather than the potential tables that induced them. Every distinct
    support ever seen is remembered by its key; what is kept for those inside the window is the masks
    themselves, which are the only thing any consumer reads. There is no cap on how many: capping it
    would make the scheme stop for want of memory rather than for want of an answer, and on the largest
    cell of the certificate tier the packed window is tens of megabytes.
    """
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    window = settings.inclusion_window
    demanded = [d for d in range(topo.n_zones) if od[:, d].sum() > 0.0]
    # Every support ever seen, by a digest of it, so a support returning after a long absence is
    # recognised rather than counted as new. The digest and not the support: a packed support is 139 KiB
    # on a network with four hundred destinations, and this dictionary is never pruned, so keeping the
    # supports themselves cost fifty-eight gigabytes at four hundred thousand iterations and killed the
    # run. Sixteen bytes collide with probability about 1e-27 over a million supports, which is far below
    # the floating-point noise the supports are being told apart by.
    #
    # What is held inside the window is the packed support itself, because the only consumer of the
    # window reads masks off it. That is bounded by the window length and is a few tens of megabytes.
    first_seen: dict[bytes, int] = {}
    recent: dict[bytes, tuple[dict, int]] = {}
    seen_before = 0

    x = np.zeros(arrays.n_links)
    move = float("inf")
    started = perf_counter()
    last_progress = started
    converged, stopped_on, iteration = False, "iteration limit", 0
    required = settings.envelope_stable_windows
    envelope = None if required is None else _EnvelopeWatch(window, int(required))
    first_iteration, resumed_from = 1, None

    periodic = PeriodicCheckpoint(checkpoint, settings.checkpoint_seconds)
    stored = load_checkpoint(checkpoint) if checkpoint is not None else None
    if stored is not None and int(stored.get("phase", 1)) == 3:
        # The window is not carried: it is the last few hundred iterations and is rebuilt in that many.
        # The flow is what took the time.
        x = stored["x"]
        first_iteration = int(stored["iteration"]) + 1
        resumed_from = first_iteration - 1
        seen_before = int(stored["seen_before"]) if "seen_before" in stored else 0
        if progress is not None:
            progress(f"resumed the inclusion scheme at iteration {first_iteration} with "
                     f"{seen_before} distinct supports already seen; the window is rebuilt over the "
                     f"next {window} iterations")

    # No iteration count by default: the run is bounded by its wall clock, and its state is checkpointed
    # so a cell too long for one job continues in the next.
    limit = settings.max_phase1_iterations
    for iteration in (count(first_iteration) if limit is None
                      else range(first_iteration, limit + 1)):
        cost = arrays.cost(np.maximum(x, 0.0))
        potential = potentials(topo, cost)
        y = load(topo, od, mu, cost, potential)
        following = x + (y - x) / iteration
        move = float(np.max(np.abs(following - x)))
        x = following

        # Hashed as the masks are built, so the concatenation of every destination's mask is never
        # materialised: on a large network that temporary was itself a sixth of a megabyte per iteration.
        hasher = blake2b(digest_size=16)
        packed = {}
        for d in demanded:
            bits = np.packbits(active_for_dest(topo, potential[:, d], d))
            packed[d] = bits
            hasher.update(bits.tobytes())
        key = hasher.digest()
        first_seen.setdefault(key, iteration)
        recent[key] = (packed, iteration)
        if observer is not None:
            observer(iteration, packed, move)
        if envelope is not None:
            envelope.see(iteration, packed)
            if envelope.held >= envelope.required:
                converged, stopped_on = True, "envelope settled"
                break
        for stale in [k for k, (_, last) in recent.items() if last <= iteration - window]:
            del recent[stale]

        newest = max(first_seen.values())
        if iteration - newest >= window:
            converged, stopped_on = True, "support set settled"
            break
        if iteration > 1 and move <= settings.flow_tolerance * max(float(np.abs(x).max()), 1.0):
            converged, stopped_on = True, "flow tolerance"
            break

        now = perf_counter()
        if progress is not None and now - last_progress >= settings.progress_seconds:
            progress(f"inclusion iteration {iteration}, move {move:.3e}, "
                     f"{seen_before + len(first_seen)} distinct "
                     f"supports, {len(recent)} in the window, newest at {newest}, "
                     f"{_resident_mb():.0f} MiB resident, {now - started:.0f}s elapsed")
            last_progress = now
        if periodic.due():
            # The count of distinct supports travels with the state. Without it a resumed cell reports
            # only what it saw since the last kill, which understates the evidence for the one thing this
            # scheme is asked about: whether the support set ever settles.
            periodic.write(phase=3, x=x, iteration=np.int64(iteration),
                           seen_before=np.int64(seen_before + len(first_seen)))
        if deadline is not None and now >= deadline:
            stopped_on = "time budget"
            break

    if not converged and checkpoint is not None:
        save_checkpoint(checkpoint, phase=3, x=x, iteration=np.int64(iteration),
                        seen_before=np.int64(seen_before + len(first_seen)))

    if not converged and stopped_on == "iteration limit" and settings.max_phase1_iterations is None:
        stopped_on = "support set did not settle"
    if not converged and stopped_on == "iteration limit":
        # The budget is exhausted and the support set never stopped changing. That is a verdict about the
        # cell and not a stop that resuming would get past, so it is named as one.
        stopped_on = "support set did not settle"
    return InclusionResult(x=x, supports=[support for support, _ in recent.values()],
                           potential=potential, n_links=topo.n_links,
                           converged=converged, stopped_on=stopped_on,
                           iterations=iteration, final_move=move, resumed_from=resumed_from,
                           n_supports_seen=seen_before + len(first_seen),
                           envelope_base=None if envelope is None
                           else getattr(envelope, "settled_base", None),
                           envelope_union=None if envelope is None
                           else getattr(envelope, "settled_union", None),
                           envelope_stable_from=None if envelope is None else envelope.stable_from,
                           envelope_last_change=None if envelope is None else envelope.last_change,
                           envelope_windows_held=0 if envelope is None else envelope.held,
                           wall_seconds=perf_counter() - started)


def polish(net: dict, od: np.ndarray, mu: float, x0: np.ndarray, potential: np.ndarray,
           settings: Settings = DEFAULTS, parallel: bool | None = None,
           checkpoint=None, deadline: float | None = None,
           progress=None) -> PolishResult:
    """Solve the fixed point on a frozen support.

    The residual is relative and is measured against the frozen support, which is not by itself a
    certificate: it says the flow is consistent with the support the solver held, not with the support
    the flow's own cost induces. :mod:`endogenous_sue.certificate` evaluates the latter.
    """
    from scipy.optimize import newton_krylov

    try:
        from scipy.optimize import NoConvergence
    except ImportError:
        from scipy.optimize.nonlin import NoConvergence

    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    if parallel is None:
        parallel = _use_parallel(net)
    cache: dict = {}

    def loaded(x):
        cost = arrays.cost(np.maximum(x, 0.0))
        if parallel:
            return load_parallel(topo, od, mu, cost, potential)
        return load(topo, od, mu, cost, potential, cache=cache, reuse_support=True)

    # The best iterate is tracked in memory and written only where the run stops short: the solver may
    # make no progress for several evaluations, and resuming from a worse point than one already reached
    # would undo work.
    best = {"residual": np.inf, "x": np.asarray(x0, dtype=float).copy()}
    started = perf_counter()
    last_progress = [started]
    evaluations = [0]

    # SciPy grades its own termination in the max-norm, while this function, the certificate and the
    # paper all grade in the relative 2-norm, which is larger by up to sqrt(|E|). Left alone the solver
    # stops on a criterion up to 137x looser than the one applied to its answer, and a solve that met its
    # own target is recorded as a failure. The scale travels with the iterate so that the norm handed to
    # SciPy is the same object the answer is graded by.
    scale = [max(float(np.linalg.norm(x0)), 1e-30)]

    def relative_norm(v):
        return float(np.linalg.norm(v)) / scale[0]

    def residual(x):
        now = perf_counter()
        evaluations[0] += 1
        difference = x - loaded(x)
        scale[0] = max(float(np.linalg.norm(x)), 1e-30)
        relative = float(np.linalg.norm(difference)) / scale[0]
        if relative < best["residual"]:
            best["residual"], best["x"] = relative, np.maximum(x, 0.0)
        if progress is not None and now - last_progress[0] >= settings.progress_seconds:
            progress(f"phase 2 evaluation {evaluations[0]}, residual {relative:.3e}, "
                     f"{now - started:.0f}s elapsed")
            last_progress[0] = now
        if periodic.due():
            periodic.write(phase=2, x=best["x"], relative=best["residual"], potential=potential)
        if deadline is not None and now >= deadline:
            raise DeadlineReached
        return difference

    periodic = PeriodicCheckpoint(checkpoint, settings.checkpoint_seconds)
    resumed = load_checkpoint(checkpoint) if checkpoint is not None else None
    if resumed is not None and int(resumed.get("phase", 1)) == 2:
        x0 = resumed["x"]
        best["residual"], best["x"] = float(resumed["relative"]), resumed["x"]
        if progress is not None:
            progress(f"resumed phase 2 from residual {best['residual']:.3e}")

    tolerance = settings.residual_tolerance * settings.residual_margin
    method = "newton_krylov"
    try:
        x = newton_krylov(residual, x0, f_tol=tolerance, tol_norm=relative_norm,
                          maxiter=settings.max_newton_iterations, method="lgmres")
    except NoConvergence as exc:
        x = np.asarray(exc.args[0])
        method = "newton_krylov_incomplete"
    except DeadlineReached:
        x, method = best["x"], "time budget"
    except ValueError:
        # The Jacobian is singular where the cost clamp binds or the flow is zero, because the cost
        # derivative vanishes there. That is a property of the cost function rather than a failure, and
        # the Krylov step cannot proceed. The frozen-support map is averaged-contractive whatever the
        # derivative, so damped averaging followed by plain iteration converges. The method actually used
        # is returned, not substituted silently.
        method = "damped_averaging"
        x = np.asarray(x0, dtype=float).copy()
        for k in range(1, 400):
            x = x + (loaded(x) - x) / k
            if periodic.due():
                periodic.write(phase=2, x=x, relative=best["residual"], potential=potential)
        rounds = (count() if settings.max_newton_iterations is None
                  else range(2 * settings.max_newton_iterations))
        for _ in rounds:
            following = loaded(x)
            if (float(np.linalg.norm(following - x))
                    <= tolerance * max(float(np.linalg.norm(x)), 1e-30)):
                x = following
                break
            x = following

    def graded(candidate):
        candidate = np.maximum(candidate, 0.0)
        return candidate, float(np.linalg.norm(loaded(candidate) - candidate)
                                / max(float(np.linalg.norm(candidate)), 1e-30))

    x, relative = graded(x)
    if relative > settings.residual_tolerance and best["residual"] < relative:
        # The iterate the solver stopped on is not always the best it passed through, and the grade is
        # taken after the clip, which the tracked residual was not. Only worth re-evaluating where the
        # answer has already failed, because that is the only case in which the better point changes it.
        candidate, candidate_relative = graded(best["x"])
        if candidate_relative < relative:
            x, relative = candidate, candidate_relative

    converged = bool(method != "time budget" and relative <= settings.residual_tolerance)
    if not converged and checkpoint is not None:
        # Phase two used to write a checkpoint only where the wall clock stopped it. A cell that ran to
        # the solver's own termination and was then graded above tolerance saved nothing, so the run
        # that said "to be continued" restarted it from the beginning and failed identically: a loop
        # that made no progress. Whatever stopped it, the best point reached is worth resuming from.
        save_checkpoint(checkpoint, phase=2, x=x, relative=relative, potential=potential)
    return PolishResult(x=x, residual=relative, method=method, converged=converged)


def _polish_stop(polished: PolishResult) -> str:
    """Why phase two ended, in the same vocabulary phase one uses."""
    if polished.method == "time budget":
        return "time budget"
    if polished.converged:
        return "polished"
    return f"phase 2 above tolerance ({polished.method})"


def solve(net: dict, od: np.ndarray, mu: float, settings: Settings = DEFAULTS,
          rule: SupportRule = SupportRule.DAMPED, parallel: bool | None = None,
          tiebreak: np.ndarray | None = None, tiebreak_sign: float = 0.0,
          checkpoint=None, deadline: float | None = None,
          progress=None) -> Equilibrium:
    """Freeze the support by damping, then polish the flow on it.

    ``checkpoint`` is a path prefix. The two phases write beside it and resume from it independently, so
    a run stopped by a wall-clock limit continues from where it reached rather than from the beginning.
    """
    stored = load_checkpoint(checkpoint) if checkpoint is not None else None
    if stored is not None and int(stored.get("phase", 1)) == 2:
        # Phase one finished in an earlier run. Its potentials travel in the checkpoint, so the support
        # is not rebuilt and phase two continues from the iterate it had reached.
        phase1 = Phase1Result(x=stored["x"], potential=stored["potential"], converged=True,
                              stopped_on="restored from checkpoint", iterations=0)
        polished = polish(net, od, mu, phase1.x, phase1.potential, settings,
                          parallel=parallel, checkpoint=checkpoint, deadline=deadline,
                          progress=progress)
        return Equilibrium(x=polished.x, residual=polished.residual,
                           converged=polished.converged,
                           stopped_on=_polish_stop(polished), polish_method=polished.method,
                           phase1=phase1, settings=settings)

    phase1 = solve_phase1(net, od, mu, settings, rule=rule, parallel=parallel,
                          tiebreak=tiebreak, tiebreak_sign=tiebreak_sign,
                          checkpoint=checkpoint, deadline=deadline, progress=progress)
    if not phase1.converged:
        # Phase two is not started. Polishing a support that has not settled drives the flow to machine
        # precision against a support that is still going to move, which costs the time of the polish and
        # buys a residual measured against the wrong thing. The cell is unfinished, its state is
        # checkpointed, and it says so.
        return Equilibrium(x=phase1.x, residual=float("nan"), converged=False,
                           stopped_on=phase1.stopped_on, polish_method="not reached",
                           phase1=phase1, settings=settings)
    polished = polish(net, od, mu, phase1.x, phase1.potential, settings,
                      parallel=parallel, checkpoint=checkpoint, deadline=deadline,
                      progress=progress)
    return Equilibrium(x=polished.x, residual=polished.residual,
                       converged=bool(phase1.converged and polished.converged),
                       stopped_on=(phase1.stopped_on if polished.converged
                                   else _polish_stop(polished)),
                       polish_method=polished.method, phase1=phase1, settings=settings)
