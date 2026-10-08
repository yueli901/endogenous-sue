"""Simplicial fixed-point search on a box: the guaranteed branch.

The face solve is a search for weights satisfying the complementarity conditions and has no convergence
theorem, so on its own it can terminate without a certificate. A simplicial fixed-point method has one
unconditionally: it is the constructive content of Brouwer's theorem and needs no monotonicity, no
P-matrix property and no diagonal dominance. The guaranteed method is therefore the parent here, and the
face solve is an accelerator layered on it.

The search dimension is the number of tied pairs rather than the number of links, which is tens rather
than thousands on the corpus, and simplicial methods are run routinely at that dimension.

The box variational inequality is exactly the fixed-point problem

    w = Psi(w),   Psi(w) = clip(w + g(w), 0, 1),

since the residual is the negated gap and the clip is the projection onto the box. The three cases are the
complementarity conditions verbatim: an interior ``w`` with zero gap is fixed, ``w = 0`` with non-positive
gap clips back to zero, and ``w = 1`` with non-negative gap clips back to one.

The method is Merrill's restart algorithm on the slab. The search space is the unit cube crossed with the
homotopy level, triangulated by the Freudenthal triangulation restricted to the base level. That
triangulation increments the level exactly once along a simplex's vertex chain, so every simplex has its
vertices on the two end faces only, which is Merrill's two-level triangulation obtained for free.

A vertex carries the label ``centre - x`` at level zero, the artificial map whose only zero is the restart
centre, and ``Psi(clip x) - x`` at level one, the real map. A facet is completely labelled when some
convex combination of its labels vanishes. On the base face the completely labelled facets are exactly the
simplices containing the restart centre, so there is one and it is available in closed form. The path
leaves it, cannot revisit it, cannot cycle under the lexicographic rule, and cannot run away, since
outside the box every label points back in; so it terminates on the far face, where the completely
labelled facet gives an approximate fixed point whose error is one mesh. Restarting halves the mesh and
re-centres, and accumulation points are exact fixed points by continuity and compactness.

Two invariants make the walk safe. Every label column has leading entry one, so the pivot direction sums
to one and the ratio test can never be unblocked: an unbounded ray is impossible rather than merely
unlikely. And within one simplex each coordinate takes only two consecutive grid values, so a completely
labelled facet lies within one mesh of the box and the base index cannot wander.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import count
from time import perf_counter

import numpy as np

from ..config import LEX_TOLERANCE, PIVOT_TOLERANCE, DeadlineReached

__all__ = ["SimplicialResult", "simplicial_fixed_point", "walk_one_level"]


@dataclass
class SimplicialResult:
    """Outcome of a restart search."""

    w: np.ndarray
    residual: float
    evaluations: int
    pivots: int
    levels: int
    status: str
    mesh: float
    simplex: tuple | None = None
    perturbed_levels: int = 0
    trail: list = field(default_factory=list)
    resumed_from: int | None = None


def _simplex_vertices(base, order):
    """The Freudenthal simplex: start at ``base``, then add one unit coordinate at a time."""
    n = len(base)
    vertices = np.empty((n + 1, n), dtype=np.int64)
    vertices[0] = base
    for i in range(n):
        vertices[i + 1] = vertices[i]
        vertices[i + 1, order[i]] += 1
    return vertices


def _pivot(base, order, leaving):
    """Replace one vertex. The three Freudenthal rules, with a zero-indexed permutation."""
    n = len(base)
    base = base.copy()
    order = list(order)
    if leaving == 0:
        base[order[0]] += 1
        order = order[1:] + order[:1]
    elif leaving == n:
        base[order[n - 1]] -= 1
        order = order[n - 1:] + order[:n - 1]
    else:
        order[leaving - 1], order[leaving] = order[leaving], order[leaving - 1]
    return base, order


def _lexicographic_ratio(basis_inverse, basic_solution, direction):
    """Lexicographic minimum ratio test. Returns the leaving row, or -1 for an unbounded ray.

    Ties are broken by the rows of the basis inverse, which is the standard right-hand-side perturbation
    argument. It makes the pivot sequence unique, and uniqueness is what lets the door-in-door-out
    argument guarantee that no facet is visited twice.
    """
    rows = len(basic_solution)
    best, candidates = None, []
    for i in range(rows):
        if direction[i] > PIVOT_TOLERANCE:
            ratio = basic_solution[i] / direction[i]
            if best is None or ratio < best - LEX_TOLERANCE:
                best, candidates = ratio, [i]
            elif abs(ratio - best) <= LEX_TOLERANCE:
                candidates.append(i)
    if best is None:
        return -1
    if len(candidates) == 1:
        return candidates[0]
    for column in range(rows):
        values = [basis_inverse[i, column] / direction[i] for i in candidates]
        smallest = min(values)
        kept = [c for c, value in zip(candidates, values) if value <= smallest + LEX_TOLERANCE]
        if len(kept) == 1:
            return kept[0]
        candidates = kept
    return candidates[0]


def _starting_simplex(centre, n, grid):
    """The unique completely labelled facet on the base face, in closed form.

    Writing ``grid * centre`` as an integer part plus a fraction, sorting the fraction decreasing names
    the simplex containing the centre, and the barycentric coordinates are the consecutive differences of
    the sorted fraction, which are non-negative by construction. The centre is first nudged off the grid;
    the nudge only decides which cell is chosen, and the nudged centre is the one the labels use, so those
    coordinates stay exactly non-negative.
    """
    scaled = np.asarray(centre, dtype=float) * grid
    scaled = scaled + 1e-7 * (((np.arange(n) + 1) * 0.6180339887498949) % 1.0 + 0.5)
    integral = np.floor(scaled).astype(np.int64)
    order = np.argsort(-(scaled - integral), kind="stable")
    base = np.zeros(n + 1, dtype=np.int64)
    base[:n] = integral
    return base, [int(j) for j in order] + [n], scaled / grid


def _vertex_noise(key: bytes, n: int) -> np.ndarray:
    """A deterministic offset attached to a lattice point rather than to a pivot step.

    Genericity has to be a property of the labelling and not of the walk: if the offset changed between
    visits to one vertex, the label would stop being a function of the vertex and door-in-door-out would
    lose its meaning. Hashing the vertex's own bytes gives an offset fixed for all time and uncorrelated
    across vertices, which is what breaks the exact coincidences that make a basis singular.
    """
    needed = 8 * n
    buffer = b""
    block = 0
    while len(buffer) < needed:
        buffer += hashlib.blake2b(key + block.to_bytes(4, "little"), digest_size=64).digest()
        block += 1
    uniform = np.frombuffer(buffer[:needed], dtype=np.uint64).astype(np.float64) / float(1 << 64)
    return uniform - 0.5


def walk_one_level(psi, n, grid, centre, max_pivots, counter, jitter=0.0, deadline=None):
    """One Merrill restart. ``max_pivots`` of ``None`` lets the walk run until the door-in-door-out
    argument closes it, which it does in finite time with no polynomial bound on when.

    ``deadline`` is checked every few hundred pivots rather than every one, because reading a clock is
    comparable in cost to a pivot. Without it a budget can only be honoured between levels, and a level
    that outlasts the whole budget makes the budget advisory on exactly the faces where it matters.
    """
    """Follow the path at one mesh. Returns the point, the pivot count, a status and a certificate."""
    size = n + 1
    right_hand_side = np.zeros(size)
    right_hand_side[0] = 1.0
    labels: dict = {}

    base, order, artificial_centre = _starting_simplex(centre, n, grid)
    entering = size

    def label(vertex):
        key = vertex.tobytes()
        stored = labels.get(key)
        if stored is None:
            point = vertex[:n].astype(float) / grid
            if vertex[n] == 0:
                stored = artificial_centre - point
            else:
                counter[0] += 1
                stored = np.asarray(psi(np.clip(point, 0.0, 1.0)), dtype=float) - point
            if jitter > 0.0:
                stored = stored + jitter * _vertex_noise(key, n)
            labels[key] = stored
        return stored

    def barycentre(columns, vertices, dropped):
        kept = [i for i in range(size + 1) if i != dropped]
        try:
            weights = np.linalg.solve(columns[:, kept], right_hand_side)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(columns[:, kept], right_hand_side, rcond=None)[0]
        weights = np.clip(weights, 0.0, None)
        total = weights.sum()
        weights = weights / total if total > 0 else np.full(size, 1.0 / size)
        points = np.asarray([vertices[i][:n] for i in kept], dtype=float) / grid
        return np.clip(weights @ points, 0.0, 1.0), np.clip(points, 0.0, 1.0), weights

    for step in (count() if max_pivots is None else range(max_pivots)):
        if deadline is not None and (step & 0x3FF) == 0 and perf_counter() >= deadline:
            raise DeadlineReached(f"pivot {step} of the current level")
        vertices = _simplex_vertices(base, order)
        level_split = order.index(n)
        columns = np.empty((size, size + 1))
        columns[0, :] = 1.0
        for i in range(size + 1):
            columns[1:, i] = label(vertices[i])
        basic = [i for i in range(size + 1) if i != entering]
        try:
            basis_inverse = np.linalg.inv(columns[:, basic])
        except np.linalg.LinAlgError:
            # Two vertices carry the same label, so the basis has no inverse and the walk has no next
            # door. The theorem holds for any continuous labelling, so the caller retries the level with
            # a generic perturbation, which enlarges the returned error by at most its own size.
            return None, step + 1, "singular", None
        if not np.all(np.isfinite(basis_inverse)):
            return None, step + 1, "singular", None

        leaving = _lexicographic_ratio(basis_inverse, basis_inverse @ right_hand_side,
                                       basis_inverse @ columns[:, entering])
        if leaving < 0:
            return None, step + 1, "ray", None
        leaving_vertex = basic[leaving]
        if level_split == 0 and leaving_vertex == 0:
            point, points, weights = barycentre(columns, vertices, leaving_vertex)
            return point, step + 1, "labelled", (points, weights)
        if level_split == size - 1 and leaving_vertex == size:
            return None, step + 1, "boundary", None

        base, order = _pivot(base, order, leaving_vertex)
        # Door-in-door-out: the next entering column is the vertex the pivot just created. Under the two
        # rotating rules that vertex does not keep the leaving index -- dropping the first appends the new
        # point at the far end, and dropping the last prepends it.
        entering = size if leaving_vertex == 0 else (0 if leaving_vertex == size else leaving_vertex)
        if base[:n].min() < -2 or base[:n].max() > grid + 2:
            return None, step + 1, "escaped", None
    return None, (max_pivots or 0), "pivot limit", None


def simplicial_fixed_point(psi, n, w0=None, levels=24, grid=4, tolerance=1e-10,
                           max_pivots=200_000, residual=None, min_mesh=1e-13,
                           progress=None, deadline=None, checkpoint=None,
                           checkpoint_seconds=600.0, memory_ceiling_mb=None) -> SimplicialResult:
    """Find ``w`` in the unit cube with ``w = psi(w)``, by Merrill restart of a simplicial search.

    ``residual`` is an optional acceptance test, defaulting to the sup-norm of ``psi(w) - w``. The face
    solver passes its own complementarity residual so that a simplicial answer is accepted on exactly the
    same terms as one from the accelerator.

    ``progress`` is an optional callable receiving one line per completed level. A single call can run for
    hours, and without it a slow search cannot be told from a stalled one.

    Termination is unconditional at every level. The restart loop stops on the tolerance, on the minimum
    mesh, or after the given number of halvings, and says which.

    ``deadline`` stops it between restarts, which is the only boundary it has: a level is one walk and has
    no interior state. It returns the best point reached with status ``time budget``, which is a stop and
    not an answer -- the acceptance test is the residual and nothing here weakens it.

    ``checkpoint`` makes that stop resumable. The state of a restart loop is the best point so far and the
    mesh it was found at, and a resumed search continues from that point at that mesh rather than from the
    centre of the cube at the coarsest one. Without it a face too long for one job never finishes however
    many jobs it is given, because every job repeats the levels the last one completed. The dimension is
    stored with the state and a checkpoint of a different width is ignored: the ladder's rungs search in
    different numbers of unknowns and must not read each other's answers.
    """
    if n == 0:
        return SimplicialResult(np.zeros(0), 0.0, 0, 0, 0, "trivial", 0.0)

    measure = residual or (lambda w: float(np.max(np.abs(np.asarray(psi(w), dtype=float) - w))))
    w = np.full(n, 0.5) if w0 is None else np.clip(np.asarray(w0, dtype=float), 0.0, 1.0)
    counter = [0]
    best_w, best_residual = w.copy(), measure(w)
    total_pivots, used, status, mesh = 0, 0, "levels", int(grid)
    perturbed, certificate, trail = 0, None, []
    from ..equilibrium import PeriodicCheckpoint, _resident_mb

    first_level, resumed_from = 0, None
    periodic = PeriodicCheckpoint(checkpoint, checkpoint_seconds)

    stored = _load_search(checkpoint, n)
    if stored is not None:
        best_w, best_residual, first_level, mesh, total_pivots = stored
        resumed_from = first_level
        if progress is not None:
            progress(f"resumed the search at level {first_level}, mesh 1/{mesh}, "
                     f"residual {best_residual:.3e}")

    for level in range(first_level, levels):
        used = level
        if best_residual <= tolerance or 1.0 / mesh <= min_mesh:
            status = "converged" if best_residual <= tolerance else "mesh"
            break
        if periodic.due():
            _save_search(checkpoint, n, best_w, best_residual, level, mesh, total_pivots)
            periodic.last = perf_counter()
        if memory_ceiling_mb is not None and _resident_mb() > memory_ceiling_mb:
            status = "memory ceiling"
            _save_search(checkpoint, n, best_w, best_residual, level, mesh, total_pivots)
            break
        if deadline is not None and perf_counter() >= deadline:
            status = "time budget"
            _save_search(checkpoint, n, best_w, best_residual, level, mesh, total_pivots)
            break
        jitter = 0.0
        for _ in range(6):
            try:
                point, pivots, outcome, found = walk_one_level(psi, n, mesh, best_w, max_pivots,
                                                               counter, jitter, deadline=deadline)
            except DeadlineReached:
                status = "time budget in pivot"
                _save_search(checkpoint, n, best_w, best_residual, level, mesh, total_pivots)
                point, outcome = None, status
                break
            total_pivots += pivots
            if outcome != "singular":
                break
            # Escalate the genericity perturbation. It is scaled to the map, and the accepted point is
            # still tested against the unperturbed map below, so nothing is accepted on trust. This sat
            # after a `break` and was unreachable: a singular level was retried six times with the same
            # zero jitter, which is the same walk six times, and `perturbed_levels` could only ever be
            # zero. No landed record has ever reported a perturbed level, which is consistent with the
            # escalation never having run rather than with singularity never arising.
            jitter = 1e-12 if jitter == 0.0 else jitter * 100.0
            perturbed += 1
        if status == "time budget in pivot":
            used = level
            break
        trail.append(dict(level=level, mesh=1.0 / mesh, pivots=pivots, status=outcome))
        if point is None:
            status = outcome
            break
        here = measure(point)
        certificate = found
        if here < best_residual:
            best_w, best_residual = point.copy(), here
        if progress is not None:
            progress(f"level {level + 1}/{levels} mesh=1/{mesh} pivots={total_pivots} "
                     f"residual={best_residual:.3e}")
        mesh *= 2
    else:
        used = levels
        status = "converged" if best_residual <= tolerance else "levels"

    # Every status that wrote state keeps it. The guard named one of them and the search reports three:
    # a budget crossed between levels is "time budget", one crossed inside a pivot walk is "time budget
    # in pivot", and the memory ceiling is its own. The latter two each save their state and then had it
    # deleted three lines later, so a rung that ran out of wall left nothing to resume from and an
    # eleven-hour search could only ever be repeated.
    if status not in ("time budget", "time budget in pivot", "memory ceiling"):
        _clear_search(checkpoint, n)
    return SimplicialResult(w=best_w, residual=best_residual, evaluations=counter[0],
                            pivots=total_pivots, levels=used, status=status, mesh=1.0 / mesh,
                            simplex=certificate, perturbed_levels=perturbed, trail=trail,
                            resumed_from=resumed_from)


def _save_search(checkpoint, n, best_w, best_residual, level, mesh, pivots) -> None:
    """Write the restart loop's own state: the best point, and the mesh it was reached at."""
    if checkpoint is None:
        return
    from ..equilibrium import save_checkpoint

    save_checkpoint(checkpoint, phase=5, n=np.int64(n), w=best_w,
                    residual=np.float64(best_residual), level=np.int64(level),
                    mesh=np.int64(mesh), pivots=np.int64(pivots))


def _load_search(checkpoint, n):
    """Read it back, or ``None`` where there is none for this phase and this width."""
    if checkpoint is None:
        return None
    from ..equilibrium import load_checkpoint

    stored = load_checkpoint(checkpoint)
    if stored is None or int(stored.get("phase", 0)) != 5 or int(stored["n"]) != n:
        return None
    return (stored["w"], float(stored["residual"]), int(stored["level"]),
            int(stored["mesh"]), int(stored["pivots"]))


def _clear_search(checkpoint, n) -> None:
    """Remove this search's finished state, so a later run does not resume into a completed one.

    Only this search's own state. A rung that refines its classes runs the search twice at different
    widths through one checkpoint path: the first descent reads the file, rejects it on the width guard
    because it belongs to the second, and then used to delete it. So the second descent -- the one the
    refinement exists to perform, and the one that had eleven hours of progress stored -- began again from
    the coarsest mesh every time, and a resumed run repeated its predecessor level for level. A file
    whose width is not this search's is not this search's to remove.
    """
    if checkpoint is None:
        return
    from pathlib import Path

    from ..equilibrium import load_checkpoint

    path = Path(checkpoint)
    stored = load_checkpoint(path)
    if stored is not None and int(stored.get("phase", 0)) == 5 and int(stored["n"]) != n:
        return
    path.unlink(missing_ok=True)
