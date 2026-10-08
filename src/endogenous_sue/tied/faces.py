"""Solving one tied face.

Given the contested pairs and the base support, the equilibrium on the face is the fixed point

    x = sum over destinations of [ (1 - sum_j w_j) L(base) + sum_j w_j L(V_j) ]

together with one tie residual per contested pair. That is a square system in the flow and the weights,
solved matrix-free.

The vertex supports are parameterised by tied *group*, a group being a tied edge of which a support admits
exactly one orientation. Their marginals are free in the unit cube with no coupling constraint, so the
domain is a box. Where the efficiency gap does not depend on the destination the search may run on one
weight per contested link instead, and every fixed point of that reduced map lifts to a fixed point of the
full face map; the lift is checked at the point returned rather than assumed over the face.

A mask is never derived by perturbing a potential. Newton's own finite differences move the potentials by
the same order as any such perturbation, so the two would fight over the same digits and the Jacobian
would see supports flipping mid-evaluation. Every loading here receives the boolean mask it must use,
explicitly, and potentials are used only for what they are: the tie residuals.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, root

from ..config import ACTIVE_TOLERANCE, DEFAULTS, TIE_TOLERANCE, WEIGHT_EPSILON, DeadlineReached, Settings
from ..costs import bpr_cost
from ..loading import load_destination, load_prepared, prepare_masks
from ..network import Topology, unpack
from ..shortestpath import potentials
from ..subnetwork import kahn_order
from .atoms import (
    RingResidual,
    atom_mask,
    build_groups,
    cycle_in_mask,
    find_tied_cycle,
    is_acyclic,
    nested_atoms,
)
from .contested import subset_family
from .simplicial import simplicial_fixed_point

__all__ = ["convex_certificate", "solve_square_system", "solve_face_nested", "solve_face_ladder",
           "solve_by_bisection"]



def convex_certificate(topo: Topology, od, mu, cost, x, demanded, base, tied):
    """Per-destination fit of ``x`` by loadings over the subset family: the convex representation.

    The square solve pins the flow but returns only an affine weight representation, because duplicated
    tie rows leave the weights underdetermined. What a relaxed equilibrium asserts is membership of the
    flow in the per-destination convex hulls, and this linear programme either exhibits that or fails, at
    the already-exact final cost.
    """
    links_by_destination: dict = {}
    for d, link in tied:
        links_by_destination.setdefault(d, []).append(link)

    n_links = topo.n_links
    remainder = np.asarray(x, dtype=float).copy()
    columns, labels, blocks = [], [], []
    for d in demanded:
        links = links_by_destination.get(d, [])
        if not links:
            remainder -= load_destination(topo, od, mu, cost, d, base[d], kahn_order(topo, base[d]))
            continue
        family = subset_family(topo, base[d], links)
        if family is None:
            return dict(feasible=False, status="family too large", residual=float("inf"))
        start = len(columns)
        for added, mask in family:
            columns.append(load_destination(topo, od, mu, cost, d, mask, kahn_order(topo, mask)))
            labels.append((d, list(map(int, added)), mask))
        blocks.append((start, len(columns)))

    if not columns:
        return dict(feasible=True, status="no tied pairs", residual=0.0, masks=[], weighted=[])

    loadings = np.column_stack(columns)
    n_columns = len(columns)
    objective = np.concatenate([np.zeros(n_columns), np.ones(n_links)])
    inequality = np.vstack([np.hstack([loadings, -np.eye(n_links)]),
                            np.hstack([-loadings, -np.eye(n_links)])])
    equality = np.zeros((len(blocks), n_columns + n_links))
    for row, (start, stop) in enumerate(blocks):
        equality[row, start:stop] = 1.0

    solution = linprog(objective, A_ub=inequality,
                       b_ub=np.concatenate([remainder, -remainder]), A_eq=equality,
                       b_eq=np.ones(len(blocks)),
                       bounds=[(0, None)] * (n_columns + n_links), method="highs")
    if not solution.success:
        return dict(feasible=False, status=str(solution.message)[:80], residual=float("inf"))

    weights = solution.x[:n_columns]
    residual = float(np.abs(remainder - loadings @ weights).max()
                     / max(float(np.abs(x).max()), 1e-30))
    used = [j for j in range(n_columns) if weights[j] > ACTIVE_TOLERANCE]
    return dict(feasible=True, status="solved", residual=residual,
                masks=[[labels[j][0], labels[j][1], float(weights[j])] for j in used],
                weighted=[(labels[j][0], labels[j][2]) for j in used])


def solve_square_system(topo: Topology, arrays, od, mu, demanded, base, vertices, tied,
                        x, w0, scale, tolerance, max_iterations):
    """One matrix-free solve of the square system in the flow and the weights, with fixed masks.

    ``vertices`` is a flat list of ``(destination, added links, mask)`` aligned with the weight vector.
    Masks never change inside the solve, so no finite-difference step of the Jacobian can flip one. Flow
    unknowns are rescaled so that both blocks sit on the same footing.
    """
    fft, cap, b, power, _, _ = arrays
    tail, head = topo.tail, topo.head
    n_links = topo.n_links
    if len(vertices) != len(tied):
        raise ValueError(f"system is not square: {len(vertices)} vertices against {len(tied)} tie rows")

    base_orders = {d: kahn_order(topo, base[d]) for d in demanded}
    vertex_orders = [kahn_order(topo, mask) for _, _, mask in vertices]
    by_destination: dict = {}
    for i, (d, _, _) in enumerate(vertices):
        by_destination.setdefault(d, []).append(i)

    def residual(unknowns):
        flow = unknowns[:n_links] * scale
        weights = unknowns[n_links:]
        cost = bpr_cost(np.maximum(flow, 0.0), fft, cap, b, power)
        potential = potentials(topo, cost)
        blended = np.zeros(n_links)
        for d in demanded:
            base_loading = load_destination(topo, od, mu, cost, d, base[d], base_orders[d])
            indices = by_destination.get(d, [])
            if not indices:
                blended += base_loading
                continue
            carried = 0.0
            for i in indices:
                blended += weights[i] * load_destination(topo, od, mu, cost, d, vertices[i][2],
                                                         vertex_orders[i])
                carried += weights[i]
            blended += (1.0 - carried) * base_loading
        out = np.empty(n_links + weights.shape[0])
        out[:n_links] = (flow - blended) / scale
        for i, (d, link) in enumerate(tied):
            out[n_links + i] = potential[tail[link], d] - potential[head[link], d]
        return out

    start = np.concatenate([x / scale, w0])
    try:
        solution = root(residual, start, method="krylov", tol=tolerance,
                        options=dict(fatol=tolerance, maxiter=max_iterations))
    except ValueError:
        # The Krylov step is singular where the cost clamp binds or the flow is zero, because the cost
        # derivative vanishes there. A method that forms no Jacobian is immune, at the cost of a slower
        # rate. Which one ran is visible in the returned solution object.
        solution = root(residual, start, method="df-sane",
                        options=dict(fatol=tolerance, maxfev=4000))
    return solution.x[:n_links] * scale, solution.x[n_links:].copy(), solution


def _blend_fixed_point(step_map, start, averaging=40, polish=60, use_krylov=True, tolerance=1e-13,
                       deadline=None):
    """Solve ``z = G(z)`` for a fixed weight vector. Returns the point and its sup-norm residual.

    Averaging first, because it converges here from any start, then a guarded polish. The polish must be
    guarded: plain iteration does not converge on the five-node witness, where the iterates leave a
    residual of ``2e-6`` and reach ``7.7e3``, and every quantity read off the result is then noise. A step
    is therefore taken only if it reduces the residual, and a matrix-free solve finishes the job because
    averaging alone stalls far above any tie tolerance.

    The residual returned is whatever the best iterate achieved. A caller needing more must check it.

    ``deadline`` is checked on every sweep. One of these solves is minutes on a wide face and a single
    weight vector needs one, so a budget that cannot interrupt it is a budget that cannot interrupt
    anything on the faces where it matters.
    """
    z = np.asarray(start, dtype=float).copy()
    for k in range(1, averaging + 1):
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached(f"blend averaging sweep {k}")
        z = z + (step_map(z) - z) / k
    residual = float(np.max(np.abs(step_map(z) - z)))

    for _ in range(polish):
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached("blend polish")
        if residual <= tolerance * max(1.0, float(np.max(np.abs(z)))):
            break
        candidate = step_map(z)
        candidate_residual = float(np.max(np.abs(step_map(candidate) - candidate)))
        if candidate_residual >= residual:
            break
        z, residual = candidate, candidate_residual

    if use_krylov and residual > tolerance * max(1.0, float(np.max(np.abs(z)))):
        try:
            solution = root(lambda u: step_map(u) - u, z, method="krylov",
                            options=dict(fatol=tolerance * max(1.0, float(np.max(np.abs(z)))),
                                         maxiter=200))
            point = np.asarray(solution.x, dtype=float)
            here = float(np.max(np.abs(step_map(point) - point)))
            if np.all(np.isfinite(point)) and here < residual:
                z, residual = point, here
        except (ValueError, RuntimeError, ZeroDivisionError):
            pass
    return z, residual


def solve_face_nested(topo, arrays, od, mu, demanded, base, tied, x0,
                      collapse=True, settings: Settings = DEFAULTS, tie_tolerance=TIE_TOLERANCE,
                      levels=None, grid=None, max_pivots=None, jacobian_probes=4,
                      max_refinements=None, progress=None, measure_dimension=False, deadline=None,
                      checkpoint=None):
    """Search the face on the cube, over the nested atoms, optionally collapsed by contested link.

    The unknown is one weight per tied group rather than per tied link, and the atoms are the nested
    masks. Their marginals are free in the unit cube with no coupling, so the domain is a box and a fixed
    point of the box map is a certificate; on a product of simplices no such statement is available, and
    a positive multiplier there would say only that no atom carrying mass is admissible.

    Collapsing runs the search on one weight per contested link, which is exact where the efficiency gap
    does not depend on the destination. That hypothesis is not a property of the whole face: the
    within-class spread moves continuously as the weights cross the cube. What is needed is the a
    posteriori reading, and it is checked here: if the spread vanishes at the point returned, the lift is
    exact there. Where it does not, the offending classes are split by their observed gaps and the search
    is repeated. Each split strictly increases the class count, which is bounded by the number of groups,
    so the loop ends, in the worst case at the full face.

    The step size mixes a gap, which is a cost, into a weight, which is dimensionless. It cannot move a
    fixed point, so its only role is conditioning the pivot path, and it is set from the measured Jacobian
    norm rather than from the largest gap, which is out by two orders of magnitude on the worst face of
    the corpus.
    """
    fft, cap, b, power, _, _ = arrays
    tail, head = topo.tail, topo.head
    n_links = topo.n_links
    max_pivots = settings.max_pivots if max_pivots is None else max_pivots
    levels = settings.simplicial_levels if levels is None else levels
    grid = settings.simplicial_grid if grid is None else grid
    max_refinements = (settings.max_class_refinements if max_refinements is None
                       else max_refinements)
    if not tied:
        raise ValueError("the face is empty")

    by_destination: dict = {}
    for i, (d, link) in enumerate(tied):
        by_destination.setdefault(int(d), []).append((int(i), int(link)))
    destinations = sorted(by_destination)

    groups_of, ring_of, slots_of, n_groups = {}, {}, {}, 0
    for d in destinations:
        groups = build_groups(topo, by_destination[d])
        groups_of[d] = groups
        ring_of[d] = find_tied_cycle(topo, base[d], groups)
        slots_of[d] = list(range(n_groups, n_groups + len(groups)))
        n_groups += len(groups)
    if n_groups == 0:
        raise ValueError("the face is empty")

    base_orders = {d: kahn_order(topo, base[d]) for d in demanded}
    mask_cache: dict = {}

    def mask_for(d, admit):
        key = (d, tuple(np.flatnonzero(admit).tolist()))
        stored = mask_cache.get(key)
        if stored is None:
            mask = atom_mask(base[d], groups_of[d], admit)
            if not is_acyclic(topo, mask):
                raise RingResidual(f"destination {d}: nested atom {key[1]} is cyclic")
            stored = (mask, kahn_order(topo, mask))
            mask_cache[key] = stored
        return stored

    def cells_for(d, weights, max_rings=8):
        """The nested atoms for one destination, cutting every ring found rather than only the first.

        The all-forward atom is not the only one that can be cyclic: a group that is not admitted
        contributes its reverse link, so an intermediate atom carries mixed orientations and can close a
        ring the all-forward atom does not. Refusing on the first cyclic atom would decline faces the
        construction can represent, so the ring is recovered from the offending atom and the family
        rebuilt. The ring learned is kept so that later evaluations start where this one ended.
        """
        ring = ring_of[d]
        for _ in range(max_rings):
            try:
                cells = nested_atoms(weights, ring)
            except ValueError as exc:
                raise RingResidual(f"destination {d}: {exc}") from exc
            cyclic = None
            for admit, mass in cells:
                if mass <= WEIGHT_EPSILON:
                    continue
                mask = atom_mask(base[d], groups_of[d], admit)
                if not is_acyclic(topo, mask):
                    cyclic = (admit, mask)
                    break
            if cyclic is None:
                ring_of[d] = ring
                return cells
            found = cycle_in_mask(topo, cyclic[1], groups_of[d])
            if not found or found == ring:
                raise RingResidual(
                    f"destination {d}: atom {np.flatnonzero(cyclic[0]).tolist()} is cyclic and its "
                    f"ring {found} was already cut ({ring})")
            ring = found
        raise RingResidual(f"destination {d}: more than {max_rings} distinct rings")

    seed = np.asarray(x0, dtype=float).copy()
    memo: dict = {}
    evaluations = [0]
    started = time.perf_counter()

    def blend(weights):
        key = weights.tobytes()
        stored = memo.get(key)
        if stored is not None:
            return stored
        cells = {d: cells_for(d, weights[slots_of[d]]) for d in destinations}

        def step_map(z):
            cost = bpr_cost(np.maximum(z, 0.0), fft, cap, b, power)
            y = np.zeros(n_links)
            for d in demanded:
                here = cells.get(d)
                if here is None:
                    y += load_destination(topo, od, mu, cost, d, base[d], base_orders[d])
                    continue
                for admit, mass in here:
                    if mass <= WEIGHT_EPSILON:
                        continue
                    mask, order = mask_for(d, admit)
                    y += mass * load_destination(topo, od, mu, cost, d, mask, order)
            return y

        z, residual = _blend_fixed_point(step_map, seed, deadline=deadline)
        evaluations[0] += 1
        cost = bpr_cost(np.maximum(z, 0.0), fft, cap, b, power)
        potential = potentials(topo, cost)
        gaps = np.empty(n_groups)
        for d in destinations:
            for slot, group in zip(slots_of[d], groups_of[d]):
                gaps[slot] = potential[tail[group.forward], d] - potential[head[group.forward], d]
        stored = (z, gaps, residual / max(float(np.max(np.abs(z))), 1e-30))
        memo[key] = stored
        return stored

    forward_link = np.empty(n_groups, dtype=np.int64)
    for d in destinations:
        for slot, group in zip(slots_of[d], groups_of[d]):
            forward_link[slot] = group.forward
    if collapse:
        classes = [np.flatnonzero(forward_link == link) for link in sorted(set(forward_link.tolist()))]
    else:
        classes = [np.array([slot]) for slot in range(n_groups)]

    def search(members):
        count = len(members)

        def lift(reduced):
            full = np.empty(n_groups)
            for position, indices in enumerate(members):
                full[indices] = reduced[position]
            return np.clip(full, 0.0, 1.0)

        def reduce_gaps(gaps):
            return np.array([float(np.mean(gaps[indices])) for indices in members])

        seed_weights = np.full(count, 0.5)
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached("before the seed blend of the first level")
        seed_gaps = blend(lift(seed_weights))[1]
        generator = np.random.default_rng(settings.seed)
        ratios = []
        for probe in range(jacobian_probes):
            if deadline is not None and time.perf_counter() >= deadline:
                raise DeadlineReached(f"Jacobian probe {probe + 1} of {jacobian_probes}, "
                                      f"before the first level")
            direction = generator.standard_normal(count)
            direction /= max(np.linalg.norm(direction), 1e-30)
            probed = np.clip(seed_weights + 1e-4 * direction, 0.0, 1.0)
            moved = float(np.max(np.abs(probed - seed_weights)))
            if moved <= 1e-14:
                continue
            ratios.append(float(np.max(np.abs(reduce_gaps(blend(lift(probed))[1])
                                              - reduce_gaps(seed_gaps)))) / moved)
        jacobian_norm = max(ratios) if ratios else None
        alpha = (1.0 / jacobian_norm) if (jacobian_norm and jacobian_norm > 1e-12) else 1.0

        def psi(reduced):
            return np.clip(reduced + alpha * reduce_gaps(blend(lift(reduced))[1]), 0.0, 1.0)

        def residual(reduced):
            return float(np.max(np.abs(np.asarray(reduced, dtype=float) - psi(reduced)))) / alpha

        # One file per refinement, not one per rung. Each refinement is a separate search at a
        # different width, and they ran through a single path: the first overwrote the second's stored
        # state with its own, then cleared it as its own on the way out, and the refinement -- the search
        # that actually carries the answer, and the one holding eleven hours of progress -- began again
        # at the coarsest mesh every time. A resumed run reproduced its predecessor level for level.
        outcome = simplicial_fixed_point(psi, count, w0=seed_weights, levels=levels, grid=grid,
                                         tolerance=tie_tolerance, residual=residual,
                                         max_pivots=max_pivots, progress=progress,
                                         deadline=deadline,
                                         checkpoint=_sibling(checkpoint, f"r{refinements[0]}"),
                                         checkpoint_seconds=settings.checkpoint_seconds,
                                         memory_ceiling_mb=settings.memory_ceiling_mb)
        weights = lift(np.clip(outcome.w, 0.0, 1.0))
        flow, gaps, flow_residual = blend(weights)
        return outcome, weights, flow, gaps, flow_residual, jacobian_norm, alpha

    # Held in a list so the nested search closure reads the current value rather than a copy.
    refinements = [0]
    while True:
        outcome, w, x, gaps, flow_residual, jacobian_norm, alpha = search(classes)
        spreads = [float(np.max(gaps[indices]) - np.min(gaps[indices])) for indices in classes]
        worst_spread = max(spreads, default=0.0)
        offending = [i for i, spread in enumerate(spreads)
                     if spread > tie_tolerance and len(classes[i]) > 1]
        if not offending or (max_refinements is not None and refinements[0] >= max_refinements):
            break
        split = []
        for i, indices in enumerate(classes):
            if i not in offending:
                split.append(indices)
                continue
            buckets: dict = {}
            for slot in indices.tolist():
                buckets.setdefault(round(float(gaps[slot]) / max(tie_tolerance, 1e-12)), []).append(slot)
            split += [np.asarray(members, dtype=int) for members in buckets.values()]
        if len(split) == len(classes):
            break
        classes, refinements[0] = split, refinements[0] + 1

    # The complementarity clause on the cube, one per group: the three branches of the clip are the three
    # branches of the face conditions, so this number is the distance from a certificate.
    complementarity = []
    for slot in range(n_groups):
        if w[slot] <= ACTIVE_TOLERANCE:
            complementarity.append(max(0.0, gaps[slot]))
        elif w[slot] >= 1.0 - ACTIVE_TOLERANCE:
            complementarity.append(max(0.0, -gaps[slot]))
        else:
            complementarity.append(abs(gaps[slot]))

    # The local dimension of the equilibrium manifold at the point returned. The face conditions pin the
    # weights only through the gaps, so the free directions are the interior weights less the rank of the
    # gaps' sensitivity to them. Measured by one extra blend per group and never on the search path, so it
    # is off unless a caller asks for it: on a face with a hundred thousand tied pairs it is not payable.
    interior = int(np.sum((w > ACTIVE_TOLERANCE) & (w < 1.0 - ACTIVE_TOLERANCE)))
    gap_rank = None
    if measure_dimension and n_groups:
        step = 1e-6
        jacobian = np.empty((n_groups, n_groups))
        for column in range(n_groups):
            probe = w.copy()
            probe[column] = w[column] + step if w[column] <= 1.0 - step else w[column] - step
            jacobian[:, column] = (blend(probe)[1] - gaps) / (probe[column] - w[column])
        gap_rank = int(np.linalg.matrix_rank(jacobian))

    pair_weights = np.empty(len(tied))
    for d in destinations:
        for slot, group in zip(slots_of[d], groups_of[d]):
            pair_weights[group.index_forward] = w[slot]
            if group.index_reverse is not None:
                # The reverse orientation's gap is the negative of the forward one identically, so the
                # pair carries a single unknown.
                pair_weights[group.index_reverse] = 1.0 - w[slot]

    # The atoms are the mixture. Handing back only the marginals would force the caller to rebuild a
    # representation, and a cube point whose weights sum above one has none on the single-link simplex;
    # the certificate would then fail for want of a family rather than for want of an equilibrium.
    mixture = [(d, base[d], 1.0) for d in demanded if d not in by_destination]
    for d in destinations:
        for admit, mass in cells_for(d, w[slots_of[d]]):
            if mass > WEIGHT_EPSILON:
                mixture.append((d, mask_for(d, admit)[0], float(mass)))

    report = dict(status=outcome.status, vi_residual=outcome.residual, evaluations=evaluations[0],
                  pivots=outcome.pivots, family="nested", collapsed=bool(collapse),
                  n_groups=n_groups, mixture=mixture, n_classes=len(classes),
                  n_refinements=refinements[0], class_spread=worst_spread,
                  jacobian_norm=jacobian_norm, alpha=alpha, levels=outcome.levels,
                  mesh=outcome.mesh, flow_residual=flow_residual,
                  complementarity=float(max(complementarity)) if complementarity else 0.0,
                  worst_raw_gap=float(np.max(np.abs(gaps))),
                  n_interior=interior,
                  gap_rank=gap_rank,
                  manifold_dimension=None if gap_rank is None else interior - gap_rank,
                  gaps=[float(value) for value in gaps],
                  wall_seconds=time.perf_counter() - started)
    return x, pair_weights, report


def _sibling(checkpoint, suffix):
    """A checkpoint beside another, named for the rung it belongs to."""
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix or '.npz'}")


def solve_face_ladder(topo, arrays, od, mu, demanded, base, tied, x0,
                      settings: Settings = DEFAULTS, tie_tolerance=TIE_TOLERANCE,
                      budgets=None, use_complete=True, progress=None, deadline=None,
                      measure_dimension=False, checkpoint=None):
    """Descend a sequence of increasingly complete searches, stopping at the first that certifies.

    Each rung either returns a flow whose complementarity residual is below tolerance, which is a
    certificate the a posteriori test then confirms on the flow alone, or hands over to the next:

        collapsed nested face, one weight per contested link, with the spread check and class refinement
        full nested face, one weight per tied group, on the cube
        the unconditional search over the whole flow space, whose label is the strict efficient support

    The last rung carries the guarantee and has no hypothesis: its label is acyclic at every cost, so no
    atom family is indexed. It searches in as many dimensions as there are links and is not affordable
    past toy scale, so a descent that reaches it usually ends without reaching its target. That is
    reported as what it is: completeness in finite time, never completeness in affordable time.

    A rung that stops on a pivot limit has declined to continue, which is not the same as having failed,
    and the trail records which is which.

    ``deadline`` is checked between rungs and inside one. A rung is a sequence of simplicial searches,
    one per class refinement, and each of those checks the clock between its restarts and inside a pivot
    walk; the three stopping statuses it can report are named in :func:`simplicial_fixed_point`. Every one
    of them writes the search's state, so a rung that outlasts its budget is continued by resubmitting
    rather than repeated.

    ``checkpoint`` is what makes that true, and each refinement searches through its own file beneath it.
    They shared one until 2026-09-16: the first refinement overwrote the second's state with its own and
    then cleared it, so a face that had stored eleven hours of progress began again at the coarsest mesh
    every time and a resumed run reproduced its predecessor level for level.
    """
    budgets = settings.ladder_budgets if budgets is None else budgets
    trail, best = [], None
    started = time.perf_counter()

    out_of_time = False
    for collapse in (True, False):
        for budget in budgets:
            if deadline is not None and time.perf_counter() >= deadline:
                out_of_time = True
                break
            name = (f"{'collapsed' if collapse else 'full'} nested"
                    + ("" if budget is None else f" @ {budget}"))
            if progress is not None:
                progress(f"enter {name}, {len(tied)} tied pairs, "
                         f"{time.perf_counter() - started:.0f}s elapsed")
            try:
                x, w, report = solve_face_nested(topo, arrays, od, mu, demanded, base, tied, x0,
                                                 collapse=collapse, settings=settings,
                                                 tie_tolerance=tie_tolerance, max_pivots=budget,
                                                 progress=progress, deadline=deadline,
                                                 measure_dimension=measure_dimension,
                                                 checkpoint=_sibling(checkpoint,
                                                                     name.replace(" ", "_")))
            except DeadlineReached as reached:
                # The budget crossed inside the rung rather than between rungs. Where that happened is
                # the whole diagnosis: a rung stopped before its first level has evaluated the blend a
                # handful of times and searched nothing, which is a different statement about the face
                # from a rung that searched and did not certify.
                trail.append(dict(rung=name, status=f"time budget: {reached}", searched=False))
                if progress is not None:
                    progress(f"leave {name}: time budget: {reached}")
                out_of_time = True
                break
            except (ValueError, RuntimeError) as exc:
                # A residual ring descends: the final rung's label is the strict efficient support,
                # which is acyclic at every cost, so the atom family is not a hypothesis there.
                trail.append(dict(rung=name, status=f"failed: {type(exc).__name__}: {str(exc)[:80]}"))
                if progress is not None:
                    progress(f"leave {name}: {type(exc).__name__}: {str(exc)[:80]}")
                continue

            trail.append(dict(rung=name, status=report["status"], dimension=report["n_classes"],
                              complementarity=report["complementarity"], pivots=report["pivots"],
                              class_spread=report["class_spread"],
                              wall_seconds=report["wall_seconds"]))
            if progress is not None:
                progress(f"leave {name}: status={report['status']} "
                         f"complementarity={report['complementarity']:.2e} "
                         f"dimension={report['n_classes']} pivots={report['pivots']}")
            if best is None or report["complementarity"] < best[2]["complementarity"]:
                best = (x, w, report, name)
            if report["complementarity"] <= tie_tolerance:
                return x, w, dict(report, rung=name, trail=trail, certified_face=True)
            if report["status"] in ("time budget", "memory ceiling"):
                out_of_time = True
                break
        if out_of_time:
            break

    if out_of_time:
        # The budget stopped a rung. Descending to a more expensive one would only be stopped sooner, so
        # what is returned is the best face reached, marked as a stop rather than an answer. The rung has
        # written its state, so a caller that wants it finished resubmits and the search continues from
        # the level it reached.
        if best is None:
            # Nothing to hand back. A caller reading a report here would read the seed point as though
            # it were an answer, which is what happened once: the complementarity and the inclusion
            # violations at a point where every weight is one half are diagnostics of that point and say
            # nothing about the face.
            raise DeadlineReached(
                f"{len(trail)} rung(s) tried, none searched: "
                + "; ".join(str(entry.get("status")) for entry in trail))
        x, w, report, name = best
        return x, w, dict(report, rung=name, trail=trail, certified_face=False,
                          stopped_on="time budget")

    if use_complete:
        from .complete_k import naive_label, solve_complete_core

        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached(f"{len(trail)} rung(s) tried, none certified")
        name = "complete flow space"
        if progress is not None:
            progress(f"enter {name}, {topo.n_links} links, "
                     f"{time.perf_counter() - started:.0f}s elapsed")
        try:
            x, report = solve_complete_core(topo, arrays, od, mu, demanded, settings,
                                            x_start=x0, progress=progress)
        except (ValueError, RuntimeError) as exc:
            trail.append(dict(rung=name, status=f"failed: {type(exc).__name__}: {str(exc)[:80]}"))
            x, report = None, None
        if x is not None:
            trail.append(dict(rung=name, status=report["status"], dimension=topo.n_links,
                              reached=report.get("reached"), evaluations=report["evaluations"],
                              wall_seconds=report["wall_seconds"]))
            # This rung forms no mixture, so the pair marginals are read back from the support at its
            # own answer, which is an exact reading of the sign pattern.
            masks = naive_label(topo, od, mu, arrays, demanded, x)[1]
            w = np.array([1.0 if masks[d][link] else 0.0 for d, link in tied])
            complete = dict(report, family="complete", n_classes=topo.n_links, n_refinements=0,
                            complementarity=float(report.get("reached", np.inf)),
                            flow_residual=float(report.get("mixture_residual", np.nan)),
                            class_spread=0.0, pivots=report["evaluations"], rung=name, trail=trail,
                            certified_face=(report["status"] == "certified"))
            if report["status"] == "certified" or best is None:
                return x, w, complete
            if complete["complementarity"] < best[2]["complementarity"]:
                return x, w, complete

    if best is None:
        why = "; ".join(f"{entry['rung']} -> {entry['status']}" for entry in trail)
        raise RuntimeError(f"every rung failed to return a point. Trail: {why}")
    x, w, report, name = best
    return x, w, dict(report, rung=name, trail=trail, certified_face=False)


def _blend_at_weight(topo, arrays, od, mu, demanded, base, destinations, link, weight, seed,
                     averaging=40):
    """The blended fixed point at one weight on a single contested link, and its feedback.

    The destinations sharing the link mix the mask without it against the mask with it; every other
    destination stays on its base mask. For a fixed weight this is a smooth fixed point, with no tie
    equations and no square system, which is exactly why the bisection below needs no regularity.
    """
    fft, cap, b, power, _, _ = arrays
    tail, head = topo.tail, topo.head

    with_link = {}
    for d in destinations:
        mask = base[d].copy()
        mask[link] = True
        try:
            kahn_order(topo, mask)
        except RuntimeError:
            return None, None
        with_link[d] = mask

    # Split the demand once. The untouched destinations are a single loading and each mixed destination
    # costs two; recomputing a mixed destination's base loading inside every evaluation would triple the
    # work when many destinations contest one link.
    mixed = set(destinations)
    od_rest = od.copy()
    for d in mixed:
        od_rest[:, d] = 0.0
    rest_masks = {d: base[d] for d in demanded if d not in mixed}
    od_single = {d: np.zeros_like(od) for d in destinations}
    for d in destinations:
        od_single[d][:, d] = od[:, d]

    prepared_rest = prepare_masks(topo, od_rest, rest_masks) if rest_masks else []
    prepared_off = {d: prepare_masks(topo, od_single[d], {d: base[d]}) for d in destinations}
    prepared_on = {d: prepare_masks(topo, od_single[d], {d: with_link[d]}) for d in destinations}

    def step_map(z):
        cost = bpr_cost(np.maximum(z, 0.0), fft, cap, b, power)
        y = (load_prepared(topo, od_rest, mu, cost, prepared_rest) if prepared_rest
             else np.zeros_like(z))
        for d in destinations:
            y = y + ((1.0 - weight) * load_prepared(topo, od_single[d], mu, cost, prepared_off[d])
                     + weight * load_prepared(topo, od_single[d], mu, cost, prepared_on[d]))
        return y

    x, _ = _blend_fixed_point(step_map, seed, averaging=averaging)
    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    potential = potentials(topo, cost)
    feedback = float(np.mean([potential[tail[link], d] - potential[head[link], d]
                              for d in destinations]))
    return x, feedback


def solve_by_bisection(net, od, mu, x0, destinations, link, base,
                       tie_tolerance=TIE_TOLERANCE, max_evaluations=40, deadline=None):
    """Bracket and bisect the class weight on a single contested link.

    Needs only continuity of the feedback and a sign change between the two ends: no Jacobian, no
    regularity condition and no square system. Returns the flow, the weight and a record, or ``None``
    with the reason.

    The step is a guarded secant rather than plain bisection. The bracket is never given up, so the
    guarantee is untouched, but the secant converges superlinearly and typically ends in under ten
    evaluations instead of forty. Each evaluation is a full blended solve, which is the dominant cost
    where one link is contested by many destinations, and the budget is checked between them: that is the
    finest boundary this routine has, because a blended solve has no interior state to resume from.
    """
    tail, head, fft, cap, b, power, n_nodes, n_zones = unpack(net)
    arrays = (fft, cap, b, power, n_nodes, n_zones)
    topo = Topology.build(tail, head, fft, n_nodes, n_zones, net.get("first_thru"))
    demanded = [d for d in range(n_zones) if od[:, d].sum() > 0.0]
    started = time.perf_counter()

    x_low, feedback_low = _blend_at_weight(topo, arrays, od, mu, demanded, base, destinations,
                                           link, 0.0, x0)
    if x_low is None:
        return None, None, "the link closes a ring for one of these destinations"
    x_high, feedback_high = _blend_at_weight(topo, arrays, od, mu, demanded, base, destinations,
                                             link, 1.0, x_low)
    if x_high is None:
        return None, None, "the link closes a ring for one of these destinations"

    if abs(feedback_low) <= tie_tolerance:
        return x_low, 0.0, dict(evaluations=2, weight=0.0, feedback=feedback_low,
                                wall_seconds=time.perf_counter() - started)
    if abs(feedback_high) <= tie_tolerance:
        return x_high, 1.0, dict(evaluations=2, weight=1.0, feedback=feedback_high,
                                 wall_seconds=time.perf_counter() - started)
    if feedback_low * feedback_high > 0.0:
        return None, None, (f"no bracket: feedback {feedback_low:.3e} at zero and "
                            f"{feedback_high:.3e} at one")

    low, high, x = 0.0, 1.0, x_high
    feedback = feedback_high
    for step in range(max_evaluations):
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineReached(f"bisecting link {link}, {step + 2} evaluation(s) spent")
        span = high - low
        if abs(feedback_high - feedback_low) > 1e-300:
            candidate = low + span * feedback_low / (feedback_low - feedback_high)
        else:
            candidate = 0.5 * (low + high)
        if not (low + 0.05 * span <= candidate <= high - 0.05 * span):
            candidate = 0.5 * (low + high)
        x_mid, feedback = _blend_at_weight(topo, arrays, od, mu, demanded, base, destinations,
                                           link, candidate, x)
        if x_mid is None:
            return None, None, "the link closes a ring for one of these destinations"
        x = x_mid
        if abs(feedback) <= tie_tolerance or span < 1e-15:
            return x, candidate, dict(evaluations=step + 3, weight=candidate, feedback=feedback,
                                      wall_seconds=time.perf_counter() - started)
        if (feedback > 0.0) == (feedback_low > 0.0):
            low, feedback_low = candidate, feedback
        else:
            high, feedback_high = candidate, feedback

    midpoint = 0.5 * (low + high)
    return x, midpoint, dict(evaluations=max_evaluations + 2, weight=midpoint, feedback=feedback,
                             wall_seconds=time.perf_counter() - started,
                             stopped_on="evaluation limit")
