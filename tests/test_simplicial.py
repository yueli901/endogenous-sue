"""The simplicial search, on synthetic maps only: no network data, no solver.

The guarantee claimed is convergence from any start on any continuous self-map of the cube, with no
monotonicity, no P-matrix property and no regularity. That is checked by sampling random maps with no
structure, not by one favourable instance.
"""
import numpy as np

from endogenous_sue.tied.simplicial import simplicial_fixed_point

RNG = np.random.default_rng(0)


def _check(psi, n, want=None, tolerance=2e-2, **kwargs):
    # Never start on the answer: one half is a fixed point of several maps here, and a run that returns
    # before pivoting once tests nothing.
    kwargs.setdefault("w0", RNG.uniform(0.02, 0.98, n))
    outcome = simplicial_fixed_point(psi, n, tolerance=1e-12, **kwargs)
    assert outcome.pivots > 0, "the search returned without pivoting"
    assert outcome.residual <= tolerance, f"residual {outcome.residual:.3e}"
    if want is not None:
        assert float(np.max(np.abs(outcome.w - want))) <= tolerance
    return outcome


def test_constant_maps_are_solved_in_several_dimensions():
    for n in (1, 2, 5, 8):
        target = RNG.uniform(0.15, 0.85, n)
        _check(lambda z, target=target: target, n, want=target)


def test_contractions_are_solved():
    for n in (2, 5, 10):
        target = RNG.uniform(0.2, 0.8, n)
        _check(lambda z, target=target: target + 0.4 * (z - target), n, want=target)


def test_a_fixed_point_on_the_boundary_is_found():
    """Where the clip is active, which is where a box method usually breaks."""
    _check(lambda z: np.clip(z - 0.3, 0.0, 1.0), 3, want=np.zeros(3))
    _check(lambda z: np.clip(z + 0.3, 0.0, 1.0), 3, want=np.ones(3))
    _check(lambda z: np.clip(z + np.array([-0.3, 0.3, -0.3, 0.3]), 0.0, 1.0), 4,
           want=np.array([0.0, 1.0, 0.0, 1.0]))


def test_a_non_monotone_rotation_is_solved():
    """The case every method needing monotonicity or a P-matrix fails on, and the reason for this one."""
    def rotate(z):
        centred = z - 0.5
        out = np.empty_like(centred)
        out[0::2] = centred[0::2] - 1.5 * centred[1::2]
        out[1::2] = centred[1::2] + 1.5 * centred[0::2]
        return np.clip(0.5 + 0.5 * out, 0.0, 1.0)

    _check(rotate, 4, want=np.full(4, 0.5))
    _check(rotate, 6, want=np.full(6, 0.5))


def test_an_indefinite_box_problem_is_solved():
    """Written the way a face is written, with a deliberately indefinite Jacobian."""
    target = np.array([0.3, 0.6, 0.45, 0.7])
    matrix = np.array([[-1.0, 2.0, 0.0, 0.0],
                       [-2.0, -1.0, 0.0, 0.0],
                       [0.0, 0.0, -1.0, 3.0],
                       [0.0, 0.0, -3.0, -1.0]]) * 0.25
    _check(lambda z: np.clip(z + matrix @ (z - target), 0.0, 1.0), 4, want=target)


def test_non_differentiable_maps_are_solved():
    """Continuous with kinks, which is what a tied residual looks like when a support flips."""
    _check(lambda z: np.clip(0.5 + 0.6 * np.abs(z - 0.5) - 0.3, 0.0, 1.0), 3)
    _check(lambda z: np.clip(0.5 + 0.4 * np.sin(6.0 * (z - 0.5)), 0.0, 1.0), 5)


def test_the_dimensions_a_face_actually_reaches():
    for n in (20, 46):
        target = RNG.uniform(0.15, 0.85, n)
        _check(lambda z, target=target: target + 0.5 * (z - target), n, want=target, tolerance=5e-2)


def test_random_non_monotone_maps_from_random_starts():
    """The property the guarantee claims, checked by sampling rather than by one lucky instance."""
    worst, not_converged = 0.0, 0
    for _ in range(200):
        n = int(RNG.integers(2, 13))
        matrix = RNG.normal(0.0, 1.0, (n, n))
        target = RNG.uniform(0.1, 0.9, n)

        def psi(z, matrix=matrix, target=target):
            return np.clip(target + 0.45 * np.tanh(matrix @ (z - target)), 0.0, 1.0)

        outcome = simplicial_fixed_point(psi, n, w0=RNG.uniform(0.0, 1.0, n), tolerance=1e-10)
        worst = max(worst, outcome.residual)
        not_converged += outcome.status != "converged"
    assert not_converged == 0, f"{not_converged} of 200 random maps did not converge"
    assert worst <= 1e-8, f"worst residual {worst:.3e}"


def test_a_search_writes_state_as_it_goes_and_resumes_from_it():
    """The run is expected to be killed by the scheduler, so the state on disk carries the answer.

    A restart loop's state is the best point so far and the mesh it was found at, overwritten in place as
    the search proceeds. Without it, a face too long for one job never finishes however many jobs it is
    given: each repeats the levels the last one completed. The dimension travels with the state, because
    the ladder's rungs search in different numbers of unknowns and must not read each other's answers.
    """
    import tempfile
    from pathlib import Path as FilePath
    from time import perf_counter

    from endogenous_sue.tied.simplicial import simplicial_fixed_point

    seen_on_disk = []

    def rotate(w):
        # Non-monotone with no fixed point at the centre, so the search keeps going, and each call
        # records whether the state was on disk at the time -- which is what a killed job depends on.
        w = np.asarray(w, dtype=float)
        seen_on_disk.append(checkpoint.exists())
        return np.clip(0.62 + 0.37 * np.roll(2.0 * w - 1.0, 1), 0.0, 1.0)

    directory = tempfile.mkdtemp()
    checkpoint = FilePath(directory) / "face.npz"

    # Writing on every level. A killed job never reaches a clean exit, so what it leaves behind is
    # whatever the last periodic write put there.
    simplicial_fixed_point(rotate, 3, levels=8, grid=4, tolerance=1e-14,
                           checkpoint=checkpoint, checkpoint_seconds=1e-9)
    assert any(seen_on_disk), "no state was on disk at any point during the search"

    # Simulating the kill: the state the last write left, read back by the next job.
    stopped = simplicial_fixed_point(rotate, 3, levels=2, grid=4, tolerance=1e-14,
                                     deadline=perf_counter() - 1.0, checkpoint=checkpoint)
    assert stopped.status == "time budget"
    assert checkpoint.exists(), "a stopped search left no state"
    resumed = simplicial_fixed_point(rotate, 3, levels=50, grid=4, checkpoint=checkpoint)
    assert resumed.resumed_from is not None, "the state was written and not read back"

    # A different width is a different search and must not read this one's answer.
    simplicial_fixed_point(rotate, 3, levels=2, grid=4, tolerance=1e-14,
                           deadline=perf_counter() - 1.0, checkpoint=checkpoint)
    assert simplicial_fixed_point(rotate, 5, checkpoint=checkpoint).resumed_from is None


def test_a_budget_interrupts_a_single_level_and_not_only_the_gaps_between_them():
    """A level that outlasts the whole budget makes the budget advisory on exactly the hard faces.

    The clock was checked only between Merrill restarts. On the widest face in the study a single blend
    evaluation ran to hours, so a twelve-hour budget was spent before the first level began and the run
    reported a point it had never searched. The pivot loop now checks the clock too.
    """
    from time import perf_counter

    from endogenous_sue.config import DeadlineReached
    from endogenous_sue.tied.simplicial import walk_one_level

    calls = [0]

    def slow(w):
        calls[0] += 1
        return np.clip(0.62 + 0.37 * np.roll(2.0 * np.asarray(w, dtype=float) - 1.0, 1), 0.0, 1.0)

    try:
        walk_one_level(slow, 3, 4, np.full(3, 0.5), None, calls,
                       deadline=perf_counter() - 1.0)
    except DeadlineReached as reached:
        assert "pivot" in str(reached)
    else:
        raise AssertionError("an expired budget did not interrupt the pivot loop")


def test_a_search_that_never_reached_a_level_says_so():
    """Stopping before the first level is a different statement from searching and not certifying."""
    from time import perf_counter

    from endogenous_sue.tied.simplicial import simplicial_fixed_point

    def rotate(w):
        return np.clip(0.62 + 0.37 * np.roll(2.0 * np.asarray(w, dtype=float) - 1.0, 1), 0.0, 1.0)

    stopped = simplicial_fixed_point(rotate, 3, levels=50, grid=4,
                                     deadline=perf_counter() - 1.0)
    assert stopped.status.startswith("time budget"), stopped.status
    assert stopped.levels == 0 and stopped.pivots == 0


def test_a_search_stopped_by_its_budget_keeps_what_it_reached():
    """Every status that writes resumable state must keep it.

    The search saves state on three stopping conditions and the clear-guard named one of them. A budget
    crossed inside a pivot walk reports "time budget in pivot", not "time budget", so its state was
    written and deleted three lines later: an eleven-hour rung on Chicago-Sketch at mu=1 left nothing to
    continue from and could only be repeated.
    """
    import inspect

    from endogenous_sue.tied import simplicial

    source = inspect.getsource(simplicial.simplicial_fixed_point)
    saved = {"time budget", "time budget in pivot", "memory ceiling"}
    for status in saved:
        assert f'status = "{status}"' in source, f"{status} is no longer a stopping condition"
    guard = [line for line in source.splitlines() if "_clear_search(checkpoint" in line]
    assert guard, "the search must still clear state on success"
    condition = source.split("_clear_search(checkpoint")[0].splitlines()[-2]
    for status in saved:
        assert status in condition, (
            f"{status!r} saves state and must be excluded from the clear, or the state is written "
            f"and then deleted")


def test_the_genericity_escalation_is_reachable():
    """A singular level must be retried with a different perturbation, not the same one six times.

    The escalation sat after a `break` and never ran, so a singular walk was repeated identically and
    `perturbed_levels` could only report zero.
    """
    import inspect

    from endogenous_sue.tied import simplicial

    lines = inspect.getsource(simplicial.simplicial_fixed_point).splitlines()
    escalation = next(i for i, line in enumerate(lines) if "jitter = 1e-12" in line)
    preceding = [line.strip() for line in lines[:escalation] if line.strip()]
    assert preceding[-1] != "break", "the escalation is unreachable again"


def test_a_search_does_not_clear_another_width_s_state():
    """A rung that refines its classes runs the search twice, at different widths, through one path.

    The first descent reads the stored state, rejects it on the width guard because it belongs to the
    second, and must not then delete it. It did: Chicago-Sketch at mu=1 stored eleven hours of progress
    at width 31, the width-29 descent cleared it on the way past, and the resumed run repeated its
    predecessor level for level -- twenty-six levels from the coarsest mesh, arriving at exactly the
    residual it had started the day with.
    """
    import tempfile
    from pathlib import Path

    import numpy as np

    from endogenous_sue.tied.simplicial import _clear_search, _load_search, _save_search

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "search.npz"
        _save_search(path, 31, np.zeros(31), 1.0e-9, 12, 16384, 3612)
        assert _load_search(path, 29) is None, "a different width must not be resumed"
        _clear_search(path, 29)
        assert path.exists(), "a width-29 search removed the width-31 state it could not use"
        assert _load_search(path, 31) is not None, "the state it belongs to must still be readable"
        _clear_search(path, 31)
        assert not path.exists(), "a search must still clear its own finished state"


def test_each_refinement_searches_through_its_own_checkpoint():
    """A refining rung runs one search per refinement, and they must not share a file.

    They did. The first descent wrote its own width over the second's stored state, then cleared that
    file as its own on the way out, so the refinement -- the search that carries the answer -- began at
    the coarsest mesh every time. Chicago-Sketch at mu=1 stored eleven hours of progress and a resumed
    run reproduced it level for level, twice, because the file it needed had been overwritten and then
    removed before it looked.
    """
    import inspect

    from endogenous_sue.tied import faces

    source = inspect.getsource(faces.solve_face_nested)
    call = source.split("simplicial_fixed_point(")[1].split(")")[0]
    assert "checkpoint=checkpoint" not in call, (
        "every refinement would share one checkpoint file again")
    assert "_sibling(checkpoint" in call and "refinements" in call, (
        "the search's checkpoint must be keyed by the refinement it belongs to")


def test_sibling_paths_are_distinct_per_refinement():
    from endogenous_sue.tied.faces import _sibling

    assert _sibling(None, "r0") is None
    first, second = _sibling("/tmp/face.npz", "r0"), _sibling("/tmp/face.npz", "r1")
    assert first != second, "two refinements resolved to the same file"
    assert str(first).endswith(".npz") and str(second).endswith(".npz")
