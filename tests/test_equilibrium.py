"""The two-phase solver: what stops it, and what the residual is measured against."""
import tempfile
from pathlib import Path

import numpy as np
from helpers import require_networks, transit_network

from endogenous_sue.certificate import conservation_error
from endogenous_sue.config import DEFAULTS, Settings
from endogenous_sue.equilibrium import SupportRule, polish, solve, solve_inclusion, solve_phase1

require_networks()

from endogenous_sue import corpus  # noqa: E402


def test_phase_one_stops_because_the_support_settled_not_because_it_ran_out():
    """Averaging reduces the flow gap at rate one over k, so a flow tolerance alone is unreachable.

    What phase one is for is freezing the support, which it does in finite time. There is no iteration
    ceiling by default, so "it ran out" is not one of the ways this can end: it stops on a criterion or it
    is still going.
    """
    assert Settings().max_phase1_iterations is None, (
        "an iteration ceiling is a property of the run and never of the method")
    for name in ("SiouxFalls", "Eastern-Massachusetts"):
        net, od = corpus.load(name)
        result = solve_phase1(net, od, 1.0)
        assert result.converged, f"{name} did not converge: {result.stopped_on}"
        assert result.stopped_on in {"support settled", "exactness bound", "flow tolerance"}


def test_reaching_the_iteration_limit_is_reported_as_unconverged():
    net, od = corpus.load("SiouxFalls")
    result = solve_phase1(net, od, 1.0, Settings(max_phase1_iterations=5))
    assert not result.converged
    assert result.stopped_on == "iteration limit"


def test_the_polished_flow_conserves_demand():
    for name in ("SiouxFalls", "Eastern-Massachusetts", "Anaheim"):
        net, od = corpus.load(name)
        equilibrium = solve(net, od, 1.0)
        worst, _, demand = conservation_error(net, od, equilibrium.x)
        assert worst <= 1e-6 * demand, f"{name} lost demand: {worst:.3e} of {demand:.3e}"
        assert equilibrium.residual < 1e-8


def test_the_polish_method_used_is_recorded_rather_than_substituted_silently():
    net, od = corpus.load("SiouxFalls")
    equilibrium = solve(net, od, 1.0)
    assert equilibrium.polish_method in {"newton_krylov", "newton_krylov_incomplete",
                                         "damped_averaging"}


def test_the_fixed_support_rule_never_rebuilds():
    net, od = corpus.load("SiouxFalls")
    result = solve_phase1(net, od, 1.0, Settings(max_phase1_iterations=200),
                          rule=SupportRule.FIXED, exact_stop=False)
    measured = [value for value in result.support_changes if value is not None]
    assert measured and max(measured) == 0, "the fixed-support rule rebuilt the support"


def test_the_inclusion_scheme_reads_a_trailing_window_rather_than_clearing_it():
    """A tie shows as several supports recurring late, so the window must accumulate, not reset."""
    from endogenous_sue import witness

    result = solve_inclusion(witness.network(), witness.demand(), 1.0,
                             Settings(max_phase1_iterations=1500))
    assert len(result.supports) > 1, (
        "the witness has unresolved ties, so more than one support must survive the window")
    unpacked = list(result.masks())
    assert len(unpacked) == len(result.supports)
    assert all(mask.dtype == bool and mask.size == result.n_links
               for support in unpacked for mask in support.values()), (
        "the window is stored packed and must unpack to masks of the network's own width")


def test_the_inclusion_window_is_not_capped():
    """The scheme stops for want of an answer, never for want of room to hold the window.

    A cap on how many supports the window may hold makes a cell whose support set moves quickly stop
    early and report a verdict the evidence does not support: on the largest cell of the certificate tier
    it fired after three hundred iterations, where the smallest cell of that tier needs six hundred and
    ninety-two to settle.
    """
    from endogenous_sue import witness

    settings = Settings(max_phase1_iterations=40, inclusion_window=5)
    result = solve_inclusion(witness.network(), witness.demand(), 1.0, settings)
    assert result.stopped_on in ("support set settled", "flow tolerance",
                                 "support set did not settle"), result.stopped_on
    assert result.stopped_on != "support set not settling"


def test_settings_are_carried_into_the_result():
    net, od = transit_network()
    settings = Settings(support_rebuild_interval=3)
    equilibrium = solve(net, od, 1.0, settings)
    assert equilibrium.settings.support_rebuild_interval == 3
    assert equilibrium.settings.as_record()["support_rebuild_interval"] == 3


def test_a_run_stopped_early_resumes_to_the_same_answer():
    """A scheduler's wall-clock limit must cost time, not correctness.

    The checkpoint carries the flow, the averaged cost, the frozen potentials and the settled-rebuild
    count, which is the whole of phase one's state. Resuming therefore reaches the same iteration with
    the same flow as an uninterrupted run, bit for bit.
    """
    import tempfile
    from pathlib import Path

    net, od = corpus.load("Eastern-Massachusetts")
    uninterrupted = solve_phase1(net, od, 1.0)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "cell.npz"
        stopped = solve_phase1(net, od, 1.0, Settings(max_phase1_iterations=100),
                               checkpoint=checkpoint)
        assert not stopped.converged and checkpoint.exists()
        resumed = solve_phase1(net, od, 1.0, checkpoint=checkpoint)
    assert resumed.resumed_from == stopped.iterations
    assert resumed.iterations == uninterrupted.iterations
    assert np.max(np.abs(resumed.x - uninterrupted.x)) == 0.0


def test_a_checkpoint_write_is_atomic():
    """A job killed during a write must leave the previous checkpoint readable."""
    import tempfile
    from pathlib import Path

    from endogenous_sue.equilibrium import load_checkpoint, save_checkpoint

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "state.npz"
        save_checkpoint(path, x=np.arange(4.0), iteration=np.int64(7))
        assert load_checkpoint(path)["iteration"] == 7
        assert not list(Path(directory).glob("*.partial"))


def test_an_unreachable_tolerance_is_reported_rather_than_returned_as_a_result():
    """Phase two returns a flow whether or not it reached the tolerance, so it must say which.

    The Krylov solve raises on exhausting its iterations and the averaging fallback has no convergence
    test at all, and both used to be recorded as converged. A residual is then the best one seen, which
    is not the quantity a table reports.
    """
    net, od = corpus.load("Braess-Example")
    phase1 = solve_phase1(net, od, 1.0)

    reached = polish(net, od, 1.0, phase1.x, phase1.potential)
    assert reached.converged and reached.residual <= DEFAULTS.residual_tolerance

    impossible = Settings(residual_tolerance=1e-300)
    unreached = polish(net, od, 1.0, phase1.x, phase1.potential, impossible)
    assert not unreached.converged
    assert unreached.method != "newton_krylov"


def test_a_cell_whose_phase_two_did_not_reach_tolerance_is_not_a_converged_equilibrium():
    net, od = corpus.load("Braess-Example")
    assert solve(net, od, 1.0).converged
    stopped = solve(net, od, 1.0, Settings(residual_tolerance=1e-300))
    assert not stopped.converged
    assert "phase 2" in stopped.stopped_on


def test_the_inclusion_scheme_remembers_supports_without_holding_them():
    """What is remembered forever must be small, because it is never pruned.

    The scheme recognises a support returning after a long absence, which needs a record of every support
    ever seen. Holding the supports themselves is 139 KiB each on a network with four hundred
    destinations: at four hundred thousand iterations that is fifty-eight gigabytes, and it killed a run
    after seven hours while looking, from the outside, exactly like a run that was working.
    """
    from endogenous_sue import witness
    from endogenous_sue.equilibrium import solve_inclusion

    result = solve_inclusion(witness.network(), witness.demand(), 1.0,
                             Settings(max_phase1_iterations=400))
    assert result.n_supports_seen > 0

    # The window holds supports; the long-term record holds only digests of them. Sixteen bytes each.
    for support in result.masks():
        assert support, "the window must hold the supports themselves, not digests of them"
        break


def test_phase_two_stops_on_the_norm_it_is_graded_by():
    """The stopping criterion and the acceptance criterion must be the same object.

    SciPy grades ``newton_krylov`` in the max-norm; this package, the certificate and the paper all grade
    in the relative 2-norm, which is larger by up to sqrt(|E|). Left to its default the solver stopped on
    a criterion up to 137x looser than the one applied to its answer, so a solve that met its own target
    was recorded as a failure and sent back to be "continued" from a checkpoint that was never written.
    Twenty-six cells of the first canonical run ended that way. A converged result therefore has to carry
    a residual that actually meets the tolerance, on a network with enough links for the two norms to
    differ.
    """
    for name in ("Eastern-Massachusetts", "Berlin-Tiergarten"):
        net, od = corpus.load(name)
        assert len(net["capacity"]) > 100, f"{name} is too small to separate the two norms"
        phase1 = solve_phase1(net, od, 5.0)
        polished = polish(net, od, 5.0, phase1.x, phase1.potential)
        assert polished.converged, f"{name}: {polished.method} stopped above tolerance"
        assert polished.residual <= DEFAULTS.residual_tolerance, (
            f"{name}: reported converged at {polished.residual:.3e}, "
            f"above {DEFAULTS.residual_tolerance:.0e}")


def test_phase_two_checkpoints_whatever_stopped_it():
    """A cell that says it is to be continued has to leave something to continue from.

    Phase two used to save state only where the wall clock stopped it. A cell that ran to the solver's
    own termination and was then graded above tolerance saved nothing, so the resubmit restarted it from
    the beginning and failed identically -- a loop that made no progress rather than a resume.
    """
    net, od = corpus.load("Eastern-Massachusetts")
    settings = Settings(residual_tolerance=1e-16, max_newton_iterations=2)
    phase1 = solve_phase1(net, od, 5.0, settings)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "phase2.npz"
        polished = polish(net, od, 5.0, phase1.x, phase1.potential, settings, checkpoint=checkpoint)
        assert not polished.converged, "the tolerance was chosen to be out of reach"
        assert checkpoint.exists(), "stopped above tolerance and saved nothing to resume from"

        resumed = polish(net, od, 5.0, phase1.x, phase1.potential, settings, checkpoint=checkpoint)
        assert resumed.residual <= polished.residual, "resuming lost the progress it had"


def test_the_direct_face_route_certifies_and_its_mixture_round_trips():
    """The declared direct route must return a record the rest of the sweep consumes unchanged.

    Everything downstream of the face solve is one code path on either route, so the ladder route's
    record has to carry the same keys with the same meaning. The reconstruction is the part that can
    silently disagree: the sweep rebuilds each support as ``base_masks[d]`` plus the listed links, so the
    links must be listed against that mask and not against the absorbing one the ladder searches over.
    Taken against the absorbing base they differ wherever absorption added a link, and the mixture
    certified would not be the mixture the ladder solved.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from endogenous_sue.certificate import certify
    from endogenous_sue.tied import contested_pairs
    from reproduction.exhibits.certificate_sweep import direct_face

    net, od = corpus.load("Berlin-Prenzlauerberg-Center")
    inclusion = solve_inclusion(net, od, 1.0, DEFAULTS)
    contested, base_masks = contested_pairs(inclusion.masks())
    assert contested, "this cell has a tied face and none was found"

    x, record = direct_face(net, od, 1.0, inclusion.x, contested, base_masks, DEFAULTS)
    assert record["phase"] == "ladder" and "ladder" in record["route"]
    assert record.get("certified_face"), record.get("status")

    mixture = []
    for d, added, weight in record["vertex_mixture"]:
        mask = base_masks[int(d)].copy()
        for link in added:
            mask[int(link)] = True
        mixture.append((int(d), mask, float(weight)))
    assert mixture, "a contested face must return supports carrying weight"
    certificate = certify(net, od, 1.0, np.maximum(x, 0.0), mixture, DEFAULTS)
    assert certificate.certified, f"{certificate.kind}: {certificate.reason}"


def test_the_envelope_watch_counts_only_complete_identical_windows():
    """The rule is five identical complete windows, and a partial one is not a window.

    It stops the scheme on what the face solver consumes -- the window's intersection and union -- rather
    than on a support identity the face never reads. The count must reset on any change, and the envelope
    handed on must be the one that was held rather than a partially accumulated successor.
    """
    from endogenous_sue.equilibrium import _EnvelopeWatch

    watch = _EnvelopeWatch(window=4, required=3)
    steady = {0: np.packbits(np.array([1, 1, 0, 0], dtype=bool))}
    for iteration in range(1, 13):
        watch.see(iteration, steady)
    assert watch.held == 2, "three windows give two comparisons"
    assert watch.last_change is None
    assert watch.stable_from is not None
    assert watch.settled_base[0].tobytes() == steady[0].tobytes()

    moved = {0: np.packbits(np.array([1, 1, 1, 0], dtype=bool))}
    for iteration in range(13, 17):
        watch.see(iteration, moved)
    assert watch.held == 0, "the count must reset when the envelope moves"
    assert watch.last_change == 16
    assert watch.stable_from is None


def test_the_envelope_rule_is_off_unless_a_cell_asks_for_it():
    """It is an empirical condition, not a theorem, so it must not apply to every cell by default."""
    from endogenous_sue.config import DEFAULTS, DIRECT_FACE_CELLS, DIRECT_FACE_ENVELOPE_WINDOWS

    assert DEFAULTS.envelope_stable_windows is None
    assert DIRECT_FACE_ENVELOPE_WINDOWS >= 2, "one repeat is not evidence"
    assert DIRECT_FACE_CELLS, "the rule has no cells declared for it"


def test_the_recorded_base_is_the_one_the_solve_loaded():
    """A mask is meaningless without the base it is stated against, and the sweep must get that base.

    The accelerating routes search against a *copy* of the base with contested links pinned in and out,
    and pass that copy down to a recursive solve. A support recorded as "base plus these links" is
    therefore relative to a base the caller does not hold: rebuilding it on the inclusion scheme's base
    drops every link pinned in and restores every link pinned out. All three cells solved through the
    pattern route failed their independent certificate that way, at mixture residuals of 1e-2 on flows
    whose own residual was 1e-15, while the solver's own check passed because it reconstructed from its
    own copy.

    The base is therefore recorded absolutely, by the solve that loaded it. A delta cannot serve: one
    written by an outer search is relative to a base its winning solve may never have seen, and one
    written by an inner search is relative to a base the caller does not have. This asserts the contract
    end to end -- solve, record, rebuild exactly as the sweep rebuilds, certify -- on a cell that reaches
    a pinning route.
    """
    import numpy as np

    from endogenous_sue.certificate import certify
    from endogenous_sue.network import NetworkArrays
    from endogenous_sue.tied import contested_pairs, solve_tied

    net, od = corpus.load("Eastern-Massachusetts")
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    inclusion = solve_inclusion(net, od, 10.0, DEFAULTS)
    contested, base_masks = contested_pairs(inclusion.masks())
    assert contested, "this cell has a tied face and none was found"

    x, _, record = solve_tied(net, od, 10.0, inclusion.x, contested, base_masks, DEFAULTS)
    listed = record.get("support_base")
    assert listed, "the solve recorded no base for the supports it returned"

    # Exactly as reproduction/exhibits/certificate_sweep rebuilds it: from the recorded base, never
    # from base_masks.
    working = {}
    for d in base_masks:
        links = listed.get(d, listed.get(str(d)))
        if links is None:
            working[d] = base_masks[d].copy()
            continue
        mask = np.zeros(arrays.n_links, dtype=bool)
        if len(links):
            mask[np.asarray(links, dtype=int)] = True
        working[d] = mask

    # Deliberately not asserting that the recorded base equals the absorbing closure of the input: on a
    # route that pins links it does not, and an equality assertion here would pass with the pinning
    # mishandled. What this cell establishes is the record-and-rebuild path; the pinned case is
    # ``test_a_pattern_route_certifies_through_the_base_it_pinned`` below.

    mixture, carried = [], {}
    for d, added, weight in record["vertex_mixture"]:
        mask = working[int(d)].copy()
        for link in added:
            mask[int(link)] = True
        mixture.append((int(d), mask, float(weight)))
        carried[int(d)] = carried.get(int(d), 0.0) + float(weight)
    for d in sorted(int(key) for key in (record.get("base_weight") or {})):
        if d not in working:
            continue
        deficit = 1.0 - carried.get(d, 0.0)
        if deficit > 1e-12:
            mixture.append((d, working[d].copy(), deficit))
    # A destination the mixture does not name is not contested and carries its whole flow on the support
    # its own cost induces, which is the last step the sweep takes before certifying.
    from endogenous_sue.shortestpath import potentials
    from endogenous_sue.subnetwork import active_for_dest
    potential = potentials(topo, arrays.cost(np.maximum(x, 0.0)))
    named = {entry[0] for entry in mixture}
    for d in range(arrays.n_zones):
        if od[:, d].sum() > 0.0 and d not in named:
            mixture.append((d, active_for_dest(topo, potential[:, d], d), 1.0))
    assert mixture

    certificate = certify(net, od, 10.0, np.maximum(x, 0.0), mixture, DEFAULTS)
    assert certificate.mixture_residual <= DEFAULTS.residual_tolerance, (
        f"the recorded supports do not reproduce the flow: {certificate.mixture_residual:.3e}")


def test_a_pattern_route_certifies_through_the_base_it_pinned():
    """The case the absolute-base record exists for, on a cell that actually reaches it.

    ``_search_patterns`` searches against a copy of the base with contested links pinned in and out and
    hands that copy to a recursive solve, so the supports it returns are stated against a base the caller
    does not hold. Rebuilding them on the inclusion scheme's base drops every link pinned in and restores
    every link pinned out, and all three cells solved through this route failed their independent
    certificate that way at mixture residuals of 1e-2.

    Three things have to hold together, and none of them alone is the property:

      the winning route is the pattern route, so the pinning code actually ran;
      at least one recorded base differs from the absorbing closure of the original, so a link was in
      fact pinned and the record is carrying something the caller could not have derived;
      the mixture rebuilt from that record passes the whole independent certificate, not merely its
      mixture-residual clause.

    Slow -- this cell takes around two minutes -- and that is the cost of testing the route rather than a
    helper. The cheaper cells reach ``bracketed link``, which starts from the absorbing closure and never
    pins, so a test written on one of those passes whether this works or not.
    """
    import numpy as np

    from endogenous_sue.certificate import certify
    from endogenous_sue.loading import absorbing
    from endogenous_sue.network import NetworkArrays
    from endogenous_sue.shortestpath import potentials
    from endogenous_sue.subnetwork import active_for_dest
    from endogenous_sue.tied import contested_pairs, solve_tied

    net, od = corpus.load("Berlin-Tiergarten")
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    inclusion = solve_inclusion(net, od, 1.0, DEFAULTS)
    contested, base_masks = contested_pairs(inclusion.masks())
    assert contested

    x, _, record = solve_tied(net, od, 1.0, inclusion.x, contested, base_masks, DEFAULTS)
    assert "pattern of" in str(record.get("route")), (
        f"this cell no longer reaches the pattern route: {record.get('route')!r}; "
        f"find another that does rather than weakening the test")

    listed = record.get("support_base")
    assert listed, "the solve recorded no base for the supports it returned"
    working = {}
    for d in base_masks:
        links = listed.get(d, listed.get(str(d)))
        if links is None:
            working[d] = base_masks[d].copy()
            continue
        mask = np.zeros(arrays.n_links, dtype=bool)
        if len(links):
            mask[np.asarray(links, dtype=int)] = True
        working[d] = mask

    pinned = [d for d in base_masks
              if not np.array_equal(working[d], absorbing(topo, base_masks[d], d))]
    assert pinned, ("no recorded base differs from the absorbing closure of the original, so this run "
                    "never pinned a link and does not exercise what the record is for")

    mixture, carried = [], {}
    for d, added, weight in record["vertex_mixture"]:
        mask = working[int(d)].copy()
        for link in added:
            mask[int(link)] = True
        mixture.append((int(d), mask, float(weight)))
        carried[int(d)] = carried.get(int(d), 0.0) + float(weight)
    for d in sorted(int(key) for key in (record.get("base_weight") or {})):
        if d not in working:
            continue
        deficit = 1.0 - carried.get(d, 0.0)
        if deficit > 1e-12:
            mixture.append((d, working[d].copy(), deficit))
    potential = potentials(topo, arrays.cost(np.maximum(x, 0.0)))
    named = {entry[0] for entry in mixture}
    for d in range(arrays.n_zones):
        if od[:, d].sum() > 0.0 and d not in named:
            mixture.append((d, active_for_dest(topo, potential[:, d], d), 1.0))

    certificate = certify(net, od, 1.0, np.maximum(x, 0.0), mixture, DEFAULTS)
    assert certificate.certified, (
        f"{certificate.kind}: {certificate.reason} "
        f"(mixture residual {certificate.mixture_residual:.3e}, "
        f"{certificate.violations} violation(s), {len(pinned)} destination(s) pinned)")
