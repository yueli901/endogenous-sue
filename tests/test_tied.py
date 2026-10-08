"""The set-valued case: the witness certifies, and the atoms represent what they claim to."""
import numpy as np

from endogenous_sue import witness
from endogenous_sue.config import Settings
from endogenous_sue.equilibrium import solve_inclusion
from endogenous_sue.network import NetworkArrays
from endogenous_sue.tied import contested_pairs, solve_tied
from endogenous_sue.tied.atoms import build_groups, marginals, nested_atoms, ring_feasible

SETTINGS = Settings(max_phase1_iterations=1500)


def test_the_witness_needs_a_mixture_and_gets_a_certified_one():
    """At unit dispersion no single support reproduces itself, so the equilibrium must be a mixture."""
    net, od = witness.network(), witness.demand()
    inclusion = solve_inclusion(net, od, 1.0, SETTINGS)
    contested, base = contested_pairs(inclusion.masks())
    assert contested, "the witness has unresolved ties and none were found"

    x, weights, record = solve_tied(net, od, 1.0, inclusion.x, contested, base, SETTINGS)
    assert record["verdict"] == "CERTIFIED", record["reason"]
    assert record["tie_gap"] < 1e-9
    assert record["violations"] == 0
    assert record["conservation_max"] < 1e-6 * float(od.sum())


def test_the_nested_atoms_have_the_marginals_they_were_asked_for():
    for weights in ([0.3], [0.25, 0.75], [0.1, 0.5, 0.9, 0.2]):
        cells = nested_atoms(np.asarray(weights))
        assert abs(sum(mass for _, mass in cells) - 1.0) < 1e-12
        assert np.allclose(marginals(cells, len(weights)), weights, atol=1e-12)


def test_the_nested_family_is_small():
    """One weight per group realises any point of the cube with at most one atom more than groups."""
    for k in (1, 2, 3, 5, 8):
        weights = np.linspace(0.1, 0.9, k)
        assert len(nested_atoms(weights)) <= k + 1


def test_a_ring_that_cannot_be_avoided_is_refused_rather_than_loaded():
    """Where the marginals force the whole ring under every coupling, no acyclic representation exists."""
    ring = [(0, True), (1, True), (2, True)]
    feasible, slack = ring_feasible(np.array([1.0, 1.0, 1.0]), ring)
    assert not feasible and slack < 0
    try:
        nested_atoms(np.array([1.0, 1.0, 1.0]), ring)
    except ValueError:
        return
    raise AssertionError("an unavoidable ring was represented instead of refused")


def test_cutting_a_ring_keeps_the_marginals_and_removes_the_cycle():
    ring = [(0, True), (1, True), (2, True)]
    weights = np.array([0.5, 0.5, 0.5])
    feasible, _ = ring_feasible(weights, ring)
    assert feasible
    cells = nested_atoms(weights, ring)
    assert np.allclose(marginals(cells, 3), weights, atol=1e-12)
    for admit, mass in cells:
        if mass > 1e-12:
            assert not admit.all(), "an atom admitted the whole ring"


def test_a_group_is_an_edge_and_carries_at_most_two_orientations():
    topo = witness.topology()
    items = [(0, 6), (1, 7)]                       # A->B and B->A, one edge, two orientations
    groups = build_groups(topo, items)
    assert len(groups) == 1 and groups[0].paired


def test_the_envelope_diagnostic_counts_the_same_face_as_the_solver():
    """The diagnostic and the face solver have to be reading one object.

    ``contested_pairs`` consumes the window's per-destination intersection and union and nothing else,
    which is the whole reason a window whose support identity never settles can still define one face.
    ``reproduction/tools/envelope_diagnostic`` derives the same two from the packed masks the inclusion
    scheme hands
    its observer. If the two ever disagreed, the diagnostic would be answering a question the solver does
    not ask, and its verdict on whether a cell is certifiable would mean nothing.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reproduction.tools.envelope_diagnostic import _face

    net, od = witness.network(), witness.demand()
    result = solve_inclusion(net, od, 1.0, SETTINGS)
    expected, _ = contested_pairs(result.masks())

    topo = NetworkArrays.from_dict(net).topology()
    base, union = {}, {}
    for support in result.masks():
        for d, mask in support.items():
            packed = np.packbits(mask)
            if d in base:
                base[d] &= packed
                union[d] |= packed
            else:
                base[d], union[d] = packed.copy(), packed.copy()

    pairs, links, groups = _face(topo, base, union)
    assert pairs == len(expected), f"diagnostic counted {pairs} contested pairs, solver {len(expected)}"
    assert links <= pairs and groups <= links, ("a face cannot have more links than pairs, or more "
                                                "groups than links")


