"""The contract a solved face must satisfy, checked from the record alone.

``docs/face_contract.md`` states the clauses. These exercise them against real solves rather than
against a constructed record, because every defect the contract exists to prevent was a disagreement
between what the solver wrote and what a consumer read, and a hand-built record has no solver side.

The reconstruction here is deliberately a second implementation, written from the document rather than
copied from ``reproduction/exhibits/certificate_sweep``. A test that reuses the consumer's own code
cannot detect
the consumer and the producer agreeing on the wrong thing, which is what happened three times.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from helpers import two_destination_witness

from endogenous_sue import witness
from endogenous_sue.certificate import certify, violating_pairs
from endogenous_sue.config import ACTIVE_TOLERANCE, TIE_TOLERANCE, Settings
from endogenous_sue.costs import bpr_cost
from endogenous_sue.equilibrium import solve_inclusion
from endogenous_sue.loading import load_destination
from endogenous_sue.network import Topology, unpack
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import kahn_order
from endogenous_sue.tied import contested_pairs, solve_face_admissible, solve_tied

SETTINGS = Settings(max_phase1_iterations=1500)


def rebuild(record, n_links, base_masks):
    """The mixture a consumer holding only the record can form. C1, C2 and C4.

    Returns ``[(destination, mask, weight)]``. The deficit rule is applied here and ``base_weight`` is
    never read, which is the whole of C4.
    """
    listed = record.get("support_base") or {}
    base = {}
    for d in base_masks:
        links = listed.get(str(d), listed.get(d))
        if links is None:
            base[d] = np.asarray(base_masks[d]).copy()
            continue
        mask = np.zeros(n_links, dtype=bool)
        if len(links):
            mask[np.asarray(links, dtype=int)] = True
        base[d] = mask

    mixture, carried = [], {}
    for d, added, weight in record.get("vertex_mixture") or []:
        d, weight = int(d), float(weight)
        if weight <= ACTIVE_TOLERANCE:
            continue
        mask = base[d].copy()
        for link in added:
            mask[int(link)] = True
        mixture.append((d, mask, weight))
        carried[d] = carried.get(d, 0.0) + weight
    for d in sorted(base):
        deficit = 1.0 - carried.get(d, 0.0)
        if deficit > ACTIVE_TOLERANCE:
            mixture.append((d, base[d].copy(), deficit))
    return mixture


def check_contract(net, od, mu, x, record, base_masks, settings=SETTINGS):
    """Every clause of docs/face_contract.md that a record can be held to on its own."""
    tail, head, fft, cap, b, power, n_nodes, n_zones = unpack(net)
    topo = Topology.build(tail, head, fft, n_nodes, n_zones, net.get("first_thru"))
    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    potential = potentials(topo, cost)
    demanded = {d for d in range(n_zones) if od[:, d].sum() > 0.0}
    mixture = rebuild(record, topo.n_links, base_masks)

    # C1: every destination carrying demand is accounted for, at total weight one.
    total: dict = {}
    for d, _, weight in mixture:
        total[d] = total.get(d, 0.0) + weight
    for d in demanded & set(base_masks):
        assert d in total, f"C1: destination {d} carries demand and no support"
        assert abs(total[d] - 1.0) < 1e-9, f"C1: destination {d} carries weight {total[d]}"

    # C2: the recorded base is absolute, so every index names a link of this network.
    for d, links in (record.get("support_base") or {}).items():
        assert all(0 <= int(link) < topo.n_links for link in links), f"C2: destination {d}"

    # C7: a support that cannot be ordered cannot be loaded.
    for _d, mask, _ in mixture:
        kahn_order(topo, mask)

    # C3: the record alone rebuilds the flow it describes.
    rebuilt = np.zeros(topo.n_links)
    for d, mask, weight in mixture:
        rebuilt += weight * load_destination(topo, od, mu, cost, d, mask, kahn_order(topo, mask))
    residual = float(np.max(np.abs(x - rebuilt))) / max(float(np.abs(x).max()), 1e-30)
    assert residual < settings.residual_tolerance, f"C3: reconstruction residual {residual:.3e}"

    # C5: every support carrying weight is admissible at the solved flow's own cost.
    refused = violating_pairs(topo, [(d, mask) for d, mask, _ in mixture], potential,
                              tolerance=TIE_TOLERANCE)
    assert not refused, f"C5: {len(refused)} pair(s) the final cost refuses: {refused[:5]}"

    # C6: the pair marginals and the mixture are two views of one object.
    for d, link, value in record.get("weights") or []:
        d, link, value = int(d), int(link), float(value)
        carried = sum(weight for other, mask, weight in mixture if other == d and mask[link])
        assert abs(carried - value) < 1e-6, (
            f"C6: pair ({d}, {link}) marginal {value} against {carried} in the mixture")
    return mixture


def certifies(net, od, mu, x, mixture, settings=SETTINGS):
    """The independent verdict, which is what decides a row."""
    return certify(net, od, mu, x, mixture, settings)


def test_the_witness_record_satisfies_every_clause():
    """The five-node witness is the smallest case whose equilibrium is genuinely a mixture."""
    net, od = witness.network(), witness.demand()
    inclusion = solve_inclusion(net, od, 1.0, SETTINGS)
    contested, base = contested_pairs(inclusion.masks())
    x, _, record = solve_tied(net, od, 1.0, inclusion.x, contested, base, SETTINGS)

    mixture = check_contract(net, od, 1.0, x, record, base)
    assert certifies(net, od, 1.0, x, mixture).certified, "the rebuilt mixture does not certify"


def test_the_contract_holds_through_a_nested_route():
    """The accelerating routes return an inner solve's record, and C2 is what makes that safe.

    The route is asserted to be one of the recursive ones. A run that happened to take the plain walk
    would satisfy every clause without exercising recursion at all, which is the absence of the
    condition under test rather than evidence about it.
    """
    net, od = witness.network(), witness.demand()
    inclusion = solve_inclusion(net, od, 1.0, SETTINGS)
    contested, base = contested_pairs(inclusion.masks())
    x, _, record = solve_tied(net, od, 1.0, inclusion.x, contested, base, SETTINGS)

    route = record.get("route", "")
    assert route.startswith(("bracketed link", "single link", "orientation", "pattern of")), (
        f"no recursive route was taken, so recursion is untested here: {route!r}")
    mixture = check_contract(net, od, 1.0, x, record, base)
    assert certifies(net, od, 1.0, x, mixture).certified


def test_a_destination_carrying_only_base_mass_is_still_in_the_mixture():
    """C1 and C4 together, on a network built so that the condition is present rather than hoped for.

    The plain witness has one destination and cannot exercise this at all: every run of it leaves the
    unnamed set empty, so a test written against it asserts nothing and passes. The two-destination
    variant splits destination 1 across a mixture and leaves destination 2 whole on its base, where it
    appears in no vertex entry. Rebuilding from the listed vertices alone silently drops its entire
    flow, which is what the last assertion here measures.
    """
    net, od = two_destination_witness()
    inclusion = solve_inclusion(net, od, 1.0, SETTINGS)
    contested, base = contested_pairs(inclusion.masks())
    x, _, record = solve_tied(net, od, 1.0, inclusion.x, contested, base, SETTINGS)

    named = {int(entry[0]) for entry in (record.get("vertex_mixture") or [])}
    listed = {int(d) for d in (record.get("support_base") or {})}
    unnamed = sorted(listed - named)
    assert unnamed, "no destination carries only base mass here, so the clause is untested"

    mixture = check_contract(net, od, 1.0, x, record, base)
    for d in unnamed:
        assert any(other == d for other, _, _ in mixture), (
            f"destination {d} is named by no vertex entry and the deficit rule did not restore it")
    assert certifies(net, od, 1.0, x, mixture).certified

    # The teeth: without the deficit rule those destinations vanish and the flow is materially wrong.
    # A reconstruction that still matched would mean this network cannot detect the defect.
    tail, head, fft, cap, b, power, n_nodes, n_zones = unpack(net)
    topo = Topology.build(tail, head, fft, n_nodes, n_zones, net.get("first_thru"))
    cost = bpr_cost(np.maximum(x, 0.0), fft, cap, b, power)
    vertices_only = [entry for entry in mixture if int(entry[0]) in named]
    partial = np.zeros(topo.n_links)
    for d, mask, weight in vertices_only:
        partial += weight * load_destination(topo, od, 1.0, cost, d, mask, kahn_order(topo, mask))
    dropped = float(np.max(np.abs(x - partial))) / max(float(np.abs(x).max()), 1e-30)
    assert dropped > 1e-3, (
        f"dropping the base-only destinations changes the flow by only {dropped:.3e}, "
        "so this network cannot detect the defect the deficit rule exists to prevent")


def test_reopening_clears_the_link_from_the_base_and_not_only_from_the_tie_set():
    """A pair reopened into the tie set while still in the base is inert.

    The two vertices of that tie are ``base`` and ``base union {link}``, which are the same support when
    the link is already in the base, so the weight cannot move anything and the violation survives the
    reopen. This asserts the base mask the reopened solve receives actually lost the link.
    """
    net, od = witness.network(), witness.demand()
    inclusion = solve_inclusion(net, od, 1.0, SETTINGS)
    contested, base = contested_pairs(inclusion.masks())

    seen: list = []
    original = solve_tied

    def spy(net_, od_, mu_, x0_, pairs_, masks_, *args, **kwargs):
        seen.append({int(d): np.asarray(m).copy() for d, m in masks_.items()})
        return original(net_, od_, mu_, x0_, pairs_, masks_, *args, **kwargs)

    import endogenous_sue.tied.solver as solver
    solver.solve_tied = spy
    try:
        forced = {d: np.asarray(mask).copy() for d, mask in base.items()}
        # Put a link the equilibrium cost excludes into one destination's base, so the first attempt
        # must refuse it and the reopen must take it back out.
        d0 = sorted(forced)[0]
        cost = bpr_cost(np.maximum(inclusion.x, 0.0), *unpack(net)[2:6])
        tail, head = unpack(net)[0], unpack(net)[1]
        topo = Topology.build(*unpack(net)[:3], *unpack(net)[6:], net.get("first_thru"))
        potential = potentials(topo, cost)
        excluded = [int(L) for L in range(topo.n_links)
                    if not forced[d0][L] and tail[L] != d0
                    and float(potential[tail[L], d0] - potential[head[L], d0]) < -TIE_TOLERANCE]
        assert excluded, ("no link on this network is excluded at the equilibrium cost, "
                          "so there is nothing to force in and the clause is untested")
        link = excluded[0]
        forced[d0][link] = True
        solve_face_admissible(net, od, 1.0, inclusion.x, contested, forced, SETTINGS,
                              tie_tolerance=TIE_TOLERANCE, rounds=2)
    finally:
        solver.solve_tied = original

    assert len(seen) >= 2, "the face was never reopened, so the clause is untested"
    assert seen[0][d0][link], "the forced link was not in the base of the first attempt"
    assert not seen[-1][d0][link], (
        "the reopened attempt still carries the link in its base, so the tie it added is inert")


def test_two_different_reopen_sets_do_not_share_a_checkpoint():
    """A reopen checkpoint is named for the pairs it reopened, not for the attempt number.

    ``solve_tied`` restores a stored walk without checking that its tie set is the one being asked for.
    A file named for the attempt alone would therefore be loaded by any first reopen of that cell,
    including one from an earlier submission that reopened different pairs, and the resumed run would
    solve the previous face while reporting the current one. The cleanup in ``Exhibit.clear_checkpoint``
    still has to find them, which it does by the ``<stem>_`` rule and not by a bare prefix.
    """
    from hashlib import blake2b

    from endogenous_sue.tied.solver import _sibling

    def path_for(pairs):
        return _sibling(Path("/tmp/cells/Anaheim_mu0.5.npz"),
                        "reopen" + blake2b(repr(sorted(pairs)).encode(),
                                           digest_size=5).hexdigest())

    one = path_for([(4, 572), (29, 454)])
    other = path_for([(4, 572), (17, 572)])
    assert one != other, "two different reopen sets share a checkpoint and will resume each other"
    assert path_for([(29, 454), (4, 572)]) == one, "the name depends on the order of the pairs"
    assert one.name.startswith("Anaheim_mu0.5_"), (
        f"{one.name} is not a sibling of its cell, so clear_checkpoint will leave it behind")
