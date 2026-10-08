"""The certificate: each clause fires on a flow that violates it, and none fires on one that does not."""
import numpy as np
from helpers import require_networks, transit_network

from endogenous_sue.certificate import (
    admissibility_violations,
    certify,
    conservation_error,
    reload_on_own_support,
    supports_from_potential,
)
from endogenous_sue.equilibrium import solve
from endogenous_sue.network import NetworkArrays
from endogenous_sue.shortestpath import potentials

require_networks()

from endogenous_sue import corpus  # noqa: E402


def test_a_strict_equilibrium_certifies():
    net, od = transit_network()
    equilibrium = solve(net, od, 1.0)
    certificate = certify(net, od, 1.0, equilibrium.x)
    assert certificate.certified and certificate.kind == "strict"
    assert certificate.reason == ""


def test_the_conservation_clause_fires_on_a_flow_that_lost_demand():
    net, od = transit_network()
    equilibrium = solve(net, od, 1.0)
    damaged = equilibrium.x.copy()
    damaged[2] *= 0.5                                   # remove half the flow on the legal route
    certificate = certify(net, od, 1.0, damaged)
    assert not certificate.certified
    assert "conserved" in certificate.reason


def test_a_relaxed_flow_is_reported_as_not_strict_rather_than_as_a_failure():
    """A mixture is not the loading on any single support, so a large defect is a measurement."""
    net, od = corpus.load("Eastern-Massachusetts")
    equilibrium = solve(net, od, 1.0)
    certificate = certify(net, od, 1.0, equilibrium.x)
    if certificate.certified:
        return                                          # this cell happens to be strict
    assert certificate.kind == "not strict"
    assert "relaxed verdict requires the mixture" in certificate.reason
    assert certificate.violations == 0


def test_conservation_is_independent_of_admissibility():
    """A flow can satisfy every support clause and still not have carried the demand."""
    net, od = corpus.load("SiouxFalls")
    equilibrium = solve(net, od, 1.0)
    worst, total, demand = conservation_error(net, od, equilibrium.x)
    assert worst <= 1e-6 * demand
    scaled = equilibrium.x * 0.9
    assert conservation_error(net, od, scaled)[0] > 1e-6 * demand


def test_the_strict_reload_is_evaluated_on_the_flow_alone():
    net, od = transit_network()
    equilibrium = solve(net, od, 1.0)
    reloaded = reload_on_own_support(net, od, 1.0, equilibrium.x)
    assert np.max(np.abs(reloaded - equilibrium.x)) < 1e-6


def test_a_support_that_drops_a_forced_link_is_a_violation():
    """The model puts the forced links in every admissible support, so omitting one is inadmissible.

    Without this the checker only tested links whose membership the gap decides, and a support that
    stranded demand by dropping an arrival connector passed the inclusion clause. The checker is meant to
    be usable on a flow from any solver, so it cannot rely on the support having been built correctly.
    """
    for name in ("SiouxFalls", "Anaheim"):
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        potential = potentials(topo, arrays.fft)
        supports = supports_from_potential(topo, od, potential)
        assert admissibility_violations(topo, supports, potential)[0] == 0

        d, mask = supports[0]
        forced = np.flatnonzero(mask & (topo.head == d))
        assert forced.size, f"{name} has no forced arrival link for destination {d}"
        dropped = mask.copy()
        dropped[forced[0]] = False
        assert admissibility_violations(topo, [(d, dropped)], potential)[0] == 1
