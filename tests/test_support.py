"""The efficient support: the centroid clauses, acyclicity, and the isolation margin."""
import numpy as np
from helpers import require_networks, transit_network

from endogenous_sue import witness
from endogenous_sue.network import NetworkArrays, Topology
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import (
    active_for_dest,
    cycle_links,
    isolation_margin,
    kahn_order,
    path_excess_margin,
)

require_networks()

from endogenous_sue import corpus  # noqa: E402

NETWORKS = ["Braess-Example", "SiouxFalls", "Eastern-Massachusetts", "Anaheim"]


def test_the_support_is_acyclic_on_every_corpus_network():
    for name in NETWORKS:
        net, _ = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        potential = potentials(topo, arrays.fft)
        for d in range(topo.n_zones):
            mask = active_for_dest(topo, potential[:, d], d)
            kahn_order(topo, mask)                       # raises on a cycle
            assert cycle_links(topo, mask).size == 0


def test_the_compiled_and_plain_support_rules_agree():
    from endogenous_sue.subnetwork import _active_for_dest_kernel

    for name in NETWORKS:
        net, _ = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        potential = potentials(topo, arrays.fft)
        for d in range(min(topo.n_zones, 8)):
            plain = active_for_dest(topo, potential[:, d], d)
            compiled = _active_for_dest_kernel(topo.tail, topo.head, potential[:, d], d,
                                               topo.n_zones, topo.zero_dep_conn, topo.no_thru,
                                               topo.n_links)
            assert np.array_equal(plain, np.asarray(compiled))


def test_flow_may_always_arrive_at_its_destination():
    """The arrival clause is unconditional and is not nested inside the transit clause."""
    net, od = transit_network()
    topo = NetworkArrays.from_dict(net).topology()
    potential = potentials(topo, np.asarray(net["fft"], dtype=float))
    mask = active_for_dest(topo, potential[:, 2], 2)
    arriving = np.flatnonzero(topo.head == 2)
    assert mask[arriving].all(), "a link into the destination was excluded"


def test_transit_through_a_foreign_centroid_is_forbidden_when_the_data_says_so():
    net, _ = transit_network()
    topo = NetworkArrays.from_dict(net).topology()
    potential = potentials(topo, np.asarray(net["fft"], dtype=float))
    mask = active_for_dest(topo, potential[:, 2], 2)
    into_foreign = np.flatnonzero((topo.head < topo.n_zones) & (topo.head != 2))
    assert not mask[into_foreign].any(), "a link into a foreign centroid was admitted"


def test_the_isolation_margin_excludes_only_what_the_support_rule_excludes():
    """A forced connector is in every admissible support, so its vanishing gap decides nothing.

    Without the exclusion the margin would be zero on every network carrying a zero-cost connector. With
    the exclusion applied more widely than the support rule applies it, the margin would instead be
    vacuous wherever every node is a centroid, because every link then arrives at one.
    """
    # Nodes one and two are zones; three, four and five are through nodes. The departure connector is
    # zero-cost, and there are two through-to-through links whose gaps are what the margin should see.
    net = dict(tail=np.array([1, 3, 3, 4, 5]), head=np.array([3, 4, 5, 2, 2]),
               fft=np.array([0.0, 5.0, 6.0, 1.0, 1.0]), capacity=np.full(5, 1e6),
               b=np.full(5, 0.15), power=np.full(5, 4.0), n_nodes=5, n_zones=2, first_thru=3)
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    assert topo.zero_dep_conn
    margin = isolation_margin(topo, potentials(topo, arrays.fft))
    assert np.isfinite(margin) and margin > 0.0, (
        f"the zero-cost departure connector drove the margin to {margin}")


def test_the_isolation_margin_is_not_vacuous_where_every_node_is_a_centroid():
    for name in NETWORKS:
        net, _ = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        if topo.no_thru or arrays.n_zones != arrays.n_nodes:
            continue
        margin = isolation_margin(topo, potentials(topo, arrays.fft))
        assert np.isfinite(margin), (
            f"{name} permits transit and has no through nodes, so its margin must be a real number")


def test_a_cycle_is_reported_rather_than_ordered():
    """A support containing a directed cycle admits no topological order."""
    topo = Topology.build(np.array([0, 1, 2]), np.array([1, 2, 0]),
                          np.ones(3), 3, 1, first_thru=1)
    mask = np.ones(3, dtype=bool)
    try:
        kahn_order(topo, mask)
    except RuntimeError:
        assert cycle_links(topo, mask).size == 3
        return
    raise AssertionError("a directed cycle was ordered without complaint")


def test_the_witness_topology_matches_its_declared_shape():
    topo = witness.topology()
    assert topo.n_nodes == 5 and topo.n_zones == 2 and topo.n_links == 12


def _exclusion_margin(topo, potential, od):
    """The margin the approximation bound is stated with: smallest gap over every excluded link."""
    best = np.inf
    for d in range(topo.n_zones):
        if od[:, d].sum() <= 0.0:
            continue
        column = potential[:, d]
        finite = np.isfinite(column[topo.tail]) & np.isfinite(column[topo.head])
        gap = np.zeros(topo.n_links)
        np.subtract(column[topo.tail], column[topo.head], out=gap, where=finite)
        eligible = finite & (topo.tail != d) & (topo.head != d)
        if topo.no_thru:
            eligible &= ~((topo.head < topo.n_zones) & (topo.head != d))
        if topo.zero_dep_conn:
            eligible &= ~((topo.tail < topo.n_zones) & (topo.tail != d))
        away = eligible & (gap < 0.0)
        if away.any():
            best = min(best, float(np.abs(gap[away]).min()))
    return best


def test_the_three_margins_are_ordered_as_their_definitions_require():
    """Isolation over every eligible link, exclusion over the excluded ones, path-excess over a subset.

    Each minimises over a smaller set than the last, so the three can only increase in that order. They
    are three different numbers and the tables have already printed one under another's heading once.
    """
    for name in NETWORKS:
        net, od = corpus.load(name)
        arrays = NetworkArrays.from_dict(net)
        topo = arrays.topology()
        potential = potentials(topo, arrays.fft)
        isolation = isolation_margin(topo, potential)
        exclusion = _exclusion_margin(topo, potential, od)
        excess = path_excess_margin(topo, potential, od)
        assert isolation <= exclusion + 1e-12, f"{name}: isolation {isolation} > exclusion {exclusion}"
        assert exclusion <= excess + 1e-12, f"{name}: exclusion {exclusion} > path-excess {excess}"


def test_the_path_excess_margin_ignores_links_no_flow_can_stand_at():
    """A network whose demand reaches one origin only cannot be judged by margins elsewhere."""
    net, od = corpus.load("SiouxFalls")
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    potential = potentials(topo, arrays.fft)
    single = np.zeros_like(od)
    single[0, 1] = od[0, 1] if od[0, 1] > 0 else 1.0
    assert path_excess_margin(topo, potential, single) >= path_excess_margin(topo, potential, od)
