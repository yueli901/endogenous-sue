"""Recursive-logit loading: conservation, the parallel path, and explicitly supplied supports."""
import numpy as np
from helpers import require_networks

from endogenous_sue.loading import LoadingReport, load, load_parallel, load_with_masks, prepare_masks
from endogenous_sue.network import NetworkArrays
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import active_for_dest

require_networks()

from endogenous_sue import corpus  # noqa: E402

NETWORKS = ["Braess-Example", "SiouxFalls", "Eastern-Massachusetts", "Anaheim"]


def _loaded(name, mu=1.0):
    net, od = corpus.load(name)
    arrays = NetworkArrays.from_dict(net)
    topo = arrays.topology()
    potential = potentials(topo, arrays.fft)
    return net, od, arrays, topo, potential


def test_the_loading_carries_the_whole_demand():
    for name in NETWORKS:
        net, od, arrays, topo, potential = _loaded(name)
        x = load(topo, od, 2.0, arrays.fft, potential)
        divergence = np.zeros(arrays.n_nodes)
        np.add.at(divergence, arrays.tail, x)
        np.add.at(divergence, arrays.head, -x)
        expected = np.zeros(arrays.n_nodes)
        expected[:arrays.n_zones] = od.sum(axis=1) - od.sum(axis=0)
        assert np.abs(divergence - expected).max() < 1e-6 * max(float(od.sum()), 1.0)


def test_the_parallel_path_gives_the_same_answer():
    for name in NETWORKS:
        net, od, arrays, topo, potential = _loaded(name)
        serial = load(topo, od, 2.0, arrays.fft, potential)
        parallel = load_parallel(topo, od, 2.0, arrays.fft, potential)
        assert np.allclose(serial, parallel, rtol=0, atol=1e-9)


def test_a_supplied_support_gives_the_same_answer_as_a_derived_one():
    for name in NETWORKS[:3]:
        net, od, arrays, topo, potential = _loaded(name)
        masks = {d: active_for_dest(topo, potential[:, d], d) for d in range(topo.n_zones)}
        derived = load(topo, od, 2.0, arrays.fft, potential)
        supplied = load_with_masks(topo, od, 2.0, arrays.fft, masks)
        assert np.allclose(derived, supplied, rtol=0, atol=1e-9)


def test_preparing_masks_refuses_a_destination_that_has_none():
    net, od, arrays, topo, potential = _loaded("SiouxFalls")
    masks = {d: active_for_dest(topo, potential[:, d], d) for d in range(1, topo.n_zones)}
    try:
        prepare_masks(topo, od, masks)
    except KeyError:
        return
    raise AssertionError("a demanded destination with no supplied mask was accepted, which would "
                         "silently drop its demand")


def test_the_report_records_absence_rather_than_zero():
    """The parallel path cannot measure the support diagnostics, and must not report them as zero."""
    net, od, arrays, topo, potential = _loaded("SiouxFalls")
    report = LoadingReport()
    load(topo, od, 2.0, arrays.fft, potential, cache={}, report=report)
    assert report.support_changes is not None
    assert LoadingReport().support_changes is None
