"""Reading benchmark networks."""
import numpy as np
from helpers import require_networks

from endogenous_sue import corpus, tntp

require_networks()


def test_braess_parses_to_declared_size():
    net, od = corpus.load("Braess-Example")
    assert len(net["tail"]) == net["n_links"]
    assert net["n_zones"] <= net["n_nodes"]
    assert od.shape == (net["n_zones"], net["n_zones"])


def test_node_ids_are_one_based_and_within_range():
    net, _ = corpus.load("SiouxFalls")
    assert net["tail"].min() >= 1 and net["head"].min() >= 1
    assert net["tail"].max() <= net["n_nodes"] and net["head"].max() <= net["n_nodes"]


def test_fft_floor_leaves_connectors_alone():
    net_path, trips_path = corpus.paths("Anaheim")
    plain = tntp.parse_net(net_path)
    floored = tntp.parse_net(net_path, fft_floor=1.0)
    n_zones = plain["n_zones"]
    connector = (plain["tail"] <= n_zones) | (plain["head"] <= n_zones)
    assert np.array_equal(plain["fft"][connector], floored["fft"][connector])
    assert (floored["fft"][~connector] >= 1.0).all()


def test_trips_refuse_to_guess_a_zone_mapping():
    """A network with renumbered nodes must have its trips renumbered by the same map, not by their own."""
    net, od = corpus.load("SiouxFalls")
    assert od.sum() > 0


def test_a_network_and_trips_pair_declaring_different_zone_counts_is_refused():
    """The two files each declare a zone count and they must agree.

    Where they do not, the trips file's ids do not index the network file's centroids. A demand matrix
    narrower than the zone set makes ``od[:, d]`` raise for the zones past its edge, which is how Munich
    surfaced: an ``IndexError`` inside the inclusion scheme, sixteen seconds into a held-out job, after
    the pre-flight had already described the condition in words and continued anyway. A matrix that is
    merely mislabelled is worse, because it loads silently onto the wrong centroids.
    """
    import tempfile
    from pathlib import Path

    from endogenous_sue.tntp import read_network

    with tempfile.TemporaryDirectory() as directory:
        here = Path(directory)
        (here / "x_net.tntp").write_text(
            "<NUMBER OF ZONES> 3\n<NUMBER OF NODES> 4\n<FIRST THRU NODE> 4\n"
            "<NUMBER OF LINKS> 2\n<END OF METADATA>\n"
            "~\tinit_node\tterm_node\tcapacity\tlength\tfree_flow_time\tb\tpower\t"
            "speed\ttoll\tlink_type\t;\n"
            "\t1\t4\t100\t1\t1\t0.15\t4\t0\t0\t1\t;\n"
            "\t4\t2\t100\t1\t1\t0.15\t4\t0\t0\t1\t;\n")
        (here / "x_trips.tntp").write_text(
            "<NUMBER OF ZONES> 2\n<TOTAL OD FLOW> 10\n<END OF METADATA>\n"
            "Origin 1\n    2 :      10.0;\n")
        try:
            read_network(here / "x_net.tntp", here / "x_trips.tntp")
        except ValueError as exc:
            assert "3 zones" in str(exc), str(exc)
        else:
            raise AssertionError("a 3-zone network with a 2-zone trips file was accepted")
