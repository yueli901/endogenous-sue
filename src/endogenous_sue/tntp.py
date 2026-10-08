"""Reading TNTP benchmark networks.

The format is the one used by the Transportation Networks for Research repository. A network is returned
as a dictionary of arrays; :func:`read_network` reads a matched network and trips pair and is the normal
entry point.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_HEADERS = {
    "<NUMBER OF ZONES>": "n_zones",
    "<NUMBER OF NODES>": "n_nodes",
    "<FIRST THRU NODE>": "first_thru",
    "<NUMBER OF LINKS>": "n_links",
}


def parse_net(net_path: str | Path, fft_floor: float = 0.0) -> dict:
    """Parse a ``*_net.tntp`` file.

    Node ids stay 1-indexed, as in the file. Keys returned are ``n_zones``, ``n_nodes``, ``first_thru``,
    ``n_links``, ``node_ids``, ``tail``, ``head``, ``capacity``, ``fft``, ``b`` and ``power``.

    ``fft_floor`` raises free-flow times below it, but only on links between two non-centroid nodes:
    centroid connectors routinely have a free-flow time of zero by convention, and that zero carries
    meaning the rest of the package relies on.
    """
    path = Path(net_path)
    lines = path.read_text().splitlines()

    meta: dict = {}
    data_start = 0
    for i, raw in enumerate(lines):
        line = raw.strip()
        for header, key in _HEADERS.items():
            if line.startswith(header):
                meta[key] = int(re.search(r"\d+", line).group())
        if line.startswith("<END OF METADATA>"):
            data_start = i + 1
            break

    rows: dict[str, list] = {k: [] for k in ("tail", "head", "capacity", "fft", "b", "power")}
    for raw in lines[data_start:]:
        line = raw.strip()
        if not line or line.startswith("~") or line.startswith("<"):
            continue
        parts = line.rstrip(";").split()
        if len(parts) < 6:
            continue
        try:
            rows["tail"].append(int(parts[0]))
            rows["head"].append(int(parts[1]))
            rows["capacity"].append(float(parts[2]))
            rows["fft"].append(float(parts[4]))
            rows["b"].append(float(parts[5]))
            rows["power"].append(float(parts[6]) if len(parts) > 6 else 4.0)
        except (ValueError, IndexError):
            continue

    fft = np.array(rows["fft"], dtype=np.float64)
    tail = np.array(rows["tail"], dtype=np.int64)
    head = np.array(rows["head"], dtype=np.int64)
    if fft_floor > 0.0:
        n_zones = meta.get("n_zones", 0)
        interior = (tail > n_zones) & (head > n_zones)
        fft = np.where((fft < fft_floor) & interior, fft_floor, fft)

    tail, head, node_ids = _contiguous_node_ids(tail, head, meta)
    return {
        **meta,
        "node_ids": node_ids,
        "tail": tail.astype(np.int32),
        "head": head.astype(np.int32),
        "capacity": np.array(rows["capacity"], dtype=np.float64),
        "fft": fft,
        "b": np.array(rows["b"], dtype=np.float64),
        "power": np.array(rows["power"], dtype=np.float64),
    }


def _contiguous_node_ids(tail, head, meta):
    """Renumber nodes to ``1..N`` when the file does not already, preserving order.

    Everything downstream indexes per-node arrays by ``id - 1``, which assumes the file numbers its nodes
    from one. Munich numbers its by OpenStreetMap id, so that assumption indexes past the end of every
    per-node array. Renumbering preserves order because TNTP lists centroids first and the convention that
    a node below ``n_zones`` is a centroid is relied on throughout. The mapping is returned so that
    :func:`parse_trips` can apply the identical one rather than derive its own.
    """
    if not len(tail):
        return tail, head, None
    ids = np.unique(np.concatenate((tail, head)))
    if ids[0] == 1 and ids[-1] == int(meta.get("n_nodes", 0)):
        return tail, head, None
    remap = {int(v): i + 1 for i, v in enumerate(ids)}
    tail = np.array([remap[int(v)] for v in tail], dtype=np.int64)
    head = np.array([remap[int(v)] for v in head], dtype=np.int64)
    meta["n_nodes"] = len(ids)
    return tail, head, ids


def parse_trips(trips_path: str | Path, node_ids=None) -> tuple[int, np.ndarray]:
    """Parse a ``*_trips.tntp`` file into ``(n_zones, od)``, with ``od`` 0-indexed by zone.

    ``node_ids`` is the renumbering :func:`parse_net` applied. The trips file uses the same original ids,
    so it must be renumbered identically, and the mapping is passed in rather than recomputed so that the
    two cannot disagree. ``None`` means the network was already contiguous.
    """
    path = Path(trips_path)
    text = path.read_text()
    n_zones = int(re.search(r"<NUMBER OF ZONES>\s*(\d+)", text).group(1))
    od = np.zeros((n_zones, n_zones), dtype=np.float64)
    zmap = {int(v): i + 1 for i, v in enumerate(node_ids)} if node_ids is not None else None

    if zmap is None:
        # Refuse rather than guess. If the ids here are not 1..n_zones then they are the network's own
        # node ids, and only the network knows their order: a mapping derived from this file alone would
        # number the origins by their own sort, attaching each origin's demand to a different node. That
        # is silently wrong output, which is worse than a crash.
        raw = [int(v) for v in re.findall(r"Origin\s+(\d+)", text)]
        if raw and (min(raw) < 1 or max(raw) > n_zones):
            distinct = sorted(set(raw))
            raise ValueError(
                f"{path.name} uses non-sequential zone ids ({len(distinct)} distinct, "
                f"{distinct[0]}..{distinct[-1]}) against <NUMBER OF ZONES> {n_zones}. Pass "
                f"node_ids=parse_net(...)['node_ids'] so origins are mapped by the network's node order.")

    for block in re.split(r"Origin\s+", text)[1:]:
        block_lines = block.strip().split("\n")
        origin_raw = int(block_lines[0].strip())
        origin = (zmap.get(origin_raw, 0) if zmap is not None else origin_raw) - 1
        for line in block_lines[1:]:
            for pair in line.strip().rstrip(";").split(";"):
                pair = pair.strip()
                if ":" not in pair:
                    continue
                dest_str, demand_str = pair.split(":")
                dest_raw = int(dest_str.strip())
                dest = (zmap.get(dest_raw, 0) if zmap is not None else dest_raw) - 1
                if 0 <= dest < n_zones and 0 <= origin < n_zones:
                    od[origin, dest] = float(demand_str.strip())
    return n_zones, od


def read_network(net_path: str | Path, trips_path: str | Path,
                 fft_floor: float = 0.0) -> tuple[dict, np.ndarray]:
    """Read a matched network and trips pair. Returns ``(net, od)``.

    The two files each declare a zone count and they must agree. Where they do not, the trips file's
    origin and destination ids do not index the network file's centroids, so nothing downstream can be
    read as an assignment on this network: a demand matrix narrower than the zone set makes
    ``od[:, d]`` raise for the zones past its edge, and one merely mislabelled would load silently onto
    the wrong centroids, which is worse. Refused here, where both numbers are in hand for the first
    time.

    Every network of the reported corpus agrees, so this never fired until Munich was swept as a
    held-out network: its ``_net`` file declares 742 zones and its trips file 284, and the mismatch
    surfaced as an ``IndexError`` inside the inclusion scheme after the pre-flight had already warned
    about it in words and continued anyway.
    """
    net = parse_net(net_path, fft_floor=fft_floor)
    trips_zones, od = parse_trips(trips_path, node_ids=net.get("node_ids"))
    if trips_zones != net["n_zones"]:
        raise ValueError(
            f"{Path(net_path).name} declares {net['n_zones']} zones and "
            f"{Path(trips_path).name} declares {trips_zones}; the two files do not describe the same "
            f"network and no assignment is defined on the pair")
    return net, od
