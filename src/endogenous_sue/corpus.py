"""Access to the benchmark networks.

Networks are the TNTP files of the Transportation Networks for Research repository, which is not
redistributed with this package. ``reproduction/tools/fetch_networks.py`` clones it at the pinned commit
recorded in
:mod:`endogenous_sue.config`; set ``ENDOGENOUS_SUE_DATA`` to point at an existing clone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import CORPUS, DATA_ROOT
from .tntp import read_network

__all__ = ["available", "paths", "load", "select"]


def available() -> list[tuple[str, Path, Path]]:
    """Every directory under the data root holding both a network and a trips file."""
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"network data not found at {DATA_ROOT}. Run reproduction/tools/fetch_networks.py to clone "
            f"it at the "
            f"pinned commit, or set ENDOGENOUS_SUE_DATA to an existing clone.")
    found = []
    for directory in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir() and not p.name.startswith("_")):
        networks = sorted(directory.glob("*_net.tntp"))
        trips = [t for t in sorted(directory.glob("*_trips*.tntp")) if t.suffix != ".zip"]
        if networks and trips:
            found.append((directory.name, networks[0], trips[0]))
    return found


def paths(name: str) -> tuple[Path, Path]:
    """The network and trips file of one benchmark, by directory name."""
    for found_name, network, trips in available():
        if found_name == name:
            return network, trips
    raise KeyError(f"network {name!r} not found under {DATA_ROOT}")


def load(name: str, fft_floor: float = 0.0) -> tuple[dict, np.ndarray]:
    """Read one network by name. Returns the network dictionary and the demand matrix."""
    network_path, trips_path = paths(name)
    return read_network(network_path, trips_path, fft_floor=fft_floor)


def select(names: str | None = None, max_links: int | None = None) -> list[str]:
    """Resolve a comma-separated selection against what is available, in corpus order."""
    present = {name for name, _, _ in available()}
    if names:
        chosen = [name.strip() for name in names.split(",")]
        missing = [name for name in chosen if name not in present]
        if missing:
            raise KeyError(f"not found under {DATA_ROOT}: {', '.join(missing)}")
    else:
        chosen = [name for name in CORPUS if name in present]
    if max_links is not None:
        chosen = [name for name in chosen if len(load(name)[0]["tail"]) <= max_links]
    return chosen
