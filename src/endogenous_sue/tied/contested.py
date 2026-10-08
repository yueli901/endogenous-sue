"""Finding the contested links, and the families of supports that resolve them.

The inclusion scheme ends on a window of late supports. A link on which those supports disagree is
*contested*: the tie is unresolved there, and the intersection of the late supports is the base support
that every resolution contains.
"""
from __future__ import annotations

from itertools import product

import numpy as np

from ..network import Topology
from ..subnetwork import kahn_order

__all__ = ["contested_pairs", "orientation_family", "subset_family"]


def contested_pairs(supports):
    """The contested ``(destination, link)`` pairs, with the per-destination base support.

    ``supports`` is an iterable of ``{destination: boolean mask}`` -- the window the inclusion scheme
    returns, which :meth:`InclusionResult.masks` yields one at a time. Neither the network nor the demand
    is needed: which destinations carry demand is already decided by which ones the window holds. The
    base support is the
    intersection of those supports and the contested links are those in their union but not their
    intersection, both accumulated as the window streams past, so the whole window is never held at once.
    """
    base_masks, unions = {}, {}
    for support in supports:
        for d, mask in support.items():
            if d in base_masks:
                base_masks[d] &= mask
                unions[d] |= mask
            else:
                base_masks[d] = mask.copy()
                unions[d] = mask.copy()
    if not base_masks:
        raise ValueError("the inclusion scheme returned no supports")
    contested = [(d, int(link)) for d in sorted(base_masks)
                 for link in np.flatnonzero(unions[d] & ~base_masks[d])]
    return contested, base_masks


def orientation_family(topo: Topology, base: np.ndarray, links: list):
    """Maximal admissible resolutions of a chained tie: one link per tied node pair, acyclic only."""
    grouped: dict = {}
    for link in links:
        origin, destination = int(topo.tail[link]), int(topo.head[link])
        key = (min(origin, destination), max(origin, destination))
        grouped.setdefault(key, []).append(link)

    family = []
    for combination in product(*grouped.values()):
        mask = base.copy()
        for link in combination:
            mask[link] = True
        try:
            kahn_order(topo, mask)
        except RuntimeError:
            continue
        family.append((tuple(int(link) for link in combination), mask))
    return family


def subset_family(topo: Topology, base: np.ndarray, links: list, limit: int = 4096):
    """Admissible supports formed from the base plus a subset of the tied links.

    Enumerated in full while the number of subsets fits the limit. Beyond it the family is the base, the
    single-link additions, and the maximal orientations, which are the compositions observed to carry
    weight.
    """
    count = len(links)
    if 2 ** count <= limit:
        family = []
        for bits in range(2 ** count):
            mask = base.copy()
            added = tuple(links[j] for j in range(count) if bits >> j & 1)
            for link in added:
                mask[link] = True
            try:
                kahn_order(topo, mask)
            except RuntimeError:
                continue
            family.append((added, mask))
        return family

    family = [((), base.copy())]
    for link in links:
        mask = base.copy()
        mask[link] = True
        family.append(((link,), mask))
    edges = {(min(topo.tail[link], topo.head[link]), max(topo.tail[link], topo.head[link]))
             for link in links}
    if 2 ** len(edges) <= limit:
        family += orientation_family(topo, base, links)
    return family
