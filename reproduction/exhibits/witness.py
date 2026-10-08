r"""The five-node witness: a network on which no strict equilibrium exists.

Small enough to enumerate exhaustively and large enough to carry the phenomenon. The enumeration is done
twice, by two arguments that do not share an assumption.

The first enumerates the orderings of the node potentials. Every strict efficient set is the set of links
along which some potential strictly decreases, so the orderings enumerate the candidates. The second drops
that argument and enumerates the link subsets directly, all \(2^{12}-1\) of them, taking each as a support
in its own right. The two must agree, and the second is what the claim of exhaustiveness rests on.

Each candidate is solved on its own support and the support the resulting cost induces is compared with
the one assumed. A support that reproduces itself is a strict equilibrium; at moderate dispersion none
does, which is why the equilibrium has to be a mixture.

Cyclic and empty supports are counted rather than discarded: a support admitting no link routes nothing,
and one containing a directed cycle admits no topological order and is not a candidate at all. Reporting
the three counts together is what makes the enumeration exhaustive rather than selective.
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from endogenous_sue import witness
from endogenous_sue.config import WITNESS_DISPERSIONS
from endogenous_sue.costs import bpr_cost
from endogenous_sue.loading import load, load_prepared, prepare_masks
from endogenous_sue.shortestpath import potentials
from endogenous_sue.subnetwork import active_for_dest
from reproduction.exhibits.records import Exhibit

OD = witness.demand()
TOPO = witness.topology()
DESTINATION = witness.DESTINATION
NODE_ORDER = ["D", "O", "A", "B", "C"]


def _cost(x):
    return bpr_cost(x, witness.FFT, witness.CAPACITY, witness.B, witness.POWER)


def _solve_on_support(potential, mu, iterations=4000, tolerance=1e-13):
    """Solve the loading fixed point on the support a potential induces."""
    x = np.zeros(TOPO.n_links)
    for k in range(1, iterations + 1):
        y = load(TOPO, OD, mu, _cost(x), potential)
        following = x + (y - x) / k
        if k > 50 and np.linalg.norm(following - x) <= tolerance * max(np.linalg.norm(x), 1.0):
            x = following
            break
        x = following
    residual = load(TOPO, OD, mu, _cost(x), potential)
    return x, float(np.linalg.norm(residual - x) / max(np.linalg.norm(x), 1e-30))


def enumerate_orderings(mu):
    """Every candidate support at one dispersion, and which of them reproduce themselves."""
    records, fixed_points = [], []
    for ranks in permutations([1.0, 2.0, 3.0, 4.0]):
        potential = np.zeros((5, 2))
        potential[0, 1], potential[2, 1], potential[3, 1], potential[4, 1] = ranks
        assumed = active_for_dest(TOPO, potential[:, DESTINATION], DESTINATION)
        ordering = "<".join(name for _, name in sorted(zip([0.0] + list(ranks), NODE_ORDER)))
        record = dict(mu=mu, ordering=ordering, n_active=int(assumed.sum()))

        if not assumed.any():
            records.append(dict(record, status="empty support", reproduces_itself=False))
            continue
        try:
            x, residual = _solve_on_support(potential, mu)
        except RuntimeError:
            records.append(dict(record, status="cyclic support", reproduces_itself=False))
            continue

        induced = active_for_dest(TOPO, potentials(TOPO, _cost(x))[:, DESTINATION], DESTINATION)
        agrees = bool(np.array_equal(assumed, induced))
        record.update(status="solved", loading_residual=residual,
                      demand_routed=float(x[:3].sum()), demand_total=float(OD.sum()),
                      reproduces_itself=agrees,
                      links_disagreeing=[witness.ARC_LABELS[i]
                                         for i in np.flatnonzero(assumed != induced)])
        if agrees:
            fixed_points.append(record)
        records.append(record)
    return records, fixed_points


def _solve_on_masks(prepared, mu, iterations, tolerance=1e-13):
    """Solve the loading fixed point on an explicitly supplied support."""
    x = np.zeros(TOPO.n_links)
    for k in range(1, iterations + 1):
        y = load_prepared(TOPO, OD, mu, _cost(x), prepared)
        following = x + (y - x) / k
        if k > 50 and np.linalg.norm(following - x) <= tolerance * max(np.linalg.norm(x), 1.0):
            return following
        x = following
    return x


def enumerate_subsets(mu, iterations=2000):
    """Every non-empty link subset as a candidate support, without assuming any potential produced it.

    Two necessary conditions are applied before a subset is solved, and both are proved rather than
    heuristic. An induced support contains every link entering the destination, because the destination
    absorbs and such a link is not transit; and it is acyclic, because the potential strictly decreases
    along every link it contains. A subset failing either cannot reproduce itself, and is counted as
    examined rather than skipped: the counts below add up to the full enumeration.
    """
    into_destination = np.flatnonzero(TOPO.head == DESTINATION)
    forced = 0
    for link in into_destination:
        forced |= 1 << int(link)

    examined = solved = cyclic = fixed = 0
    fixed_masks = []
    for bits in range(1, 1 << TOPO.n_links):
        examined += 1
        if bits & forced != forced:
            continue
        mask = np.array([(bits >> link) & 1 for link in range(TOPO.n_links)], dtype=bool)
        try:
            prepared = prepare_masks(TOPO, OD, {DESTINATION: mask})
        except RuntimeError:
            cyclic += 1
            continue
        solved += 1
        x = _solve_on_masks(prepared, mu, iterations)
        induced = active_for_dest(TOPO, potentials(TOPO, _cost(x))[:, DESTINATION], DESTINATION)
        if np.array_equal(induced, mask):
            fixed += 1
            fixed_masks.append([witness.ARC_LABELS[i] for i in np.flatnonzero(mask)])
    return dict(n_subsets=examined, n_subsets_missing_arrival=examined - solved - cyclic,
                n_subsets_cyclic=cyclic, n_subsets_solved=solved,
                n_subset_fixed_points=fixed, subset_fixed_points=fixed_masks)


def main(argv=None):
    exhibit = Exhibit("witness", __doc__.splitlines()[0], networks=False)
    exhibit.parser.set_defaults(dispersions=",".join(str(mu) for mu in WITNESS_DISPERSIONS))
    exhibit.add_argument("--subsets", action=argparse.BooleanOptionalAction, default=True,
                         help="also enumerate the link subsets directly (default: true)")
    exhibit.parse(argv)
    exhibit.sink.write_text("")
    written, disagreed = 0, []

    for mu in exhibit.dispersions:
        records, fixed_points = enumerate_orderings(mu)
        solved = [r for r in records if r["status"] == "solved"]
        serving = [r for r in solved if r["demand_routed"] > 0.999 * r["demand_total"]]
        subsets = enumerate_subsets(mu) if exhibit.args.subsets else {}
        if subsets:
            subsets["enumerations_agree"] = (subsets["n_subset_fixed_points"] == len(fixed_points))
        exhibit.emit(dict(
            exhibit="witness", mu=mu, n_orderings=len(records), n_solved=len(solved), **subsets,
            n_empty=sum(1 for r in records if r["status"] == "empty support"),
            n_cyclic=sum(1 for r in records if r["status"] == "cyclic support"),
            n_serving_all_demand=len(serving), n_fixed_points=len(fixed_points),
            fixed_point_orderings=[r["ordering"] for r in fixed_points],
            fewest_links_disagreeing=min((len(r["links_disagreeing"]) for r in serving),
                                         default=None)))
        for record in records:
            exhibit.emit(record)
        written += len(records) + 1
        verdict = "a strict equilibrium exists" if fixed_points else "no strict equilibrium"
        print(f"mu={mu:<5} orderings={len(records)} solved={len(solved)} "
              f"serving all demand={len(serving)} reproducing themselves={len(fixed_points)} "
              f"-> {verdict}", flush=True)
        if subsets:
            print(f"      subsets: {subsets['n_subsets']} examined, "
                  f"{subsets['n_subsets_solved']} solved, "
                  f"{subsets['n_subset_fixed_points']} reproducing themselves", flush=True)
            if not subsets["enumerations_agree"]:
                disagreed.append(mu)
                print(f"      DISAGREEMENT at mu={mu:g}: {len(fixed_points)} fixed points by "
                      f"ordering, {subsets['n_subset_fixed_points']} by subset. Every strict "
                      f"efficient set is some potential's, so the two cannot differ and one of the "
                      f"enumerations is wrong.", flush=True)

    exhibit.done(written)
    return 1 if disagreed else 0


if __name__ == "__main__":
    raise SystemExit(main())
