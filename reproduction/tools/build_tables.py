"""Build the manuscript's table bodies and generated macros from the stored results.

Every number the manuscript states about the corpus is produced here and spliced in, so that no value is
typed by hand. A count that lives in prose drifts from the data it describes; a count that lives in a
macro emitted by this script cannot, because the validator compares the two.

Run with no arguments to print every table and macro. Pass names to print a subset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "reproduction" / "tools"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from style import display, num, sci, span

from endogenous_sue.config import (
    ACCURACY_NETWORKS,
    BUDGET_SENSITIVE,
    CERTIFICATE_TIER,
    DISPERSIONS,
    EXCLUDED_NETWORKS,
    FACE_CELL,
    RECOVERY_DISPERSIONS,
    RECOVERY_NETWORKS,
    SOLVED_BELOW,
    SUPPORT_COMPARISON_DISPERSIONS,
    WALK_CELLS,
)
from reproduction.exhibits.records import load_records

RESULTS = REPO_ROOT / "results" / "reproduce"

#: Networks shown in the main text's abridged radius table. The supplement carries all of them.
RHO_EXAMPLES = ("SiouxFalls", "Eastern-Massachusetts", "Anaheim", "Winnipeg-Asymmetric",
                "Chicago-Sketch", "Sydney")

#: Emitted between groups of rows. The splicer keeps these, so a table typeset with an internal rule
#: keeps it when its body is regenerated.
RULE = r"\midrule"


def load(name: str) -> list[dict]:
    """Records for one exhibit, across its shards, with the excluded networks filtered out once, here."""
    return [record for record in load_records(name, RESULTS)
            if record.get("network") not in EXCLUDED_NETWORKS]


def latest(records):
    """One record per cell, the last written winning, so a re-run supersedes rather than duplicates."""
    cells = {}
    for record in records:
        if "network" in record and "mu" in record:
            cells[(record["network"], float(record["mu"]))] = record
    return cells


def row(cells) -> str:
    return " & ".join(cells) + r" \\"


def spelled(n: int) -> str:
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
             "eleven", "twelve"]
    return words[n] if n < len(words) else str(n)


def macro(name: str, value) -> None:
    print(f"\\newcommand{{\\{name}}}{{{value}}}")


# ---------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------

def corpus():
    """Network sizes and the evidence tier each is in.

    The tiers are the ones declared in :mod:`endogenous_sue.config` before the sweep ran. Reading them
    back from whichever cells happen to be in the sink would make the tier a description of the run
    rather than a commitment the run is judged against.
    """
    sweep = latest(load("corpus_sweep"))
    if not sweep:
        return
    by_network = {}
    for (network, _), record in sweep.items():
        by_network.setdefault(network, record)

    def tier(network):
        if network in BUDGET_SENSITIVE:
            return "budget-sensitive diagnostic"
        return "certificate" if network in CERTIFICATE_TIER else "diagnostic"

    for index, group in enumerate((
            [item for item in by_network.items() if item[0] not in BUDGET_SENSITIVE],
            [item for item in by_network.items() if item[0] in BUDGET_SENSITIVE])):
        if index and group:
            print(RULE)
        for network, record in sorted(group, key=lambda item: item[1]["n_links"]):
            print(row([f"{display(network):<32}", f"{record['n_nodes']}", f"{record['n_zones']}",
                       f"{record['n_links']}", tier(network)]))


def solved():
    """The headline count, which requires the residual *and* conservation."""
    sweep = latest(load("corpus_sweep"))
    if not sweep:
        return
    # The corpus size includes terminal unresolved rows. Residual-based counts below deliberately do not:
    # a budget-stopped row has no phase-two residual and is neither solved nor unsolved.
    residuals = [record["residual"] for record in sweep.values() if record.get("residual") is not None]
    passing_records = [record for record in sweep.values()
                       if record.get("residual") is not None and record["residual"] < SOLVED_BELOW
                       and record.get("conserves") is True]
    print("% --- the solved headline ---")
    macro("NRHOCELLS", len(sweep))
    macro("NNETS", len({network for network, _ in sweep}))
    macro("NDISP", spelled(len({mu for _, mu in sweep})))
    macro("NSOLVED", len(passing_records))
    macro("NUNSOLVED", spelled(len(residuals) - len(passing_records)))
    # Cells that reached the end of a declared budget without the support settling. They carry no
    # residual, so they are not in the counts above and cannot be read as solved or as unsolved: the
    # quantity those two divide was never measured for them. Counted here so the text can say how many
    # there are rather than leaving the difference between NRHOCELLS and the corpus unexplained.
    unresolved = [key for key, record in sweep.items() if record.get("unresolved") is True]
    macro("NUNRESOLVED", len(unresolved))
    macro("NUNRESOLVEDNETS", len({network for network, _ in unresolved}))
    reload_measured = [record for record in passing_records
                       if record.get("strict_reload_defect") is not None]
    macro("NRELOADCELLS", len(reload_measured))
    macro("NRELOADBIGGER", sum(record["strict_reload_defect"] >= SOLVED_BELOW
                                for record in reload_measured))
    if passing_records:
        macro("NSOLVEDWORST", sci(max(record["residual"] for record in passing_records)))


def edsp():
    """What the congestion-adaptive support changes against the free-flow one."""
    records = latest(load("support_comparison"))
    if not records:
        return
    by_network = {}
    for (network, mu), record in sorted(records.items()):
        if mu in SUPPORT_COMPARISON_DISPERSIONS:
            by_network.setdefault(network, []).append(record)
    for network, cells in sorted(by_network.items(), key=lambda item: item[1][0]["n_links"]):
        moved = [record["support_moved"] for record in cells]
        shares = [100.0 * record["moved_share"] for record in cells]
        differences = [record["flow_difference"] for record in cells]
        moved_cell = span(moved, digits=0, tol=0.5)
        share_cell = span(shares, digits=1, tol=0.05).replace("$", "") + r"\%"
        print(row([f"{display(network):<28}", moved_cell, share_cell, span(differences)]))


def rho():
    """The spectral radius at equilibrium, one column per dispersion.

    A value at or above one means the full-graph value function does not exist, so the comparator has
    nothing to offer at that cell. Those entries are set in bold.
    """
    records = latest(load("full_graph"))
    if not records:
        return
    # The declared grid, not whatever the sink holds. The accuracy comparison adds dispersions to the
    # same sink and they would otherwise widen this table past the columns it is typeset with.
    grid = sorted(DISPERSIONS)
    by_network = {}
    for (network, mu), record in records.items():
        by_network.setdefault(network, {})[mu] = record
    for network, cells in sorted(by_network.items(),
                                 key=lambda item: min(r["n_links"] for r in item[1].values())):
        columns = []
        for mu in grid:
            record = cells.get(mu)
            if record is None:
                columns.append("---")
                continue
            radius = record.get("radius_equilibrium")
            if radius is None:
                columns.append("n.r.")
                continue
            text = f"{radius:.2f}" if radius == radius else "---"
            columns.append(f"\\(\\mathbf{{{text}}}\\)" if record.get("well_posed") is False else text)
        print(row([f"{display(network):<32}"] + columns))
    # Counted over the declared corpus grid only. The accuracy comparison adds dispersions to this same
    # sink, and counting those too would report a fraction whose denominator is not the one the sentence
    # beside it states.
    on_grid = {cell: record for cell, record in records.items()
               if cell[1] in set(DISPERSIONS) and record.get("radius_equilibrium") is not None}
    failing_cells = sum(1 for record in on_grid.values() if record.get("well_posed") is False)
    failing_networks = len({network for (network, _), record in on_grid.items()
                            if record.get("well_posed") is False})
    print("% --- full-graph feasibility ---")
    macro("NRHONETS", len({network for network, _ in on_grid}))
    macro("NRHORESOLVED", len(on_grid))
    macro("NRHOFAILNETS", failing_networks)
    macro("NRHOFAILCELLS", failing_cells)


def rho_examples():
    """The abridged global radius screen in the main text."""
    records = latest(load("full_graph"))
    for network in RHO_EXAMPLES:
        record = records.get((network, 1.0))
        if record is None:
            continue
        radius = record.get("radius_equilibrium")
        if radius is None:
            print(row([f"{display(network):<28}", "n.r.", "not resolved"]))
            continue
        status = "well defined" if record["well_posed"] else "global screen fails"
        print(row([f"{display(network):<28}", f"{radius:.2f}", status]))


def _log_fit(mus, differences):
    """Descriptive log-log slope and coefficient of determination for the observed differences."""
    import numpy as np

    pairs = [(mu, value) for mu, value in zip(mus, differences)
             if value is not None and value > 0.0 and mu > 0.0]
    if len(pairs) < 3:
        return None, None
    x = np.log(np.array([mu for mu, _ in pairs]))
    y = np.log(np.array([value for _, value in pairs]))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    total = float(((y - y.mean()) ** 2).sum())
    residual = float(((y - predicted) ** 2).sum())
    return float(-slope), (1.0 - residual / total if total > 0 else None)


def comparison():
    """Restricted/full-graph differences where the declared global screen passes."""
    records = latest(load("full_graph"))
    margins = latest(load("solution_set"))
    if not records:
        return
    by_network = {}
    for (network, mu), record in records.items():
        if record.get("flow_difference") is not None:
            by_network.setdefault(network, []).append((mu, record))
    for network, cells in sorted(by_network.items(),
                                 key=lambda item: item[1][0][1]["n_links"]):
        cells.sort()
        mus = [mu for mu, _ in cells]
        differences = [record["flow_difference"] for _, record in cells]
        slope, fit = _log_fit(mus, differences)
        # The path-excess margin, which is what the column is headed with. The isolation margin is a
        # different quantity: it admits tied links and is zero wherever a tie exists, so printing it here
        # would report one margin under another's name.
        reported = [margins[(network, mu)]["path_excess_margin"] for mu in mus
                    if margins.get((network, mu), {}).get("path_excess_margin") is not None]
        print(row([f"{display(network):<28}",
                   f"{mus[0]:g}--{mus[-1]:g}", f"{len(cells)}",
                   sci(differences[0]), sci(differences[-1]),
                   f"{slope:.2f}" if slope is not None else "---",
                   f"{fit:.3f}" if fit is not None else "---",
                   span(reported, digits=2)]))


def accuracy():
    """Descriptive fit on the one network where the comparator runs across a finer grid."""
    records = latest(load("full_graph"))
    cells = sorted((mu, record) for (network, mu), record in records.items()
                   if network in ACCURACY_NETWORKS and record.get("flow_difference") is not None)
    if len(cells) < 3:
        return
    mus = [mu for mu, _ in cells]
    differences = [record["flow_difference"] for _, record in cells]
    print("% --- the restricted/full-graph comparison ---")
    macro("ACCNET", display(ACCURACY_NETWORKS[0]))
    macro("ACCLO", sci(differences[0]).strip("$"))
    macro("ACCHI", sci(differences[-1]).strip("$"))
    macro("ACCMULO", f"{mus[0]:g}")
    macro("ACCMUHI", f"{mus[-1]:g}")
    macro("ACCPOINTS", len(cells))


def recovery():
    """Whether the dispersion parameter can be recovered from the flows it produced."""
    records = latest(load("full_graph"))
    rows = [records[(network, mu)] for network in RECOVERY_NETWORKS for mu in RECOVERY_DISPERSIONS
            if records.get((network, mu), {}).get("recovered_mu") is not None]
    for record in rows:
        print(row([f"{display(record['network']):<28}", f"${record['mu']:g}$",
                   f"${record['recovered_mu']:.8f}$", sci(record["recovery_error"]),
                   f"{record['recovery_solves']}", f"{record['recovery_seconds']:.1f}"]))


def witness():
    """Independent exhaustive checks of the five-node strict non-existence witness."""
    summaries = sorted((record for record in load("witness") if "n_orderings" in record),
                       key=lambda record: float(record["mu"]))
    for record in summaries:
        print(row([f"{float(record['mu']):g}", f"{record['n_orderings']}",
                   f"{record['n_serving_all_demand']}", f"{record['n_subsets']}",
                   f"{record['n_fixed_points']}"]))


def certificate_records():
    """Certificate-sweep cells on the declared tier.

    A cell from a network outside the tier is a probe, not a reported row: the tier is what the counts in
    the text are stated over, and admitting whatever else was swept would make those counts a description
    of the run.
    """
    return {cell: record for cell, record in latest(load("certificate_sweep")).items()
            if cell[0] in CERTIFICATE_TIER}


def _tie_rows(networks):
    records = certificate_records()
    by_network = {}
    for (network, mu), record in records.items():
        if network in networks:
            by_network.setdefault(network, []).append((mu, record))
    for position, network in enumerate(sorted(by_network,
                                              key=lambda name: by_network[name][0][1]["n_links"])):
        cells = sorted(by_network[network])
        if position:
            print(RULE)
        for index, (mu, record) in enumerate(cells):
            label = display(network) if index == 0 else ""
            pairs, links = record["n_tied_pairs"], record["n_tied_links"]
            overcount = f"{pairs / links:.1f}" if links else "---"
            print(row([f"{label:<28}", f"${mu:g}$", f"${pairs}$", overcount,
                       num(record.get("smallest_tie_cost")), num(record.get("exclusion_margin"))]))


def _certificate_networks():
    """The tier's networks that the sink actually holds, smallest first."""
    records = certificate_records()
    return sorted({network for network, _ in records},
                  key=lambda name: min(record["n_links"] for (other, _), record in records.items()
                                       if other == name))


def ties_a():
    """The first half of the tie inventory, by network size."""
    networks = _certificate_networks()
    _tie_rows(set(networks[:len(networks) // 2 + len(networks) % 2]))


def ties_b():
    """The second half of the tie inventory, and the certificate counts."""
    networks = _certificate_networks()
    _tie_rows(set(networks[len(networks) // 2 + len(networks) % 2:]))
    records = certificate_records()
    if not records:
        return
    print("% --- the certificate tier ---")
    macro("NCERTTOTAL", len(records))
    macro("NCERTPASS", sum(1 for record in records.values() if record["certified"]))
    macro("NCERTNETS", len(networks))


#: Phase-one stopping criteria that establish the support is final. On any other criterion the run
#: stopped for a reason that says nothing about the support, and no iteration count can be quoted as the
#: one it froze at.
SUPPORT_FROZEN_ON = ("support settled", "exactness bound")


def _fixed_at(record) -> str:
    """The iteration the support froze at, or why no such iteration can be quoted.

    ``n.r.`` is not zero and not "never": it is the diagnostic having gone unrecorded, which is what a
    record produced before the criterion existed looks like. Printing a number there would report a
    measurement that was never taken.
    """
    stopped_on = record.get("phase1_stopped_on")
    if stopped_on is None or record.get("phase1_iterations") is None:
        return "n.r."
    if stopped_on in SUPPORT_FROZEN_ON:
        return f"${record['phase1_iterations']}$"
    return "not fixed"


def timing():
    """What the two phases cost on an idle machine."""
    records = latest(load("convergence"))
    if not records:
        return
    cells = [record for (network, mu), record in records.items() if mu == 1.0]
    for record in sorted(cells, key=lambda item: item["n_links"]):
        print(row([f"{display(record['network']):<32}", f"${record['n_links']}$", _fixed_at(record),
                   num(record["phase1_seconds"], digits=2), num(record["phase2_seconds"], digits=2),
                   sci(record["residual"])]))
    print("% --- timing conditions ---")
    macro("NCLOCKCELLS", len(records))
    cores = {record.get("cores") for record in records.values() if record.get("cores")}
    if len(cores) == 1:
        macro("TCORES", cores.pop())
    loads = [record.get("load_average") for record in records.values()
             if record.get("load_average") is not None]
    if loads:
        macro("TLOAD", f"{max(loads):.2f}")


def certification():
    """What a certificate costs to check, against what a certifiable flow costs to reach.

    The two are separate quantities and the paragraph that quotes them says so. Checking is timed on the
    timing cells, where the machine is known to be idle; reaching is the certificate sweep's own wall
    time, which is not comparable across machines and is reported as a range.
    """
    timings = latest(load("convergence"))
    certificates = certificate_records()
    if not timings and not certificates:
        return
    print("% --- certification cost ---")
    # Named ``strict_certify_seconds`` from 2026-09-20; rows written before that carry the older
    # name. Both are the same measurement -- the strict check on the timed flow -- so both are read.
    checks = [(record.get("strict_certify_seconds") or record.get("certify_seconds")) * 1e3
              for record in timings.values()
              if (record.get("strict_certify_seconds") or record.get("certify_seconds"))
              is not None]
    if checks:
        low, high = min(checks), max(checks)
        macro("TCHECKLO", f"{low:.2g}")
        macro("TCHECKHI", (sci(high).strip("$") if high >= 100.0 else f"{high:.2g}"))
    if not certificates:
        return
    feasible = [record for record in certificates.values() if record.get("certified")]
    if feasible:
        macro("TFEASLO", f"{min(record['seconds'] for record in feasible):.2g}")
        upper = max(record["seconds"] for record in feasible)
        macro("TFEASHI", sci(upper).strip("$"))
        macro("NFEASCELLS", len(feasible))
    macro("NANSCELLS", sum(1 for record in certificates.values() if record.get("kind") == "strict"))
    macro("NTIEFREE", sum(1 for record in certificates.values()
                          if record.get("n_tied_pairs") == 0))
    slowest = max(certificates.values(), key=lambda record: record.get("seconds", 0.0))
    macro("TFACEHI", f"{slowest['seconds'] / 3600.0:.1f}")
    macro("TFACENET", display(slowest["network"]))
    macro("TFACEMU", f"{slowest['mu']:g}")


def walk():
    """What the instrumented active-set walks did, against the bound that counts the same events.

    Two cells, declared in the configuration rather than chosen from whichever ran, because a count
    quoted as evidence about the mechanics has to be about a cell someone decided on beforehand.
    """
    records = latest(load("certificate_sweep"))
    instrumented = [(network, mu) for network, mu in WALK_CELLS
                    if ((records.get((network, mu), {}).get("tied") or {}).get("walk"))]
    if not instrumented:
        return
    print("% --- walk instrumentation ---")
    for index, (network, mu) in enumerate(instrumented, start=1):
        counters = records[(network, mu)]["tied"]["walk"]
        suffix = "A" if index == 1 else "B"
        macro(f"WALKNET{suffix}", display(network))
        macro(f"WALKMU{suffix}", f"{mu:g}")
        macro(f"WALKLIVE{suffix}", counters["live_initial"])
        macro(f"WALKPIVOTS{suffix}", counters["pivots"])
        macro(f"WALKCROSS{suffix}", counters["crossings"])
        macro(f"WALKSET{suffix}", counters["working_set"])
        macro(f"WALKWEIGHTS{suffix}", counters["interior"])
        record = records[(network, mu)]
        macro(f"WALKBOUND{suffix}", 2 * record["n_zones"] * record["n_links"])


def face():
    """The local dimension of the equilibrium manifold on the one cell it is measured on."""
    record = latest(load("certificate_sweep")).get(FACE_CELL)
    tied = (record or {}).get("tied") or {}
    if tied.get("manifold_dimension") is None:
        return
    print("% --- the certified face ---")
    macro("FACENET", display(FACE_CELL[0]))
    macro("FACEMU", f"{FACE_CELL[1]:g}")
    macro("FACEGROUPS", tied["n_interior"])
    macro("FACERANK", tied["gap_rank"])
    macro("FACEDIM", tied["manifold_dimension"])


def uniqueness():
    """How large the equilibrium set is, and where a flow is provably alone."""
    records = latest(load("solution_set"))
    if not records:
        return
    certified = [record for record in records.values() if record.get("unique")]
    print("% --- the solution set ---")
    macro("NUNIQUECELLS", len(certified))
    macro("NDIAMCELLS", len(records))
    widths = [record["diameter_flow_relative"] for record in records.values()]
    if widths:
        macro("SETWIDTH", sci(max(widths)).strip("$"))


TABLES = dict(corpus=corpus, solved=solved, edsp=edsp, rho=rho, rho_examples=rho_examples,
              comparison=comparison, accuracy=accuracy, witness=witness, ties_a=ties_a, ties_b=ties_b,
              timing=timing,
              certification=certification, walk=walk, face=face, uniqueness=uniqueness)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"subset of {', '.join(TABLES)}")
    args = parser.parse_args(argv)
    chosen = args.names or list(TABLES)
    unknown = [name for name in chosen if name not in TABLES]
    if unknown:
        raise SystemExit(f"unknown table(s): {', '.join(unknown)}; known: {', '.join(TABLES)}")
    for name in chosen:
        print(f"\n{name}\n{'=' * 100}")
        TABLES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
