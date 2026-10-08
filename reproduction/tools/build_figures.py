"""Produce the paper's figures from the stored experiment records.

Nothing is recomputed here, so the figures and the text cannot drift apart.

Axis conventions. Iteration counts are drawn on linear axes. The set-diameter figure keeps a logarithmic
ordinate against a linear abscissa, because that is the pair of axes on which an exponential contraction is
a straight line and its rate is the slope; on a linear ordinate a decay over twelve orders of magnitude is
a vertical drop onto the axis and nothing can be read off it. Results a table reports at least as well as a
plot -- the accuracy bound and the parameter recovery -- are tabulated instead.

Figures are written into ``results/figures/``. Pass ``--out DIR`` to write them somewhere else, which is
how they reach a manuscript source tree.

    python reproduction/tools/build_figures.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from style import PALETTE, RC_PARAMS, display

from endogenous_sue import witness
from endogenous_sue.config import DIAMETER_NETWORKS, OSCILLATION_PANELS
from reproduction.exhibits.records import load_records

RESULTS = REPO_ROOT / "results" / "reproduce"
FIGURES = REPO_ROOT / "results" / "figures"

plt.rcParams.update(RC_PARAMS)


def load(filename: str) -> list[dict]:
    """One exhibit's records, across its shards, so a figure sees the whole run."""
    return load_records(filename, RESULTS)


def fig1_witness():
    """The five-node witness network.

    One panel. The behaviour an earlier second panel tried to show is a set of numbers, and is stated in
    the caption instead. The caption also carries the key -- which links are the two-way ring and what
    the three numbers are -- because a legend set in the figure cannot be read at the width the paper
    includes it at, and duplicates the caption when it can.
    """
    fig, ax = plt.subplots(figsize=(3.9, 1.8))
    position = {"O": (0.0, 0.0), "A": (1.60, 1.00), "B": (2.35, 0.0),
                "C": (1.60, -1.00), "D": (3.95, 0.0)}
    exit_capacity = {n: witness.CAPACITY[list(witness.ARC_LABELS).index(f"{n}->D")]
                     for n in "ABC"}
    radius = 0.17

    def arc(u, v, **kw):
        """Draw u to v, shortened so the arrowhead meets the node circle rather than its centre."""
        (x0, y0), (x1, y1) = position[u], position[v]
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        ax.annotate("", (x1 - ux * radius, y1 - uy * radius),
                    (x0 + ux * radius, y0 + uy * radius), arrowprops=kw)

    for node in "ABC":
        arc("O", node, arrowstyle="-|>", color="0.5", lw=0.9, shrinkA=0, shrinkB=0)
        arc(node, "D", arrowstyle="-|>", color="0.5", lw=0.9, shrinkA=0, shrinkB=0)
    # The A-C link bulges away from B, so it crosses only the O-B-D axis and never touches B.
    for u, v, curve in [("A", "B", 0.20), ("B", "C", 0.20), ("A", "C", 0.32)]:
        arc(u, v, arrowstyle="<->", color=PALETTE[1], lw=1.0, shrinkA=0, shrinkB=0,
            connectionstyle=f"arc3,rad={curve}")
    for node in "ABC":
        along, offset = 0.30, (0.20 if position[node][1] > 0
                               else (-0.20 if position[node][1] < 0 else -0.19))
        x = position[node][0] + along * (position["D"][0] - position[node][0])
        y = position[node][1] + along * (position["D"][1] - position[node][1])
        ax.text(x, y + offset, f"{exit_capacity[node]:.0f}", fontsize=6.8, color="0.4",
                ha="center", va="bottom" if offset > 0 else "top")
    for node, (x, y) in position.items():
        ax.add_patch(plt.Circle((x, y), radius, fc="white", ec="black", zorder=3, lw=1.0))
        ax.text(x, y, node, ha="center", va="center", zorder=4, fontsize=8.5)
    ax.text(position["O"][0], position["O"][1] - radius - 0.10, "origin", ha="center",
            va="top", fontsize=6.5, color="0.4")
    ax.text(position["D"][0], position["D"][1] - radius - 0.10, "destination", ha="center",
            va="top", fontsize=6.5, color="0.4")
    ax.set_xlim(-0.45, 4.45)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(FIGURES / "fig1_five_node.pdf")
    plt.close(fig)
    print("fig1 written")
    return True


def fig2_naive_vs_damped():
    """Links changing status per iteration, under the two schemes.

    The panels are the combinations on which both schemes are doing visible work: the naive iteration
    locks at a positive plateau and the damped count starts high and decays to zero. Sioux Falls is shown
    at the lowest dispersion because at unit dispersion the naive iteration converges on it.

    Plateau values and freeze iterations are in the accompanying table; repeating them here only crowds
    the panels.
    """
    rows = load("convergence.jsonl")
    if not rows:
        print("fig2 skipped: no records")
        return False
    panels = list(OSCILLATION_PANELS)
    max_iteration = 40
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4, 2.1))
    plotted = 0
    for ax, (network, mu) in zip(axes, panels):
        highest = 0.0
        record = next((r for r in rows if r["network"] == network and r["mu"] == mu), None)
        contrast = record.get("contrast", {}) if record is not None else {}
        for scheme, colour, label in (("instantaneous", PALETTE[1], "fixed-set iteration"),
                                      ("damped", PALETTE[0], "averaged-cost (ours)")):
            series = contrast.get(scheme, {}).get("support_changes")
            if not series or any(v is None for v in series):
                continue
            y = np.array(series, float)[:max_iteration]
            ax.plot(np.arange(1, len(y) + 1), y, color=colour,
                    lw=1.3 if scheme == "instantaneous" else 1.2, label=label)
            highest = max(highest, y.max())
            plotted += 1
        if highest <= 0.0:
            highest = 1.0
        ax.set_ylim(-0.035 * highest, 1.06 * highest)
        ax.set_xlim(0, max_iteration)
        ax.set_xlabel("iteration")
        ax.set_title(rf"{display(network)},  $\mu={mu:g}$", fontsize=8)
        ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    axes[0].set_ylabel("links changing status")
    axes[-1].legend(frameon=False, fontsize=6.8, loc="upper right")
    if plotted == 0:
        plt.close(fig)
        print("fig2 skipped: no plottable convergence series")
        return False
    fig.savefig(FIGURES / "fig2_naive_vs_damped.pdf")
    plt.close(fig)
    print("fig2 written")
    return True


def fig3_set_diameter():
    """Diameter of the equilibrium set against the dispersion.

    The networks plotted are the ones the accompanying text discusses, and each is labelled through the
    shared display-name map rather than by a prefix test, which is how one of them came to be drawn under
    another's name.
    """
    rows = load("solution_set.jsonl")
    if not rows:
        print("fig3 skipped: no records")
        return False
    networks = list(DIAMETER_NETWORKS)
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    plotted = 0
    for index, network in enumerate(networks):
        series = sorted([r for r in rows if r["network"] == network
                         and r.get("diameter_flow_relative", 0) > 0], key=lambda r: r["mu"])
        if len(series) < 2:
            continue
        ax.plot([r["mu"] for r in series], [r["diameter_flow_relative"] for r in series],
                "o-", ms=3.5, color=PALETTE[index % len(PALETTE)], label=display(network))
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        print("fig3 skipped: no plottable solution-set series")
        return False
    ax.set_yscale("log")
    ax.set_xlabel(r"dispersion $\mu$")
    ax.set_ylabel("diameter of the equilibrium set")
    ax.set_ylim(top=1e-1)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.savefig(FIGURES / "fig3_set_diameter.pdf")
    plt.close(fig)
    print("fig3 written")
    return True


FIGURE_FUNCTIONS = (fig1_witness, fig2_naive_vs_damped, fig3_set_diameter)


def main(argv=None):
    global FIGURES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=FIGURES,
                        help="directory to write the figures into")
    parser.add_argument("names", nargs="*",
                        help=f"subset of {', '.join(f.__name__ for f in FIGURE_FUNCTIONS)}")
    args = parser.parse_args(argv)
    FIGURES = args.out
    FIGURES.mkdir(parents=True, exist_ok=True)
    chosen = ([make for make in FIGURE_FUNCTIONS if make.__name__ in args.names]
              if args.names else list(FIGURE_FUNCTIONS))
    unknown = set(args.names) - {make.__name__ for make in FIGURE_FUNCTIONS}
    if unknown:
        raise SystemExit(f"unknown figure(s): {', '.join(sorted(unknown))}")

    failures = 0
    for make in chosen:
        try:
            if not make():
                failures += 1
        except Exception as exc:
            print(f"{make.__name__} failed: {type(exc).__name__}: {exc}")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
