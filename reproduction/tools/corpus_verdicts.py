"""Why each benchmark network is in the study corpus, or is not.

The corpus is a list of names in :mod:`endogenous_sue.config`, and a list is not a criterion: a reader
cannot check a scope sentence against it. This runs the same pre-flight gate the sweeps run over every
network the data root offers, prints the verdict for each with the values behind it, and reports whether
the gate's verdicts and the declared corpus agree.

    python reproduction/tools/corpus_verdicts.py            every network, with its verdict
    python reproduction/tools/corpus_verdicts.py --check    non-zero where a verdict disagrees with the list

A network the gate would admit but the list omits, or the reverse, is reported. The point is that the
scope of the study is decided by stated properties and not by which files someone happened to include.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.config import (
    CORPUS,
    EXCLUDED_NETWORKS,
    HELD_OUT_NETWORKS,
    REPORTED_CORPUS,
)
from endogenous_sue.preflight import inspect_network


def verdict(report) -> tuple[str, str]:
    """Whether the gate admits this network to the reported corpus, and why not where it does not."""
    if report.warnings:
        return "excluded", report.warnings[0]
    return "reported", ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero where the gate and the declared corpus disagree")
    args = parser.parse_args(argv)

    disagreements = []
    print(f"{'network':<52} {'N':>7} {'Z':>6} {'E':>7}  verdict")
    for name, _, _ in corpus.available():
        try:
            network, demand = corpus.load(name)
        except ValueError as refusal:
            # A pair the loader refuses is a verdict about the data, and the survey exists to report
            # exactly that. Crashing here would make one unusable network hide every network after it.
            where = ("reported" if name in REPORTED_CORPUS else
                     "held out" if name in HELD_OUT_NETWORKS else "not in the corpus")
            print(f"{name:<52} {'-':>7} {'-':>6} {'-':>7}  unreadable / declared {where}")
            print(f"    {refusal}")
            if name in REPORTED_CORPUS:
                disagreements.append(f"{name}: the corpus reports it and its files cannot be read")
            continue
        report = inspect_network(name, network, demand)
        decision, reason = verdict(report)
        declared = ("reported" if name in REPORTED_CORPUS else
                    "swept, excluded" if name in CORPUS else
                    "held out" if name in HELD_OUT_NETWORKS else "not in the corpus")
        print(f"{name:<52} {report.n_nodes:>7} {report.n_zones:>6} {report.n_links:>7}  "
              f"{decision} / declared {declared}")
        if reason:
            print(f"    {reason}")
        # A held-out network is admitted by the gate and deliberately absent from the corpus: that is
        # what "held out" means, and reporting it as a disagreement would invite closing the gap by
        # folding the prospective set back into the development corpus.
        if decision == "reported" and name not in REPORTED_CORPUS and name not in HELD_OUT_NETWORKS:
            disagreements.append(f"{name}: the gate admits it, the corpus does not list it")
        if decision == "excluded" and name in REPORTED_CORPUS:
            disagreements.append(f"{name}: the gate excludes it, the corpus reports it")

    print(f"\n{len(REPORTED_CORPUS)} reported, {len(EXCLUDED_NETWORKS)} swept and excluded by the "
          f"stated criterion, {len(HELD_OUT_NETWORKS)} held out")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) between the gate and the declared corpus:")
        for line in disagreements:
            print(f"  - {line}")
        print("\nA network the gate admits but the corpus omits is a scope decision made by omission. "
              "Either state the criterion that excludes it, or include it.")
    return 1 if (args.check and disagreements) else 0


if __name__ == "__main__":
    raise SystemExit(main())
