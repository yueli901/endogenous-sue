"""Replace the generated table bodies in the manuscript with freshly built ones.

Each table is delimited in the source by its rules. The generators emit body rows only, so the splice
replaces everything between the header line and the closing rule and touches nothing else.

    python reproduction/tools/splice_tables.py --tex FILE [--only corpus,rho] [--dry-run]

This edits the manuscript in place. It does nothing unless a file is named.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_tables

# LaTeX label to the generator that produces its body. Every label here must exist in the manuscript and
# every table the manuscript carries must appear here: a label mapped to nothing is a phantom, and a table
# absent from this map is maintained by hand and will drift from the data.
LABELS = {
    # main text
    "tab:corpus": "corpus",
    "tab:support-comparison": "edsp",
    "tab:rhoexamples": "rho_examples",
    "tab:timing": "timing",
    # supplement
    "stab:rho": "rho",
    "stab:comparison": "comparison",
    "stab:witness": "witness",
    "stab:ties-a": "ties_a",
    "stab:ties-b": "ties_b",
}

# Frozen protocol tables are intentionally transcribed from terminal outcomes rather than rebuilt from
# the canonical in-sample sinks. Naming them here keeps coverage explicit without pretending they have a
# generator in build_tables.py.
MANUAL_LABELS = {
    "stab:larger-networks": ("fixed protocol and terminal outcomes for the additional "
                             "larger-network assessment"),
}


def body_rows(generator: str) -> list[str]:
    """The generated body: data rows, and the rules a table groups them with."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        build_tables.TABLES[generator]()
    return [line for line in buffer.getvalue().splitlines()
            if line.strip() and not line.startswith("%")
            and ("\\\\" in line or is_rule(line))]


def is_rule(line: str) -> bool:
    """Whether a line is a horizontal rule rather than a row of data."""
    return line.strip().startswith("\\midrule") or line.strip().startswith("\\cmidrule")


def column_count(row: str) -> int:
    """Number of LaTeX cells in a simple generated table row."""
    return len(re.findall(r"(?<!\\)&", row)) + 1


def splice(text: str, label: str, rows: list[str]) -> str:
    at_label = text.index("\\label{" + label + "}")
    top = text.rindex("\\toprule", 0, at_label)
    middle = text.index("\\midrule", top)
    bottom = text.index("\\bottomrule", middle)
    current_rows = [line.strip() for line in text[middle:bottom].splitlines()
                    if "\\\\" in line and "&" in line]
    if current_rows:
        expected = column_count(current_rows[0])
        generated = {column_count(row) for row in rows if not is_rule(row)}
        if generated != {expected}:
            raise ValueError(f"generated row width {sorted(generated)} does not match "
                             f"the table's {expected} columns")
    return text[:middle + len("\\midrule") + 1] + "\n".join(rows) + "\n" + text[bottom:]


def check_coverage(paths) -> list[str]:
    """Every mapped label must exist, and every table in the sources must be mapped.

    A label mapped to nothing is a phantom that silently splices no table. A table absent from the map is
    maintained by hand and will drift from the data, which is the failure the whole pipeline exists to
    prevent. Both directions are checked because each has occurred.
    """
    text = "\n".join(Path(path).read_text() for path in paths if Path(path).exists())
    if not text:
        return ["no manuscript source was readable"]
    present = set(re.findall(r"\\label\{(s?tab:[A-Za-z0-9:-]+)\}", text))
    problems = []
    known = set(LABELS) | set(MANUAL_LABELS)
    for label in sorted(set(LABELS) - present):
        problems.append(f"{label} is mapped to {LABELS[label]!r} but appears in no source")
    for label in sorted(present - known):
        problems.append(f"{label} is a table in the manuscript with no generator")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tex", required=True, help="manuscript source to edit", action="append")
    parser.add_argument("--only", help="comma-separated generator names")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    parser.add_argument("--check", action="store_true",
                        help="verify label coverage against the sources and exit")
    args = parser.parse_args()

    if args.check:
        problems = check_coverage(args.tex)
        for problem in problems:
            print(f"  - {problem}")
        print("label coverage is complete" if not problems
              else f"{len(problems)} coverage problem(s)")
        return 1 if problems else 0

    path = Path(args.tex[0])
    text = path.read_text()
    wanted = set(args.only.split(",")) if args.only else None
    changed = 0
    for label, generator in LABELS.items():
        if wanted and generator not in wanted:
            continue
        if "\\label{" + label + "}" not in text:
            continue
        rows = body_rows(generator)
        if not rows:
            print(f"skip {label}: the generator emitted nothing")
            continue
        try:
            text = splice(text, label, rows)
        except ValueError as exc:
            print(f"skip {label}: {exc}")
            continue
        changed += 1
        print(f"{label:<18} {len(rows)} body rows")
    if args.dry_run:
        print(f"\ndry run: {changed} table(s) would be rewritten in {path}")
        return
    path.write_text(text)
    print(f"\nwrote {changed} table(s) into {path}")


if __name__ == "__main__":
    main()
