"""Check the stored results before any number is read off them.

Six checks, and each exists because something once passed without it.

*Vintage.* Every record carries the gate revision it was produced under. Records from different revisions
are not comparable, and a file mixing them is refused rather than averaged.

*Settings.* Likewise for the solver settings: two records produced under different tolerances do not
belong in one table.

*Convergence.* A record that stopped on an iteration limit is not a result, and is reported so that no
table can quote it as one. The frozen-support solve is checked too: it can exhaust its iterations or fall
back to averaging and still return a flow, so the method it used is read as well as the flag it set.

*Coverage.* Every exhibit declares the cells it must produce. A sink missing any of them is a failure and
not an absence: a run that silently skipped a network passes every check that looks only at the rows it
has, and the table it feeds then reports a corpus nobody chose.

*Referenced flows.* A record naming a deposited flow is checked against that flow, or against the tracked
manifest of digests where the arrays themselves are not present.

*Generated macros.* Every value the table builder emits and the manuscript uses must agree. A hand-copied
value drifts exactly the way the sentence it replaced did. Under the manuscript gate a generated macro the
manuscript declares but never uses is a failure too: it means the number is typed into a sentence, and the
rerun will change the macro and leave the sentence.

Exit status is non-zero when any check reports a problem, so this can gate a build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from endogenous_sue import config
from endogenous_sue.config import EXCLUDED_NETWORKS, EXHIBIT_CELLS, GATE_REVISION
from reproduction.exhibits.records import sink_paths

RESULTS = REPO_ROOT / "results" / "reproduce"
MACRO_PATTERN = re.compile(r"\\newcommand\{\\([A-Z][A-Z]+)\}\{(.*)\}\s*$")

#: The sinks the manuscript is built from.
SINKS = ["corpus_sweep", "certificate_sweep", "support_comparison", "full_graph",
         "convergence", "solution_set", "witness"]

#: Sinks that declare no cell set, with the reason. Every other exhibit is checked against
#: ``config.EXHIBIT_CELLS``.
PARTIAL_BY_DESIGN = {
    "witness": "a synthetic network swept over a dispersion grid the exhibit declares itself",
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path) if line.strip()]


def check_vintage(name, records):
    revisions = sorted({record.get("gate_revision") for record in records})
    if len(revisions) > 1:
        return [f"{name}: records from {len(revisions)} gate revisions in one file, {revisions}"], []
    if revisions and revisions[0] != GATE_REVISION:
        return [f"{name}: gate revision {revisions[0]!r} is not the current {GATE_REVISION!r}"], []
    return [], []


#: Settings a cell may legitimately differ from the run on, and the declaration that says which cells.
#: A run-wide setting that varies between rows means two runs were mixed and their numbers cannot be
#: read together; a setting declared to vary per cell is the opposite, and refusing it would make a
#: declared cell unreportable. The value is checked as well as the cell, so a row cannot quietly carry
#: some other number under a name the declaration permits.
PER_CELL_SETTINGS = {"envelope_stable_windows": ("DIRECT_FACE_CELLS", "DIRECT_FACE_ENVELOPE_WINDOWS")}


def check_settings(name, records):
    problems = []
    for field, (cells_name, value_name) in PER_CELL_SETTINGS.items():
        cells = getattr(config, cells_name, ())
        expected = getattr(config, value_name, None)
        declared = {(n, float(m)) for n, m in cells}
        for record in records:
            carried = (record.get("settings") or {}).get(field)
            if carried is None:
                continue
            cell = (record.get("network"), float(record.get("mu", 0.0)))
            if cell not in declared:
                problems.append(f"{name}: {cell[0]} mu={cell[1]:g} carries {field}={carried!r}, "
                                f"which only {cells_name} declares")
            elif carried != expected:
                problems.append(f"{name}: {cell[0]} mu={cell[1]:g} carries {field}={carried!r}, "
                                f"not the declared {value_name}={expected!r}")

    def comparable(record):
        settings = dict(record.get("settings") or {})
        for field in PER_CELL_SETTINGS:
            settings.pop(field, None)
        return json.dumps(settings, sort_keys=True)

    seen = {comparable(record) for record in records}
    if len(seen) > 1:
        problems.append(f"{name}: records produced under {len(seen)} different settings in one file")
    if problems:
        return problems, []
    return [], []


#: Phase-two outcomes that returned a flow without reaching the tolerance. A record carrying one of these
#: is not a result whatever its residual says, because the residual is then the best seen and not the one
#: the solve was asked for.
INCOMPLETE_POLISH = ("newton_krylov_incomplete", "damped_averaging", "time budget")


def check_convergence(name, records):
    """Records that claim an equilibrium without having reached one.

    A row marked ``unresolved`` is not one of them. It states that the support did not settle within a
    budget declared for that network, carries ``None`` for everything the solve never measured, and is
    excluded from the solved counts by the table builder. Treating it as a defective result would be
    the same error in the opposite direction from leaving the cell absent: the first says a measurement
    failed, the second says nobody looked, and the row exists precisely to say neither.
    """
    unconverged = []
    unresolved = []
    for record in records:
        cell = f"{record.get('network', '?')} mu={record.get('mu', '?')}"
        if record.get("unresolved") is True:
            unresolved.append(cell)
            continue
        if record.get("stopped_on") == "iteration limit" or record.get("converged") is False:
            unconverged.append(cell)
        elif record.get("polish_method") in INCOMPLETE_POLISH:
            unconverged.append(f"{cell} (phase 2 {record['polish_method']})")
        elif record.get("polish_converged") is False:
            unconverged.append(f"{cell} (phase 2 above tolerance)")
        elif record.get("budget_limited") is True:
            unconverged.append(f"{cell} (the face solve stopped on its time budget)")
        for key, value in record.items():
            if key.endswith("_stopped_on") and value == "iteration limit":
                unconverged.append(f"{cell} ({key})")
            if key.endswith("_converged") and value is False:
                unconverged.append(f"{cell} ({key})")
    if unresolved:
        # Printed rather than returned. Under the manuscript gate both slots of this check escalate to
        # problems, and an unresolved cell is neither a problem nor a warning: it is the recorded
        # outcome for a network declared budget-sensitive, and the run that produced it did what it was
        # asked to. It is stated so that nobody has to infer it from a count that does not add up.
        listed = ", ".join(sorted(unresolved)[:4]) + ("" if len(unresolved) <= 4 else ", and more")
        print(f"  {len(unresolved)} cell(s) recorded as unresolved: the support did not settle within "
              f"the declared budget ({listed})")
    if not unconverged:
        return [], []
    listed = ", ".join(unconverged[:4]) + ("" if len(unconverged) <= 4 else ", and more")
    return [], [f"{name}: {len(unconverged)} record(s) did not converge and are not results ({listed})"]


def check_coverage(name, records):
    """What the exhibit declared against what its sink holds, both directions."""
    declared = EXHIBIT_CELLS.get(name)
    if declared is None:
        reason = PARTIAL_BY_DESIGN.get(name)
        return [], [f"{name}: coverage is not checked -- {reason}" if reason else
                    f"{name}: declares no cell set in config.EXHIBIT_CELLS, so nothing says what it "
                    f"was meant to produce"]

    held = {(record["network"], float(record["mu"])) for record in records
            if record.get("network") is not None and record.get("mu") is not None}
    wanted = {(network, float(mu)) for network, mu in declared}
    missing = sorted(wanted - held)
    extra = sorted(cell for cell in held - wanted if cell[0] not in EXCLUDED_NETWORKS)

    problems, warnings = [], []
    if missing:
        listed = "; ".join(f"{network} mu={mu:g}" for network, mu in missing[:6])
        problems.append(f"{name}: {len(missing)} of {len(wanted)} declared cell(s) absent ({listed}"
                        + (", and more)" if len(missing) > 6 else ")"))
    if extra:
        listed = "; ".join(f"{network} mu={mu:g}" for network, mu in extra[:6])
        warnings.append(f"{name}: {len(extra)} cell(s) present but not declared ({listed}"
                        + (", and more)" if len(extra) > 6 else ")"))
    return problems, warnings


def flow_manifest(root: Path) -> dict:
    path = root / "flows.manifest.json"
    return json.loads(path.read_text()).get("files", {}) if path.exists() else {}


def check_referenced_flows(name, records, root: Path = RESULTS):
    manifest = flow_manifest(root)
    problems, warnings, unchecked = [], [], 0
    for record in records:
        named = record.get("flow_file")
        if not named:
            continue
        path = root / "flows" / named
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        elif named in manifest:
            digest = manifest[named]["sha256"]
        elif manifest:
            problems.append(f"{name}: names {named}, which is in neither the flow directory nor the "
                            f"manifest")
            continue
        else:
            unchecked += 1
            continue
        if not record.get("flow_sha256"):
            warnings.append(f"{name}: {record.get('network')} mu={record.get('mu')} names {named} "
                            "but carries no flow_sha256")
        elif digest != record["flow_sha256"]:
            problems.append(f"{name}: {record.get('network')} mu={record.get('mu')} was computed "
                            f"against a different version of {named}")
    if unchecked:
        warnings.append(f"{name}: {unchecked} record(s) name a flow that is neither present nor in a "
                        f"manifest; run reproduction/tools/build_flow_manifest.py where the arrays are")
    return problems, warnings


def default_macro_sources() -> list[Path]:
    """The submission package, then the working manuscript, whichever is present."""
    for package in (REPO_ROOT / "submission", REPO_ROOT / "manuscript" / "submission"):
        present = [package / name for name in ("macros.tex", "main.tex", "supplement.tex")
                   if (package / name).exists()]
        if present:
            return present
    return []


def _without_declarations(text: str) -> str:
    kept = []
    for line in text.splitlines():
        uncommented = re.sub(r"(?<!\\)%.*", "", line)
        if MACRO_PATTERN.match(uncommented.strip()):
            continue
        kept.append(uncommented)
    return "\n".join(kept)


def _macro_declarations(text: str) -> dict[str, str]:
    declarations = {}
    for line in text.splitlines():
        uncommented = re.sub(r"(?<!\\)%.*", "", line)
        if match := MACRO_PATTERN.match(uncommented.strip()):
            declarations[match.group(1)] = match.group(2)
    return declarations


def check_generated_macros(sources: list[Path], strict: bool = False):
    """Every generated macro the manuscript uses must match what the table builder emits.

    Only macros the sources actually use are enforced. One nobody uses cannot drift into a sentence, so
    requiring it to match forever is a standing false failure; it is reported as unused instead.
    """
    problems, warnings = [], []
    missing = [path for path in sources if not path.exists()]
    if missing:
        problems.append("macro check: requested source absent: "
                        + ", ".join(str(path) for path in missing))
    present = [path for path in sources if path.exists()]
    if not present:
        return problems, warnings

    builder = REPO_ROOT / "reproduction" / "tools" / "build_tables.py"
    emitted = subprocess.run([sys.executable, str(builder)], capture_output=True, text=True)
    if emitted.returncode != 0:
        problems.append(f"macro check: build_tables.py exited {emitted.returncode}")
        return problems, warnings

    generated = _macro_declarations(emitted.stdout)
    declared: dict = {}
    body = ""
    for path in present:
        text = path.read_text()
        declared.update(_macro_declarations(text))
        body += _without_declarations(text) + "\n"

    unused = []
    for name, value in sorted(generated.items()):
        used = re.search(r"\\" + name + r"(\{\}|[^A-Za-z])", body) is not None
        if name not in declared:
            if used:
                problems.append(f"macro {name}: used by the manuscript but not declared")
            continue
        if not used:
            unused.append(name)
        elif declared[name] != value:
            problems.append(f"macro {name}: the manuscript declares {declared[name]!r}, "
                            f"build_tables.py emits {value!r}")
    if unused:
        # A generated macro the manuscript declares and does not use means the number it carries is
        # typed into a sentence instead. The rerun then changes the macro and leaves the sentence, which
        # is the exact drift this whole pipeline exists to prevent, so under the manuscript gate it is a
        # failure and not a note.
        listed = ", ".join("\\" + name for name in unused)
        message = (f"macro check: {len(unused)} generated macro(s) declared but unused, so the value "
                   f"they carry is typed into the text instead: {listed}")
        (problems if strict else warnings).append(message)
    return problems, warnings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", default=str(RESULTS), help="results directory")
    parser.add_argument("--tex", action="append", type=Path, default=None,
                        help="TeX source to check generated macros against; repeatable")
    parser.add_argument("--strict", action="store_true",
                        help="the manuscript gate: an absent sink and an unconverged record are "
                             "failures rather than warnings")
    args = parser.parse_args(argv)
    root = Path(args.dir)

    problems, warnings, checked = [], [], 0
    for name in SINKS:
        paths = sink_paths(name, root)
        if not paths:
            message = f"{name}: absent, so nothing derived from it can be checked"
            (problems if args.strict else warnings).append(message)
            continue
        records = [record for path in paths for record in read(path)]
        checked += 1
        shards = "" if len(paths) == 1 else f" across {len(paths)} shards"
        print(f"checked {name}.jsonl: {len(records)} records{shards}")
        for check in (check_vintage, check_settings, check_convergence, check_coverage,
                      check_referenced_flows):
            found, noted = (check(name, records, root) if check is check_referenced_flows
                            else check(name, records))
            problems += found
            # Under the manuscript gate an unconverged record is not a warning: a table that quotes one
            # states a number the run did not establish.
            if args.strict and check is check_convergence:
                problems += noted
            else:
                warnings += noted

    macro_problems, macro_warnings = check_generated_macros(args.tex or default_macro_sources(),
                                                            strict=args.strict)
    problems += macro_problems
    warnings += macro_warnings

    print(f"\n{checked} of {len(SINKS)} result files present")
    for warning in warnings:
        print(f"  ~ {warning}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nno problems found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
