"""Shared plumbing for the reproduction scripts: arguments, result sinks and provenance.

Records are appended and flushed one at a time rather than collected and written at the end, so a run
that is interrupted keeps everything it had finished.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from endogenous_sue import corpus
from endogenous_sue.config import (
    DATA_ROOT,
    DEFAULTS,
    DISPERSIONS,
    EXHIBIT_CELLS,
    PINNED_COMMIT,
    RESULTS_ROOT,
    Settings,
)
from endogenous_sue.preflight import inspect_network, report_lines
from endogenous_sue.provenance import stamp

RESULTS = RESULTS_ROOT / "reproduce"

#: Exit status meaning the sweep is incomplete and resubmitting will continue it.
EXIT_INCOMPLETE = 2

__all__ = ["Exhibit", "EXIT_INCOMPLETE", "sink_paths", "load_records", "file_fingerprint",
           "jsonable", "peak_rss_mb", "allocated_cores"]


def allocated_cores() -> int:
    """Cores this process may actually use.

    ``os.cpu_count()`` reports the machine. Under a scheduler that is the node, not the allocation: a
    sixteen-core job on a fifty-six-core node recorded fifty-six, and the timing table states a
    wall-clock second against that number.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                    # pragma: no cover - macOS and Windows
        return os.cpu_count() or 1


def peak_rss_mb() -> float | None:
    """The process's high-water resident memory, in mebibytes, or ``None`` where it cannot be read.

    Recorded with every result because a memory request is otherwise a guess repeated: the cluster
    scripts ask for a figure someone measured once, by hand, on one network, and nothing since has
    checked it. This is the process maximum rather than the current size, so it survives the peak having
    already been released.
    """
    try:
        import resource
    except ImportError:                       # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kibibytes, macOS and the BSDs report bytes. Getting this wrong is a factor of 1024
    # in the number a job request is sized from.
    scale = 1.0 if sys.platform == "darwin" else 1024.0
    return float(peak) * scale / (1024.0 * 1024.0)


def sink_paths(name: str, root: Path = RESULTS) -> list[Path]:
    """Every file holding records for one exhibit, in a stable order.

    A sweep run as a job array writes one shard per task, because concurrent appends to a single file are
    atomic only while every record stays under the pipe buffer, which is a property of the record and not
    something a run can guarantee. The shards go in a directory named after the exhibit, and are read
    together with the plain file, so a sharded sweep and a single-process one are read alike and no
    concatenation step stands between the cluster and the tables.

    The directory is what keeps one exhibit from reading another's shards. A shared prefix would: a glob
    for ``full`` matches ``full_graph_Anaheim.jsonl``, and the reader would never know.
    """
    name = name[:-len(".jsonl")] if name.endswith(".jsonl") else name
    root = Path(root)
    single = root / f"{name}.jsonl"
    directory = root / name
    shards = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    return ([single] if single.exists() else []) + shards


def jsonable(value, max_array: int = 64):
    """Coerce solver output into something the JSON encoder accepts.

    Diagnostic dictionaries gain fields over time, and one array reaching the encoder would abort a run
    after the work is already done. Coercing generally rather than dropping known keys means a newly
    added field cannot cost a completed run. An array too long to belong in a record is replaced by its
    shape, so the record still says the field was there.
    """
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return (value.tolist() if value.size <= max_array
                else {"array_shape": list(value.shape), "dtype": str(value.dtype)})
    if isinstance(value, dict):
        return {key: jsonable(item, max_array) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item, max_array) for item in value]
    return value


class Exhibit:
    """One reproduction script: a named result sink plus the arguments every script accepts."""

    def __init__(self, name: str, description: str, networks: bool = True,
                 dispersions: bool = True):
        self.name = name
        self.sink = RESULTS / f"{name}.jsonl"
        self.parser = argparse.ArgumentParser(prog=name, description=description)
        if networks:
            self.parser.add_argument("--networks", help="comma-separated names (default: the corpus)")
            self.parser.add_argument("--max-links", type=int, default=None,
                                     help="skip networks larger than this")
        if dispersions:
            self.parser.add_argument("--dispersions", default=None,
                                     help="comma-separated dispersion parameters, selecting within "
                                          "what this exhibit declares (default: all of them)")
        self.parser.add_argument("--out", help="override the result file")
        self.parser.add_argument("--setting", action="append", default=[], metavar="NAME=VALUE",
                                 help="override one solver setting; repeatable")
        self.parser.add_argument("--checkpoint-dir",
                                 help="write resumable per-cell state here, so a run stopped by a "
                                      "wall-clock limit continues rather than restarts")
        self.parser.add_argument("--redo", action="store_true",
                                 help="recompute cells already present in the sink")
        self.parser.add_argument("--terminal", action="store_true",
                                 help="this is the last round for these cells: a cell stopped by its "
                                      "budget is recorded as an unresolved result rather than left "
                                      "absent for a later run to resume")
        self.parser.add_argument("--time-budget", type=float, default=None, metavar="SECONDS",
                                 help="stop cleanly and checkpoint once this much wall time has "
                                      "passed, rather than waiting to be killed by a scheduler")
        self.parser.add_argument("--cell-seconds", type=float, default=None, metavar="SECONDS",
                                 help="give each cell its own budget, so one long cell does not consume "
                                      "the run. Without it the first cell of a network can exhaust the "
                                      "whole wall and the remaining dispersions are never attempted at "
                                      "all: that is how twelve corpus cells came to be reported missing "
                                      "when they had never been tried")
        self.args = None
        self.settings = DEFAULTS
        self.run_spec: dict = {}
        self.started = perf_counter()
        self.unfinished: list = []
        self._peak_at_last_record = peak_rss_mb()

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
        return self

    def parse(self, argv=None):
        self.args = self.parser.parse_args(argv)
        if self.args.out:
            self.sink = Path(self.args.out)
        self.sink.parent.mkdir(parents=True, exist_ok=True)
        self.settings = _apply_overrides(DEFAULTS, self.args.setting)
        self.run_spec = _publishable(dict(exhibit=self.name, **vars(self.args)))
        # Printed at launch and not only stored. A run whose specification is visible only in the records
        # it eventually writes cannot be identified while it is running, or at all if it is stopped.
        print(f"[{self.name}] {json.dumps(self.run_spec, default=str, sort_keys=True)}", flush=True)
        return self.args

    @property
    def declared(self) -> list[tuple]:
        """The cells this exhibit is required to produce, from the declarations in ``config``.

        An exhibit with no declaration falls back to the corpus crossed with the dispersion grid, which is
        what a new exhibit gets before anyone has decided what it should run. The validator reports that
        as undeclared rather than treating it as a choice.
        """
        declared = EXHIBIT_CELLS.get(self.name)
        if declared is not None:
            return [tuple(cell) for cell in declared]
        return [(name, mu) for name in corpus.select(None) for mu in DISPERSIONS]

    @property
    def cells(self) -> list[tuple]:
        """The declared cells this run will attempt, after the command line narrows them.

        ``--networks`` and ``--dispersions`` select within the declaration and never beyond it, so one
        task of a job array runs a subset of what was declared and the union of the array is the whole of
        it. A name that is not declared is an error rather than an extra cell, because a sink holding a
        cell nobody declared is a result set nobody chose.
        """
        cells = self.declared
        chosen = getattr(self.args, "networks", None)
        if chosen:
            wanted = [name.strip() for name in chosen.split(",")]
            unknown = [name for name in wanted if name not in {cell[0] for cell in cells}]
            if unknown:
                raise SystemExit(f"{self.name} does not declare {', '.join(unknown)}; it declares "
                                 f"{', '.join(sorted({cell[0] for cell in cells}))}")
            cells = [cell for cell in cells if cell[0] in wanted]
        if getattr(self.args, "dispersions", None) is not None:
            wanted_mu = {float(value) for value in self.args.dispersions.split(",")}
            cells = [cell for cell in cells if cell[1] in wanted_mu]
        if getattr(self.args, "max_links", None) is not None:
            sizes = {name: len(corpus.load(name)[0]["tail"]) for name in {cell[0] for cell in cells}}
            cells = [cell for cell in cells if sizes[cell[0]] <= self.args.max_links]
        present = {name for name, _, _ in corpus.available()}
        return [cell for cell in cells if cell[0] in present]

    @property
    def pending(self) -> list[tuple]:
        """The cells still to run: the declared ones this sink does not already hold."""
        if self.args.redo:
            return self.cells
        done = self.completed
        return [cell for cell in self.cells if cell not in done]

    @property
    def networks(self) -> list[str]:
        """The networks of the pending cells, in declaration order."""
        seen, ordered = set(), []
        for name, _ in self.pending:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    @property
    def dispersions(self) -> list[float]:
        """The dispersions this run covers.

        An exhibit that declares its cells takes them from the declaration. One that does not -- the
        witness, which is a fixed synthetic network and not a corpus sweep -- takes them from its own
        argument, whose default is the grid it declares for itself. Deriving them from the corpus cells
        in that case silently drops any dispersion the corpus grid does not have.
        """
        if EXHIBIT_CELLS.get(self.name) is None:
            return [float(value) for value in self.args.dispersions.split(",")]
        return sorted({mu for _, mu in self.cells})

    def dispersions_for(self, network: str) -> list[float]:
        """The pending dispersions of one network, in ascending order."""
        return sorted(mu for name, mu in self.pending if name == network)

    @property
    def completed(self) -> set:
        """Cells already written to this sink, so a resumed run does not redo them.

        A scheduler that stops a job at its wall-clock limit leaves the records already written intact,
        because each is flushed as it is produced. Re-submitting the same job therefore continues the
        sweep rather than restarting it, and re-submitting until this set stops growing is a complete
        run.
        """
        if not self.sink.exists():
            return set()
        done = set()
        for line in open(self.sink):
            if not line.strip():
                continue
            record = json.loads(line)
            if "network" in record and "mu" in record:
                done.add((record["network"], float(record["mu"])))
        return done

    @property
    def deadline(self):
        """When to stop, as a monotonic timestamp, or ``None`` when no budget was given."""
        budget = getattr(self.args, "time_budget", None)
        return None if budget is None else self.started + float(budget)

    def cell_deadline(self):
        """When the cell now starting must stop, whichever of its own budget and the run's comes first.

        A per-cell budget is what lets a sweep reach every dispersion. The corpus sweep runs a whole
        network per task, and on Austin, Hessen-Asymmetric and Sydney the first dispersion took the whole
        twelve-hour wall, so mu=1, 2, 5 and 10 were never attempted. They were then reported as missing
        alongside the one that had genuinely been tried, which is a different thing.
        """
        budget = getattr(self.args, "cell_seconds", None)
        own = None if budget is None else perf_counter() + float(budget)
        run = self.deadline
        return own if run is None else (run if own is None else min(own, run))

    def unresolved(self, network: str, mu: float, reason: str, **fields) -> None:
        """Record that one cell reached the end of a declared budget without settling.

        This is a result and not an absence. A cell that is simply missing from the sink is
        indistinguishable from one nobody ran, and the reader cannot tell a network the method does not
        settle on from a job the scheduler killed. What the run did reach is recorded, with the
        quantities it could not measure left as ``None`` so the tables print ``n.r.`` rather than a
        number that was never computed.
        """
        self.emit(dict(exhibit=self.name, network=network, mu=mu, converged=False,
                       stopped_on=reason, unresolved=True, **fields))
        print(f"  {network} mu={mu:g}: UNRESOLVED on {reason}; recorded as a diagnostic row",
              flush=True)

    def out_of_time(self) -> bool:
        """Whether to stop before starting another cell."""
        deadline = self.deadline
        return deadline is not None and perf_counter() >= deadline

    def progress_for(self, network: str, mu: float):
        """A printer for the solver's heartbeat, so a long cell is visibly alive rather than silent."""
        def report(message: str) -> None:
            print(f"    [{network} mu={mu:g}] {message}", flush=True)
        return report

    def to_be_continued(self, network: str, mu: float, reason: str = "the time budget",
                        **fields) -> None:
        """Record that one cell stopped short of an answer.

        By default no result record is written. The cell is therefore absent from the sink, a later run
        does not skip it, and it resumes from its checkpoint. That is right for a cell that will finish
        given another round.

        It is wrong for a cell that will not, and under ``--terminal`` this writes an unresolved result
        instead. A sweep whose cells are resubmitted until something settles has made its budget a free
        parameter, and an evaluation run declares its last round in advance. Absence then stops being a
        promise to continue and becomes indistinguishable from a cell nobody attempted: ten cells of the
        first held-out array ran twelve hours each and left the sink unable to say they had been tried.
        """
        if getattr(self.args, "terminal", False):
            self.unresolved(network, mu, reason, **fields)
            return
        self.unfinished.append((network, mu))
        print(f"  {network} mu={mu:g}: stopped on {reason}, checkpointed, to be continued", flush=True)

    def checkpoint_for(self, network: str, mu: float, part: str = ""):
        """Where one cell's resumable state lives, or ``None`` when checkpointing is off.

        ``part`` names one of several solves within a cell, where an exhibit runs more than one -- the two
        tie-break arms of the solution set, the two support rules of the comparison. Without it they would
        share a file and each would resume from the other's state.
        """
        if not getattr(self.args, "checkpoint_dir", None):
            return None
        directory = Path(self.args.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f"_{part}" if part else ""
        return directory / f"{self.name}_{network}_mu{mu:g}{suffix}.npz"

    def clear_checkpoint(self, network: str, mu: float) -> None:
        """Remove a finished cell's state, every part of it, so a later run does not resume into it.

        Matched against this cell's own names and not by prefix. A dispersion is formatted with ``%g``,
        so the stem for mu=1 is ``..._mu1`` and a prefix glob on it also matches ``..._mu10.npz`` and the
        ``.partial`` file a concurrent write is holding. The mu=1 cell of one array task therefore
        deleted the mu=10 cell's state from another, silently where the two merely overlapped and with a
        ``FileNotFoundError`` where it landed between the write and the rename -- which is how
        full_graph Hessen-Asymmetric mu=10 died after eleven hours.
        """
        prefix = self.checkpoint_for(network, mu)
        if prefix is None:
            return
        stem = prefix.name[:-len(".npz")]
        for path in prefix.parent.iterdir():
            name = path.name
            body = name[:-len(".partial")] if name.endswith(".partial") else name
            if not body.endswith(".npz"):
                continue
            body = body[:-len(".npz")]
            if body == stem or body.startswith(stem + "_"):
                path.unlink(missing_ok=True)

    def preflight(self) -> dict:
        """Print what this run found and what it will do, before it spends anything.

        Every network it is about to sweep is opened, sized and checked for the elements the model has no
        answer for, with each value beside the threshold that judges it. A cell can run for hours; a run
        started on the wrong data still produces numbers, and the only cheap moment to notice is now.

        Returns the reports, so a caller can record them or refuse to continue. Nothing here raises: the
        decision to stop is the reader's, and a warning that aborted the run would make a diagnostic into
        a gate on cells that are merely unusual.
        """
        cells, pending = self.cells, self.pending
        print(f"[{self.name}] pre-flight", flush=True)
        print(f"  data {DATA_ROOT}", flush=True)
        print(f"  pinned commit {PINNED_COMMIT[:12]}", flush=True)
        if not DATA_ROOT.is_dir():
            print("    WARNING: the data root does not exist; run reproduction/tools/fetch_networks.py",
                  flush=True)
        declared = len(self.declared)
        print(f"  cells: {declared} declared, {len(cells)} selected, {len(cells) - len(pending)} "
              f"already in {self.sink.name}, {len(pending)} to run", flush=True)
        missing = sorted({cell[0] for cell in self.declared} - {name for name, _, _ in
                                                                corpus.available()})
        if missing:
            print(f"    WARNING: declared but not present under the data root: {', '.join(missing)}",
                  flush=True)
        reports = {}
        for name in sorted({cell[0] for cell in pending}):
            try:
                network, demand = corpus.load(name)
            except (ValueError, KeyError, OSError) as exc:
                # The docstring above promises this method does not raise, and it has to be true: the
                # pre-flight runs before the loop that would record anything, so an exception here loses
                # the whole job and leaves every one of its cells indistinguishable from a cell nobody
                # ran. The sweep reaches the same refusal and records it per cell.
                print(f"  {name}: REFUSED, {exc}", flush=True)
                continue
            report = inspect_network(name, network, demand)
            reports[name] = report
            for line in report_lines(report):
                print(line, flush=True)
        if not pending:
            print("  nothing to do: every declared cell is already recorded", flush=True)
        baseline = peak_rss_mb()
        if baseline is not None:
            print(f"  memory: {baseline:.0f} MiB resident before the first cell", flush=True)
        print(f"[{self.name}] pre-flight done, starting\n", flush=True)
        return reports

    def emit(self, record: dict) -> dict:
        """Stamp, write and flush one record.

        Peak memory is added here rather than in each exhibit, so no exhibit can forget it. Two figures:
        the process high-water mark, which is what a job request has to cover, and how much of it this
        cell added, which is what says whether the cost grows with the cell or was paid once.
        """
        peak = peak_rss_mb()
        if peak is not None:
            previous = self._peak_at_last_record
            record = dict(record, peak_rss_mb=round(peak, 1),
                          peak_rss_added_mb=(None if previous is None
                                             else round(max(peak - previous, 0.0), 1)))
            self._peak_at_last_record = peak
        record = stamp(jsonable(record), self.settings, run_spec=self.run_spec)
        with open(self.sink, "a") as handle:
            handle.write(json.dumps(record) + "\n")
            handle.flush()
        return record

    def done(self, count: int) -> int:
        """Report the run and return the exit status: nonzero while work remains.

        A distinct status for "incomplete" lets a submission script tell a finished sweep from one that
        ran out of time, and resubmit only in the second case.
        """
        print(f"{self.name}: {count} records written to {self.sink}", flush=True)
        if self.unfinished:
            listed = ", ".join(f"{network} mu={mu:g}" for network, mu in self.unfinished)
            print(f"{self.name}: {len(self.unfinished)} cell(s) TO BE CONTINUED ({listed}). "
                  f"Resubmit the same command to resume from the checkpoints.", flush=True)
            return EXIT_INCOMPLETE
        return 0


#: Arguments naming a location on the machine that ran the sweep. A record is published; where someone's
#: scratch directory is says nothing about the result and should not travel with it. Whether one was given
#: is kept, because that does bear on how the run was produced.
_LOCAL_PATHS = ("out", "checkpoint_dir")


def _publishable(spec: dict) -> dict:
    """The run specification with local paths reduced to whether there was one."""
    published = dict(spec)
    for name in _LOCAL_PATHS:
        if name in published:
            published[name] = bool(published[name])
    return published


def _apply_overrides(settings: Settings, overrides: list[str]) -> Settings:
    """Apply ``name=value`` settings overrides, refusing anything that is not a known setting."""
    if not overrides:
        return settings
    known = {field.name: _setting_type(field)
             for field in settings.__dataclass_fields__.values()}
    changes = {}
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"setting override {override!r} is not of the form NAME=VALUE")
        name, _, raw = override.partition("=")
        if name not in known:
            raise SystemExit(f"unknown setting {name!r}; known settings are "
                             f"{', '.join(sorted(known))}")
        changes[name] = _coerce(name, known[name], raw)
    return replace(settings, **changes)


_SETTING_TYPES = {"bool": bool, "tuple": tuple, "int": int, "float": float, "str": str}


def _setting_type(field) -> type:
    """The type one setting's override should be read as.

    Taken from the annotation and not from the value the setting currently holds. Every uncapped budget
    defaults to ``None``, whose type is ``NoneType``, and ``NoneType(raw)`` raises; read off the value,
    the six settings a run is most likely to need to bound could not be overridden at all.
    """
    for candidate in str(field.type).replace("|", " ").replace("[", " ").split():
        if candidate in _SETTING_TYPES:
            return _SETTING_TYPES[candidate]
    return str


def _coerce(name: str, kind: type, raw: str):
    """Read one override into the type its setting holds.

    The type is not enough on its own. ``bool("false")`` is ``True``, so a run asked to switch something
    off switches it on and records that it did; and ``tuple("4000000,40000000")`` is a tuple of sixteen
    characters, which then reaches a comparison as a string. Both are silent, and both would be stamped
    onto every record the run produced.
    """
    if kind is bool:
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise SystemExit(f"setting {name}: {raw!r} is not a boolean")
    if kind is tuple:
        parts = [piece.strip() for piece in raw.split(",") if piece.strip()]
        if not parts:
            raise SystemExit(f"setting {name}: {raw!r} is empty")
        return tuple(int(piece) if piece.isdigit() else float(piece) for piece in parts)
    try:
        return kind(raw)
    except ValueError as exc:
        raise SystemExit(f"setting {name}: {raw!r} is not a {kind.__name__} ({exc})") from exc


def load_records(name: str, root: Path = RESULTS) -> list[dict]:
    """Read one exhibit's records back, across its shards."""
    return [json.loads(line) for path in sink_paths(name, root)
            for line in open(path) if line.strip()]


def file_fingerprint(path) -> dict:
    """Size and content hash of an input file, recorded alongside any result derived from it.

    A record that names a file but was computed against an earlier version of it is indistinguishable
    from a current one unless the file is fingerprinted at the time of reading.
    """
    path = Path(path)
    return {"source_file": path.name,
            "source_bytes": path.stat().st_size,
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
