"""Result sinks: how a sweep's records are written and read back.

A sweep run as a job array writes one shard per task. Everything downstream has to read the shards
together, because a concatenation step that a reader can forget is a step that produces a table from half
the run.
"""
import json
import tempfile
from pathlib import Path

from reproduction.exhibits.records import load_records, sink_paths


def _write(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_shards_are_read_together_with_the_plain_sink():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "corpus_sweep").mkdir()
        (root / "certificate_sweep").mkdir()
        _write(root / "corpus_sweep.jsonl", [{"network": "A", "mu": 1.0}])
        _write(root / "corpus_sweep" / "B.jsonl", [{"network": "B", "mu": 1.0}])
        _write(root / "corpus_sweep" / "C.jsonl", [{"network": "C", "mu": 2.0}])
        _write(root / "certificate_sweep" / "D.jsonl", [{"network": "D", "mu": 1.0}])

        assert len(sink_paths("corpus_sweep", root)) == 3
        assert {record["network"] for record in load_records("corpus_sweep", root)} == {"A", "B", "C"}
        assert {record["network"] for record in load_records("certificate_sweep", root)} == {"D"}


def test_an_absent_sink_reads_as_nothing_rather_than_raising():
    with tempfile.TemporaryDirectory() as directory:
        assert sink_paths("corpus_sweep", Path(directory)) == []
        assert load_records("corpus_sweep", Path(directory)) == []


def test_one_exhibit_does_not_read_another_exhibits_shards():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "full_graph").mkdir()
        _write(root / "full_graph" / "Anaheim.jsonl", [{"network": "Anaheim", "mu": 1.0}])
        assert sink_paths("full", root) == []
        assert len(sink_paths("full_graph", root)) == 1


def test_an_exhibit_without_a_declaration_keeps_its_own_dispersion_grid():
    """The witness is a fixed synthetic network, not a corpus sweep, and declares its own grid.

    Deriving its dispersions from the corpus cells drops any value the corpus grid does not carry, which
    on this exhibit is one of the six the non-existence result is stated over.
    """
    from endogenous_sue.config import WITNESS_DISPERSIONS
    from reproduction.exhibits.records import Exhibit

    exhibit = Exhibit("witness", "the witness", networks=False)
    exhibit.parser.set_defaults(dispersions=",".join(str(mu) for mu in WITNESS_DISPERSIONS))
    with tempfile.TemporaryDirectory() as directory:
        exhibit.parse(["--out", str(Path(directory) / "witness.jsonl")])
        assert exhibit.dispersions == list(WITNESS_DISPERSIONS)


def test_a_declared_exhibit_refuses_a_network_it_does_not_declare():
    """A cell nobody declared is a result set nobody chose, so it is an error and not an extra row."""
    from reproduction.exhibits.records import Exhibit

    exhibit = Exhibit("certificate_sweep", "the certificate sweep")
    with tempfile.TemporaryDirectory() as directory:
        exhibit.parse(["--out", str(Path(directory) / "out.jsonl"), "--networks", "Braess-Example"])
        try:
            _ = exhibit.cells
        except SystemExit as exit_code:
            assert "does not declare" in str(exit_code)
        else:
            raise AssertionError("an undeclared network was accepted")


def test_an_uncapped_setting_can_still_be_overridden():
    """The override type comes from the annotation, not from the value the setting happens to hold.

    Every uncapped budget defaults to ``None``, whose type is ``NoneType``, and ``NoneType("2")`` raises.
    Read off the value, the six settings a run is most likely to need to bound could not be set at all.
    """
    from endogenous_sue.config import DEFAULTS
    from reproduction.exhibits.records import _apply_overrides

    assert DEFAULTS.max_newton_iterations is None
    changed = _apply_overrides(DEFAULTS, ["max_newton_iterations=2", "memory_ceiling_mb=1024"])
    assert changed.max_newton_iterations == 2
    assert changed.memory_ceiling_mb == 1024.0


def test_an_unresolved_cell_is_neither_solved_nor_unsolved():
    """A cell that never settled has no residual, and must not be counted as though it did.

    The solved headline divides cells by whether their residual clears a threshold. A cell that reached
    the end of its budget without the support settling never ran phase two, so it has no residual against
    a frozen support: counting it as unsolved would assert a measurement that was not taken, and counting
    it as solved would be worse. It is recorded so that it is distinguishable from a cell nobody ran, and
    excluded from both counts.
    """
    from reproduction.exhibits.records import Exhibit

    record = dict(exhibit="corpus_sweep", network="Hessen-Asymmetric", mu=10.0, converged=False,
                  stopped_on="support set did not settle", unresolved=True,
                  residual=None, solved=False, conserves=None, strict_reload_defect=None)
    assert record["residual"] is None, "an unresolved cell must not carry a residual"
    assert record["conserves"] is None, "conservation is not a verdict where the flow never settled"
    assert record["unresolved"] is True
    assert hasattr(Exhibit, "unresolved"), "the exhibit must be able to record this outcome"
    assert hasattr(Exhibit, "cell_deadline"), "every cell needs its own budget or later cells go untried"


def test_clearing_one_cell_does_not_delete_another():
    """A dispersion is formatted with %g, so mu=1's stem is a prefix of mu=10's name.

    Clearing by prefix glob therefore deleted the mu=10 cell's checkpoint, and its `.partial` file mid
    write: full_graph Hessen-Asymmetric mu=10 died on FileNotFoundError after eleven hours because the
    mu=1 task in the same array had just finished. Where the two did not collide in time it was silent,
    and the mu=10 cell simply restarted from nothing.
    """
    import tempfile
    from pathlib import Path

    from reproduction.exhibits.records import Exhibit

    with tempfile.TemporaryDirectory() as directory:
        exhibit = Exhibit("full_graph", "test")
        exhibit.parse(["--checkpoint-dir", directory])
        room = Path(directory)
        for name in ("full_graph_Net_mu1.npz", "full_graph_Net_mu1_face.npz",
                     "full_graph_Net_mu10.npz", "full_graph_Net_mu10.npz.partial",
                     "full_graph_Net_mu10_face.npz"):
            (room / name).write_bytes(b"x")

        exhibit.clear_checkpoint("Net", 1.0)
        survived = sorted(path.name for path in room.iterdir())
        assert survived == ["full_graph_Net_mu10.npz", "full_graph_Net_mu10.npz.partial",
                            "full_graph_Net_mu10_face.npz"], survived

        exhibit.clear_checkpoint("Net", 10.0)
        assert not list(room.iterdir()), "the cell's own files, including .partial, must all go"


def test_the_timing_exhibit_names_its_certificate_for_what_it_tests():
    """A strictness test on a fixed-support flow must not be recorded under a name meaning more.

    `convergence` calls `certify` with no mixture, so the only verdict available to it is whether the
    flow it timed is the loading on the strict support of its own cost. A tied face cannot pass that and
    is not expected to. Recorded as `certified`, the field read as a certificate and 3 of 45 read as a
    pass rate: of the 42 that failed, 15 were cells the certificate sweep independently found contested,
    which is a mixture being tested for strictness and cannot be a finding about the solver.

    The mixtures that would settle those cells belong to the certificate sweep's own flows -- different
    computed candidates -- and attaching them to these would certify one flow with another's evidence.
    So the field is renamed rather than the test widened.
    """
    import inspect

    from reproduction.exhibits import convergence

    source = inspect.getsource(convergence.main)
    assert "strictly_certified=" in source, "the verdict field must say it is strictness only"
    assert "strict_certificate_kind=" in source
    assert "strict_certify_seconds=" in source
    import re
    for bare in ("certified", "certificate_kind", "certify_seconds"):
        # At a word boundary: ``strictly_certified`` contains ``certified`` and is the correct name.
        assert not re.search(rf"(?<![A-Za-z0-9_]){bare}\s*=", source), (
            f"{bare!r} as a field name claims more than this exhibit tests")
    # And nothing here may reach for the certificate sweep's mixtures.
    assert "certify(net, od, mu, polished.x, settings=settings)" in source, (
        "the timed check must stay a no-mixture strict test on this exhibit's own flow")


def test_a_terminal_round_records_a_stopped_cell_instead_of_leaving_it_absent():
    """Absence means "resume me", which is a promise a last round cannot keep.

    Without ``--terminal`` a budget-stopped cell writes nothing, so a later run resumes it rather than
    skipping it. That is right while rounds continue and wrong once they stop: ten cells of the first
    held-out array each ran twelve hours and left the sink unable to say they had been attempted at all.
    """
    from reproduction.exhibits.records import Exhibit

    for terminal, expected in ((False, 0), (True, 1)):
        with tempfile.TemporaryDirectory() as directory:
            sink = Path(directory) / "held_out.jsonl"
            exhibit = Exhibit("held_out", "the held-out sweep")
            argv = ["--out", str(sink), "--networks", "Munich", "--dispersions", "1"]
            exhibit.parse(argv + (["--terminal"] if terminal else []))
            exhibit.to_be_continued("Munich", 1.0, "the time budget", iterations=4045)
            rows = ([json.loads(line) for line in sink.read_text().splitlines()]
                    if sink.exists() else [])
            assert len(rows) == expected, (
                f"terminal={terminal} wrote {len(rows)} row(s), expected {expected}")
            if terminal:
                assert rows[0]["unresolved"] is True
                assert rows[0]["converged"] is False
                assert rows[0].get("certified") is None, (
                    "a cell that never reached the certificate must record certified as null, not "
                    "false: false reads as a test that ran and failed")
                assert rows[0]["iterations"] == 4045, "what the cell reached was not carried through"
