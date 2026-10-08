"""The declarations a run is judged against: the corpus, the tiers, and the cells a job array indexes.

These exist because the first job array repeated the network names by hand. The copy drifted: it ran
Winnipeg, which is filtered out where records are loaded, and never ran Austin, which is a reported row in
two tables. Nothing caught it, because nothing compared the two lists.
"""
import re
from pathlib import Path

from endogenous_sue.config import (
    BUDGET_SENSITIVE,
    CERTIFICATE_CELLS,
    CERTIFICATE_TIER,
    CORPUS,
    CORPUS_CELLS,
    DIAMETER_NETWORKS,
    DISPERSIONS,
    EXCLUDED_NETWORKS,
    EXHIBIT_CELLS,
    FACE_CELL,
    OSCILLATION_PANELS,
    REPORTED_CORPUS,
    WALK_CELLS,
)

HPC = Path(__file__).resolve().parents[1] / "reproduction" / "cluster"


def test_the_reported_corpus_is_the_corpus_less_the_exclusions():
    assert set(REPORTED_CORPUS) == set(CORPUS) - set(EXCLUDED_NETWORKS)
    assert list(REPORTED_CORPUS) == [name for name in CORPUS if name in REPORTED_CORPUS]


def test_every_tier_names_a_reported_network():
    for name in CERTIFICATE_TIER + BUDGET_SENSITIVE:
        assert name in REPORTED_CORPUS, f"{name} is in a tier but is not reported"


def test_the_tiers_do_not_overlap():
    assert not set(CERTIFICATE_TIER) & set(BUDGET_SENSITIVE)


def test_the_cells_cover_the_declarations():
    from endogenous_sue.config import CERTIFICATE_EXCLUDED

    assert [cell[0] for cell in CORPUS_CELLS] == list(REPORTED_CORPUS)
    assert (len(CERTIFICATE_CELLS)
            == len(CERTIFICATE_TIER) * len(DISPERSIONS) - len(CERTIFICATE_EXCLUDED))
    assert {cell[0] for cell in CERTIFICATE_CELLS} == set(CERTIFICATE_TIER)
    assert {cell[1] for cell in CERTIFICATE_CELLS} == set(DISPERSIONS)


def test_an_excluded_cell_is_declared_out_of_every_tier_sweep_that_mirrors_the_tier():
    """A cell outside the certificate tier must be outside the measurements stated over the same cells.

    The solution-set counts and the certificate counts are quoted in one sentence, so they have to be
    over one set of cells. They are not excluded from the corpus sweep or the full-graph comparator,
    which use the two-phase solver rather than the inclusion scheme and the tied face, and which have no
    reason to be unaffordable there.
    """
    from endogenous_sue.config import CERTIFICATE_EXCLUDED, EXHIBIT_CELLS

    for excluded in CERTIFICATE_EXCLUDED:
        cell = (excluded[0], float(excluded[1]))
        for name in ("certificate_sweep", "solution_set"):
            assert cell not in {(n, float(m)) for n, m in EXHIBIT_CELLS[name]}, \
                f"{name} still declares {cell}"
        for name in ("corpus_sweep", "full_graph"):
            assert cell in {(n, float(m)) for n, m in EXHIBIT_CELLS[name]}, \
                f"{name} should still declare {cell}: it does not use the tied face"


def _array_size(script: Path) -> int:
    """The number of tasks a job script's --array line declares."""
    text = script.read_text()
    match = re.search(r"#SBATCH --array=(\d+)-(\d+)", text)
    assert match, f"{script.name} declares no array range"
    return int(match.group(2)) - int(match.group(1)) + 1


def test_each_job_array_is_sized_to_its_declared_cells():
    assert _array_size(HPC / "corpus_sweep.slurm") == len(CORPUS_CELLS)
    assert _array_size(HPC / "certificate_sweep.slurm") == len(CERTIFICATE_CELLS)
    from endogenous_sue.config import HELD_OUT_CELLS
    assert _array_size(HPC / "held_out.slurm") == len(HELD_OUT_CELLS)


def test_no_job_script_repeats_a_network_name():
    """A name written into a job script is a second declaration, and the two will drift."""
    for script in sorted(HPC.glob("*.slurm")):
        text = script.read_text()
        for name in CORPUS:
            assert name not in text, (f"{script.name} names {name}; resolve it with "
                                      f"reproduction/tools/cells.py")


def test_every_exhibit_declares_its_cells():
    """An exhibit sweeping whatever is present produces a result set nobody chose."""
    for name in ("corpus_sweep", "certificate_sweep", "support_comparison", "convergence",
                 "full_graph", "solution_set", "held_out"):
        assert name in EXHIBIT_CELLS, f"{name} declares no cells"
        assert EXHIBIT_CELLS[name], f"{name} declares an empty cell set"


def test_no_exhibit_declares_a_network_outside_the_reported_corpus():
    """One exemption, by name. The held-out sweep exists precisely to run networks the corpus does not
    report, and a blanket rule here would either forbid it or have to be relaxed for everything."""
    for name, cells in EXHIBIT_CELLS.items():
        if name == "held_out":
            continue
        for network, _ in cells:
            assert network in REPORTED_CORPUS, f"{name} declares {network}, which is not reported"


def test_the_held_out_networks_are_outside_the_reported_corpus():
    """The point of the set. A held-out network that the study already reports measures nothing."""
    from endogenous_sue.config import CORPUS, HELD_OUT_NETWORKS
    assert HELD_OUT_NETWORKS, "the held-out set is empty"
    for network in HELD_OUT_NETWORKS:
        assert network not in CORPUS, f"{network} is in the corpus and cannot be held out"


def test_the_quoted_cells_are_cells_that_are_actually_run():
    """A number quoted for one cell must come from a cell some exhibit produces."""
    certificate = {tuple(cell) for cell in EXHIBIT_CELLS["certificate_sweep"]}
    for cell in WALK_CELLS:
        assert tuple(cell) in certificate, f"the walk quotes {cell}, which the sweep does not run"
    assert tuple(FACE_CELL) in certificate, f"the face dimension quotes {FACE_CELL}, which is not run"


def test_the_figures_have_the_cells_they_plot():
    """Each figure names its panels; a panel with no cell is an empty axis in the paper."""
    convergence = {tuple(cell) for cell in EXHIBIT_CELLS["convergence"]}
    for cell in OSCILLATION_PANELS:
        assert tuple(cell) in convergence, f"figure 2 plots {cell}, which the timing sweep does not run"
    solution_set = {network for network, _ in EXHIBIT_CELLS["solution_set"]}
    for network in DIAMETER_NETWORKS:
        assert network in solution_set, f"figure 3 plots {network}, which is not measured"


def test_the_exclusion_criterion_admits_the_synthetic_network_it_is_not_about():
    """Unit capacity alone is not the criterion; unit capacity with a steep exponent is.

    The Braess example has unit capacity on every link by construction and a unit exponent, so its cost is
    linear in flow -- an ordinary cost function, not an unnormalised flow raised to a large power. Testing
    on capacity alone excluded a network the corpus reports.
    """
    from helpers import require_networks

    # This one reads the benchmark files; the rest of this module only reads declarations. Guarding the
    # module would skip those too, so the guard belongs here.
    require_networks()

    from endogenous_sue import corpus as benchmark
    from endogenous_sue.preflight import inspect_network

    for name in ("Braess-Example", "SiouxFalls"):
        network, demand = benchmark.load(name)
        report = inspect_network(name, network, demand)
        assert not report.warnings, f"{name}: {report.warnings}"

    for name in ("Barcelona", "Winnipeg"):
        network, demand = benchmark.load(name)
        report = inspect_network(name, network, demand)
        assert any("exclusion criterion" in message for message in report.warnings), \
            f"{name} is excluded by the corpus but not by the criterion"


def test_no_job_script_resolves_a_path_through_its_own_location():
    """Slurm runs a copy of the script from a spool directory, so `dirname $0` is not the checkout.

    Every one of these scripts sourced its environment that way and none had ever been submitted, so the
    first job ever queued from them died in two seconds. Paths are resolved against the working directory
    the script sets, which is the checkout.
    """
    for script in sorted(HPC.glob("*.slurm")):
        for number, line in enumerate(script.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]
            assert "dirname" not in code, (
                f"{script.name}:{number} resolves a path through $0; Slurm copies the script to a spool "
                f"directory, so it will not find what it is looking for")


def test_every_job_script_checks_its_inputs_before_it_runs_anything():
    """A job that discovers a missing input after an hour of queueing has wasted the queueing."""
    for script in sorted(HPC.glob("*.slurm")):
        text = script.read_text()
        assert "== inputs ==" in text, f"{script.name} names no inputs to check"
        assert "MISSING" in text, f"{script.name} does not fail on a missing input"


def test_every_job_script_exits_with_the_status_of_its_python():
    """Slurm records the exit status of the last command the script ran.

    Every script ends by printing ``sacct`` output, which succeeds whatever the run did. Three of them
    stopped there, so a cell that exited 2 for "to be continued" was recorded as COMPLETED: twenty-one
    such cells looked like successes in the accounting while writing no record at all.
    """
    for script in sorted(HPC.glob("*.slurm")):
        text = script.read_text()
        if "status=$?" not in text:
            continue
        assert re.search(r"^exit \$status\s*$", text, re.MULTILINE), (
            f"{script.name} captures the python exit status but never exits with it")


def test_the_held_out_protocol_holds_on_this_revision():
    """The guarantees docs/held_out_protocol.md makes about the code, asserted against the code.

    A protocol is only worth writing if it can be checked. Each clause here is one the document states
    in prose and a reader would otherwise have to take on trust, and each is the kind that a later
    change could silently break: turning the envelope rule on by default, adding a held-out cell to the
    declared direct-face route, or letting a held-out network into the corpus it is meant to be held out
    from.
    """
    from endogenous_sue.config import (
        BUDGET_SENSITIVE,
        CORPUS,
        DEFAULTS,
        DIRECT_FACE_CELLS,
        HELD_OUT_CELLS,
        HELD_OUT_NETWORKS,
        TIE_TOLERANCE,
    )

    assert DEFAULTS.envelope_stable_windows is None, (
        "the protocol declares the envelope rule off for the held-out run; it is on by default here")
    direct = {tuple(cell) for cell in DIRECT_FACE_CELLS}
    on_route = [cell for cell in HELD_OUT_CELLS if tuple(cell) in direct]
    assert not on_route, f"held-out cells are on the declared direct-face route: {on_route}"
    assert len(HELD_OUT_CELLS) == 20, f"the protocol fixes twenty cells, found {len(HELD_OUT_CELLS)}"
    assert not [n for n in HELD_OUT_NETWORKS if n in CORPUS], (
        "a held-out network is in the corpus, so it is not held out from anything")
    assert not [n for n in HELD_OUT_NETWORKS if n in BUDGET_SENSITIVE], (
        "a held-out network is declared budget-sensitive; that is a corpus judgement and pre-empts the "
        "outcome this run exists to measure")
    assert TIE_TOLERANCE == 1e-10 and DEFAULTS.residual_tolerance == 1e-10, (
        "the protocol declares the corpus tolerances unchanged for the held-out run")
