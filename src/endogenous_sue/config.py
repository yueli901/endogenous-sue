"""Every setting, tolerance and corpus constant used by this package.

No tolerance, budget, threshold, grid or corpus membership is defined outside this module. Two things are
deliberately not here: the structural definition of the synthetic witness network, whose node indices and
demand define the object rather than parametrise a method, and the exit status a sweep returns. Values are
grouped by what they govern, and each carries the reason it holds the value it does.

Two kinds of value appear. Fixed constants describe the data or the model and are not meant to be varied:
corpus membership, the pinned data commit, the representability clamp on the cost. Run settings are
gathered in :class:`Settings`, which is passed to the solver, recorded verbatim with every result, and is
the only mechanism for varying solver behaviour. No default is read from the environment, so a result
cannot depend on a shell.

Iteration counts here are limits, not stopping rules. The solver stops on a convergence criterion; a run
that instead reaches its limit is reported as unconverged and is not a result. The distinction matters
because the support-freezing theorem is qualitative -- it gives finite termination and no bound on the
number of iterations required -- so any fixed count is a property of the run and never of the method.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------------------------------
# Benchmark data
# ---------------------------------------------------------------------------------------------------

#: Commit of bstabler/TransportationNetworks the reported results were produced against.
PINNED_COMMIT = "d1639b4ef218c17928ba573e806ddf8ba5e7ae6d"

#: Location of the benchmark clone. The data is not redistributed with this package.
DATA_ROOT = Path(os.environ.get("ENDOGENOUS_SUE_DATA", REPO_ROOT / "data" / "TransportationNetworks"))

#: Location of result records.
RESULTS_ROOT = REPO_ROOT / "results"

#: Dispersion grid. Larger values concentrate route choice more tightly on the efficient support.
DISPERSIONS = (0.5, 1.0, 2.0, 5.0, 10.0)

#: Networks the study sweeps, in ascending order of size.
CORPUS = (
    "Braess-Example",
    "SiouxFalls",
    "Eastern-Massachusetts",
    "Berlin-Tiergarten",
    "Berlin-Prenzlauerberg-Center",
    "Berlin-Friedrichshain",
    "Berlin-Mitte-Center",
    "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center",
    "Anaheim",
    "Terrassa-Asymmetric",
    "Hessen-Asymmetric",
    "Winnipeg",
    "Winnipeg-Asymmetric",
    "Barcelona",
    "Chicago-Sketch",
    "Sydney",
    "Austin",
)

#: Swept, then excluded from the reported corpus. These files fold capacity into the BPR coefficient and
#: leave unit capacity on every link, so the load ratio is not a load ratio and the cost is not the
#: model's cost. Stated as a criterion a reader can check on any new network.
EXCLUDED_NETWORKS = ("Barcelona", "Winnipeg")

#: The corpus the study reports on. Job scripts index this rather than repeating the names, so a network
#: cannot be swept and unreported, or reported and never swept.
REPORTED_CORPUS = tuple(name for name in CORPUS if name not in EXCLUDED_NETWORKS)

#: Networks the a posteriori certificate is evaluated on, declared before the sweep runs rather than read
#: back from whichever cells happened to finish. The tier is the largest prefix of the corpus on which a
#: tied face is expected to be solvable within one job, plus Chicago-Sketch, which is included because its
#: low-dispersion faces are the hardest cells in the study and excluding them would flatter the method.
CERTIFICATE_TIER = (
    "SiouxFalls",
    "Eastern-Massachusetts",
    "Berlin-Friedrichshain",
    "Berlin-Prenzlauerberg-Center",
    "Berlin-Tiergarten",
    "Berlin-Mitte-Center",
    "Anaheim",
    "Chicago-Sketch",
)

#: Networks whose corpus rows are reported as budget-sensitive: the sweep reaches its stopping criterion
#: only for some dispersions, so their rows are diagnostics and never equilibrium claims. Declared here
#: for the same reason as the tier above.
BUDGET_SENSITIVE = (
    "Terrassa-Asymmetric",
    "Hessen-Asymmetric",
    "Austin",
    # Declared on evidence rather than in advance. Under a two-hour per-cell budget Sydney
    # settled at no dispersion, and under eleven hours per cell in the full-graph sweep it
    # settled at none either: the inclusion scheme reaches about 11,000 iterations in twelve
    # hours on 75,379 links and 3,343,560 demanded pairs. Its gap does fall with dispersion,
    # from 4.0e-02 at mu=0.5 to 1.7e-03 at mu=10, so this is a statement about what is
    # reachable within one job here and not about the method. It is dropped from TIMING_NETWORKS
    # for the same reason: a network that cannot be solved cannot be timed.
    "Sydney",
)

# ---------------------------------------------------------------------------------------------------
# What each exhibit runs
# ---------------------------------------------------------------------------------------------------
#
# Every exhibit's cells are declared here. An exhibit left to sweep "whatever is present" produces a
# result set nobody chose: the reported tables are over particular networks and particular dispersions,
# and a run that silently swept one network too few passes any check that only looks at the rows it has.
# The validator compares what a sink holds against these declarations, so a missing network is a failure
# and not an absence.

#: The controlled support comparison, over the dispersions its table is stated across.
SUPPORT_COMPARISON_NETWORKS = (
    "SiouxFalls",
    "Eastern-Massachusetts",
    "Berlin-Tiergarten",
    "Berlin-Friedrichshain",
    "Berlin-Mitte-Center",
    "Anaheim",
)
SUPPORT_COMPARISON_DISPERSIONS = (0.5, 1.0, 2.0, 5.0)

#: The timed networks. Every dispersion is timed, though the table reports one: a time at a single
#: dispersion cannot be told from a time that happens to be fast there.
TIMING_NETWORKS = (
    "SiouxFalls",
    "Eastern-Massachusetts",
    "Berlin-Tiergarten",
    "Berlin-Mitte-Center",
    "Anaheim",
    "Berlin-Mitte-Prenzlauerberg-Friedrichshain-Center",
    "Winnipeg-Asymmetric",
    "Chicago-Sketch",
    "Terrassa-Asymmetric",
)

#: The full-graph comparator runs on the whole reported corpus, because the point of its table is how
#: often the comparator fails to exist and that is a statement about the corpus.
FULL_GRAPH_NETWORKS = REPORTED_CORPUS

#: The accuracy comparison needs more dispersions than the corpus grid, because it fits a rate against
#: them and three points is not a fit. The extra points live in the same sink; the radius table takes its
#: columns from ``DISPERSIONS`` rather than from whatever the sink holds, so they do not widen it.
ACCURACY_NETWORKS = ("SiouxFalls",)
ACCURACY_DISPERSIONS = (0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0)

#: Dispersion recovery: generate a flow at a known value and re-estimate it. Restricted to networks where
#: the full-graph reference exists at these dispersions, since there is nothing to recover otherwise.
RECOVERY_NETWORKS = ("SiouxFalls", "Eastern-Massachusetts", "Anaheim", "Chicago-Sketch")
RECOVERY_DISPERSIONS = (1.0, 2.0, 5.0)

#: The solution-set measurement runs on the certificate tier, so its counts are over the same cells the
#: certificate counts are, and the two can be stated in one sentence.
SOLUTION_SET_NETWORKS = CERTIFICATE_TIER

#: The two cells whose active-set walk is fully instrumented. Every certificate cell records the walk
#: counters; these are the two the supplement quotes, and naming them here is what stops the quoted
#: numbers drifting to whichever cell was looked at last.
WALK_CELLS = (("SiouxFalls", 0.5), ("Eastern-Massachusetts", 2.0))

#: Cells whose face goes straight to the ladder, bypassing the accelerators and the active-set walk.
#:
#: :func:`endogenous_sue.tied.solve_tied` tries single-link bracketing and pattern enumeration, then
#: walks an active set retiring one contested *pair* per round. That is the right order for a face that
#: is small in pairs. Chicago-Sketch at mu=1 is 1,574 pairs over 58 links, and the ladder searches it in
#: 31 classes: the walk's cost is in the quantity that is large and the ladder's in the quantity that is
#: small. Declared here rather than inferred from a size threshold, because which side of that trade a
#: face falls on is a property of the face and the reader should be able to see which cells were routed
#: this way.
DIRECT_FACE_CELLS = (("Chicago-Sketch", 1.0),)

#: Identical consecutive windows the declared cells wait for before attempting the face. Five: the
#: envelope is what the face is built from, and one repeat is not evidence.
DIRECT_FACE_ENVELOPE_WINDOWS = 5

#: Times a face may be reopened against the cost its own solved flow induces before the attempt is
#: abandoned. A face is a hypothesis about which destination-link pairs are tied; the walk pins some to a
#: bound and the inclusion scheme's base supplies the rest, and neither is rechecked at the solved cost.
#: Where that cost refuses a link the recorded mixture carries, the face was identified wrongly, and
#: neither more iterations on the same face nor a wider tie band repairs it. Three, because each round
#: names pairs the previous one could not have known about and a face that has not settled after three
#: is not converging on one: the honest answer there is to descend a rung, not to keep reopening.
FACE_REOPEN_ROUNDS = 3

#: The cell whose certified face the main text describes by its dimension.
FACE_CELL = ("SiouxFalls", 0.5)

#: The cell to pilot before the certificate array is queued. It is the largest tie set in the study by
#: two orders of magnitude, its face solve is the longest single operation anywhere in the work, and the
#: active-set walk retires at most one pair per round -- so whether the whole array is affordable is
#: decided here and nowhere else.
PILOT_CELL = ("Chicago-Sketch", 0.5)

#: The cells each figure draws. Declared here because they are also cells some exhibit has to run, and a
#: panel whose cell nobody swept is an empty axis in the paper.
OSCILLATION_PANELS = (("Eastern-Massachusetts", 1.0), ("Anaheim", 1.0), ("SiouxFalls", 0.5))
DIAMETER_NETWORKS = ("SiouxFalls", "Eastern-Massachusetts", "Berlin-Mitte-Center", "Chicago-Sketch")

#: The witness's own dispersion grid. It is not a benchmark network and does not belong in the corpus
#: grid, but it is still a declared set rather than a default buried in the script.
WITNESS_DISPERSIONS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0)

#: Cells the certificate tier declares but does not attempt, and why. Declared before the run, so the
#: tier stays a commitment rather than a description of whatever finished.
#:
#: Chicago-Sketch at the lowest dispersion is the widest tied face in the study. Its support correspondence
#: does not settle -- 31 repeats in 2.19 million iterations -- but its tie set is identified all the same:
#: the intersection and union of the supports over a trailing window are invariant across 7,500 iterations,
#: giving 3,700 contested pairs on 128 links, 1,850 tied groups collapsing to 70 classes. Neither branch
#: reaches a verdict on that face. The active-set walk retires at most one pair per round at about 1,036 s
#: a pair; the ladder's blend evaluation is itself hours at this width, so a twelve-hour budget was spent
#: before its first level began. This is a bound on affordable computation at the widest face in the
#: corpus, and neither a failure of the certificate nor of the theory.
CERTIFICATE_EXCLUDED = (("Chicago-Sketch", 0.5),)

#: Cells of each sweep, in the order the job arrays index them.
CORPUS_CELLS = tuple((name,) for name in REPORTED_CORPUS)
CERTIFICATE_CELLS = tuple((name, mu) for name in CERTIFICATE_TIER for mu in DISPERSIONS
                          if (name, mu) not in CERTIFICATE_EXCLUDED)
SUPPORT_COMPARISON_CELLS = tuple((name, mu) for name in SUPPORT_COMPARISON_NETWORKS
                                 for mu in SUPPORT_COMPARISON_DISPERSIONS)
TIMING_CELLS = tuple((name, mu) for name in TIMING_NETWORKS for mu in DISPERSIONS)
FULL_GRAPH_CELLS = (tuple((name, mu) for name in FULL_GRAPH_NETWORKS for mu in DISPERSIONS)
                    + tuple((name, mu) for name in ACCURACY_NETWORKS for mu in ACCURACY_DISPERSIONS
                            if mu not in DISPERSIONS))
#: The solution-set measurement mirrors the certificate tier, so that its counts and the certificate
#: counts are over the same cells and can be stated in one sentence. It therefore drops the same cell.
SOLUTION_SET_CELLS = tuple((name, mu) for name in SOLUTION_SET_NETWORKS for mu in DISPERSIONS
                           if (name, mu) not in CERTIFICATE_EXCLUDED)

#: Networks the study never developed against, swept under frozen rules as a reliability check.
#:
#: Every result in the reported corpus was produced by code that was repeatedly repaired while those
#: cells were being run. Certifying a returned flow establishes that flow; it does not establish that
#: the solver finds and records solutions on a network nobody tuned it against. These four are the
#: remainder of the benchmark collection, they pass the same exclusion criterion the corpus does -- no
#: link carries unit capacity -- and none of them appears in any cell-specific route. Their rows are a
#: reliability measurement and are never mixed into the reported corpus, which is why they have their
#: own sink and are exempted from the reported-corpus check by name.
#:
#: Munich is the interesting small case: 742 nodes of which all 742 are zones, so every node is a
#: centroid and the transit rule has nothing to exclude. The other three are 28,000 to 40,000 links,
#: where Sydney at 75,379 did not settle in twelve hours. A cell that runs out of budget is reported as
#: that, with its runtime, and is not dropped.
HELD_OUT_NETWORKS = ("Munich", "Berlin-Center", "chicago-regional", "Philadelphia")

HELD_OUT_CELLS = tuple((name, mu) for name in HELD_OUT_NETWORKS for mu in DISPERSIONS)

#: Every exhibit, by the name of its result sink, and the cells it is required to produce. The validator
#: reads this; so does ``reproduction/tools/cells.py``, so a job array and a table cannot disagree about
#: what was
#: meant to run.
EXHIBIT_CELLS = {
    "corpus_sweep": tuple((name, mu) for name in REPORTED_CORPUS for mu in DISPERSIONS),
    "certificate_sweep": CERTIFICATE_CELLS,
    "support_comparison": SUPPORT_COMPARISON_CELLS,
    "convergence": TIMING_CELLS,
    "full_graph": FULL_GRAPH_CELLS,
    "solution_set": SOLUTION_SET_CELLS,
    "held_out": HELD_OUT_CELLS,
}

# ---------------------------------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------------------------------

#: Largest exponent argument float64 can carry, as a power of ten. The load ratio is clamped so that
#: ``ratio ** power`` stays representable. This is a property of the arithmetic, not of the model: above
#: the clamp the cost is constant in flow, its derivative vanishes, and strict monotonicity fails.
FLOAT64_EXPONENT_HEADROOM = 250.0

#: Floor on capacity, guarding the division only. No benchmark network has a smaller positive capacity.
MIN_CAPACITY = 1e-9

# ---------------------------------------------------------------------------------------------------
# Efficient support
# ---------------------------------------------------------------------------------------------------

#: Width of the band within which two costs to a destination count as tied. Reduced costs are differences
#: of shortest-path potentials, so their error accumulates with path length; this sits several orders
#: above the float64 noise of such a sum and several orders below the smallest exclusion margin measured
#: across the corpus.
TIE_TOLERANCE = 1e-10

# ---------------------------------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------------------------------

#: Work below which a destination-parallel kernel costs more in thread-pool overhead than it saves,
#: measured as ``n_zones * n_links``. Entering a parallel region costs a fixed barrier of a few
#: milliseconds whatever the problem size, which on a small network exceeds the sweeps themselves.
#: Crossing either threshold changes the running time and never the result.
#:
#: The two kernels have separately measured crossovers and the values are not interchangeable. For the
#: potentials it lies between 2.8e5, where the serial form is faster, and 1.1e6, where the parallel form
#: is faster by a factor of about two and a half. For the loading it is higher: at 3.9e5 the parallel
#: form runs at six tenths of the serial speed and only at 1.1e6 does it reach 1.3 times.
PARALLEL_MIN_WORK_POTENTIALS = 500_000
PARALLEL_MIN_WORK_LOADING = 1_000_000

# ---------------------------------------------------------------------------------------------------
# Simplicial search
# ---------------------------------------------------------------------------------------------------

#: Smallest pivot direction treated as positive in the ratio test. Below it the direction is numerical
#: noise and admitting it would pick a leaving row at random.
PIVOT_TOLERANCE = 1e-11

#: Width within which two ratios count as tied, so that the lexicographic rule decides between them.
#: Degeneracy is the norm in this search rather than the exception, and the rule is what makes the pivot
#: sequence unique; uniqueness is what the door-in-door-out argument requires.
LEX_TOLERANCE = 1e-14

#: Smallest mass a mixture weight must carry to be treated as present.
WEIGHT_EPSILON = 1e-12

#: Width within which a weight counts as sitting on a bound of the unit interval. Wider than
#: ``WEIGHT_EPSILON``, which is a mass, because this is a decision about which branch of the
#: complementarity condition applies and the two must not be conflated.
ACTIVE_TOLERANCE = 1e-9

#: Slack allowed on the implied base weight before a mixture is judged to lie outside the simplex. A raw
#: negative entry of this size is arithmetic, not a signed combination.
SIMPLEX_SLACK = 1e-12

# ---------------------------------------------------------------------------------------------------
# Tie-pattern search
# ---------------------------------------------------------------------------------------------------
#
# Bounds on the enumeration that runs before the joint system. Each is a count and none is a wall clock:
# a search whose branch depends on how fast the machine is makes the certificate depend on it too, and
# the reproduction guarantee is stated over the network, the demand, the dispersion and the settings.

#: Contested links tried alone, and base links flipped while retrying one.
PATTERN_MAX_LINKS = 6
PATTERN_MAX_BASE_FLIPS = 2

#: Full solves the pattern search may spend before handing over to the joint system.
PATTERN_MAX_SOLVES = 60

#: Contested groups above which the pattern search is not attempted at all.
PATTERN_MAX_GROUPS = 12

#: Two-way contested pairs above which their orientations are not enumerated.
PATTERN_MAX_ORIENTATION_EDGES = 8

# ---------------------------------------------------------------------------------------------------
# Pre-flight thresholds
# ---------------------------------------------------------------------------------------------------

#: Share of demand no route can carry, above which a cell is not worth solving: a network losing more
#: than this at free flow loses at least as much at any congested cost, and every total it reaches is
#: short by that much.
UNREACHABLE_LIMIT = 1e-6

#: Share of links whose capacity is at or below the floor, above which the division is being guarded
#: rather than performed.
ZERO_CAPACITY_LIMIT = 0.01

#: Share of links carrying capacity exactly one, above which the file has folded capacity into the BPR
#: coefficient: the load ratio is then the flow itself. This is the corpus exclusion criterion as a
#: number a reader can check on any new network, and it applies only where the exponent is above one --
#: with a unit exponent the cost is linear in flow whatever the capacity convention, which is an ordinary
#: cost function and not a load ratio raised to a power of an unnormalised flow.
UNIT_CAPACITY_LIMIT = 0.5
UNIT_CAPACITY_EXPONENT = 1.0

#: Zones above which the pre-flight does not measure unreachable demand. The measurement needs a
#: cost-to-destination table, which is one shortest-path pass per zone -- the same work the solver's first
#: iteration does, so a sweep pays it either way, but a survey of every file in the collection does not
#: and would spend minutes per network on networks it is not going to sweep. Above this the fields are
#: recorded as not measured, which is what they are.
REACHABILITY_MAX_ZONES = 1000

#: Load ratio below which the representability clamp is reachable in practice. Above the clamp the cost
#: is constant in flow and the strict monotonicity the uniqueness argument needs is gone.
REACHABLE_RATIO = 1e3

# ---------------------------------------------------------------------------------------------------
# Reporting thresholds
# ---------------------------------------------------------------------------------------------------

#: Frozen-support residual below which a combination counts as solved, given that it also conserves
#: demand. The headline count is stated against this and nothing else may define it.
SOLVED_BELOW = 1e-8

#: Reduced cost within which a link counts as lying on a cost-minimal path, when the longest such path is
#: measured. Tighter than the tie band, because this is a property of one cost and not of two.
COST_MINIMAL_TOLERANCE = 1e-9

# ---------------------------------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------------------------------

#: Iterations of the full-graph value recursion. It is a cyclic-graph fixed point with no ordered sweep,
#: so it is iterated rather than solved, and this bounds that iteration.
VALUE_ITERATIONS = 400

# ---------------------------------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------------------------------

#: Identifier for the support-gate revision. Results produced under different revisions are not
#: comparable, and the validator refuses to mix them.
GATE_REVISION = "transit_2026_09_03"

#: Directories whose contents decide whether the working tree differs from the recorded commit. Results
#: are excluded: a run writes its own output, so including them would mark every record as modified.
CODE_PATHS = ("src", "reproduction", "tests")


class DeadlineReached(Exception):
    """Raised where a solve crosses its wall-clock budget, so the caller unwinds and checkpoints.

    Declared here rather than beside any one solver because several of them raise it and the tied face
    would otherwise have to import the two-phase solver to name it.
    """


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one run, recorded with every result it produces.

    Defaults are the values the reported results were produced under. Pass an instance to vary them.
    """

    #: Consecutive support rebuilds that must leave the support unchanged before phase one stops.
    #: This is phase one's real stopping criterion, and it is the one the theory licenses: the job of
    #: phase one is to freeze the support, which it does in finite time, and not to converge the flow,
    #: which the averaging does only at rate ``1 / k``. Requiring more than one rebuild guards against a
    #: support that pauses for one interval and then moves again.
    support_stable_rebuilds: int = 3

    #: Relative convergence threshold on the phase-1 flow update. Reaching it also stops phase one, but
    #: on a congested network the averaging will not get there, so it is a secondary criterion.
    flow_tolerance: float = 1e-9

    #: Relative residual the frozen-support solve is driven to.
    residual_tolerance: float = 1e-10

    #: Fraction of ``residual_tolerance`` the solver is actually asked for. The answer is graded after
    #: the iterate is clipped to the nonnegative orthant, and the clip moves the point by an amount of
    #: the order of the tolerance itself, so a solver stopped exactly at the bar can be graded just over
    #: it. Asking for a tenth puts that perturbation well inside the margin.
    residual_margin: float = 0.1

    #: Exponent of the averaging step ``1 / k ** step_power``. One gives the harmonic step.
    step_power: float = 1.0

    #: Half-width of the potential dead band, as a fraction of the median free-flow cost. A node
    #: potential is updated only when it moves by more than this, which stops near-tied arcs flipping
    #: indefinitely and freezes the support onto one admissible selection at an error of the same order.
    potential_dead_band: float = 0.05

    #: Outer iterations between support rebuilds. The support moves slowly relative to the flow, so
    #: rebuilding less often saves time without approximating.
    support_rebuild_interval: int = 10

    #: Iterations phase one and the inclusion scheme may take, or ``None`` for no limit, which is the
    #: default. A count is the wrong bound: the support-freezing result is qualitative and gives no a
    #: priori iteration count, so any number here is a property of the run and never of the method, and a
    #: run that reaches one has been stopped by an opinion rather than by an answer. What bounds a run is
    #: the wall clock it was given, and what makes that safe is that the state is checkpointed: a cell too
    #: long for one job continues in the next rather than starting again.
    max_phase1_iterations: int | None = None

    #: Iterations of the frozen-support solve, or ``None`` for no limit. Same rule.
    max_newton_iterations: int | None = None

    #: Pivots the guaranteed branch may take within one level, or ``None`` for no limit. The branch
    #: terminates finitely by a combinatorial argument with no polynomial bound, so a fixed count decides
    #: nothing about the method and only about how early the run gives up.
    max_pivots: int | None = None

    #: Pivot ceilings the face ladder tries in turn before descending to the next rung. A single entry of
    #: ``None`` means each rung runs until it converges, reaches the mesh floor, or the wall clock stops
    #: it -- there is then nothing to retry at a larger ceiling, because there was no ceiling.
    ladder_budgets: tuple = (None,)

    #: Mesh halvings the simplicial search may perform, and the starting grid. The error at each level is
    #: one mesh, so more levels is more precision and not merely more time. Fifty halvings of a grid of
    #: four take the mesh below float64's resolution, so the mesh floor stops the search before this does
    #: and it is a limit in name only.
    simplicial_levels: int = 50
    simplicial_grid: int = 4

    #: Times a collapsed class may be split when its members' gaps disagree at the returned point, or
    #: ``None`` to let it run to the bound the argument already gives it: each split strictly increases
    #: the class count, which cannot exceed the number of tied groups, so the loop ends on its own.
    max_class_refinements: int | None = None

    #: Target for the two approximation clauses of the guaranteed branch.
    pivot_epsilon: float = 1e-6

    #: Perturbation separating the two tie-break selections used to bound the solution set. Applied to
    #: the cost that defines the support and not to the cost flow is loaded at, so it selects among
    #: supports at a tie without moving the equilibrium.
    tiebreak_scale: float = 0.05

    #: Length of the trailing window over which the inclusion scheme collects distinct supports. A tie
    #: shows as two supports recurring within one window, so this is part of the definition of "the
    #: support has not settled" and not a budget. It must be long enough that a support alternating
    #: slowly is still seen twice.
    inclusion_window: int = 300

    #: Consecutive complete windows whose envelope must be identical before the inclusion scheme may stop
    #: on the envelope rather than on support recurrence. ``None`` leaves the rule off, which is the
    #: default: it is an empirical condition and not a theorem, and it should apply only where a cell has
    #: been declared to need it.
    #:
    #: Five, because the envelope is what the face solver consumes and one repeat is not evidence. This
    #: stops the scheme on what the face is built from rather than on an identity the face never reads:
    #: Chicago-Sketch at mu=1 produced 441,722 distinct supports in a million iterations while its
    #: intersection and union did not move at all over eight thousand. Reaching it is a statement about
    #: the windows observed and never a claim that the envelope stays fixed.
    envelope_stable_windows: int | None = None

    #: Relative conservation error above which a flow is judged not to carry its demand, as a fraction of
    #: total demand. Well above the accumulation of ``n_links`` additions in float64 and well below the
    #: smallest genuine shortfall observed, which was six parts in ten thousand.
    conservation_tolerance: float = 1e-6

    #: Resident memory above which a long solve writes its state and stops, in mebibytes, or ``None``
    #: for no ceiling. A wall-clock budget does not guard memory, and memory is the failure this work has
    #: actually had: a loop that accumulates without bound looks exactly like a loop that is working until
    #: the machine kills it, and a killed process writes nothing. Crossing this is treated the same way as
    #: crossing a deadline -- state is written and the run stops resumably -- so a cell too large for one
    #: machine says so instead of being lost.
    memory_ceiling_mb: float | None = None

    #: Seconds between checkpoint writes inside any loop that can run long. The state is overwritten in
    #: place, so a cell keeps one checkpoint however long it runs, and the write is atomic, so a job
    #: killed during one leaves the previous state intact. This is what lets a run be killed by the
    #: scheduler rather than stopping itself: whatever it was doing, the last few minutes of it are on
    #: disk and the next job continues from there.
    #:
    #: Expressed in seconds rather than iterations because the state is not the same size on every
    #: network. Sydney's phase-one state is most of a gigabyte, so a fixed iteration count would write it
    #: every few seconds and spend the run on I/O; a time interval costs the same fraction of the run
    #: whatever the cell.
    checkpoint_seconds: float = 600.0

    #: Seconds between progress lines while a cell is solving. A large cell runs for hours between one
    #: printed result and the next, and silence is indistinguishable from a hang.
    progress_seconds: float = 120.0

    #: Seed for every randomised search in the package.
    seed: int = 0

    def as_record(self) -> dict:
        """The settings as a plain dictionary, for stamping onto a result record."""
        return asdict(self)


DEFAULTS = Settings()
