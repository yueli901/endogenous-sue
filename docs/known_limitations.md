# Known limitations

Written so that a reader does not have to discover these by running into them.

## In the method

**The certificate is a posteriori, and that is the point.** What is certified is not that the solver did
the right thing but that the flow it returned satisfies the definition. That is why the check is run
against the support the flow's own cost induces, rather than against the support the solver believed it
was using. The two differ by orders of magnitude after a single pass, and a test asserts that they do.

**A small residual is not a certificate.** The two-phase solver reports its residual against the support
it froze. That number can reach machine precision on a flow that loses several per cent of its demand, on
a network where every arc carrying flow is still efficient. Conservation is the only clause that sees it.

**Phase 1 has no a priori iteration bound, so its budget is an instance parameter.** Theorem 7 gives
finite-time freezing of the support whenever the equilibrium is off the tie boundary; it gives no bound on
how many iterations that takes, and none is claimed. The budget therefore has to be chosen per instance,
and a cell that stops on the budget rather than on the tolerance has not converged. Measured on the corpus:
at 400 iterations Terrassa-Asymmetric (mu=5, mu=10), Hessen-Asymmetric and Austin stop on the budget and
lose between 0.06 and 0.13 per cent of demand; at 1600 all of the first three conserve, Terrassa to 2.7e-04
and Hessen to 4.6e-04. Terrassa at mu=5 is also *faster* at 1600 than at 400 -- 9.4 s against 12.5 s --
because the Newton polish thrashes on an unconverged warm start, so the short budget bought a worse answer
at a higher price. Every row records `phase1_iter` and `phase1_stopped_on` for this reason, and a
budget-stopped row is not admissible evidence for a converged result.

**The guaranteed branch does not scale.** The unconditional rung searches in as many dimensions as there
are links, and no polynomial bound on the pivot count is claimed or exists — the problem is PPAD-complete.
It is what the guarantee descends to, not the method of choice, and it is not expected to run on a city
network. The accelerators above it inherit the guarantee; they never replace it.

**The collapse is applied a posteriori.** The strong form of the reduction — that a collapsed class holds
a constant gap across the whole face — is false, and was refuted by measurement. What is used instead is:
collapse, solve the reduced face, lift, then check the *full* gaps at the returned point and split any
class whose gaps disagree. Correctness never depends on the collapse being globally valid.

**Coverage must be checked at the result-file boundary.** Smoke runs and interrupted sweeps can contain
only part of the manuscript grid. `reproduction/tools/validate_results.py` reports partial coverage rather than
letting it read as complete.

## In the stored results

These are properties of the deposited files, not of the code. Each is reported by
`reproduction/tools/validate_results.py` or by the table generators; none is silently carried into a number.

**The tracked results are currently a smoke artifact, not the manuscript sweep.** The repository contains
the witness enumeration, two one-cell support comparisons, two one-cell corpus rows, and one one-cell
certificate row. It does not yet contain the canonical `full_graph`, `convergence`, or `solution_set`
sinks. The validator reports those absences, and the figure builder exits non-zero when asked to draw a
manuscript figure whose sink is missing.

**Old rows written before the flow digest field cannot authenticate their arrays.** Certificate rows
written from now on carry `flow_file`, `flow_bytes` and `flow_sha256`. Rows already in the smoke artifact
name the flow array but have no digest, so the validator reports that limitation until they are rerun.

## The certified flows are sidecar artifacts

`results/reproduce/flows/` holds the flow arrays written by `reproduction/exhibits/certificate_sweep.py`. The arrays
can be large, so a release may carry only `results/reproduce/flows.manifest.json`, written by
`reproduction/tools/build_flow_manifest.py`, when the arrays themselves are omitted. The manifest preserves the
digest link between a result row and the bytes it was computed from.
