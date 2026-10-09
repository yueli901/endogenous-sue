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

**The two Phase-1 stopping guarantees are conditional, and on different things.** Support
identification stops on one of two criteria, and neither is an unconditional convergence guarantee.
*Support recurrence*: the support is unchanged across `support_window` consecutive rebuilds. This is a
stability observation, not a proof that it will not move again. *The exactness bound*: when the smallest
potential gap over every still-eligible link exceeds what the remaining cost movement could close, no
efficiency gap can change sign and the support is final from that point. `equilibrium.py` records which
of the two fired in `stopped_on`. The finite-time freezing result behind the first applies when the
equilibrium sits off the tie boundary by a margin large enough relative to the averaging band; the paper
states the condition. A run that exhausts its budget before either criterion fires is recorded as
unconverged and is not a result.

## In the stored results

These are properties of the deposited files, not of the code. Each is reported by
`reproduction/tools/validate_results.py` or by the table generators; none is silently carried into a number.

**The deposited results are the reported sweep.** `results/reproduce/` carries the corpus sweep (75
records), the certificate sweep (39), the full-graph comparison (77), the solution-set sweep (39), the
support comparison (24), the timing and convergence subset (45) and the witness enumeration (150);
`results/held_out/` carries the 19 recorded cells of the prospective run. The validator checks each sink
against the coverage `config.py` declares, so a partial sweep is reported as partial rather than read as
complete.

**The `commit` field is null in every deposited record.** `provenance.commit()` reads the revision
with `git rev-parse` and returns `None` rather than guessing when that fails. The reported runs executed
on a cluster tree deployed by file copy with no `.git` directory, so the field is recorded as null
throughout. Every other provenance field — package version, platform, timestamp, resolved run arguments,
gate revision and the full solver settings — is present on all 468 records, and all 39 certificate rows
carry `flow_file`, `flow_bytes` and `flow_sha256`.

## The certified flows are sidecar artifacts

`results/reproduce/flows/` holds the flow arrays written by `reproduction/exhibits/certificate_sweep.py`. The arrays
can be large, so a release may carry only `results/reproduce/flows.manifest.json`, written by
`reproduction/tools/build_flow_manifest.py`, when the arrays themselves are omitted. The manifest preserves the
digest link between a result row and the bytes it was computed from.
