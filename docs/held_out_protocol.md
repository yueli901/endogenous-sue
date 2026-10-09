# Held-out evaluation protocol

Written before the run, and fixed once written. Anything decided after seeing a result is not part of
this protocol and must be reported as a separate, post hoc analysis.

> **Note added for the public release, 2026-10-09. The protocol text below is unchanged.** Three of its
> references do not resolve in this repository, and the document is left as written rather than edited,
> because a pre-registration that is revised after the fact is no longer evidence of anything.
> The frozen revision `24ea33e` named in section 1 belonged to a history that was re-initialised before
> release, so that hash is not reachable here. `reproduction/cluster/STATUS.md` and `archive/`, cited in
> sections 1 and 9, are working material and are not part of the released repository. The test suite was
> 109 tests at that revision and is 110 now; the test named in section 1,
> `tests/test_declarations.test_the_held_out_protocol_holds_on_this_revision`, is present and still
> asserts the clauses of sections 2, 4 and 5 against the code. The run's outcomes are in
> `results/held_out/`.

The question is narrow and worth stating plainly, because the certificate tier does not answer it. The
tier establishes that the flows it certified are relaxed equilibria of the model. It was also the corpus
the solver was developed against: defects were found and repaired while those cells were being run, and
each repair passed the cells it was written against. This run asks a different question. **On networks
nobody tuned the method against, under rules fixed in advance, what does it do?**

## 1. Code revision

One revision, named here before the run and not changed during it.

* Frozen revision: recorded in `reproduction/cluster/STATUS.md` under "Frozen for the held-out run", and checked
  against the deployed tree before the run starts. It is kept there rather than in this file because a
  commit cannot name its own hash: writing it here would always name the commit before the one that
  actually ships, which is how this paragraph read on its first attempt.
  `24ea33e` was the candidate the suite passed on, 109 tests. The commits that followed, before any cell
  ran, are a correction to this document, the Slurm memory request in section 3, and the test named
  below. None of them touches the method.
* `tests/test_declarations.test_the_held_out_protocol_holds_on_this_revision` asserts the clauses of
  sections 2, 4 and 5 against the code, so they are checkable and not merely stated.
* No methodological change during the run. If a cell fails, it fails.
* The only admissible reason to break the freeze is a **correctness** defect revealed by a held-out
  failure: a certified row that is not an equilibrium, a contract violation, a wrong number recorded.
  An ordinary unresolved cell is not a correctness defect and is not grounds to touch the solver.
* If the freeze is broken, the run is void and restarts from scratch under the new revision. Results
  from before the break are diagnostics, never evidence. The previous attempt was void for exactly this
  reason: it spanned three revisions.

## 2. Networks and cells

Every network of the benchmark collection outside `CORPUS`, at every dispersion in `DISPERSIONS`.
Declared in `endogenous_sue.config.HELD_OUT_NETWORKS` and nowhere else.

| network | nodes | zones | links | demand |
|---|---|---|---|---|
| Munich | 742 | 742 | 1,872 | 330,053 |
| Berlin-Center | 12,981 | 865 | 28,376 | 168,222 |
| chicago-regional | 12,982 | 1,790 | 39,018 | 1,360,428 |
| Philadelphia | 13,389 | 1,525 | 40,003 | 18,503,872 |

Twenty cells. The denominator is twenty and stays twenty, whatever happens to any cell. A network that
turns out to be unusable is reported as such against that denominator and is not removed from it.

## 3. Budget

* Wall: 12 hours per cell, the hard limit of the `cpu2` QoS.
* The sweep's own deadline: 11.5 hours, passed as `--time-budget`, so it stops itself, writes a final
  checkpoint and records an outcome rather than being killed mid-write by the scheduler.
* Resources: 32 cores, 32 GB per cell, ten cells at a time.
  Amended before any cell of this run had started, and recorded here rather than made
  silently. The first submission asked for 110 GB, which was a guess; the developmental
  run measured a peak of 3,299 MiB across every cell, Philadelphia reaching 2,875 MiB.
  The over-request delayed scheduling and bought nothing. The solver's own limit is
  `memory_ceiling_mb` in the settings and does not follow the Slurm request, so this
  affects when the run starts and nothing about what it computes. No result existed when
  this was changed.
* **Exactly one round.** A cell that has not reached an outcome in its 11.5 hours terminates as
  unresolved. It is not resubmitted, and its checkpoint is not continued. Resubmitting until something
  settles makes the budget a free parameter and the result uninterpretable.

## 4. Stopping rules

Unchanged from the reported corpus, and named here so that "unchanged" is checkable:

* Support settling: `contested_pairs` over the inclusion scheme's trailing window, `support_window`
  iterations wide.
* Envelope stability: **off**. `envelope_stable_windows` is `None` by default and no held-out cell
  declares it. The envelope rule is a declared route for one corpus cell and enabling it here would be a
  network-specific intervention.
* Phase two: `polish` to `residual_tolerance * residual_margin`, graded in the relative 2-norm.
* Tie tolerance `TIE_TOLERANCE`, residual tolerance `residual_tolerance`, conservation tolerance as
  declared. No tolerance is widened for any cell.

## 5. Route selection

No network-specific routing. `DIRECT_FACE_CELLS` names one corpus cell, Chicago-Sketch at mu=1, and no
held-out cell, so the declared direct-face route cannot fire here. It is an affordability decision
taken in advance for a 1,574-pair face, not a correctness exception, and it is not replaceable by the
measured escalation: that fires only after a face has been solved, which is the step the declaration
exists to skip.

The escalation past the pair walk is **not** a network-specific route and does apply: it fires on a
condition the run measures, namely that the face just solved returned a flow still carrying pairs its
own cost refuses. It is part of the method under test, not an intervention on a cell.

## 6. Certificate

`endogenous_sue.certificate.certify`, taking the network, the demand, the dispersion, the flow and the
mixture rebuilt from the record, and nothing from the solver. A row is certified when and only when that
function says so. The solver's own `verdict` decides nothing and is recorded only for comparison.

## 7. Terminal outcomes

Every cell ends in exactly one of four states, and every one of them is a result.

| outcome | meaning | recorded as |
|---|---|---|
| **certified** | the independent certificate passes every clause | `certified: true`, with `kind` strict or relaxed |
| **uncertified** | a flow was produced and the certificate refused it | `certified: false`, with `reason` and the failing clause |
| **unresolved on budget** | the cell reached 11.5 hours without producing a candidate | `unresolved: true`, `certified: null` |
| **invalid dataset** | the benchmark pair cannot define an assignment | `unresolved: true`, `certified: null`, with the refusal text |

`certified` is `null` and never `false` for the last two: the certificate was not run, and recording
`false` would read downstream as a test that was performed and failed.

An unresolved cell records what it reached: iterations, the final move, distinct supports seen, supports
in the window, elapsed seconds, peak resident memory. A cell that stopped is not the same as a cell that
was never tried, and the sink must be able to tell them apart.

**Munich is an invalid dataset, not an algorithm failure and not an exclusion.** Its `_net` file declares
742 zones and its `_trips` file 284, so the two do not describe the same network and no assignment is
defined on the pair. It occupies five of the twenty cells, it is reported under "invalid dataset", and it
is not removed from the denominator. Whether this is a fair benchmark is a question about the collection,
not about the method, and answering it by dropping the network would hide the question.

## 8. What the result licenses

Fixed in advance, so that the reading is not chosen to suit the number.

* **A useful fraction certifies without intervention.** The held-out experiment is reported, most likely
  in the supplement, as evidence that the method transfers beyond its development corpus.
* **Nearly all unresolved on budget.** The implementation is not presented as a broadly fast equilibrium
  solver. The method is framed as an existence and certification result whose practical bottleneck is
  face identification at large network scale, and the held-out run is reported as the measurement that
  establishes the bottleneck.
* **A correctness defect appears.** Submission pauses. The contract that failed is audited before
  anything in the solver is changed, per the order that produced every repair in this project: state the
  contract, test it, then fix the code.

In none of these does the in-sample count become evidence of general reliability. The certificate tier
is 39 of 39 on the corpus the solver was developed against, and that is what it is.

## 9. Reporting

* Sink: `results/held_out/`, one file per cell. Never mixed into `results/reproduce/`.
* All twenty cells appear, whatever their outcome.
* Wall time and peak memory reported per cell, not only for the cells that finished.
* The previous attempt's material is kept under `archive/` as developmental diagnostics and is never
  reported as this run.
