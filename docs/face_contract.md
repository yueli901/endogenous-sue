# The contract for a solved face

A tied face is solved by one of five routes, three of them recursive. What leaves the solver is a
record. Everything downstream -- the independent certificate, the sweep's mixture reconstruction, the
tables -- sees the record and never the solver's internal state.

Most of the defects found in this solver were not wrong arithmetic. They were two sides of this
boundary disagreeing about what a field meant: which base a support was stated against, whether the
base mass was listed among the vertices or held back, which destinations the mixture named. Each was
repaired on one side while the other still disagreed, and each repair passed the cells it was written
against. This document states the contract once, so that a change to either side has something to be
checked against rather than a set of cells that happened to pass.

Field names are those of the dictionary `solve_tied` returns.

## What a record must contain

| field | type | meaning |
|---|---|---|
| `support_base` | `{destination: [link, ...]}` | the base support the winning solve loaded, as **absolute** link indices |
| `vertex_mixture` | `[[destination, [link, ...], weight], ...]` | one support per entry, stated as `base ∪ added`, with its weight |
| `base_weight` | `{destination: weight}` | mass the solve attributes to the bare base; **diagnostic only**, see C4 |
| `weights` | `[[destination, link, value], ...]` | the marginal of each contested pair |
| `inadmissible_pairs` | `[[destination, link, gap], ...]` | pairs the solved flow's own cost refuses |
| `verdict` | `"CERTIFIED"` / `"NOT CERTIFIED"` | the solver's own opinion, which decides nothing |

## The clauses

**C1 -- Completeness.** Every destination carrying demand appears in the reconstruction with total
weight 1. A destination the route pinned but never split carries its whole flow on its base and appears
in no `vertex_mixture` entry at all; it is still a destination the mixture must account for. Four of
Berlin-Tiergarten mu=1's eight contested destinations were exactly that.

**C2 -- Absolute base.** `support_base[d]` is the absorbing closure of the base mask that the solve
which produced the returned flow actually loaded, in absolute link indices. Not a delta against
anything.

There is one exception and a consumer must handle it. `direct_face` enters the ladder without the pair
walk, so no base was ever mutated and there is nothing absolute to state: it omits `support_base`
entirely and records each entry's `added` against the caller's own raw `base_masks[d]`. A consumer
therefore falls back to `base_masks[d]` for any destination the record does not list, which is what
`reproduction/exhibits/certificate_sweep` does. Reading `support_base` alone and treating an absent entry as an
empty base yields a mixture of nearly empty supports: on Anaheim that produced a reconstruction residual
of exactly 1.000 and 67,479 spurious admissibility violations against a flow that was fine.

The accelerating routes, by contrast, search against a mutated copy of the base and pass that copy down,
so a delta recorded by an outer search is relative to a base its winning solve never saw, and a delta
recorded by an inner one is relative to a base the caller does not hold. Absolute composes through any
depth of recursion; that is the only property that makes recursion safe here.

**C3 -- Reconstruction.** Loading each listed support at its listed weight and summing reproduces the
returned flow:

    sum over d, k of  w[d,k] * load(d, support_base[d] union added[d,k])  ==  x

to within `residual_tolerance`, relative to `max |x|`. This is the clause that makes the record
self-contained. A consumer that cannot rebuild the flow does not have the mixture the solver found.

**C4 -- The deficit rule.** The mass a destination's listed vertices do not carry sits on its bare base.
The consumer computes it as `1 - sum of the listed weights` and **must not** add `base_weight` on top.
The ladder route lists its base vertices among the others and defines `base_weight` as the sum of
exactly those, so adding it again loads them twice. Anaheim at mu=0.5 reported an internal
representation residual of 0.34880746876681196 against a recorded `base_weight` of 0.3488074687668118 --
the residual was the double-counted mass itself, to every digit.

**C5 -- Admissibility at the solved cost.** Every support carrying weight is admissible under the
potential that the returned flow's own cost induces, within `tie_tolerance`. This is not the same as the
face having been solved to tolerance: a face is a hypothesis about which pairs are tied, and solving it
accurately says nothing about whether the hypothesis was right. See "Reopening" below.

**C6 -- Marginal agreement.** For each contested pair, `weights[(d, link)]` equals the total weight of
the listed supports for `d` whose mask contains `link`. The pair marginals and the mixture are two
views of one object and are checked against each other, not derived one from the other.

**C7 -- Acyclicity.** Every support carrying weight is acyclic, so `kahn_order` succeeds on it. A cyclic
support cannot be loaded and a mixture containing one is not a mixture over admissible supports.

**C8 -- Recursion.** A record returned through a nested call is the record of the solve that produced
the flow, with `route` describing how it was reached. No field is restated by the caller. C2 is what
makes this hold without the caller needing to know how deep the recursion went.

## Reopening

C5 fails in a way the other clauses cannot express: the arithmetic is right and the face is wrong. Two
sources feed a face and neither was rechecked once it was solved. The outer walk pins contested pairs to
a bound. The inclusion scheme's frozen support supplies the base for every pair it never contested. At
the flow the face solve returns, the cost may exclude a link that one of those two put in.

Anaheim at mu=0.5 had both at once, which is why it needed one rule and not two fixes:

| pair | in the tied set | in the base | gap at the solved cost |
|---|---|---|---|
| link 572, destinations 4, 17, 18, 19 | no | yes | -4.886e-04 |
| link 454, destination 29 | yes | yes | -1.601e-03 |

`solve_face_admissible` reopens them. Which repair applies is read off the gap against the tie tolerance
the model already declares, so nothing here is a tuned choice:

| gap | what it means | repair |
|---|---|---|
| below `-tolerance` | the support carries a link the cost strictly excludes | out of the base **and** out of the tie set |
| above `+tolerance` | the link is strictly efficient and the support omits it | into the base |
| within tolerance | genuinely tied | make it a contested pair and let the face decide |

Each attempt searches through its own checkpoint, named for a hash of the pairs it reopened rather than
for the attempt number. `solve_tied` restores a stored walk without checking that its tie set is the one
being asked for, so a file named by attempt alone would be loaded by any first reopen of that cell,
including one from an earlier submission that reopened a different set.

After `FACE_REOPEN_ROUNDS` the pairs are reported rather than repaired, and the caller descends a rung.

### What reopening cannot repair

Anaheim at mu=0.5 is the case that shows the limit, and it is worth stating because the failure looks
like a bug and is not one. Reopening those five pairs as ties left `(29, 454)` violating: the outer walk
pinned its reopened weight to **1.0**, and a tie at weight one is in every support for that destination,
so it folds back into the base and the reopen changes nothing. Forcing the same pairs out instead gives:

| pair | tied at weight 1.0 | forced out of the base |
|---|---|---|
| link 572, destinations 4, 17, 18, 19 | -4.886e-04 | **+7.417e-03** |
| link 454, destination 29 | -6.910e-04 | **+1.262e-02** |

The gaps flip sign. Put the links in and the flow's own cost excludes them; take them out and it
includes them. Neither pure selection is self-consistent, which is this paper's own phenomenon at the
scale of one link: the support correspondence has no fixed point there and the answer has to be a
mixture at an *interior* weight.

No reopening policy can supply that, because reopening chooses a side. A pair whose correct weight is
strictly between the bounds has to reach a face solve that can carry it, and the outer walk retires
pairs to bounds before the ladder searches. That is a limit of the walk, not of the face.

## What the record does not establish

`verdict` is the solver's own opinion and decides nothing. A row is certified when
`certificate.certify` -- which takes the network, the demand, the dispersion, the flow and the rebuilt
mixture, and nothing from the solver -- says so. The solver's own admissibility count and the
independent one agree on Anaheim (eleven incidences of five distinct pairs), but agreement is a thing to
be checked per cell and not a property to be assumed.

Certifying a returned flow establishes that flow. It does not establish that the solver finds and
records solutions reliably on a network it was not developed against. That is a separate question and
the held-out run is what addresses it.
