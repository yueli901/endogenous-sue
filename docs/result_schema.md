# What is in a result row

One JSON object per line. Fields divide into three groups: what identifies the cell, what was measured,
and what identifies the code that measured it.

## Provenance, on every row

| field | meaning |
|---|---|
| `version` | package version |
| `commit` | commit the run was made from, with `-dirty` when the working tree differed from it |
| `gate_revision` | centroid-gate revision (see below) |
| `settings` | the full resolved solver settings dataclass |
| `platform`, `python`, `timestamp` | operating system, architecture, and when. The hostname is deliberately not recorded |
| `run_spec` | the fully resolved arguments of the run |

The gate revision is not bookkeeping. It names a modelling rule that changed results across the whole
corpus while leaving the record format identical, so a row that carries no revision cannot be compared
with one that does.

**`gate_revision`** identifies the release in which the three centroid questions were separated. Before it,
one flag answered all three, and on networks with centroid connectors that lost demand — in the worst
case most of it — while every arc still carrying flow remained efficient, so no residual and no
certificate saw anything wrong. Networks with connectors give different flows either side of this.

Rows that write or read a stored flow also carry `flow_file`, `flow_bytes` and `flow_sha256`, so a row
computed against a version of a file that has since been overwritten is detectable rather than merely
suspect.

## Certificate fields

The certificate clauses are recorded separately because they fail separately.

| field | meaning |
|---|---|
| `certified` | boolean outcome of the a posteriori certificate |
| `kind` | `strict`, `relaxed`, `not strict`, or `not certified` |
| `reason` | empty on success; otherwise the failed certificate clause(s) |
| `n_contested` | number of contested destination-link pairs returned by the inclusion phase |
| `n_tied_pairs`, `n_tied_links`, `n_excluded` | measured tie and exclusion inventory at the returned cost |
| `exclusion_margin` | smallest gap over excluded pairs, or null when none is defined |
| `smallest_tie_cost` | smallest cost of a tied link, or null when the returned cost is tie-free |
| `violations`, `worst_gap` | support-inclusion failures and their largest gap |
| `mixture_residual` | distance between the returned flow and the certified mixture loading |
| `conservation_max`, `conservation_relative` | node-balance error in absolute and relative form |
| `seconds`, `cores` | the cell's own wall time, and the cores the machine had |
| `peak_rss_mb`, `peak_rss_added_mb` | the process high-water memory when the cell finished, and how much of it this cell added |
| `tied` | diagnostics from the tied-face solver, including route, weights and residuals where present |

A status string is not a certificate. The count reported in the paper is the number of rows whose
`certified` field is true, over the certificate-tier cells declared as `CERTIFICATE_TIER` in
`endogenous_sue.config` before the sweep runs. Where the certificate row is missing, the cell is not
counted: an unevaluated clause is not a passed one. A cell from a network outside the tier is a probe and
is excluded from the tables, because a tier read back from whichever cells finished would describe the run
rather than commit to anything.

## Absence is recorded as absence

Several diagnostics are unavailable on the parallel loading path, which keeps no per-destination cache to
compare against. Those rows carry `null`, and `support_measured` says so. Recording
zero support changes there would read downstream as a support that froze at the first iteration, which is
the opposite of not having looked. The table generators print `n.r.` for such rows.

`tab:timing`'s "fixed at" column is the case to watch. It quotes an iteration only where phase one stopped
on a criterion that establishes the support is final -- `support settled` or `exactness bound`. A run that
stopped on the iteration limit reads "not fixed"; one whose stopping criterion is absent from the record
reads `n.r.`, which is neither of those and must not be printed as a number.
