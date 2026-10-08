# Results

Field-by-field meaning of every record is in [`docs/result_schema.md`](../docs/result_schema.md).

This directory is the target location for the canonical sweep produced by the scripts in `reproduction/exhibits/`
against the package in `src/`. Every record carries the gate revision, the resolved solver settings, the
package version, the commit and the platform it was produced on, so a number can always be matched to the
code and the configuration that produced it.

| file | produced by | what it holds |
|---|---|---|
| `reproduction/exhibits/corpus_sweep.jsonl` | `reproduction/exhibits/corpus_sweep.py` | one equilibrium per network and dispersion, with the residual, the conservation clause and the stopping criterion |
| `reproduction/exhibits/certificate_sweep.jsonl` | `reproduction/exhibits/certificate_sweep.py` | the certificate outcome per cell, the tie inventory, and the mixture where the equilibrium is relaxed |
| `reproduction/exhibits/support_comparison.jsonl` | `reproduction/exhibits/support_comparison.py` | the congestion-adaptive support against the free-flow one |
| `reproduction/exhibits/full_graph.jsonl` | `reproduction/exhibits/full_graph.py` | the spectral radius, and the difference from the full-graph model where it is well posed |
| `reproduction/exhibits/convergence.jsonl` | `reproduction/exhibits/convergence.py` | the support-rule contrast and the two-phase timings |
| `reproduction/exhibits/solution_set.jsonl` | `reproduction/exhibits/solution_set.py` | the tie-break diameter and the isolation radius |
| `reproduction/exhibits/witness.jsonl` | `reproduction/exhibits/witness.py` | the exhaustive five-node enumeration |
| `figures/` | `reproduction/tools/build_figures.py` | the paper's figures, as included |
| `tables/` | `reproduction/tools/build_tables.py` | the generated LaTeX table bodies and numeric macros |
| `held_out/` | `reproduction/exhibits/held_out.py` | the prospective run: one file per cell, four networks the method was **not** developed against, under the protocol in `docs/held_out_protocol.md`. Never mixed into `reproduction/exhibits/` |
| `reproduction/exhibits/flows/` | `reproduction/exhibits/certificate_sweep.py` | the certified flow arrays, untracked; digests in `flows.manifest.json` |
| `figures/` | `reproduction/tools/build_figures.py` | the manuscript's figures |
| `provenance/` | `reproduction/tools/record_provenance.py` | software versions, platform, and checksums of every result file |

Run `python reproduction/tools/validate_results.py` before reading any number off these files. It refuses a file
mixing gate revisions or settings, reports partial coverage rather than letting it read as complete, names
any record that stopped on an iteration limit rather than converging, and checks every generated macro the
manuscript uses against what the table builder emits.

Results produced before 2026-09-04 are in `archive/`, which is not part of the release. They are not
comparable with these: they predate a fix to the transit rule that affected eight of the seventeen corpus
networks. The archive's README says which and why.
