# Changelog

Notable changes to the released artifact. The format follows [Keep a Changelog](https://keepachangelog.com/1.1.0/),
and this project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-09-24

First public release, accompanying the submitted manuscript.

### Added
- The method, under `src/endogenous_sue/`: the two-phase solver and its inclusion scheme, the
  set-valued tied branch with its face ladder and guaranteed simplicial core, and a solver-independent
  certificate that tests a returned flow against the definition.
- One script per exhibit under `reproduction/exhibits/`, and the canonical results they produced under
  `results/reproduce/` and `results/held_out/`. Every record carries the package version, the commit, the
  platform, the resolved run arguments, the gate revision and the full solver settings.
- A prospective held-out evaluation on four networks the method was not developed against, under a
  protocol written and frozen before the run (`docs/held_out_protocol.md`).
- `reproduction/tools/validate_results.py`, which refuses to mix result vintages and compares every
  generated macro the manuscript declares against what the table builder emits.
