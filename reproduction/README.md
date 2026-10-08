# Reproduction

Everything needed to regenerate the paper's numbers from the code in `src/`. Three directories, by what
they do rather than by who runs them.

| directory | what is in it |
|---|---|
| [`exhibits/`](exhibits/) | one script per exhibit in the paper, named for what it produces. Each writes a sink under `results/` and nothing else |
| [`tools/`](tools/) | fetch the benchmark data, validate the stored results, build the LaTeX tables and the figures, record provenance |
| [`cluster/`](cluster/) | the Slurm scripts the reported long sweeps were dispatched with. A record of how they were run, not a portable harness |

## The short path

```sh
python -m pip install -e ".[fast,figures,dev]"
python reproduction/tools/fetch_networks.py    # benchmark networks, at the pinned commit
make reproduce                                 # the exhibits that finish in minutes
python reproduction/tools/validate_results.py --strict
python reproduction/tools/build_tables.py
```

`make reproduce` writes `results/smoke/`. That is a demonstration that the pipeline runs, not a result.
The reported numbers are in `results/reproduce/` and `results/held_out/`, which the long sweeps write.

## What costs what

| | wall time |
|---|---|
| the witness enumeration | seconds |
| one network at one dispersion, two-phase solve | seconds to minutes |
| the corpus sweep, all networks and dispersions | hours |
| the certificate sweep with the face ladder | hours to days on the largest cells |
| the held-out sweep | one 11.5-hour budget per cell, twenty cells; most exhaust it |

The corpus and certificate sweeps resume: re-running skips cells already recorded, keyed on the network
*and* the dispersion. The held-out sweep deliberately does not — see `docs/held_out_protocol.md`.

## Two rules that have already been broken once each

**Validate before you build.** `tools/validate_results.py` refuses a file that mixes gate revisions or
solver settings, reports partial coverage rather than letting it read as complete, and names any record
that stopped on an iteration limit instead of converging. Every failure it looks for has happened.

**`tools/splice_tables.py` needs `--only`.** Without it the tool rewrites every table whose label it
recognises, including ones whose typeset form was adjusted by hand.
