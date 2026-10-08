# Cluster scripts

The Slurm scripts the reported long sweeps were dispatched with. They are here as a record of how the
results were produced, not as a portable harness: they encode one scheduler's partitions, one site's
module names, and a twelve-hour QoS wall.

Nothing in the package needs them. Every exhibit in `../exhibits/` runs standalone, and on a machine with
enough time and memory the sweeps can be run without a scheduler at all.

## What is site-specific

| thing | how to set it |
|---|---|
| the account to charge | export `SBATCH_ACCOUNT` before submitting; `sbatch` reads it natively |
| the scratch path | export `HPCWORK`; `env.sh` derives the data and checkpoint paths from it |
| the Python | `env.sh` prefers a conda environment and falls back to a module plus a virtualenv |
| the partition and wall | `#SBATCH -p` and `-t` in each script, currently a twelve-hour icelake QoS |

The account is deliberately not written into a `#SBATCH` line. Those directives are read before any shell
runs, so a variable there would never expand, and a hardcoded one publishes somebody's project code.

## Cells

No job script repeats a network list. `../tools/cells.py` resolves an array index to a cell against the
declarations in `endogenous_sue.config`, so a sweep cannot cover a different set from the one the tables
report. The first job script here did repeat the list, and it ran an excluded network while never running
a reported one.
