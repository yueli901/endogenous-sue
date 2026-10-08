#!/bin/bash
# The account to charge is not written here: #SBATCH directives are read before any shell runs,
# so a variable would never expand. Export SBATCH_ACCOUNT before submitting -- sbatch reads it
# natively -- or pass `sbatch -A <account> ...` on the command line.
#SBATCH -p icelake
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -J es-held
#SBATCH -o logs/held_%A_%a.out
#SBATCH -e logs/held_%A_%a.err
#SBATCH --array=0-19%10
set -euo pipefail

# The working directory comes from the environment rather than from a path written into this file, so
# the script is runnable by anyone with a checkout and says nothing about whose account produced the
# reported results.
cd "${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" || {
    echo "cannot enter ${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" >&2; exit 1; }

# Every input this job needs, named and checked, as its first output. Slurm copies the script to a spool
# directory before running it, so "$(dirname "$0")" is that directory and not the checkout -- which is why
# `source "$(dirname "$0")/env.sh"` failed here in two seconds on the first job ever submitted from these
# scripts. Paths are relative to the checkout, which is what the cd above guarantees.
echo "== inputs =="
echo "  cwd $(pwd)"
for required in reproduction/cluster/env.sh reproduce scripts src; do
    [ -e "$required" ] || { echo "  MISSING $required" >&2; exit 1; }
    echo "  ok $required"
done
source reproduction/cluster/env.sh
[ -d "$ENDOGENOUS_SUE_DATA" ] || { echo "  MISSING data at $ENDOGENOUS_SUE_DATA" >&2; exit 1; }
echo "  ok data $(ls "$ENDOGENOUS_SUE_DATA" | wc -l) network directories"
python -c "import numpy, scipy; print('  ok python', numpy.__version__, scipy.__version__)" || exit 1
python -c "import numba; print('  ok numba', numba.__version__)" \
    || echo "  WARNING: no numba, the compiled kernels degrade to Python and every time below is wrong"

# One task per cell. The array index decodes to a cell of the held-out set, declared in
# endogenous_sue.config and not repeated here.
#
# 32 GB, not the 110 GB first requested. The developmental run measured a peak of 3,299 MiB across
# every cell, Philadelphia at 40,003 links and 18.5 M demand reaching 2,875 MiB, so 110 GB was a guess
# roughly thirty times the evidence and it made the array hard for the scheduler to backfill. The
# solver's own limit is `memory_ceiling_mb` in the settings and is unaffected by what is requested here,
# so this changes when the run starts and nothing about what it computes.
#
# One round, and it is the last: --terminal makes a cell stopped by its budget record an unresolved
# result rather than stay absent for a later run to resume. An evaluation whose cells are resubmitted
# until something settles has made its budget a free parameter, so the last round is declared in
# advance. docs/held_out_protocol.md fixes the rest of the rules.
#
# The budget is 11.5 hours against a 12-hour wall, so the sweep stops itself and checkpoints rather
# than being killed mid-write. Without it the first ten tasks of this array ran to TIMEOUT and wrote
# nothing at all: a cell stopped by the scheduler records no row, which leaves it indistinguishable in
# the sink from a cell nobody ran.
#
# This runs reproduction/exhibits/held_out.py, which is reproduction/exhibits/certificate_sweep.py's own main() against a
# different declaration and a different sink. Nothing about the solver, the settings or the stopping
# rules changes: a held-out measurement made with modified code measures the modification. Cells that
# fail or exhaust the wall are results of this run and are reported with their runtime, not dropped.


declared=$(python reproduction/tools/cells.py held_out --count)
if [ "$declared" -ne 20 ]; then
    echo "the held-out set declares $declared cells but the array is sized 20; fix the --array line" >&2
    exit 1
fi
read -r NETWORK MU <<< "$(python reproduction/tools/cells.py held_out "$SLURM_ARRAY_TASK_ID")"

echo "host $(hostname)  cores $SLURM_CPUS_PER_TASK  cell $NETWORK mu=$MU  started $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +e
python reproduction/exhibits/held_out.py \
    --networks "$NETWORK" --dispersions "$MU" \
    --out "results/held_out/${NETWORK}_mu${MU}.jsonl" \
    --checkpoint-dir "$HPCWORK/checkpoints_held_out" \
    --time-budget "${BUDGET:-41400}" \
    --terminal
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"

# What the job actually used, so the next request is evidence and not the last guess repeated. Slurm
# accounting lags the step's exit by a few seconds; the sleep is for that and nothing else.
sleep 20
echo "== resources used =="
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,Partition%10,AllocCPUS%9,ReqMem%9,MaxRSS%10,MaxVMSize%10,Elapsed%12,State%15
exit $status
