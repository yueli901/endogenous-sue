#!/bin/bash
# The account to charge is not written here: #SBATCH directives are read before any shell runs,
# so a variable would never expand. Export SBATCH_ACCOUNT before submitting -- sbatch reads it
# natively -- or pass `sbatch -A <account> ...` on the command line.
#SBATCH -p icelake
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -J es-corpus
#SBATCH -o logs/corpus_%A_%a.out
#SBATCH -e logs/corpus_%A_%a.err
#SBATCH --array=0-14%8
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

# One task per network, all five dispersions. Every long loop overwrites its checkpoint every ten minutes,
# so the wall may kill this task and the next one continues from where it was. Cells already in the shard
# are skipped, so a resubmission never repeats finished work.


# The array size is a literal because Slurm requires one. This asserts it against the declared corpus, so
# a network added to the study cannot be silently left unswept.
declared=$(python reproduction/tools/cells.py corpus --count)
if [ "$declared" -ne 15 ]; then
    echo "the corpus declares $declared cells but the array is sized 15; fix the --array line" >&2
    exit 1
fi
NETWORK=$(python reproduction/tools/cells.py corpus "$SLURM_ARRAY_TASK_ID")

echo "host $(hostname)  cores $SLURM_CPUS_PER_TASK  network $NETWORK  started $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# One shard per task. Concurrent appends to a shared file are atomic only while every record stays under
# the pipe buffer, which is a property of the record rather than something the run can promise; the table
# builder and the validator read the shards together, so nothing has to be concatenated afterwards.
CELL_SECONDS=${CELL_SECONDS:-}
set +e
python reproduction/exhibits/corpus_sweep.py \
    --networks "$NETWORK" \
    --out "results/reproduce/corpus_sweep/${NETWORK}.jsonl" \
    --checkpoint-dir "$HPCWORK/checkpoints" \
    ${CELL_SECONDS:+--cell-seconds "$CELL_SECONDS"}
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"

# What the job actually used, so the next request is evidence and not the last guess repeated. Slurm
# accounting lags the step's exit by a few seconds; the sleep is for that and nothing else.
sleep 20
echo "== resources used =="
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,Partition%10,AllocCPUS%9,ReqMem%9,MaxRSS%10,MaxVMSize%10,Elapsed%12,State%15
exit $status
