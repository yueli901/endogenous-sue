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
#SBATCH -J es-exhibit
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
set -euo pipefail

# One cell per task, for any exhibit that declares its cells. The exhibit is named by the environment and
# the array index resolves against the declaration, so nothing here repeats a network name or a grid.
#
#   EXHIBIT=solution_set sbatch --array=0-38%13 -J es-solution reproduction/cluster/exhibit.slurm
#
# The array range is passed on the command line rather than written here, because one script serves
# several exhibits of different sizes; the assertion below still refuses an index outside the declaration.
# Every long loop overwrites its checkpoint every ten minutes, so the wall may kill a task and the next
# submission continues from where it was.

cd "${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" || {
    echo "cannot enter ${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" >&2; exit 1; }

EXHIBIT=${EXHIBIT:?set EXHIBIT to the sweep this array runs}

echo "== inputs =="
echo "  cwd $(pwd)  exhibit $EXHIBIT"
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

declared=$(python reproduction/tools/cells.py "$EXHIBIT" --count)
read -r NETWORK MU <<< "$(python reproduction/tools/cells.py "$EXHIBIT" "$SLURM_ARRAY_TASK_ID")"
echo "host $(hostname)  cores $SLURM_CPUS_PER_TASK  cell $NETWORK mu=$MU"
echo "  ($((SLURM_ARRAY_TASK_ID + 1)) of $declared declared cells)"
echo "started $(date -u +%FT%TZ)"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

CELL_SECONDS=${CELL_SECONDS:-}
set +e
python -u "reproduction/exhibits/${EXHIBIT}.py" \
    --networks "$NETWORK" --dispersions "$MU" \
    --out "results/reproduce/${EXHIBIT}/${NETWORK}_mu${MU}.jsonl" \
    --checkpoint-dir "$HPCWORK/checkpoints" \
    ${CELL_SECONDS:+--cell-seconds "$CELL_SECONDS"}
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"

sleep 20
echo "== resources used =="
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,Partition%10,AllocCPUS%9,ReqMem%9,MaxRSS%10,MaxVMSize%10,Elapsed%12,State%15
exit $status
