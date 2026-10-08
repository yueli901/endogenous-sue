#!/bin/bash
# The account to charge is not written here: #SBATCH directives are read before any shell runs,
# so a variable would never expand. Export SBATCH_ACCOUNT before submitting -- sbatch reads it
# natively -- or pass `sbatch -A <account> ...` on the command line.
#SBATCH -p icelake
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -J es-envelope
#SBATCH -o logs/envelope_%x_%j.out
#SBATCH -e logs/envelope_%x_%j.err
set -euo pipefail

# Whether a cell whose support never settles nonetheless has a fixed envelope, which is what the face
# solver actually consumes. Bounded by an iteration count and not by the wall: the point of this
# diagnostic is that it does not wait for the strict-support criterion, which on these cells never fires.
#
#   NETWORK=<network> MU=<mu> ITERATIONS=8000 \
#     CHECKPOINT=checkpoints/<exhibit>_<network>_mu<mu>.npz \
#     sbatch -J es-env-<network> reproduction/cluster/envelope.slurm
#
# CHECKPOINT is optional and must be phase 3 if given; the script refuses a phase-1 flow, whose dead band
# has already frozen the support onto one selection and so reports ties that are not there to be found.

cd "${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" || {
    echo "cannot enter ${ENDOGENOUS_SUE_ROOT:-$SLURM_SUBMIT_DIR}" >&2; exit 1; }

NETWORK=${NETWORK:?set NETWORK to the network this diagnostic runs}
MU=${MU:?set MU to the dispersion}
ITERATIONS=${ITERATIONS:-4000}
INTERVAL=${INTERVAL:-500}
CHECKPOINT=${CHECKPOINT:-}
BUDGET=${BUDGET:-}

echo "== inputs =="
echo "  cwd $(pwd)  cell $NETWORK mu=$MU  iterations $ITERATIONS  interval $INTERVAL"
for required in reproduction/cluster/env.sh scripts src; do
    [ -e "$required" ] || { echo "  MISSING $required" >&2; exit 1; }
    echo "  ok $required"
done
source reproduction/cluster/env.sh
[ -d "$ENDOGENOUS_SUE_DATA" ] || { echo "  MISSING data at $ENDOGENOUS_SUE_DATA" >&2; exit 1; }
[ -d "$ENDOGENOUS_SUE_DATA/$NETWORK" ] || { echo "  MISSING network $NETWORK" >&2; exit 1; }
python -c "import numpy, scipy; print('  ok python', numpy.__version__, scipy.__version__)" || exit 1
python -c "import numba; print('  ok numba', numba.__version__)" \
    || echo "  WARNING: no numba, the compiled kernels degrade to Python"
if [ -n "$CHECKPOINT" ]; then
    [ -f "$CHECKPOINT" ] || { echo "  MISSING checkpoint $CHECKPOINT" >&2; exit 1; }
    echo "  ok checkpoint $CHECKPOINT"
fi

echo "host $(hostname)  cores $SLURM_CPUS_PER_TASK"
echo "started $(date -u +%FT%TZ)"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

set +e
python -u reproduction/tools/envelope_diagnostic.py \
    --network "$NETWORK" --mu "$MU" \
    --iterations "$ITERATIONS" --interval "$INTERVAL" \
    ${CHECKPOINT:+--checkpoint "$CHECKPOINT"} \
    ${BUDGET:+--time-budget "$BUDGET"} \
    --out "results/_envelope/${NETWORK}_mu${MU}.json"
status=$?
echo "finished $(date -u +%FT%TZ)  elapsed ${SECONDS}s  exit $status"

sleep 20
echo "== resources used =="
sacct -j "$SLURM_JOB_ID" --units=G \
    --format=JobID%20,AllocCPUS%9,ReqMem%9,MaxRSS%10,Elapsed%12,State%15
exit $status
