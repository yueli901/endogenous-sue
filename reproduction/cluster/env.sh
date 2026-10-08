# The one place the environment is named. Sourced by every job script.
# Set HPCWORK before submitting, or in your shell profile. It is the only site-specific path.
HPCWORK=${HPCWORK:?set HPCWORK to the scratch directory holding the data and checkpoints}
export HPCWORK

# sbatch reads SBATCH_ACCOUNT from the submitting environment. It is needed when the job is
# submitted, not when it runs, so export it in your shell before calling sbatch.
: "${SBATCH_ACCOUNT:?export SBATCH_ACCOUNT to the project account the job should charge}"
export ENDOGENOUS_SUE_DATA="$HPCWORK/data/TransportationNetworks"

# Either a conda environment or a virtualenv beside the checkout, whichever the site has. Some sites have
# neither conda nor a Python new enough in the base image, so the module is loaded first and the
# virtualenv is built against it; the module must stay loaded because the venv links its interpreter.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate myenv
else
    module load "${ENDOGENOUS_SUE_PYTHON_MODULE:-python/3.9.12/gcc/pdcqf4o5}"
    source "$HPCWORK/venv/bin/activate"
fi

mkdir -p logs "$HPCWORK/checkpoints" results/reproduce
