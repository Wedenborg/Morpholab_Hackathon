#!/usr/bin/env bash
#SBATCH --account=morpheus
#SBATCH --job-name=embryo_datasets
#SBATCH --partition=burst
#SBATCH --requeue
#SBATCH --exclude=ins093,ins094
#SBATCH --output=starting-points-%j.out
#SBATCH --error=starting-points-%j.err
#SBATCH --open-mode=append
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --chdir=/insomnia001/depts/morpheus/users/mp4140/Morpholab_Hackathon

# Run locally: bash dataset/data/download_starting_points.sh
# Submit: sbatch dataset/data/download_starting_points.sh
# Preview without network or data writes: ... --dry-run
# All 11 resources and all available stages are selected by default.
# Options are forwarded to dataset/download_starting_points.py; see --help.
# Override DATA_DIR, REPO_ROOT, CONDA_ENV, CONDA_SH, or PYTHON_BIN if needed.
# Slurm requeues preempted jobs; persistent .part files and atomic outputs let
# the downloader reuse completed work when the batch script starts again.
set -euo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
    else
        REPO_ROOT=/insomnia001/depts/morpheus/users/mp4140/Morpholab_Hackathon
    fi
fi
cd -- "$REPO_ROOT"

# Keep --help and --dry-run usable with any Python, without activating Conda.
preview=false
for argument in "$@"; do
    case "$argument" in --help|-h|--dry-run) preview=true ;; esac
done
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ "$preview" == false ]]; then
        CONDA_SH="${CONDA_SH:-$REPO_ROOT/../miniforge/etc/profile.d/conda.sh}"
        if [[ ! -f "$CONDA_SH" ]]; then
            echo "Cannot find $CONDA_SH; set CONDA_SH or PYTHON_BIN." >&2
            exit 1
        fi
        source "$CONDA_SH"
        conda activate "${CONDA_ENV:-hackathon}"
    fi
    PYTHON_BIN=python
fi
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Node: $(hostname) | Job: ${SLURM_JOB_ID:-local} | Restart: ${SLURM_RESTART_COUNT:-0} | Python: $PYTHON_BIN"
exec "$PYTHON_BIN" -u -m dataset.download_starting_points \
    --data-dir "${DATA_DIR:-/insomnia001/depts/morpheus/projects/hackathon/Datasets}" "$@"
