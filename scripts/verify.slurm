#!/bin/bash
#SBATCH --job-name=morpholab-verify
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=verification-%j.out
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/morpholab-mpl-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/morpholab-numba-${SLURM_JOB_ID}"
PYTHON="${MORPHO_PYTHON:-.venv311/bin/python}"
status=0
"$PYTHON" -m pytest backend/tests eval_team haochen_method/tests test_active_acquisition.py test_vae_panel.py -q || status=1
"$PYTHON" -m pytest preprocessing/tests -m 'not network and not slow' -q || status=1
"$PYTHON" main.py --data-dir "${MORPHO_DATA_DIR:?Set MORPHO_DATA_DIR to the embryo data directory}" --max-cells 300 --max-genes 200 --round-sizes 4 4 --output "outputs/smoke-${SLURM_JOB_ID}" || status=1
exit "$status"
