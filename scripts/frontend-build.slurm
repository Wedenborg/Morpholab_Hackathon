#!/bin/bash
#SBATCH --job-name=morpholab-ui
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=frontend-build-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export UV_CACHE_DIR="${TMPDIR:-/tmp}/morpholab-uv-${SLURM_JOB_ID}"
.venv/bin/uv pip install --python .venv311/bin/python --reinstall nodejs-wheel
export PATH="$PWD/.venv311/bin:$PATH"
export npm_config_cache="${TMPDIR:-/tmp}/morpholab-npm-${SLURM_JOB_ID}"
cd frontend
npm ci
npm run build
