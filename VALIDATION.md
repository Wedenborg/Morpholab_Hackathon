# Integration validation — 2026-11-02

Branch: `integration/hackathon`. All remote branch tips fetched at the start of
integration are ancestors of this branch: Emilie, stasmethod, yazdan_branch,
haochen-method, pressm/preprocessing-priors, eval-team-benchmark, preprocessing,
and main. Original method implementations and commit history are preserved.

All successful checks below ran on Slurm compute nodes using Python 3.11.18.

| Check | Result | Slurm job |
| --- | --- | --- |
| Backend, evaluator, HVG/MLP, EDDI/PPCA/linear, VAE/acquisition tests | 88 passed, 6 skipped | 13756688 |
| Preprocessing tests, excluding network and slow markers | 100 passed, 6 skipped, 10 deselected | 13756606 |
| React/TypeScript production build | Passed | 13756624 |
| Real E7.5 selection → oracle measurements → ridge reconstruction → saved artifacts | Passed, exit 0 | 13756639 |

The first combined verification job (13756606) has an overall failed status because
its initial core-test collection lacked the backend import path. `pytest.ini`
fixes that path; the corrected core suite passed in job 13756688. The preprocessing
suite and real-data runner within the original job subsequently passed.

The real-data smoke run used E7.5 rep1 as target, rep2 as reference, 300 cells per
embryo, 200 aligned genes, and two rounds of four genes. It measured 8 genes in
4 oracle experiments and saved:

- `outputs/real-smoke-13756639/reconstruction.h5ad`
- `outputs/real-smoke-13756639/experiments.csv`
- `outputs/real-smoke-13756639/summary.json`

These local outputs are ignored by Git. This verifies execution on a bounded
real-data subset, not prediction quality or full-dataset resource requirements.
Large external reference benchmarks and network-dependent checks were not run.

Use `scripts/verify.slurm` to repeat verification. Set `MORPHO_DATA_DIR`,
`MORPHO_PYTHON` if using a different environment, and your Slurm account/QoS.
The local environment used here is `.venv311`; its interpreter resides in
`.python-runtime`. Both directories are ignored by Git. Follow `AGENTS.md`:
all compute work, including tests and builds, must run through Slurm.
