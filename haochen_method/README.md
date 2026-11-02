# Haochen method: detection-filtered HVG selection + MLP

Place this entire `haochen_method/` directory at the root of
[`Wedenborg/Morpholab_Hackathon`](https://github.com/Wedenborg/Morpholab_Hackathon).
It is self-contained: it does not import the separate local Hackathon project's
`main.py`, `preprocessing.py`, `imputation/`, or `selection/`. It uses the target
repository's `eval_team` for measurement and scoring. No frontend changes are needed.

## Method and defaults

- Reference: `auto` prefers the SC file supplied with `--reference` or
  `MORPHO_SC_REFERENCE`. Without one it uses E7.5 rep1 ST. `--reference-source sc`
  requires SC; `--reference-source rep1` explicitly selects rep1, even if the SC
  environment variable is set. An invalid supplied SC raises rather than falling back.
- Target: hidden E7.5 **rep2**, accessed only through `MeasurementOracle` by the
  method. A different hidden file can be explicitly supplied with `--hidden`.
- Selection: reference-only Scanpy **Seurat normalized dispersion** (`flavor="seurat"`),
  reference detection minimum 10 cells, top **2000** shared HVGs, followed by a
  strict **>50%** reference-cell detection filter (`expression > 0`). The remaining
  genes retain their original HVG order. The pool is not refilled from outside top K.
- Baselines available: `--selection hvg` or `--selection random_hvg`, each with MLP.
- Initial panel: **Sox2 Hoxa1 Hhex Pdx1**, using exact gene IDs.
- **Five acquisition rounds after baseline round 0**, four additional genes per
  round: **4, 8, 12, 16, 20, 24** measured genes. The initial panel is measured
  through the oracle and is included in the experimental budget.
- MLP: two hidden ReLU layers of width 128, Adam, learning rate 0.001, 100 epochs,
  batch size 256, seed 42. Inputs and targets are standardized using only reference
  statistics. A fresh model is fitted after every acquisition, including the last.
- CPU by default; `--device cuda` or `--device cuda:0` selects a GPU. There are no
  hardcoded data paths or GPU/node choices in the method.

`configs/default.json` contains these configurable defaults. CLI arguments override
the JSON configuration. Unknown JSON keys raise an error.

## Environment

Use **Python 3.11+** for the complete upstream benchmark, consistent with the target
repository's documented environment. Its evaluation code uses Python 3.10+ type
syntax. The local LUNA environment is Python 3.9: it can run our standalone core
tests, but cannot import the upstream evaluation package. This directory does not
modify that environment or patch the benchmark.

From the target repository root, in a compatible environment:

```bash
python -m pip install -r haochen_method/requirements.txt
python -m haochen_method --help
```

The benchmark's own dependencies/reference tables must also be available. PyTorch
must support the selected GPU. On Insomnia, run training on an allocated Slurm node;
exclude `ins046,ins047`, which failed with the existing PyTorch installation.

## Run with SC reference (preferred)

Set paths to your own data and reference files. SC `.h5ad` and legacy Seurat `.rds`
are supported. An RDS can reuse `counts.npz`, `genes.txt`, and `cells.txt` in an
existing cache. First conversion requires R and the Matrix package; a bundled
`convert_reference.R` handles the same legacy object as the local experiment.
Use a different cache when changing the source RDS.

```bash
export MORPHO_DATA_DIR=/path/to/GSE278603_RAW
export MORPHO_REF_DIR=/path/to/reference_data
export MORPHO_SC_REFERENCE=/path/to/seurat_object_E7.5.rds

# Once per benchmark reference-data directory; this may download reference data.
python -m eval_team.prepare_references

python -m haochen_method \
  --config haochen_method/configs/default.json \
  --reference-source sc \
  --reference-cache /path/to/pijuan_sala_E7.5_cache \
  --annotation /path/to/MRK_ENSEMBL.rpt \
  --device cpu \
  --output /path/to/results/haochen_sc_rep2
```

For a symbol-based `.h5ad`, provide `--reference /path/to/sc.h5ad`; annotation and
RDS cache are unnecessary. Counts layers or explicit log metadata are recognized.
Use `--reference-scale counts` or `--reference-scale log1p` when the expression
scale is not recorded. Unmapped/ambiguous mouse Ensembl IDs are dropped; duplicate
symbols are aggregated from raw counts before log1p.

## Run with rep1 reference

```bash
python -m haochen_method \
  --config haochen_method/configs/default.json \
  --reference-source rep1 \
  --device cpu \
  --output /path/to/results/haochen_rep1_rep2
```

For either run, use `--device cuda` on a compatible allocated GPU node. To shorten
a validation run, use `--rounds 1 --mlp-epochs 1`; the output still includes round 0.
`--skip-evaluation` writes the submission without benchmark or diagnostic scores,
but still needs `eval_team` for its oracle. Every run requires a **new** output
directory, and refuses to overwrite an existing one. There is no resume mode.

## Expression scale and gene coverage

The benchmark requires the original file `X` scale, documented as natural
**log1p(counts)**. This runner uses raw SC counts -> log1p, without library-total
normalization; rep1 uses its existing log1p `X`. It never computes normalization
factors from the hidden embryo. A prelogged SC file must already have a comparable
scale; we do not infer or undo unknown normalization.

This intentionally differs from the earlier standalone experiment's default
normalize-to-1e4 + log1p workflow. Those historical metrics should not be treated
as the same benchmark run. HVG scores can also change with reference preprocessing.

Training and acquisition use genes shared by the reference and oracle library.
Output retains **every hidden cell and library gene in oracle order**. Genes absent
from the reference are predicted as zero, excluded from acquisition/training, and
listed in `reference_coverage.json`. No second reference is silently mixed in.
The benchmark therefore scores this explicit lack of reference coverage too.
Initial genes missing from the reference/library raise an error.

## Acquisition and evaluation

Each round performs:

```text
select genes -> oracle measures every section -> update explicit panel
             -> fit MLP on reference -> reconstruct in cell batches -> score
```

Actual measured zero values are preserved, and never used to infer whether a gene
was measured. A round selecting four genes costs one experiment per section.
With S sections, baseline plus five acquisitions costs **6 x S** experiments.
Insufficient candidate pools raise before requesting any measurements.

Unlike the upstream convenience runner, prediction is recomputed **after** the
current round's measurements. Thus the final MLP uses all 24 observed genes.

Every round calls the upstream `Evaluator.evaluate(recon, oracle.log())`, including
the log so measured entries are excluded from held-out scoring. Benchmark metric
names in `history.csv` are prefixed with `benchmark_`, for example
`benchmark_heldout_corr`, `benchmark_peng_spearman`, and `benchmark_celltype_tv`.

When training on Pijuan-Sala SC, its source overlaps the atlas used for the
plausibility score. Report that score alongside held-out fidelity and cell-type
scores rather than interpreting it as an independent validation dataset.

Supplementary metrics are also written: `mse`, `l1` (MAE), `pearson`, `spearman`,
each with `_all` and `_unmeasured` scopes. They cover the full oracle gene library,
including zero predictions for unsupported genes. Losses average over entries;
correlations average **per-cell correlations across genes**, omitting constant
vectors. These differ from the benchmark's neighbourhood-level per-gene
`heldout_corr`. Hidden truth reaches only the evaluator and these diagnostics,
after the method has fitted and produced its reconstruction.

Predictions are written in cell batches directly into an HDF5 dataset rather
than building an entire dense prediction in RAM/GPU memory. The existing upstream
evaluator may load dense host arrays for scoring; budget host memory accordingly.

## Outputs

```text
<output>/
  config.json                    # resolved source, paths, options, versions, preprocessing
  hvg.csv                        # reference scores, detection, acquisition membership
  reference_coverage.json         # unsupported library genes and explicit fallback
  history.csv                    # round 0..5 metrics, panels, budget, runtime
  summary.json                   # final-round summary
  reconstruction.h5ad            # final reconstruction in complete oracle order
  experiments.csv                # complete oracle experiment log
  round_00/ ... round_05/
    summary.json
    experiments.csv
    region_per_gene.csv
    heldout_per_gene.csv
    heldout_by_sparsity.csv
```

Only one working reconstruction is retained during the run; final completion
renames it to `reconstruction.h5ad`. Per-round scores/logs are retained, rather
than six large reconstruction files. A failure leaves completed histories and
the working file for diagnosis. Re-run in a fresh directory.

The final files can also be rescored independently:

```bash
python -m eval_team.evaluate \
  --truth "$MORPHO_DATA_DIR/GSM9046244_Embryo_E7.5_stereo_rep2.h5ad" \
  --recon /path/to/results/haochen_sc_rep2/reconstruction.h5ad \
  --log /path/to/results/haochen_sc_rep2/experiments.csv \
  --out-dir /path/to/results/haochen_sc_rep2/rescored
```

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s haochen_method/tests -v
```

Tests cover the strict 50% boundary, reference-only ranking and pool membership,
reproducible random acquisition, exact measured zeros, reference regression,
gene-order alignment, final-round refitting, section-level budgets, output
protection, and unchanged predictions when hidden unmeasured truth changes.
They use small synthetic data and require no real-data downloads or GPU.

## Add to your Git branch

Run these commands in your real `Morpholab_Hackathon` checkout. Copy only this
directory; datasets, caches, checkpoints, and generated outputs are not needed.

```bash
git switch -c haochen-method
# If the folder is not already here, copy haochen_method/ into this repository root.
git add haochen_method/
git diff --cached --stat
git commit -m "Add reference-trained MLP and detection-filtered HVG method"
```

The folder's `.gitignore` excludes common model/data/output artifacts. Keep your
run output outside this source directory. This integration does not change the
frontend, the upstream evaluator, or any existing method.
