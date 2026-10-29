# E7.5 reconstruction benchmark

A method designs experiments on a hidden E7.5 Stereo-seq embryo. Each experiment measures up to 4 genes on one section or on a chosen set of cells. From those measurements the method reconstructs the expression of every gene in every cell. The benchmark scores the reconstruction in three ways:

- **Plausibility.** Do the genes' patterns across anatomical regions and cell types match published E7.5 data? The references are Peng et al. 2019 Geo-seq, GEO accession GSE120963, and the Pijuan-Sala et al. 2019 scRNA-seq atlas.
- **Fidelity.** Does the reconstruction match this embryo where the method did not measure? This uses the held-out gene score, following Tangram (Biancalani et al. 2021), and the region averages of the embryo itself.
- **Cell-type proportions.** Can the cell types and their proportions be recovered from the reconstruction?

## Setup

The Stereo-seq files are expected in `GSE278603_RAW/` next to the repo, as for `mouse_embryo.py`. The reference tables are built once:

```
cd eval-team
python prepare_references.py              # Peng (4 MB) and the atlas (1.7 GB download)
python prepare_references.py --skip-atlas # Peng only
```

Tables go to `reference_data/` next to the repo, or to `$MORPHO_REF_DIR` if set.

## Scoring a reconstructed tensor

Use this when all you have is the reconstruction: one expression value for every cell of the hidden embryo and every gene, with no experiment log.

**1. Pick the hidden embryo.** The default is E7.5 rep2, `GSE278603_RAW/GSM9046244_Embryo_E7.5_stereo_rep2.h5ad`. The reconstruction must describe this embryo's cells. Rep1 is the visible embryo a method may train on. To score the other way round, use the rep1 file, `GSM9046243_Embryo_E7.5_stereo_rep1.h5ad`.

**2. Put the tensor in the expected format.** It is a cells × genes matrix:

- **Rows are the hidden embryo's cells,** named exactly as in that file's `obs_names`, such as `13761--29`. Every cell must be present. Order doesn't matter.
- **Columns are genes,** named as in that file's `var_names`. Genes you leave out are scored as all-zero and counted in `library_genes_missing`. Extra genes are ignored.
- **Values are on the scale of the file's `X`,** which is log1p of counts. Dense or sparse both work.

To turn a NumPy array whose rows and columns follow the hidden file's order into an `.h5ad`:

```python
import anndata as ad, pandas as pd
hidden = ad.read_h5ad("../../GSE278603_RAW/GSM9046244_Embryo_E7.5_stereo_rep2.h5ad")
recon = ad.AnnData(X=tensor,                                   # shape (hidden.n_obs, hidden.n_vars)
                   obs=pd.DataFrame(index=hidden.obs_names),
                   var=pd.DataFrame(index=hidden.var_names))
recon.write_h5ad("reconstruction.h5ad")
```

**3. Score it.** Omit `--log`:

```
cd eval-team
python evaluate.py --truth ../../GSE278603_RAW/GSM9046244_Embryo_E7.5_stereo_rep2.h5ad \
                   --recon reconstruction.h5ad --out-dir results/
```

Or from Python, where a cells × genes `pandas.DataFrame` also works:

```python
from evaluate import Evaluator
summary, tables = Evaluator(hidden).evaluate(recon)
```

This takes about 10 seconds. It prints the summary and, with `--out-dir`, writes `summary.json` and per-gene tables.

**4. Read these keys.**

| Benchmark | Key | Meaning | Better |
|---|---|---|---|
| 1, plausibility | `peng_spearman` | Per gene, the Spearman correlation between the reconstruction's averages in the 8 anatomical regions and Peng et al.'s Geo-seq measurements of the same regions in another E7.5 embryo, averaged over about 250 genes | higher |
| 1, plausibility | `atlas_spearman` | The same comparison with cell-type averages from the Pijuan-Sala E7.5 atlas, about 11 to 13 cell types and 400 to 500 genes | higher |
| 3, cell types | `celltype_tv` | Total variation distance between predicted and true cell-type proportions. A fixed classifier is trained on the reconstruction and tested on held-out sections. | lower |
| 3, cell types | `celltype_balanced_accuracy` | Per-cell accuracy of that classifier, averaged over cell types | higher |

The hidden embryo's labels and coordinates decide which cells form each region and cell type. Its expression values are not used in benchmark 1. Each key also has a `_true_data` version: the same score for the hidden embryo's real data. For benchmark 1 there is also a `_relative` version, the reconstruction's score divided by the true-data score.

**5. Compare with the reference rows.** With rep2 hidden:

| Reconstruction | `peng_spearman` | `atlas_spearman` | `celltype_tv` | `celltype_balanced_accuracy` |
|---|---|---|---|---|
| true data | 0.53 | 0.53 | 0.06 | 0.28 |
| counts thinned to 20% | 0.48 | 0.38 | 0.15 | 0.17 |
| rep1 region averages copied | 0.67 | 0.51 | 0.33 | 0.20 |
| cells shuffled | 0.01 | 0.00 | 0.50 | 0.04 |

Report benchmarks 1 and 3 together. Benchmark 1 rewards a copy of another embryo, as high as 0.67, but benchmark 3 penalises it. Benchmark 3 ignores gene names, but benchmark 1 penalises scrambled ones. Benchmark 3 also varies by about 0.1 between embryos, so treat smaller differences as noise.

**Also in the output:**

- **`truth_region_spearman`.** This correlates the reconstruction's region averages directly with the hidden embryo's own measured averages.
- **`heldout_corr`.** Without a log, this is computed over every neighbourhood, measured ones included. It can't tell predicted values from measured values passed through unchanged. Supply `--log` to restrict it to unmeasured genes.

## What a method gets and returns

A method only touches the hidden embryo through `submission.MeasurementOracle`:

```python
from submission import MeasurementOracle
oracle = MeasurementOracle(hidden_adata)        # the benchmark creates this
oracle.cells                                    # cell names with x, y, z; no labels
oracle.genes                                    # the gene library
values = oracle.measure(["T", "Otx2", "Ttr", "Mesp1"], section=120.0)   # or cells=[...]
log = oracle.log()                              # every experiment, in order
```

An experiment can measure at most 4 genes, and every request is logged. A submission is two files:

- **`reconstruction.h5ad`.** Every cell of the hidden embryo by every gene in the library, on the scale of the data's `X`, which is log1p of counts.
- **`experiments.csv`.** The oracle's log, with columns `step, stage, section_z, genes, cells`. Genes and cells are separated by `;`, and `cells` is empty when a whole section was measured. See `examples/experiments_example.csv`.

The benchmark reads the log to tell measured values from predicted ones. The budget, meaning the number of experiments, is up to the method team. It's reported next to the scores, so methods are compared at equal budgets.

## Running the evaluation

```
python evaluate.py --truth ../../GSE278603_RAW/GSM9046244_Embryo_E7.5_stereo_rep2.h5ad \
                   --recon reconstruction.h5ad --log experiments.csv --out-dir results/
```

```python
from evaluate import Evaluator
summary, tables = Evaluator(hidden_adata).evaluate(recon_adata, log_dataframe)
```

The evaluation writes `summary.json` and three per-gene tables: region scores, held-out scores, and held-out scores by gene sparsity. It takes about 10 seconds.

## Dummy method

`dummy_method.py` stands in for the method team. It produces a real submission through the oracle:

```
python dummy_method.py --hidden-rep 2 --visible-rep 1 --sets 8 --genes markers --out-dir my_dummy/
```

It runs one experiment per section, and sections take turns through `--sets` sets of 4 genes. The genes are cell-type markers of the visible embryo, its most variable genes, or random genes. Each hidden cell gets the average profile of the visible-embryo cells closest to it on its section's 4 measured genes. Cells tied at the k-th distance are all included, since many cells read zero on all 4 genes. Measured values are passed through unchanged.

An example submission with its evaluation is in `dummy_submissions/E7.5_rep2_markers_8sets/` next to the repo. The reconstruction file is 700 MB, which is why it lives outside git.

## Regions

Only cells of the embryo proper are used. The extra-embryonic types are dropped: ExE ectoderm, endoderm and mesoderm, blood and haematoendothelial progenitors, PGCs, parietal endoderm, and the proximal "Surface ectoderm" and "Mesenchyme".

| Layer | Regions | Peng pieces |
|---|---|---|
| Ectoderm | anterior, anterior-lateral, posterior-lateral, posterior | A, L1+R1, L2+R2, P |
| Mesoderm | anterior, posterior | MA, MP |
| Endoderm | anterior, posterior | EA, EP |

Each cell's layer comes from the `germ_layer` label. Its position is the angle around the centre of the embryo in its own section, measured from the anterior direction. Left and right are merged. The anterior direction runs from the centroid of streak and caudal epiblast cells to the centroid of rostral neurectoderm and definitive endoderm cells. Each embryo's axes point in a different direction, so this is computed per embryo. `--levels` also splits each region into proximal–distal thirds, which is noisier.

## Scores

| Key | What it compares | Genes |
|---|---|---|
| `peng_spearman` | Region averages against Peng, Spearman per gene | 8-fold or larger difference between Peng regions |
| `atlas_spearman` | Cell-type averages against the atlas, Spearman per gene | 2-fold or larger difference between atlas cell types |
| `truth_region_spearman` | Region averages against the hidden embryo | All genes |
| `heldout_corr` | Neighbourhoods of about 25 cells against the hidden embryo, Pearson per gene, using only neighbourhoods where that gene was not measured | All genes, also broken down by sparsity |
| `celltype_tv`, `celltype_balanced_accuracy` | Predicted against true cell types and proportions. A fixed classifier is trained on the reconstruction and tested on held-out sections: every fifth section is held out, and its neighbouring sections are left out of training. | 2,000 most variable genes |

Every score uses only genes detected in at least 2% of the embryo's cells. Genes, regions and neighbourhoods are fixed by the hidden embryo and the references, never by the submission. Keys ending `_true_data` give the same score for the real data. That is a reference point, not an upper bound.

## Sanity checks

`python sanity_checks.py` runs every score on reconstructions whose quality is known in advance. It takes about 5 minutes. The hidden embryo here is rep2, and the visible embryo is rep1:

| Reconstruction | Peng | Atlas | Truth regions | Held-out | Cell-type TV | Balanced acc. |
|---|---|---|---|---|---|---|
| true data | 0.53 | 0.53 | 1.00 | 1.00 | 0.06 | 0.28 |
| counts thinned to 50% | 0.51 | 0.49 | 0.76 | 0.80 | 0.09 | 0.25 |
| counts thinned to 20% | 0.48 | 0.38 | 0.53 | 0.56 | 0.15 | 0.17 |
| counts thinned to 5% | 0.40 | 0.25 | 0.28 | 0.30 | 0.25 | 0.10 |
| 10-neighbour average | 0.54 | 0.53 | 0.72 | 0.93 | 0.08 | 0.23 |
| 50-neighbour average | 0.38 | 0.35 | 0.48 | 0.78 | 0.18 | 0.09 |
| cells shuffled | 0.01 | 0.00 | 0.02 | 0.00 | 0.50 | 0.04 |
| genes shuffled | 0.04 | -0.02 | 0.02 | 0.01 | 0.14 | 0.18 |
| rep1 region averages copied | 0.67 | 0.51 | 0.24 | 0.01 | 0.33 | 0.20 |
| dummy, 1 set of 4 marker genes | 0.51 | 0.35 | 0.16 | 0.08 | 0.36 | 0.10 |
| dummy, 8 sets of 4 marker genes | 0.55 | 0.36 | 0.23 | 0.08 | 0.32 | 0.08 |
| dummy, 8 sets of 4 variable genes | 0.28 | 0.19 | 0.12 | 0.04 | 0.40 | 0.05 |
| dummy, 8 sets of 4 random genes | 0.13 | 0.11 | 0.08 | 0.03 | 0.52 | 0.05 |

With rep1 hidden, the pattern is the same. The script prints both.

## Reading the scores

- **The held-out score is the fidelity score.** It is the only one that gives a reconstruction copied from another embryo about 0, even though that copy beats the real data against Peng. It falls steadily as data get shallower, and it ranks the dummy designs markers > variable > random.
- **Held-out scores depend strongly on gene sparsity.** For the dummy with 8 marker sets, genes detected in 2 to 5% of cells score 0.05 and genes detected in more than 25% score 0.22. Tangram reports the same effect, so compare methods within sparsity bins.
- **The true data is not an upper bound for Peng and the atlas.** Rep1's region averages score higher against Peng than rep2's own data, because rep1 has twice as many embryonic cells. A reconstruction that denoises can beat the measurements.
- **Peng and the atlas measure plausibility only.** Report them next to the held-out score, never alone.
- **The proportion score measures information, not gene identity.** The classifier is trained on the reconstruction itself, so shuffled gene names still get proportions roughly right. It is also noisy: the true data's distance is 0.06 for rep2 but 0.19 for rep1. Use it as a secondary score.
- **Shuffled genes score up to 0.13 against Peng.** Many patterned genes share the same broad layer pattern. Treat about 0.1 as the floor for that score.
- **The atlas comparison isn't fully independent.** The Stereo-seq cell-type labels look like they were transferred from this atlas.
- **Rotating more gene sets doesn't help the dummy.** Each cell is matched on its own section's 4 genes only. A real method has to combine measurements across sections to benefit from measuring more distinct genes.
- **Only E7.5 is covered.** Peng's data stops at E7.5. The atlas tables for E7.75 and E8.0 are already built.

## Tests

```
python -m unittest test_region_benchmark -v
```

There are 26 tests. The synthetic ones build a three-layer ring embryo with known regions. They cover the oracle and its limits, the experiment log, the regions under rotation, the held-out score with measured neighbourhoods excluded, the proportion score and the evaluator end to end. The tests on the two real E7.5 embryos check three things. Marker genes must land in the right regions, with T posterior and Otx2 and Sox2 anterior. Shuffled data must score near 0 and thinner data lower. The dummy method's submission must be valid, and its measured values must pass through unchanged. These tests are skipped when the data or the reference tables are missing.
