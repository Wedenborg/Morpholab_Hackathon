# QC and three expression layers

Read one full-gene source sample directly, filter cells, filter genes, and save a
separate `.h5ad`. The source file is never changed. Existing source layers are
replaced in the output with exactly these three layers:

| Layer | Expression |
| --- | --- |
| `counts` | Unnormalized original RNA counts for verified TOME inputs |
| `logcounts` | Natural `log1p(counts)` |
| `logcounts_normalized` | Natural `log1p(counts * 10000 / retained cell total)` |

`X` preserves source expression for retained cells and genes. Select a named
layer explicitly in downstream models. Existing `raw`, annotations, coordinates,
and other AnnData metadata follow normal AnnData slicing; `raw` retains its
original gene axis. Source layer names and expression provenance are recorded in
`uns['qc_normalization']`. The untouched input retains all original source layers.

## Paper-based cell thresholds

- SEU-3D: keep cells with at least **400** genes using original `nFeature_RNA`
  metadata, and genes detected in at least **4** retained cells using available
  stored counts. These numeric thresholds come from the
  [Xie et al. methods](https://doi.org/10.1016/j.cell.2025.05.035).
- TOME stages exclusively attributed to Pijuan-Sala on the
  [official TOME portal](https://tome.gs.washington.edu/): keep cells with at least
  **1,000** genes, as specified in the
  [Pijuan-Sala methods](https://www.babraham.ac.uk/sites/default/files/media/files/30787436.pdf).
  The CLI recognizes `seurat_object_E6.75.h5ad`, E7.0, E7.25, E7.5, E7.75,
  E8/E8.0, E8.25 and E8.5a. Other filenames need `--source-study pijuan-sala`
  when that attribution is established, or an explicit `--min-genes` override.
  Mixed-source stages such as E6.5 do not receive a guessed default.

This implements the detected-gene rule, not the papers' complete original QC.
It does not rerun mitochondrial, section/image, annotation or cell-calling QC.
These downloads are already curated atlases. Count totals must exactly match
`nCount_RNA` for TOME or `nCount_SCT` for SEU-3D; integer values alone are not
accepted as evidence of raw counts. Matrix checks were performed on local TOME
E7.5, E7.75, and E8; other files must pass validation independently.

## Gene filtering and normalization order

Calculate detected genes and total counts across the full input gene set.
Remove cells below the source-study threshold (and zero-count cells). Then
calculate each gene's population variance on **direct natural log1p counts**
across retained cells. Remove exactly constant genes and genes with variance
at or below `--min-gene-variance` (default **0**). A stronger positive cutoff is
an analysis choice, not a published universal QC threshold. No HVG selection is
performed. Genes must also meet `--min-gene-cells` (default 4 for SEU-3D, 1 for
TOME). This detection cutoff on corrected SEU-3D counts cannot exactly reproduce
filtering on the unavailable original RNA matrix.

Finally normalize the **retained gene set** to 10,000 counts per cell and apply
natural log1p. Thus reversing log1p gives a cell total of 10,000 in the normalized
layer. This denominator is explicitly recorded; it differs from normalization
over all source genes followed by gene restriction. Cell QC is not recalculated
on the reduced gene panel. If gene filtering creates a zero-count cell, the run
fails and asks for a weaker gene filter.

## SEU-3D count provenance

Available stored counts are likely SCT-corrected, not original RNA counts:
their totals match `nCount_SCT`, and published `X` matches their natural log1p.
Replicate 1 uses `layers['counts']`, replicate 2 uses gene-aligned `raw.X`.
SEU-3D therefore requires `--allow-corrected-counts`. With that option, all three
layers derive from corrected counts and provenance explicitly says so. Neither
`counts` nor `raw.X` should be assumed to mean original raw RNA counts.
Published `X` remains unchanged except for subsetting. Normalizing corrected
counts does not make them equivalent to RNA counts or harmonize datasets.

## Usage

Run from the repository root. Relative source paths use the shared data root
`/insomnia001/depts/morpheus/projects/hackathon/Datasets` by default. Outputs
must be new paths in a writable directory.

```bash
python -m preprocessing.qc_normalization tome \
  tome/seurat_object_E7.5.h5ad preprocessing/results/qc/tome_E7.5.h5ad

python -m preprocessing.qc_normalization seu-3d \
  GSE278603_RAW/GSM9046243_Embryo_E7.5_stereo_rep1.h5ad \
  preprocessing/results/qc/seu_E7.5_rep1.h5ad --allow-corrected-counts
```

Override numeric thresholds with `--min-genes`, `--min-gene-cells`, and
`--min-gene-variance`. For in-memory use, import `qc_normalize` and specify the
dataset and `source_study='pijuan-sala'` or explicit `min_genes`.
Use `write_qc_h5ad(result, output_path)` to save an in-memory result; this opts
into nullable-string serialization used by the source TOME annotations and
requires an AnnData >=0.11-compatible reader.

```bash
python -m pytest preprocessing/test_qc_normalization.py -q
```
