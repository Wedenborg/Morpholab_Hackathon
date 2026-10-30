# User manual

**You have:** a measured gene panel — a small set of genes (often 3–10) quantified in many cells,
with each cell's position — and a free-text description of the experiment.

**You also have:** a set of candidate reference datasets. Single-cell, spatial, perturbation, or a
mixture.

**You want:** (1) a score, or a set of scores, saying how well each reference aligns with your data,
and (2) a picture of where your measured genes sit inside each reference, under each embedding
method.

Two calls do that.

```python
from panelprep.api import align_references, panel_dashboard

result = align_references(my_data, references=[...])   # -> scores
panel_dashboard(result)                                # -> dashboard.html
```

---

## 1. Describe your data

Your measurement becomes a `PanelQuery`: the `N × S` panel matrix (rows = cells, columns = your
measured genes), the `N × 2` coordinates, and the descriptor.

```python
from panelprep.schemas import PanelQuery

my_data = PanelQuery(
    expression=panel_df,        # N x S, rows = cells, columns = gene symbols
    coords=xy,                  # N x 2 numpy array
    descriptor="Spatial transcriptomics of a whole mouse embryo at E7.5 during "
               "gastrulation. We study germ layer patterning and the emergence of "
               "mesoderm and endoderm progenitors.",
    units="log1p_cp10k",        # declared, not guessed — see §6
    name="my_experiment",
)
```

Loaders exist for common cases, so you rarely build it by hand:

```python
from panelprep.io.loaders import query_from_seu3d
my_data = query_from_seu3d("GSM9046243_Embryo_E7.5_stereo_rep1.h5ad",
                           panel=["T", "Krt8", "Mest", "Dnmt3b"],
                           descriptor=DESCRIPTOR, max_obs=4000)

# or from a dataset registered in the Morpholab repo
import panelprep.integrations.morpholab as morpholab
my_data = morpholab.query_from_morpholab("seu-3d", PANEL, DESCRIPTOR, timepoint="7.5")
```

The descriptor is not decoration: `match_candidates` ranks the catalogue against it, so write the
tissue, the stage and the biological question in plain prose.

---

## 2. Name your references

Either catalogue names:

```python
from panelprep.io.catalog import catalog_table
catalog_table()        # name, category, organism, descriptor, source

references = ["seu3d_E7.5_rep2", "seu3d_E7.75_rep2",
              "mittnenzweig_wt_E7.5", "mittnenzweig_wt_E6.5",
              "mittnenzweig_foxc_dko"]
```

or datasets you already hold in memory:

```python
from panelprep.schemas import CandidateDataset
references = {
    "my_scrnaseq": CandidateDataset(name="my_scrnaseq", category="single_cell",
                                    expression=df, units="log1p_cp10k"),
    "my_visium":   CandidateDataset(name="my_visium", category="spatial",
                                    expression=df2, coords=xy2,
                                    obs_unit="spot", units="log1p_cp10k"),
}
```

`category` is one of `spatial`, `single_cell`, `non_transcriptomic`, `causal`. It controls which
scores apply: a spatial score needs coordinates, and a reference with no expression matrix at all
(a protein language model, an interaction graph) is excluded from scoring rather than listed as
"not applicable".

**Include a control you expect to lose.** The pilot deliberately carries a stage-mismatched
reference (`mittnenzweig_wt_E6.5`) next to the matched one. If a score cannot rank the matched
reference above the mismatched one, that score is not measuring what you think on your data.

---

## 3. Score the alignment

```python
result = align_references(
    my_data,
    references=references,
    scores=None,          # None -> DEFAULT_SCORES
    methods=None,         # None -> DEFAULT_METHODS; embeddings=False to skip
    outdir="outputs/my_run",
    dim=32,
    max_genes=2000,
    dashboard=True,
)
print(result.summary())
```

`result` is an `Alignment`:

| attribute | what it is |
|---|---|
| `.table` | `datasets × scores` of raw values — **the answer** |
| `.ranking` | one row per reference, best first, with `mean_agreement`, `n_scores_ok`, `agreement_rank` |
| `.best` | the top-ranked reference name |
| `.scores` | long format: one row per `(dataset, score)` with status and gene coverage |
| `.embeddings` | `{name: GeneEmbedding}`, each `|G| × d` |
| `.figures` | `{label: path}` for everything written |
| `.blocked` | anything requested that could not run, and why |

On the real pilot — a 4-gene panel from an E7.5 embryo against seven references — `.table` reads:

| dataset | avg_js ↓ | corr_cosine ↑ | corr_frobenius ↓ | ot_cluster_ari ↑ | sp_autocorr ↓ |
|---|---|---|---|---|---|
| **seu3d_E7.5_rep2** | **0.0028** | **0.9685** | **0.0234** | **0.7910** | **0.0592** |
| seu3d_E7.75_rep2 | 0.0071 | 0.7707 | 0.0700 | 0.5398 | 0.1165 |
| seu3d_E8.0_rep1 | 0.0158 | −0.3624 | 0.0896 | 0.3870 | 0.1415 |
| seu3d_E8.0_rep2 | 0.0197 | 0.0708 | 0.0939 | 0.2839 | 0.0469 |
| mittnenzweig_foxc_dko | 0.1105 | 0.9805 | 0.1187 | 0.1962 | n/a |
| mittnenzweig_wt_E7.5 | 0.1822 | 0.9608 | 0.1267 | 0.2458 | n/a |
| mittnenzweig_wt_E6.5 | 0.1515 | 0.4419 | 0.1467 | 0.1792 | n/a |

The stage-matched spatial replicate wins on every score, and agreement falls monotonically with
developmental distance. **But look at the last two rows** — that is §6.

### The scores available

`SCORES.meta_table()` lists them with their direction and description. The defaults:

| score | compares | direction | notes |
|---|---|---|---|
| `avg_js` | per-gene expression distributions | lower | symmetric, bounded by ln 2. Prefer over `avg_kl` |
| `avg_kl` | the same, asymmetrically | lower | exposed for completeness; `direction` is a parameter |
| `corr_frobenius` | the `S × S` panel correlation matrices | lower | RMS correlation difference, strict upper triangle, Spearman |
| `corr_cosine` | the same, as a direction | higher | isolates the *pattern* of co-expression from its magnitude |
| `corr_mae` | the same, Pearson + MAE | lower | reproduces Morpholab `select_dataset` bit-for-bit |
| `ot_cluster_ari` | cell clusters, transported | higher | OT coupling → soft contingency table → ARI |
| `spatial_autocorr_frobenius` | section-wise Moran's I per gene | lower | spatial references only |
| `spatial_corr_frobenius` | spatial cross-correlation of gene pairs | lower | spatial references only |

### Reading the heatmap

Two are written, always:

- `figures/score_heatmap.png` — **red = agrees most**. A small divergence is red.
- `figures/score_heatmap_raw.png` — **red = numerically large**. The same cell is blue.

Both are correct; they answer different questions. Each column header carries its own direction
arrow. If you ship only one of these, a reader will eventually read `0.0234` coloured red as the
*worst* dataset.

---

## 4. The dashboard

```python
paths = panel_dashboard(result, outdir="outputs/my_run/dashboard")
# open paths["dashboard_html"]
```

or pass `dashboard=True` to `align_references`. On the real pilot — 24 embeddings over 2000 genes —
this took **98 s** and wrote:

| file | what |
|---|---|
| `dashboard.html` | the index: score table, every figure, the quality numbers, the neighbour table |
| `grid_pca_baseline.png` | that method across all 7 references, one panel each |
| `grid_ae_separate.png`, `grid_vae_separate.png` | same, other methods |
| `grid_integrated.png` | the joint methods (`dcca`, `joint_ae`, `contrastive_ae`) — one map each, built from all references at once |
| `panel_in_context.png` | one embedding, full width |
| `panel_neighbours.csv` | **the actionable table** — see below |
| `embedding_quality.csv` | silhouette and kNN label agreement per panel, with permutation p-values |

In every panel your measured genes are drawn as **large labelled stars**, filled with their own
pathway colour, over the rest of the transcriptome. The colour mapping is built **once** over the
union of plotted genes, so a colour means the same pathway in every cell of the grid and the panels
are directly comparable. Reading a stack of separately-coloured figures is how you conclude that
two embeddings disagree when only their palettes did.

### What the dashboard is actually for

The grid answers *"given that I measured these genes, what does each dataset think they are near?"*
— and the `panel_neighbours.csv` table answers it in text, computed in the **full `d`-dimensional
space**, never from the 2D coordinates. From the pilot's integrated `dcca` embedding:

| measured gene | nearest genes (cosine) |
|---|---|
| `T` | Gjc1 (0.49), Dnajc8 (0.45), Cad (0.43), Timm22 (0.40), Ckap5 (0.38) |
| `Krt8` | Tpm1 (0.64), Anxa2 (0.63), Mapkapk2 (0.62), Myadm (0.61), Kdelr2 (0.57) |
| `Mest` | Tmem88 (0.56), Arhgap29 (0.55), Derl1 (0.54), Egfl7 (0.53), Prdm6 (0.43) |
| `Dnmt3b` | Pls3 (0.70), 4833439L19Rik (0.63), Dcakd (0.61), Tcof1 (0.61), Prps1 (0.61) |

That is a candidate list of unmeasured genes behaving like the ones you measured — the input to
choosing the next panel.

### How to read a projection

UMAP and t-SNE distances are **not metric** and cluster sizes carry **no meaning**. So every panel
title prints the silhouette and kNN label agreement *against a label-permutation null*. A panel that
looks beautifully structured but scores at chance says so in its own title.

On the pilot this matters: silhouette is **negative for every expression embedding** (−0.65 to
−0.32) and kNN label agreement sits at 0.426–0.439 against a null of 0.403–0.407. KEGG pathway
membership is only marginally recoverable from these embeddings. The dashboard reports that rather
than hiding it behind an attractive picture.

---

## 5. Choosing methods

```python
methods = ["pca_baseline",       # separate: one embedding per reference. Fast, linear, the baseline
           "ae_separate",        # separate: dense autoencoder, masked-MSE
           "vae_separate",       # separate: variational
           "joint_ae",           # joint: one embedding, masked reconstruction across references
           "dcca",               # joint: deep CCA, genes paired across all views
           "contrastive_ae",     # joint: InfoNCE across references
           "factor_analysis"]    # separate: the Morpholab select_dataset representation
```

`mode` is the thing to understand, and `EMBEDDERS.meta_table()` prints it:

- **`separate`** → one matrix *per reference*, keyed `<method>::<dataset>`. Use these to ask *"does
  this reference place my genes where that one does?"*
- **`joint`** → **one** matrix, keyed `<method>`, built from all references at once. Use these when
  you want a single map and do not care which reference contributed what.

Four more sources never touch the expression values at all — protein sequence (ESM-2), the STRING
interaction graph, the literature (BioGPT), and KEGG regulation. They depend **only on your gene
list**, cost 35–50 minutes each, and so are built once and loaded thereafter:

```bash
python scripts/build_embedding_store.py --import-dir data/priors_export   # adopt the shipped ones
python scripts/build_embedding_store.py --config configs/real_pilot.yaml  # or rebuild from source
```

Ask for one that is not in the store and the run reports it in `result.blocked` with the build
command, then carries on. It will not silently start an hours-long job.

---

## 6. Three things that will mislead you

**Platform beats biology in the distributional scores.** Detection rates differ several-fold between
assays — on the pilot panel, Stereo-seq detects each gene in 8–22% of cells against 36–88% for
MARS-seq. That shifts every gene's marginal, which is exactly what `avg_js` measures. In the table
in §3, `avg_js` ranks the stage-**mismatched** E6.5 reference (0.1515) *above* the matched E7.5 one
(0.1822) — wrong, and the reason is the assay. `corr_cosine` gets it right (0.4419 vs 0.9608).

> Read `corr_cosine`, `corr_frobenius`, `corr_mae` and `ot_cluster_ari` across platforms.
> Read `avg_js` and `avg_kl` only within one.

**Platform also dominates the embeddings.** Dataset-level CKA is 0.972–0.991 within the four
Stereo-seq embryos and 0.932–0.971 within the three MARS-seq datasets, but only **0.427–0.481
across platforms**. If you compare an embedding from your spatial data with one from a dissociated
reference, you are mostly measuring the assay.

**The methods do not agree with each other.** Across the six expression methods on real data, linear
CKA spans 0.105–0.994 and kNN-Jaccard only 0.018–0.161 against a null of 0.004–0.005. There is no
single right embedding; that is why the dashboard shows all of them side by side rather than
picking one. Check `figures/agreement_cka.png` and `agreement_knn.png` for your own data.

---

## 7. Units and gene identifiers

**Units are declared, not inferred.** The contract is `log1p(CP10K)`. Scores that compare values
rather than ranks are registered `units_sensitive=True` and will **refuse** to run across a declared
mismatch unless you set `allow_unit_mismatch=True`. Correlation-based scores are scale-free and
proceed regardless. Declare `units` truthfully on every dataset you construct yourself.

**Gene symbols are canonicalised** by `normalize.gene_key` — upper-cased, Ensembl version suffix
stripped — and aliases are deliberately **not** resolved. A gene absent from a reference is reported
in `genes_missing`, never imputed. Check `result.scores[["n_genes_used", "n_genes_requested"]]`: if a
score used 2 of your 4 panel genes, say so when you quote it.

**Orientation is never inferred from shape.** Rows are observations, columns are genes, always.
`CandidateDataset.obs_unit` records whether an observation is a cell, a spot or a bin.

---

## 8. Adding your own score

One decorated function. The comparison table, the heatmap colours and the markdown report are all
generated from the registry, so they update themselves.

```python
# panelprep/scores/custom.py
from panelprep.schemas import ScoreResult
from panelprep.scores import SCORES, prepare_panel

@SCORES.register("my_score", orientation="lower_is_better",
                 applies_to=("single_cell", "spatial"), needs_coords=False,
                 units_sensitive=True, description="one line, shown in the report")
def my_score(query, candidate, *, genes, rng, **params) -> ScoreResult:
    qp, cp, used, missing = prepare_panel(query, candidate, genes, rng=rng)
    if len(used) < 2:
        return ScoreResult(score_name="my_score", dataset_name=candidate.name,
                           value=None, status="insufficient_genes",
                           orientation="lower_is_better",
                           n_genes_requested=len(genes), n_genes_used=len(used))
    ...
```

Then pass `scores=[..., "my_score"]`. Two rules: use `prepare_panel` to align the panel (never index
the matrices directly), and **never raise for data reasons** — set `status` and `value=None` so the
table shows *why* a cell is empty.

Embedders work the same way via `@EMBEDDERS.register(..., mode="separate"|"joint")` in
`panelprep/embed/custom.py`, returning a `GeneEmbedding` whose matrix is `|G| × d` indexed by gene
symbol.

---

## 9. If you want the pieces instead

`align_references` is a wrapper. Everything underneath is callable:

| call | returns |
|---|---|
| `match_candidates(query, config)` | catalogue ranked against your descriptor |
| `load_candidates(config)` | `{name: CandidateDataset}` |
| `run_scores(query, candidates, score_names=...)` | the long-format score frame |
| `gene_universe(datasets, how=..., must_include=panel)` | the gene list to embed over |
| `run_embedders(datasets, embedder_names=..., genes=..., dim=...)` | `{name: GeneEmbedding}` |
| `describe(embeddings)` / `compare(a, b, metric=...)` | what came back / how similar two are |
| `annotate_genes(genes, config)` | pathway and ligand/receptor/TF flags |
| `EmbeddingStore(root).load_all(...)` | embeddings already on disk |
| `panel_neighbours(embeddings, panel)` | the nearest-gene table |
| `prior_kernel(features)` / `fuse_kernels({...})` | gene × gene kernels and their fusion |

`notebooks/06_run_embedders_explained.ipynb` walks `run_embedders` argument by argument and shows
what changes in the output when each one changes.

---

## 10. Reproducing the worked example

```bash
python scripts/build_embedding_store.py --import-dir data/priors_export
jupyter lab notebooks/05_run_analysis.ipynb      # edit the Parameters cell, run
```

Smallest possible end-to-end check, no downloads, a few seconds:

```python
from panelprep.api import align_references
from panelprep.io.synthetic import make_synthetic_case

case = make_synthetic_case(seed=0)
result = align_references(case.query, references=case.candidates,
                          outdir="outputs/demo", dim=8, max_genes=200,
                          methods=["pca_baseline"], annotate=False,
                          projection="pca", dashboard=True)
assert result.best == "sibling"      # the fixture's known answer
```

The fixture has a ground-truth ordering by construction, so this asserts the scores measure what
they claim — `sibling` is a true replicate, `same_program_diff_mixture` runs the same gene programs
in different proportions, `shuffled_structure` preserves every marginal exactly while destroying
co-expression, and `unrelated` shares nothing.
