# preprocessing — `panelprep`

Preprocessing for spatial gene-panel planning: find the reference datasets that match an
experiment, score how much they agree with the measured panel, and build gene embeddings for the
whole transcriptome from four kinds of evidence.

This is the `preprocessing/` folder of
[Morpholab_Hackathon](https://github.com/Wedenborg/Morpholab_Hackathon). It sits alongside
`dataset_selection.py` and does not change it — see
[Merging into Morpholab_Hackathon](#merging-into-morpholab_hackathon) for how the two fit together
and for the one decision to make before merging.

The anchor experiment is Xie et al. 2025, *Digital reconstruction of full embryos during early
mouse organogenesis* ([GSE278603](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE278603)).

**Status:** `pytest -q -m "not network"` → `104 passed, 2 skipped, 10 deselected, 4 warnings in 37.26s`. The synthetic fixture reproduces its known
answer exactly; the real pilot runs end to end on two mouse embryo studies;
`notebooks/05_run_analysis.ipynb` executes all 17 of its code cells. Two prior sources are
explicitly **deferred**, not implemented — see [Scope](#scope-every-method-labelled).

## What it does

The team is handed three things: an `N x S` expression matrix for a small measured gene subset
(`S` ~ 4), the 2D positions of those cells, and a free-text description of the experiment. It
produces:

| # | Deliverable | Where |
|---|---|---|
| 0 | Gene embeddings for the whole transcriptome, tagged `spatial` / `single_cell` / `non_transcriptomic` / `causal` | `panelprep/embed/`, `panelprep/priors/` |
| 1 | A matched, tagged and **scored** candidate dataset set D | `panelprep/io/catalog.py`, `panelprep/match/`, `panelprep/scores/` |
| 2 | Separate and joint autoencoder gene embeddings per dataset | `panelprep/embed/expression.py`, `panelprep/embed/joint.py` |
| 3 | Sequence (ESM-2), interaction (STRING) and literature (BioGPT) embeddings | `panelprep/embed/external.py`, `panelprep/priors/sources.py` |
| 4 | Pathway membership and ligand / receptor / transcription-factor roles | `panelprep/annotate/` |

## Install and run

```bash
pip install -e ".[deep,dev]"
pytest -q -m "not network"          # -> 104 passed, 2 skipped, 10 deselected, 4 warnings in 37.26s

# once: adopt the shipped prior matrices (no compute), or rebuild them from source
python scripts/build_embedding_store.py --import-dir data/priors_export

# every run: edit the Parameters cell, then execute
jupyter lab notebooks/05_run_analysis.ipynb
```

The expensive embeddings — ESM-2 over UniProt, BioGPT, node2vec over STRING — depend **only on the
gene symbol list**, cost 35–50 minutes each, and return the same answer every time. They are built
once by `scripts/build_embedding_store.py` and loaded thereafter. A run that is asked for one it
does not have reports it as `blocked` with the build command and carries on; it never silently
starts an hours-long job.

Notebooks, in order:

1. `01_synthetic_end_to_end.ipynb` — the whole pipeline on a fixture with a **known** answer.
2. `02_real_data_pilot.ipynb` — the same code on real mouse embryo data.
3. `03_gene_embeddings.ipynb` — all four embedding categories, with recovery checks.
4. `04_annotations_and_viz.ipynb` — pathway/role annotation and pathway-coloured embedding maps.
5. `05_run_analysis.ipynb` — **start here.** One configured run end to end: one Parameters cell
   chooses the reference data, the scores, the embedders and the rest; everything below it calls
   package functions. Loads cached embeddings rather than retraining them.
6. `06_run_embedders_explained.ipynb` — `run_embedders` argument by argument, showing what changes
   in the output when each one changes.

## Merging into Morpholab_Hackathon

The merge is **additive**. Nothing in `preprocessing/dataset_selection.py`,
`dataset/load_dataset.py`, `backend/` or `eval_team/` has to change.

### Where the files go

```
Morpholab_Hackathon/
├─ dataset/load_dataset.py          # unchanged — panelprep calls it
├─ preprocessing/
│  ├─ dataset_selection.py          # unchanged
│  ├─ dataset_selection.ipynb       # unchanged
│  ├─ README.md                     # ← this file
│  ├─ panelprep/                    # ← the package
│  ├─ notebooks/                    # ← 05_run_analysis.ipynb + the five others
│  ├─ configs/                      # ← default.yaml, real_pilot.yaml
│  ├─ scripts/                      # ← build_embedding_store.py, build_deck.py
│  ├─ tests/
│  └─ skills/
└─ ...
```

`panelprep/integrations/morpholab.py` finds the repo root by walking up from its own location
(`<repo>/preprocessing/panelprep/integrations` → `<repo>`), so with this layout auto-discovery
works with no configuration. Override with the `MORPHOLAB_ROOT` environment variable or
`morpholab.set_repo_root(path)`.

### Making sure the two sources agree, but good to check a few times

| Morpholab | panelprep | status |
|---|---|---|
| `_gene_key` (upper-case, strip `ENSxxx.N`) | `panelprep.normalize.gene_key` | **identical** — pinned by `test_gene_key_rule_matches_morpholab_character_for_character` |
| observations × genes orientation, never transposed on shape | `PanelQuery` / `CandidateDataset` | identical |
| `REFERENCE_DATASETS` entries | `panelprep.io.catalog.CATALOG` | bridged by `morpholab.reference_to_catalog` |
| `select_dataset` score | score `corr_mae` | **same formula, bit-for-bit** — pinned by `test_corr_mae_equals_the_morpholab_formula_exactly` |
| `loadings` (genes × d) | `GeneEmbedding.matrix` (G × d) | same contract — `morpholab.loadings_to_embedding` is a metadata-only copy |
| `gene_similarity` (gene × gene) | panel correlation matrices in score diagnostics | same object, wider scope |

So their one score becomes one row of a score registry, and their factor analysis becomes one row
of an embedder registry. Both then appear in the comparison table next to everything else.

### Decisions about Metrics

`select_dataset` uses `DataFrame.corr()`, i.e. **Pearson**. On the synthetic fixtures, which have a
known ground-truth ordering, that choice inverts the top two datasets:

| correlation | reducer | `sibling` | `same_program_diff_mixture` | ordering correct? |
|---|---|---|---|---|
| Pearson | MAE (theirs) | 0.0295 | **0.0261** | no |
| Pearson | RMS | 0.0383 | **0.0280** | no |
| Spearman | MAE | **0.0276** | 0.0502 | yes |
| Spearman | RMS | **0.0376** | 0.0665 | yes |

The MAE-versus-RMS reducer makes no difference; **Pearson versus Spearman decides it**.
`same_program_diff_mixture` runs the same gene programs as the query in different cell-type
proportions, so its *rank* co-expression structure diverges while its linear correlation over four
sparse genes stays close — driven by a handful of high-expressing cells, which is the usual failure
mode of Pearson on zero-inflated counts.

`corr_mae` keeps `method='pearson'` as its default so the existing number reproduces exactly. Pass
`score_params={'corr_mae': {'method': 'spearman'}}` to switch, or read the `corr_frobenius` row,
which is Spearman by default. This is a *choice*, not a bug in their code — but it should be a
deliberate one, so the finding is pinned as a regression test rather than silently corrected.

### Make sure the functions here don't collide with the rest of the folder

**Their datasets as panelprep candidates** — uses their downloader, unpacker and stage selection:

```python
import panelprep.integrations.morpholab as morpholab
cand  = morpholab.load_morpholab_dataset('tome', timepoint='7.5', layer='logcounts')
query = morpholab.query_from_morpholab('seu-3d', PANEL, DESCRIPTOR, timepoint='7.5')
print(morpholab.available_datasets())
```

**Use gene representation inside panelprep's figures:**

```python
result = select_dataset(adata, genes=PANEL, d=64)        # their function
emb    = morpholab.loadings_to_embedding(result)         # -> GeneEmbedding
store.put(emb, f"factor_analysis::{result['gene_set_name']}", group='expression')
```

or compute the same estimator through the registry, for every dataset at once:

```python
run_embedders(candidates, embedder_names=['factor_analysis'], genes=genes, dim=64)
```

**Verify the identifier rule after merging** — a silent divergence here mis-pairs every gene in
every score:

```python
morpholab.check_gene_key_agreement()      # -> frame with an `agree` column
```

### Dependencies added

Beyond what the repo already uses (`anndata`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `scanpy`):
`pyyaml`, `pyarrow`, `pot` (optimal transport), `umap-learn`, `networkx`, `torch`, `matplotlib`,
`seaborn`, `jinja2`, `gensim` (node2vec), `transformers` (ESM-2 / BioGPT).

Shared packages carry the **same floors as `backend/requirements.txt`** (`numpy>=2.0`,
`pandas>=2.2`, `scipy>=1.14`, `anndata>=0.11`) so a fresh install cannot resolve an older
version that the backend then rejects. `panelprep` itself needs Python ≥ 3.10; the repo as a
whole needs ≥ 3.11 for the backend.

`torch` is needed only by the autoencoder / deep-CCA / contrastive embedders; `transformers` and
`gensim` only by `scripts/build_embedding_store.py`. Scoring, `pca_baseline` and `factor_analysis`
work without any of them.

### Compatibility check against the rest of the repo

Run on a real merged tree (this repo + `preprocessing/panelprep`), against the live modules:

| check | result |
|---|---|
| filename collisions in `preprocessing/` | **none** — only `README.md` is shared, and it is this file |
| `from preprocessing.dataset_selection import select_dataset` (used by `mouse_embryo.py`) | **unaffected** — no `__init__.py` is added, so the namespace package is unchanged |
| repo-root auto-discovery with no env var and no `set_repo_root` | **finds it** |
| `_gene_key` vs `normalize.gene_key`, against the live function | **identical on every input tested** |
| keys `loadings_to_embedding` reads vs keys `select_dataset` sets | **none missing** |
| `load_dataset(dataset_name, output, timepoint)` vs what the adapter calls | **signature matches** |
| `import eval_team` with `panelprep` present | **OK** |
| shared dependency constraints | **no conflict** (floors now aligned, see above) |

Two defects the check found, both fixed:

- **Spatial datasets were being tagged `single_cell`.** `dataset.load_dataset.DATASET_NAMES`
  registers **53** datasets, not the 3 in `REFERENCE_DATASETS`, and the original
  `_SPATIAL_HINTS` list caught only 7 of the 14 spatial ones — `mosta`, `mpsta`, `dbit-seq`,
  `xenium-neonatal`, `starmap-plus`, `osmfish` and `st-magic` were mis-inferred. That is not
  cosmetic: a spatial dataset tagged `single_cell` has its spatial scores silently skipped.
  The hint list now covers the platform names, and
  `tests/test_store_and_integration.py` pins the category of all 53 so a newly registered
  technology fails loudly rather than quietly. Passing `category=` to
  `load_morpholab_dataset` always overrides inference, and a dataset that actually carries
  coordinates is detected regardless of its name.
- **Unpinned shared dependencies**, now floored to match the backend.

Two things to be aware of, which are not defects:

- **Both halves want GSE278603, in different places.** `eval_team/paths.py` defaults `DATA_DIR`
  to a `GSE278603_RAW` folder *beside* the repo; `panelprep`'s loaders use `data/<accession>`
  *inside* it. Set `MORPHO_DATA_DIR`, or symlink, rather than downloading ~840 MB twice.
- **`pytest` from the repo root now collects both suites** (118 tests, 10 deselected) and will
  error on `backend/tests/test_workflow.py` unless `fastapi` is installed. Run the preprocessing
  suite from `preprocessing/`, or install the backend requirements too.

## One configuration entry point

Everything is driven by `panelprep.config.Config`, built in Python or from YAML
(`configs/default.yaml`, `configs/real_pilot.yaml`). There is no second place to look.

```python
from panelprep import Config
from panelprep.pipeline import run_pipeline
cfg = Config.from_yaml("configs/real_pilot.yaml")
out = run_pipeline(query, cfg)         # scores, embeddings, annotations, figures, report
```

## Adding a new score: IMPORTANT!

Three small registries (`scores.SCORES`, `embed.EMBEDDERS`, `priors.PRIORS`) drive everything
downstream. Adding a metric is one decorated function plus one line of config — the comparison
table, the heatmap colours and the markdown report are all generated from the registry, so they
update themselves.

```python
from panelprep.schemas import ScoreResult
from panelprep.scores import SCORES, prepare_panel

@SCORES.register("my_score", orientation="lower_is_better",
                 applies_to=("single_cell", "spatial"), needs_coords=False,
                 units_sensitive=True, description="one line, shown in the report")
def my_score(query, candidate, *, genes, rng, **params) -> ScoreResult:
    qp, cp, used, missing = prepare_panel(query, candidate, genes, rng=rng)
    ...
```

Then add `"my_score"` to `Config.scores`. `panelprep/scores/custom.py` is the drop-in file;
`notebooks/01` demonstrates the whole loop end to end.

## Assumptions for how this is built

These are the three places cross-dataset comparisons go wrong silently, so they are fixed in one
place each and documented at every call site.

**Observation unit.** A row of any expression matrix is one measured observation — a cell for
scRNA-seq and segmented spatial data, a spot/bead/bin for Visium, Slide-seq or binned Stereo-seq.
`CandidateDataset.obs_unit` records which; it is never inferred from matrix shape. In the
*embedding* half the observation flips: a `G x d` gene embedding treats the **gene** as the
observation and the cells as features, which is why the joint models pair genes across datasets and
never cells.

**Expression units.** The contract is `log1p(CP10K)`. Loaders convert and declare; scores that
compare values rather than ranks are registered `units_sensitive=True` and refuse to run across a
declared mismatch unless `Config.allow_unit_mismatch` is set. Correlation-based scores are
scale-free and proceed.

**Gene identifiers.** Gene symbols, canonicalised by `normalize.gene_key` (upper-case, Ensembl
version suffix stripped). Aliases are deliberately **not** resolved: a gene absent from a candidate
is reported in `genes_missing`, never imputed.

## Where the requested formulations were corrected

Each of the three requested scores needed a correction to be well defined. The intended scientific
question is preserved in every case; the change and the reason are in the module docstring.

| Requested | Problem | What is implemented |
|---|---|---|
| Average per-gene **KL divergence** between query and candidate | KL between two continuous samples needs a common partition; it is infinite where the candidate has no mass; and it is asymmetric with no principled choice of direction | Shared quantile bin edges from the pooled sample, Laplace smoothing with the unsupported mass reported, `direction` exposed — plus `avg_js`, the symmetric bounded Jensen-Shannon form, as the recommended default |
| **Frobenius norm** between the two `S x S` panel correlation matrices | The diagonal is 1 in both and contributes nothing, and the raw norm grows with panel size | Strictly upper triangle, normalised to an RMS correlation difference; Spearman by default because log-CP10K counts are zero-inflated; raw norm kept in diagnostics |
| Cluster both sets, transport between them, compute **ARI** | ARI is defined for two labelings of *the same objects*; query and candidate cells are different objects, so this is undefined as stated | Cluster each side, push candidate labels onto query cells through the OT coupling, and compute ARI from the resulting soft contingency table over the query cells — a valid same-set comparison. `ari_hard` (argmax-transported label) is reported alongside |
| IterPert's **literature prior** via BioGPT's "[CLS] token" | BioGPT is a decoder-only causal LM and has no `[CLS]` token | Last non-padding token's hidden state (the causal-LM analogue), `pooling` exposed, with a templated sentence rather than a bare symbol |
| Morpholab's `select_dataset` score | Nothing is wrong with it, but `DataFrame.corr()` is Pearson, which on four zero-inflated genes is driven by a few high-expressing cells | Registered unchanged as `corr_mae` so their number reproduces bit-for-bit, with the Pearson-vs-Spearman ordering difference pinned as a regression test — see [Decisions about Metrics](#decisions-about-metrics) |
| IterPert's **Perturb-seq prior** | The feature map needs one pseudobulk delta *per gene*, i.e. a genome-scale screen; the only mouse perturbation data here is a single double knockout | Implemented with the correct interface, returns an explicitly **deferred** status naming the reason and the number of perturbations found — not silently reduced to a one-column feature |

## Scoring excludes embedding-only sources

A catalogue entry such as ESM-2, STRING or KEGG has no cell × gene matrix, so it is not a scoring
candidate and does not appear in the agreement table — not even as "not applicable", which would
claim it had been considered as a reference and rejected. `scores.split_scoreable` makes the split;
such entries appear in the embedding inventory instead.

## Evidence of correctness

The synthetic fixture (`panelprep/io/synthetic.py`) has a known answer by construction. Its
`shuffled_structure` candidate permutes each gene independently *after* normalisation, so every
gene's marginal distribution is preserved exactly while co-expression is destroyed — which
separates the two score families:

| | `sibling` | `shuffled_structure` | ratio |
|---|---|---|---|
| `avg_js` (distribution) | 0.0049 | 0.0073 | **1.5×** — correctly blind |
| `corr_frobenius` (structure) | 0.0376 | 0.4386 | **11.7×** — correctly sensitive |

`tests/test_scores.py` asserts this, and the real-data pilot carries its own control: two
single-cell references from the *same study and platform* differing only in developmental age, so
any score measuring developmental similarity rather than platform must rank E7.5 above E6.5.

## Scope: every method labelled

| Status | What |
|---|---|
| implemented + evaluated | 8 scores — `avg_kl`, `avg_js`, `corr_frobenius`, `corr_cosine`, `corr_mae`, `ot_cluster_ari`, and two spatial scores |
| implemented + evaluated | 7 expression embedders, separate and joint — `pca_baseline`, `ae_separate`, `vae_separate`, `joint_ae`, `dcca`, `contrastive_ae`, `factor_analysis` |
| implemented + evaluated | 5 prior sources over 4,006 mouse symbols, plus kernel construction and fusion |
| implemented + evaluated | pathway / ligand / receptor / TF annotation |
| **deferred** | `perturbation_pseudobulk` — needs one pseudobulk delta per gene, i.e. a genome-scale screen; the only mouse perturbation data here is a single double knockout. Returns an all-NaN matrix with the reason and the perturbation count, not a one-column substitute. |
| **deferred** | `ops_morphology` — PERISCOPE is human-only; this project is mouse. |

Two further limits are recorded rather than smoothed over:

- `regulatory_kegg` as shipped is pathway **co-membership** — undirected and unsigned, not the
  signed regulatory graph the source is meant to provide. Status reads
  `ok_fallback_pathway_membership`; rebuilding takes seconds.
- The shipped prior matrices cover **4,006** symbols, not the full 26,312 protein-coding
  transcriptome. Extending ESM-2 to all 16,520 Swiss-Prot-reachable symbols is ≈5.3 h of CPU at the
  measured 1.16 s/protein.

### Kernel fusion is not PSD-preserving, including as the paper describes it

Convex combinations and the Hadamard product preserve positive semi-definiteness; element-wise
`max` does not (smallest eigenvalue −4.142e-01 on the test case). More importantly, IterPert's own
min–max normalisation step is an affine map of the off-diagonal that adds eigenvalue −c and is
itself not PSD-preserving: mean fusion *after* min–max gives −2.109e-02, and the real five-source
fused kernel gives **−115**. `fuse_kernels` warns when the result is indefinite; project it (clip
negative eigenvalues) before using it in any method that requires PSD.

## Layout

```
panelprep/
  config.py        schemas.py      normalize.py     registry.py     pipeline.py     report.py
  run.py           run_analysis(): one configured run, cached embeddings, every figure
  io/              catalog.py (the candidate set D)  loaders.py  synthetic.py
  match/           descriptor.py   (text -> ranked candidates)
  scores/          distributional.py  structure.py  alignment.py  spatial.py  custom.py
  embed/           expression.py   joint.py        external.py    custom.py
                   store.py        EmbeddingStore: compute once, load forever
  priors/          sources.py      kernels.py      (IterPert-style multi-modal priors + fusion)
  annotate/        pathways.py     roles.py
  viz/             score_table.py  embeddings.py   gallery.py (visualize_all)
  integrations/    morpholab.py    the only module that knows about the host repo
configs/  notebooks/  scripts/  tests/  skills/
```

## Data

Downloaded by the loaders into `data/raw/`; This should be changed to make it work with cluster data: TODO

- **GSE278603** — Xie et al. 2025 Stereo-seq embryos, E7.5/E7.75/E8.0, six `.h5ad`, ~840 MB.
  `X` is `log1p` of *SCT-corrected* counts whose per-cell totals are not constant, so the loader
  inverts the log, renormalises to CP10K and re-applies `log1p`. The SCT variance stabilisation
  upstream is recorded, not undone.
- **GSE169210** — Mittnenzweig et al. 2021 MARS-seq, 154 plates plus metadata, ~166 MB
  ([doi:10.1016/j.cell.2021.04.004](https://doi.org/10.1016/j.cell.2021.04.004)). The wild-type arm
  gives stage-matched and stage-mismatched references; the Foxc1/Foxc2 double-knockout chimera arm
  gives the causal candidate.

Known platform caveat, surfaced in the report rather than hidden: on the pilot panel the
Stereo-seq query detects each gene in 8–22% of cells against 36–88% for MARS-seq. That difference
shifts every gene's marginal, so the distributional scores rank the *stage-mismatched* single-cell
reference above the matched one while every structure-based score ranks them correctly. Read
`corr_cosine` and `ot_cluster_ari` across platforms; read `avg_js` within one.
