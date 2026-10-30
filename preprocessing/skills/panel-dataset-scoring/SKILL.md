---
name: panel-dataset-scoring
description: Score how well candidate reference datasets agree with a small measured gene panel, and render the red/blue agreement table. Use when a user has an N x S expression matrix for a handful of named genes (plus optionally 2D cell positions and a text description of the experiment) and wants to find, tag and rank reference datasets that match it - spatial, single-cell, non-transcriptomic or causal. Also use when adding a new agreement metric to an existing panelprep run, or when a KL / correlation-Frobenius / optimal-transport-ARI comparison between two expression datasets needs to be set up correctly.
---

# Scoring candidate datasets against a measured gene panel

## When this applies

The user has a *small* measured panel (S ~ 4-400 genes) and wants to know which reference datasets
behave the same way on those genes. The question is **agreement on the measured genes**, not
"which dataset is biggest" and not "which dataset the text describes" - those are separate and are
kept separate here.

## Inputs

| Object | Shape / type | Must be true |
|---|---|---|
| `PanelQuery.expression` | `(N, S)` DataFrame, observations x panel genes | columns are gene symbols; rows are cells/spots |
| `PanelQuery.coords` | `(N, 2)` array | same rows, same order as `expression` |
| `PanelQuery.descriptor` | `str` | free text; drives candidate shortlisting only |
| `CandidateDataset` | per reference | `category` in spatial / single_cell / non_transcriptomic / causal; `units`; `obs_unit` |

All matrices must be `log1p(CP10K)` or declare their real units. **Never transpose on shape.**

## Procedure

1. **Catalogue the candidates.** One dict per dataset in `panelprep/io/catalog.py`, with
   `category`, a `loader` name and metadata used for text matching. A loader converts to
   observations x genes, normalises to `log1p(CP10K)`, and records `obs_unit` truthfully.
2. **Match the descriptor** with `panelprep.match.match_descriptor` (TF-IDF over catalogue text
   plus an embryonic-stage bonus). Keep this *separate* from the agreement scores: a dataset can
   match the story and still disagree with the data, and the report must be able to show that.
3. **Split out embedding-only sources** with `scores.split_scoreable`. A protein language model or
   an interaction graph has no cell x gene matrix, so it is not a scoring candidate at all - it
   must not appear in the agreement table, not even as "not applicable".
4. **Run the scores** with `scores.run_scores(query, candidates, score_names=..., genes=...)`.
   Always pass `genes=query.panel`, a `seed`, and `max_obs` to bound the optimal-transport cost.
5. **Render** with `viz.score_heatmap(results, out_path=...)`.
6. **Read the diagnostics before the values** (`ScoreResult.diagnostics`), especially
   `frac_unsupported` for KL and `coupling_kl_to_product` for the OT score.

## Getting the three classic metrics right

These are the formulations that are usually requested and are usually ill-posed as stated. Fix
them this way and say in the docstring that you did.

**Average per-gene KL divergence.** The observation unit is the cell; the distribution is over
cells *within* a dataset; there is no cell pairing.
- Bin both samples on **shared** edges derived from the *pooled* values (quantile edges by
  default). Binning each sample on its own quantiles makes every pair of datasets look identical.
- KL is infinite wherever the candidate has no mass and the query does. Apply Laplace smoothing
  and report the pre-smoothing query mass on candidate-empty bins as `frac_unsupported`.
- KL is asymmetric with no principled direction here. Prefer **Jensen-Shannon**: symmetric,
  bounded by `ln 2`, finite without smoothing. Keep KL available with `direction` exposed.
- Mark `units_sensitive=True`. These scores compare values, so a normalisation mismatch invalidates
  them, and a *platform detection-rate difference will dominate them* - say so in the report.

**Frobenius norm between panel correlation matrices.**
- Use the **strictly upper triangle**: the diagonal is 1 in both and contributes nothing.
- Normalise by `sqrt(n_pairs)` and report an **RMS correlation difference**, so the value is
  comparable across panel sizes. Keep the raw norm in diagnostics.
- Default to **Spearman**: log-CP10K counts are zero-inflated and far from bivariate normal.
- Drop constant genes, list them, and return `insufficient_genes` if fewer than 2 survive.
- `units_sensitive=False` - correlation is scale-free, so this is the score that survives a
  cross-platform comparison.

**Optimal transport + ARI.** The usual request - "cluster both, transport between them, compute
ARI" - is **undefined as stated**: ARI compares two labelings of *the same objects*, and the query
and candidate cells are different objects. The valid construction:
1. cluster query cells on the shared panel genes -> labels `A`;
2. cluster candidate cells independently -> labels `B`;
3. solve entropic OT between the two clouds in the shared panel space, z-scoring on the **pooled**
   sample and normalising the cost matrix to unit mean so `reg` means the same thing everywhere;
4. push `B` onto the query cells through the coupling, build the cluster-by-cluster transported-mass
   contingency table, rescale the mass to `N_q`, and apply the ordinary ARI algebra to it
   (`comb2(x) = x(x-1)/2` extended to real `x`). That is a genuine same-set comparison.
5. report the hard variant too: argmax-transported label per query cell, then `adjusted_rand_score`.

## Adding a metric

```python
@SCORES.register("my_score", orientation="lower_is_better",
                 applies_to=("single_cell", "spatial", "causal"), needs_coords=False,
                 units_sensitive=True, description="one line, shown in the report")
def my_score(query, candidate, *, genes, rng, max_obs=None, **params) -> ScoreResult:
    qp, cp, used, missing = prepare_panel(query, candidate, genes, max_obs=max_obs, rng=rng)
    if not used:
        return ScoreResult("my_score", candidate.name, None, "lower_is_better",
                           status="insufficient_genes", n_genes_requested=len(genes))
    ...
```

Add the name to `Config.scores`. The table, the colour mapping and the markdown report are all
generated from registry metadata, so nothing else changes.

## Checks to run, every time

1. **Known-answer fixture.** `panelprep.io.synthetic.make_synthetic_case()` builds candidates in a
   known order of similarity. Its `shuffled_structure` candidate permutes each gene independently
   *after* normalisation, so marginals are preserved exactly and co-expression is destroyed.
   Distributional scores must be nearly blind to it; structure scores must catch it. Measured:
   `avg_js` 1.5x apart, `corr_frobenius` 11.7x apart.
2. **Soft ARI reduces to standard ARI** on an integer contingency table, to 1e-9.
3. **Missing panel genes** reduce `n_genes_used` and appear in `genes_missing` - never imputed.
4. **Units guard** blocks a declared mismatch for `units_sensitive` scores and lets correlation
   scores through.
5. **Determinism** for a fixed seed.
6. **A real control.** Include two references from the same study and platform differing only in a
   biological variable (e.g. developmental stage). Any score claiming to measure biological
   similarity must order them correctly; one that does not is measuring platform.

## Failure behaviour

Data problems are **statuses, not exceptions**. Return `ScoreResult(value=None, status=...)` with
`status` in `insufficient_genes` / `not_applicable` / `units_mismatch` / `error` and a `reason` in
diagnostics. `run_scores` catches exceptions per cell so one bad metric cannot take down the table.
Non-`ok` cells render grey and annotated, never as 0.

## Pitfalls

- Colouring raw score values makes "red" mean agreement in one column and disagreement in the next.
  Map every column onto a common agreement direction using `orientation` *before* colouring, and
  print the raw value in the cell.
- Uniform OT marginals assert that both datasets have the same cell composition. They usually do
  not; that is why a dataset with the same biology but a different mixture scores lower here than
  on the correlation scores.
- Large entropic `reg` drives the coupling to the product measure and ARI to 0 regardless of
  agreement. Check `coupling_kl_to_product` before believing a low ARI.
- With a 4-gene panel the transport geometry is 4-dimensional and very coarse.
