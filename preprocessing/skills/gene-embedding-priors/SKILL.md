---
name: gene-embedding-priors
description: Build gene embeddings for a whole transcriptome from several independent evidence sources - expression autoencoders, protein language models (ESM-2), interaction networks (STRING), curated regulation (KEGG), literature language models (BioGPT) - and fuse them as normalised kernels in the style of ITERPERT. Use when a user asks for gene-level embeddings, multi-modal gene priors, per-gene feature maps for a screen or panel, kernel fusion across prior sources, or pathway-coloured embedding maps with a quality number attached.
---

# Gene embeddings from multiple prior sources

## The one thing that decides the whole design

The deliverable is `G x d`: **the gene is the observation.** Given a `G x N_k` matrix the `N_k`
cells are the *features*, so an autoencoder trains on `G` samples of dimension `N_k`. Two datasets
therefore have different, non-comparable feature spaces - different cells - and the only pairing
that exists across datasets is **the same gene**.

Consequences, all of which must be implemented:
- Standardise each dataset's cell-features across genes (z-score within each column) before
  training, or sequencing depth defines the geometry.
- `N_k` can exceed `G`: project cells to `min(256, N_k, G)` principal components first. This is a
  **feature-space** reduction, not a gene-space one.
- Joint models pair **genes**, never cells. A contrastive model's positive pair is the same gene in
  two datasets; pairing cells across datasets would be wrong.
- Deep CCA is only defined on the **intersection** of genes across views. Genes outside it are
  extrapolation and must be flagged, not silently included.

## Sources, and what each one actually requires

Following ITERPERT (Huang et al., bioRxiv 2023.12.12.571389), first part only - the prior feature
maps and their fusion, not the sequential-design loop.

| Source | Feature map | Reality check |
|---|---|---|
| Expression AE / VAE / joint AE / DCCA / contrastive | learned latent of the `G x N_k` matrix | needs a real reference dataset per view |
| scRNA-seq atlas co-expression | gene's expression vector across cells, z-scored per gene | the **linear kernel of the z-scored map IS the correlation matrix** - that is why per-gene standardisation is not optional |
| Protein sequence (ESM-2) | mean over residue representations | exclude BOS/EOS and padding; ESM-2's BOS is not a trained sentence embedding the way BERT's `[CLS]` is |
| PPI (STRING) | node2vec or spectral embedding of the thresholded graph | report how many genes are *in* the source separately from how many have a partner *inside the requested list* |
| Curated regulation (KEGG KGML) | signed directed relations, separate in/out summaries | literature-curated structure, **not** an estimated causal effect - say so |
| Literature (BioGPT) | last non-padding token's hidden state | BioGPT is decoder-only and has **no `[CLS]` token**; use a templated sentence, not a bare symbol |
| Perturb-seq | control-centred pseudobulk delta per perturbation | needs a screen where **every gene** is perturbed; one knockout gives one vector, not a per-gene map - defer explicitly |
| Optical pooled screens | CellProfiler morphology profiles | PERISCOPE is human-only; for mouse this needs orthologue mapping or it is deferred |

## Procedure

1. **Fix the gene universe first** - the union or intersection across expression candidates, or the
   organism's protein-coding set. Not the measured panel: the point of a prior is to cover genes
   that were never measured.
2. **Build each source independently**, returning a `(G, f)` matrix with **NaN rows** for uncovered
   genes and a provenance dict carrying source URL/accession/checkpoint, `n_covered`, and a status
   in `ok` / `deferred_*` / `blocked_*`.
3. **Kernelise**: cosine or linear kernel per source.
4. **Normalise each kernel** by min-max over the **off-diagonal** entries (the diagonal is 1 by
   construction for a cosine kernel and would otherwise fix the range). Without this, one
   large-scale kernel dominates the fusion.
5. **Fuse**: the mean operator is ITERPERT's choice and is the safe one - a convex combination of
   PSD kernels is PSD. The Hadamard product is PSD (Schur), but the element-wise **max is not
   guaranteed PSD**; check the smallest eigenvalue and warn.
6. **Handle missing data explicitly** - the paper does not. A NaN row poisons a whole kernel row.
   Default to `missing='mask'`: compute each fused entry from only the sources covering both genes,
   and keep a companion per-entry source-count matrix. `missing='zero'` silently asserts
   dissimilarity and is the wrong default.
7. **Report kernel alignment** (Frobenius-inner-product cosine) between every pair of sources. That
   is what tells the user whether a source adds information or duplicates another.

## Visualising, with numbers attached

- `embedding_scatter(embedding, annotation, color_by='top_pathway', role_overlay=True)` - drop NaN
  rows and state how many in the subtitle; keep the top ~12 pathway categories and pool the rest.
- `embedding_panel(embeddings, annotation)` - one panel per method, shared colour mapping.
- **Pairwise comparison** is usually the real question: `embedding_agreement_matrix` (mean kNN
  Jaccard, linear CKA, or Mantel correlation between two gene embeddings - CKA is the right default
  because an autoencoder latent basis is arbitrary and CKA is rotation-invariant), and
  `embedding_pair_scatter`, which Procrustes-aligns one onto the other and draws arrows for the
  genes that move most. Those arrows answer "which genes do these two sources disagree about".
- Always attach `embedding_quality`: silhouette and kNN label-agreement of the annotation labels,
  **each against a label-permutation null**. A scatter with no number is not a result.

## Checks to run

1. **Shape and index**: `(len(genes), dim)`, gene index preserved and ordered.
2. **Mask honoured**: genes absent from a dataset are NaN rows for separate methods and present for
   joint methods (learned from the other views).
3. **Determinism** for a fixed seed - including node2vec (`workers=1`).
4. **Recovery against a known generative geometry**: on a fixture whose genes come from a known
   loadings matrix, the embedding's kNN graph must beat a label-permutation baseline. Report the
   Jaccard at k=15 per method *and* the baseline. Expect a linear SVD baseline to be strong on a
   linear fixture - that is the fixture's property, not a bug in the deep methods.
5. **Kernel algebra**: the linear kernel of a z-scored feature map equals `np.corrcoef` to 1e-10.
6. **PSD**: mean fusion of PSD kernels is PSD; exhibit a case where max fusion is not.
7. **Biological sanity, with a p-value**: a known complex (e.g. Sox2/Pou5f1/Nanog) must have higher
   mean within-set similarity than size-matched random sets, over >=200 permutations - reported
   **per source**, so a source that fails is visible instead of hidden inside the fusion.
8. **Coverage table**: per source, genes covered / requested, feature dimension, wall time.

## Failure behaviour

An unavailable source is a **status, not an exception**: return an all-NaN matrix with
`provenance['status']` naming the reason and let the caller report it. Never substitute a different
source and keep the original name; never claim genome-wide coverage you did not compute - state the
exact number of genes run. Embedding dimensionality differs across sources (ESM-2 returns its
native width); set an explicit `pca_dim` before comparing methods in one plot.

## Pitfalls

- Mean-pooling a padded transformer batch mixes padding into the embedding. Exclude special tokens
  and padding, and test that a batched embedding matches the single-sequence one to <1e-4.
- Reporting only "genes with an edge inside my list" understates a network source; reporting only
  "genes present in the source" overstates what the embedding covers. Report both.
- InfoNCE temperature is the most consequential knob for a contrastive gene embedding: a low
  temperature maximises uniformity and erodes the local neighbourhood structure the embedding is
  wanted for. Measure it rather than inheriting a default.
- A gene symbol fed to a literature model yields the model's prior about that *string*, not
  retrieved evidence. It is a weak prior and should be labelled as one.
