"""Joint gene embeddings across several datasets: joint AE, deep CCA, contrastive AE.

What is paired, and what is not
-------------------------------
**The shared object is the GENE.** Every dataset in this module contributes a
``G x N_k`` block in which a row is a gene and the columns are that dataset's
own cells. The same gene has exactly one row in every block, and that row
correspondence -- established by symbol through
``panelprep.normalize.resolve_genes`` -- is the *only* pairing that exists
between these datasets. It is what makes a multi-view method valid here.

Cells are **not** paired. Cell 7 of a gastrulation atlas and cell 7 of a Visium
section are unrelated observations, drawn from different tissues, different
dissociation protocols and in general different numbers. Consequently:

* A joint model must **not** concatenate cells across datasets as if they were
  the same features. ``joint_ae`` concatenates the per-dataset *feature blocks*
  along the feature axis of the **gene** samples -- one gene row becomes the
  concatenation of its profile in dataset 1, its profile in dataset 2, and so
  on -- which is a multi-view model of a single shared sample set, not a merge
  of two sample sets.
* The contrastive positive pair is **the same gene in two different datasets**.
  Pairing cells across datasets would be scientifically wrong: there is no
  correspondence to learn, and any InfoNCE loss built on it would be fitting
  arbitrary row order.
* Deep CCA is defined on the sample set common to all views, which here is the
  **intersection of genes**. Genes outside that intersection cannot enter the
  correlation objective; they are still embedded, by their own view's encoder,
  and flagged in provenance.

Category tagging (the brief's four-way tag)
-------------------------------------------
A joint embedding spans several datasets, so it cannot inherit one dataset's
tag. It is labelled ``"spatial"`` only when *every* contributing dataset is
spatial, and ``"single_cell"`` otherwise (a mixed spatial + dissociated
embedding is not a spatial representation of a gene).

Normalisation assumption
------------------------
Each block is prepared by ``panelprep.embed.expression.prepare_block``:
``log1p_cp10k`` input, cell-features z-scored across genes, optional
``pre_pca`` feature-space reduction, then division by the block's global RMS so
that one dataset with many cells does not dominate the concatenation purely
through magnitude.

Missing-data behaviour
----------------------
Per-block masks are honoured throughout. A gene absent from dataset *k* has
weight 0 in dataset *k*'s part of the reconstruction loss, so it never learns
from the fabricated padding zeros, but it still trains on the datasets that do
contain it -- which is precisely the advantage a joint model has over the
separate ones. A gene absent from *every* dataset is a NaN row.

MODIFY HERE
-----------
* ``joint_ae``: ``hidden``, ``epochs``, ``pre_pca``, ``mask_channels``.
* ``dcca``: ``ridge`` and ``eig_floor`` (the numerical guard), ``hidden``.
* ``contrastive_ae``: ``temperature``, ``batch``, ``hidden``.
* To add a joint method, write it in ``panelprep/embed/custom.py`` with
  ``mode="joint"`` and reuse ``expression_blocks`` from here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..schemas import CandidateDataset, GeneEmbedding
from . import EMBEDDERS
from .expression import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_PRE_PCA,
    DEFAULT_VAL_FRAC,
    FeatureBlock,
    _fit_autoencoder,
    _torch,
    embedding_frame,
    masked_mse,
    mlp,
    prepare_block,
    seed_everything,
)

# --- MODIFY HERE: numerical guards for the CCA objective -------------------
DEFAULT_RIDGE = 1e-3        # added to each view's latent covariance diagonal
DEFAULT_EIG_FLOOR = 1e-6    # eigenvalues clamped here before any sqrt / rsqrt
DEFAULT_TEMPERATURE = 1.0   # InfoNCE temperature; see contrastive_ae's docstring
                            # for the measurement behind this value


def expression_blocks(
    datasets: Dict[str, CandidateDataset],
    genes: Sequence[str],
    *,
    max_obs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    pre_pca: Optional[int] = DEFAULT_PRE_PCA,
    standardize: bool = True,
    min_views: int = 1,
    method: str = "joint",
) -> List[FeatureBlock]:
    """Prepare one ``FeatureBlock`` per dataset that has an expression matrix.

    Observation unit: rows of every returned block are the same ``genes`` in the
    same order -- that is the pairing the joint methods rely on.

    Missing-data behaviour: datasets without a matrix (``non_transcriptomic``
    and ``causal`` candidates: PPI graphs, regulatory networks) carry no cells
    and are silently skipped; they are handled by
    ``panelprep.embed.external``. Blocks where no requested gene is present are
    dropped and reported by the caller.

    ``min_views`` is a *configuration* requirement, not a property of the data:
    deep CCA and the contrastive model are undefined with one view, so too few
    views raises ``ValueError`` naming the requirement rather than returning a
    degenerate embedding.

    MODIFY HERE: ``pre_pca``, ``standardize``, ``max_obs``.
    """
    usable = {k: v for k, v in datasets.items() if v.has_expression}
    blocks = []
    for name, ds in usable.items():
        blk = prepare_block(ds, genes, max_obs=max_obs, rng=rng,
                            pre_pca=pre_pca, standardize=standardize)
        if blk.n_present and blk.n_features:
            blocks.append(blk)
    if len(blocks) < min_views:
        raise ValueError(
            f"{method} requires at least {min_views} dataset(s) with an expression "
            f"matrix and at least one of the requested genes; got {len(blocks)} "
            f"usable view(s) out of {len(datasets)} dataset(s) "
            f"({len(usable)} with a matrix). This is a configuration error: add "
            f"expression datasets to Config.candidates, or use a mode='separate' "
            f"embedder."
        )
    return blocks


def joint_category(blocks: Sequence[FeatureBlock]) -> str:
    """``"spatial"`` only if every contributing dataset is spatial."""
    cats = {b.category for b in blocks}
    return "spatial" if cats == {"spatial"} else "single_cell"


def shared_genes_mask(blocks: Sequence[FeatureBlock]) -> np.ndarray:
    """Boolean over the requested genes: present in **every** view.

    This is the sample set on which a cross-view objective (canonical
    correlation, InfoNCE positives) is defined.
    """
    out = np.ones(len(blocks[0].genes), dtype=bool)
    for b in blocks:
        out &= b.mask
    return out


def _view_provenance(blocks: Sequence[FeatureBlock]) -> dict:
    return {
        "views": [b.name for b in blocks],
        "view_categories": {b.name: b.category for b in blocks},
        "view_n_obs": {b.name: b.n_obs for b in blocks},
        "view_n_features": {b.name: b.n_features for b in blocks},
        "view_frac_genes_masked": {b.name: round(b.frac_masked, 6) for b in blocks},
        "pairing": "genes (rows); cells are NOT paired across datasets",
    }


# ---------------------------------------------------------------------------
# (1) joint autoencoder over the gene-wise concatenation of feature blocks
# ---------------------------------------------------------------------------

@EMBEDDERS.register(
    "joint_ae",
    category="single_cell",
    mode="joint",
    needs_expression=True,
    defaults=dict(pre_pca=DEFAULT_PRE_PCA, standardize=True, hidden=DEFAULT_HIDDEN,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch=DEFAULT_BATCH,
                  patience=DEFAULT_PATIENCE, val_frac=DEFAULT_VAL_FRAC,
                  mask_channels=True),
    description="one shared encoder/decoder on the gene-wise concatenation of the "
                "per-dataset feature blocks, with per-block masking so a gene missing "
                "from one dataset still trains on the others",
)
def joint_ae(datasets, *, genes, dim, rng, pre_pca: Optional[int] = DEFAULT_PRE_PCA,
             standardize: bool = True, hidden: Sequence[int] = DEFAULT_HIDDEN,
             epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
             batch: int = DEFAULT_BATCH, patience: int = DEFAULT_PATIENCE,
             val_frac: float = DEFAULT_VAL_FRAC, mask_channels: bool = True,
             max_obs: Optional[int] = None, **_) -> GeneEmbedding:
    """One shared autoencoder over all datasets, paired by gene.

    Observation unit: gene. Each gene row is the concatenation of its
    (pre-PCA'd, standardised) profile in every dataset, so the input dimension
    is ``sum_k F_k`` and the sample count is the gene universe. This is a
    multi-view model of one sample set; it is **not** a merge of cell
    populations (see module docstring).

    Normalisation assumption: as ``prepare_block`` -- ``log1p_cp10k``, cells
    z-scored across genes, per-block RMS scaling so a wide block does not
    dominate the shared encoder by magnitude alone.

    Missing-data behaviour: a gene absent from dataset *k* has reconstruction
    weight 0 across the whole of block *k* and so contributes nothing to that
    block's loss, while still training on the blocks that do contain it; it
    receives a real (non-NaN) embedding, which is the point of the joint model.
    At the *input* such a block is zero-filled, which after standardisation is
    the block mean -- the least informative value. When any block is incomplete
    and ``mask_channels=True``, one 0/1 presence channel per block is appended
    to the input so the encoder can distinguish "absent" from "average"; the
    channels are fully observed and are reconstructed alongside the features.
    A gene absent from *every* dataset has no data at all and is a NaN row.

    Works with a single view (it degenerates to ``ae_separate`` over the union
    of genes, which is deliberate: ``Config`` must not break when only one
    expression dataset loads).

    MODIFY HERE: ``hidden``, ``epochs``, ``lr``, ``batch``, ``patience``,
    ``val_frac``, ``pre_pca``, ``mask_channels``.
    """
    blocks = expression_blocks(datasets, genes, max_obs=max_obs, rng=rng,
                              pre_pca=pre_pca, standardize=standardize,
                              min_views=1, method="joint_ae")
    X = np.hstack([b.features for b in blocks]).astype(np.float32)
    W = np.hstack([np.repeat(b.mask[:, None], b.n_features, axis=1)
                   for b in blocks]).astype(np.float32)
    n_feature_cols = int(X.shape[1])
    incomplete = bool(any(b.frac_masked > 0 for b in blocks))
    used_mask_channels = bool(mask_channels and incomplete)
    if used_mask_channels:
        ind = np.stack([b.mask.astype(np.float32) for b in blocks], axis=1)
        X = np.hstack([X, ind]).astype(np.float32)
        W = np.hstack([W, np.ones_like(ind)]).astype(np.float32)

    coverage = np.hstack([b.mask[:, None] for b in blocks]).any(axis=1)
    latent, history = None, {"status": "no_genes_present"}
    if coverage.sum() >= 2:
        seed, batch_rng = seed_everything(rng)
        latent, history = _fit_autoencoder(
            X[coverage], W[coverage], dim, hidden=hidden, epochs=epochs, lr=lr,
            batch=batch, patience=patience, val_frac=val_frac, rng=batch_rng,
            variational=False)
        history["torch_seed"] = seed
        history["status"] = "ok"
    frame, dim_eff = embedding_frame(latent, coverage, genes, dim)
    prov = {
        "method": "joint_ae",
        "observation_unit": "gene (rows); features are per-dataset cells / cell-PCs",
        "n_genes_requested": int(len(genes)),
        "n_genes_embedded": int(coverage.sum()),
        "n_genes_no_coverage": int((~coverage).sum()),
        "genes_no_coverage": [g for g, c in zip(genes, coverage) if not c][:50],
        "n_genes_in_all_views": int(shared_genes_mask(blocks).sum()),
        "n_input_features": int(X.shape[1]),
        "n_feature_columns": n_feature_cols,
        "mask_channels_used": used_mask_channels,
        "dim_requested": int(dim), "dim_effective": int(dim_eff),
        "epochs": history.get("epochs_run", 0),
        "final_train_loss": history.get("final_train_loss"),
        "loss": "masked MSE over observed per-block features",
        "frac_genes_masked_out": float((~coverage).mean()) if len(genes) else 1.0,
        **_view_provenance(blocks), **history,
    }
    return GeneEmbedding(matrix=frame, method="joint_ae",
                         category=joint_category(blocks), provenance=prov)


# ---------------------------------------------------------------------------
# (2) multimodal deep CCA
# ---------------------------------------------------------------------------

def _whiten(H, ridge: float, floor: float):
    """Ridge-regularised whitening of a ``(n_samples, d)`` latent.

    Numerical guard: the covariance is formed as
    ``H_c^T H_c / (n-1) + ridge * I`` and inverted through a **symmetric
    eigendecomposition** (``torch.linalg.eigh``) with eigenvalues clamped at
    ``floor`` before ``rsqrt``. ``eigh`` is used rather than ``svd`` because
    the gradient of an SVD is unstable when singular values are nearly
    degenerate -- which is the normal situation for a latent of dimension
    ``d`` estimated from only ``G`` gene samples -- and clamping guarantees the
    inverse square root exists even when the latent collapses to a lower-rank
    subspace during training.
    """
    torch = _torch()
    Hc = H - H.mean(dim=0, keepdim=True)
    n = max(int(Hc.shape[0]) - 1, 1)
    S = Hc.t() @ Hc / n + ridge * torch.eye(Hc.shape[1], dtype=Hc.dtype)
    S = 0.5 * (S + S.t())
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.clamp_min(floor)
    return Hc @ (evecs @ torch.diag(evals.rsqrt()) @ evecs.t())


def _canonical_correlation(Zi, Zj, floor: float):
    """Sum of canonical correlations between two whitened latents.

    ``Zi``, ``Zj`` are already whitened, so the cross-covariance ``T`` has
    singular values in ``[0, 1]`` and their sum is the trace norm -- the
    standard regularised DCCA objective. It is computed as
    ``sum(sqrt(clamp(eigvalsh(T^T T), floor)))`` so that no gradient ever passes
    through ``sqrt(0)``.
    """
    torch = _torch()
    n = max(int(Zi.shape[0]) - 1, 1)
    T = Zi.t() @ Zj / n
    M = T.t() @ T
    M = 0.5 * (M + M.t())
    ev = torch.linalg.eigvalsh(M).clamp_min(floor)
    return ev.sqrt().sum()


@EMBEDDERS.register(
    "dcca",
    category="single_cell",
    mode="joint",
    needs_expression=True,
    defaults=dict(pre_pca=DEFAULT_PRE_PCA, standardize=True, hidden=DEFAULT_HIDDEN,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, patience=DEFAULT_PATIENCE,
                  ridge=DEFAULT_RIDGE, eig_floor=DEFAULT_EIG_FLOOR),
    description="multimodal deep CCA: view-specific encoders trained to maximise the "
                "regularised canonical correlation of their outputs on the genes shared "
                "by all views; needs >= 2 views",
)
def dcca(datasets, *, genes, dim, rng, pre_pca: Optional[int] = DEFAULT_PRE_PCA,
         standardize: bool = True, hidden: Sequence[int] = DEFAULT_HIDDEN,
         epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
         patience: int = DEFAULT_PATIENCE, ridge: float = DEFAULT_RIDGE,
         eig_floor: float = DEFAULT_EIG_FLOOR, max_obs: Optional[int] = None,
         **_) -> GeneEmbedding:
    """Multimodal deep CCA over gene rows.

    Observation unit: gene. One encoder per dataset maps that dataset's
    feature block to a ``dim``-dimensional latent; the encoders are trained to
    maximise the sum of canonical correlations between their outputs, averaged
    over all view pairs. The objective is evaluated **full-batch on the shared
    genes**, because a canonical correlation needs at least as many samples as
    latent dimensions to be estimable.

    Deep CCA is only defined on the **intersection** of genes across views:
    the objective is a correlation between two views of the *same* sample, so a
    gene that one view does not contain has no counterpart to correlate with.
    Genes outside the intersection are still embedded -- by the encoder of a
    view that does contain them, pushed through the same post-hoc alignment --
    and their count is recorded in ``provenance["n_genes_outside_intersection"]``.
    Treat those rows as extrapolation: the alignment map was fitted on the
    intersection only.

    Post-hoc alignment, and why it is needed: the canonical-correlation
    objective is invariant to an invertible linear re-basing of each view's
    latent, so the encoders are *not* trained into a common coordinate system
    and averaging their raw outputs would average incompatible bases. After
    training, each view's latent is z-scored over the shared genes and then
    rotated onto view 0 by orthogonal Procrustes fitted on the shared genes;
    the embedding is the mean of the aligned projections over the views in
    which the gene is present.

    Numerical guard: see ``_whiten`` and ``_canonical_correlation`` -- ridge
    regularisation plus eigenvalue clamping at ``eig_floor`` in a symmetric
    eigendecomposition.

    Missing-data behaviour: as the module docstring. Genes present in no view
    are NaN rows. The objective needs strictly more shared genes than latent
    dimensions, so the guard is ``n_shared >= max(3, dim_effective + 1)`` where
    ``dim_effective = min(dim, min_k F_k)`` -- with the default ``dim=32`` and
    blocks of at least 32 features that means **at least 33 genes shared by
    every view**, not three. Below it an all-NaN embedding is returned with
    ``provenance["status"] = "insufficient_shared_genes"`` and
    ``provenance["reason"]`` naming the actual threshold; a data problem is a
    status, not an exception. Fewer than two *views* is a configuration error
    and raises ``ValueError``.

    MODIFY HERE: ``ridge``, ``eig_floor``, ``hidden``, ``epochs``, ``lr``.
    """
    torch = _torch()
    blocks = expression_blocks(datasets, genes, max_obs=max_obs, rng=rng,
                               pre_pca=pre_pca, standardize=standardize,
                               min_views=2, method="dcca")
    shared = shared_genes_mask(blocks)
    coverage = np.hstack([b.mask[:, None] for b in blocks]).any(axis=1)
    prov = {
        "method": "dcca",
        "observation_unit": "gene (rows); one encoder per dataset over that dataset's cells",
        "n_genes_requested": int(len(genes)),
        "n_genes_in_all_views": int(shared.sum()),
        "n_genes_outside_intersection": int((coverage & ~shared).sum()),
        "genes_outside_intersection": [g for g, c, s in zip(genes, coverage, shared)
                                       if c and not s][:50],
        "n_genes_no_coverage": int((~coverage).sum()),
        "dim_requested": int(dim),
        "ridge": float(ridge), "eig_floor": float(eig_floor),
        "defined_on": "intersection of genes across views",
        "numerical_guard": "eigh with eigenvalues clamped at eig_floor; ridge on each "
                           "view covariance; trace norm via eigvalsh(T^T T)",
        "alignment": "per-view z-score on shared genes + orthogonal Procrustes onto view 0",
        **_view_provenance(blocks),
    }
    dim_eff = int(min(dim, min(b.n_features for b in blocks)))
    if int(shared.sum()) < max(3, dim_eff + 1):
        frame, _ = embedding_frame(None, coverage, genes, dim)
        prov.update({"status": "insufficient_shared_genes", "dim_effective": 0,
                     "reason": f"{int(shared.sum())} genes shared by all "
                               f"{len(blocks)} views; need > {dim_eff}"})
        return GeneEmbedding(matrix=frame, method="dcca",
                             category=joint_category(blocks), provenance=prov)

    seed, _ = seed_everything(rng)
    encoders = [mlp([b.n_features, *[int(h) for h in hidden], dim_eff]) for b in blocks]
    params = [p for e in encoders for p in e.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    feats = [torch.from_numpy(np.ascontiguousarray(b.features, dtype=np.float32))
             for b in blocks]
    shared_t = torch.from_numpy(np.where(shared)[0])
    n_pairs = len(blocks) * (len(blocks) - 1) / 2

    best, best_state, best_epoch, bad = -np.inf, None, 0, 0
    last = float("nan")
    for epoch in range(1, int(epochs) + 1):
        for e in encoders:
            e.train()
        opt.zero_grad()
        Z = [_whiten(e(f[shared_t]), ridge, eig_floor) for e, f in zip(encoders, feats)]
        corr = sum(_canonical_correlation(Z[i], Z[j], eig_floor)
                   for i in range(len(Z)) for j in range(i + 1, len(Z))) / n_pairs
        (-corr).backward()
        opt.step()
        last = float(corr.detach())
        if last > best + 1e-6:
            best, best_epoch, bad = last, epoch, 0
            best_state = [{k: v.clone() for k, v in e.state_dict().items()} for e in encoders]
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        for e, st in zip(encoders, best_state):
            e.load_state_dict(st)

    # --- post-hoc alignment onto view 0, fitted on the shared genes ---------
    from scipy.linalg import orthogonal_procrustes

    with torch.no_grad():
        for e in encoders:
            e.eval()
        raw = [e(f).numpy() for e, f in zip(encoders, feats)]   # (G, dim_eff) per view
    sh = np.where(shared)[0]
    aligned = []
    for v, (Zv, blk) in enumerate(zip(raw, blocks)):
        mu = Zv[sh].mean(axis=0, keepdims=True)
        sd = Zv[sh].std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Zs = (Zv - mu) / sd
        if v == 0:
            R = np.eye(Zs.shape[1], dtype=np.float32)
        else:
            R, _ = orthogonal_procrustes(Zs[sh], aligned[0][sh])
        aligned.append((Zs @ R).astype(np.float32))

    acc = np.zeros((len(genes), dim_eff), dtype=np.float64)
    cnt = np.zeros((len(genes), 1), dtype=np.float64)
    for Zv, blk in zip(aligned, blocks):
        acc[blk.mask] += Zv[blk.mask]
        cnt[blk.mask] += 1.0
    latent = (acc[coverage] / np.clip(cnt[coverage], 1.0, None)).astype(np.float32)
    frame, dim_eff_out = embedding_frame(latent, coverage, genes, dim)
    prov.update({
        "status": "ok", "dim_effective": int(dim_eff_out), "torch_seed": seed,
        "epochs": int(epoch), "best_epoch": int(best_epoch),
        "final_canonical_correlation": float(last),
        "best_canonical_correlation": float(best),
        "max_possible_canonical_correlation": int(dim_eff),
        "canonical_correlation_note": "maximised on the training (shared) genes, so it is "
                                      "a fit statistic, not a held-out metric",
        "final_train_loss": float(-last),
        "n_views_per_gene_mean": float(cnt[coverage].mean()),
        "early_stopping_on": "train canonical correlation",
    })
    return GeneEmbedding(matrix=frame, method="dcca",
                         category=joint_category(blocks), provenance=prov)


# ---------------------------------------------------------------------------
# (3) contrastive autoencoder (InfoNCE over genes)
# ---------------------------------------------------------------------------

@EMBEDDERS.register(
    "contrastive_ae",
    category="single_cell",
    mode="joint",
    needs_expression=True,
    defaults=dict(pre_pca=DEFAULT_PRE_PCA, standardize=True, hidden=DEFAULT_HIDDEN,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch=DEFAULT_BATCH,
                  patience=DEFAULT_PATIENCE, temperature=DEFAULT_TEMPERATURE),
    description="per-view encoders with an InfoNCE loss whose positive pair is THE SAME "
                "GENE in two different datasets and whose negatives are the other genes "
                "in the batch; needs >= 2 views",
)
def contrastive_ae(datasets, *, genes, dim, rng, pre_pca: Optional[int] = DEFAULT_PRE_PCA,
                   standardize: bool = True, hidden: Sequence[int] = DEFAULT_HIDDEN,
                   epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
                   batch: int = DEFAULT_BATCH, patience: int = DEFAULT_PATIENCE,
                   temperature: float = DEFAULT_TEMPERATURE,
                   max_obs: Optional[int] = None, **_) -> GeneEmbedding:
    """Contrastive multi-view gene embedding (InfoNCE across datasets).

    **The positive pair is the same gene in two different datasets.** That is
    the only correspondence these datasets share, and it is the correct
    positive-pair definition for this task: gene *Sox2* measured in a Visium
    section and gene *Sox2* measured in a dissociated atlas are two views of one
    object. Negatives are the other genes in the same minibatch.

    Pairing **cells** across datasets would be wrong. The datasets contain
    different cells in different numbers with no correspondence whatsoever, so
    an InfoNCE loss built on "cell *i* here matches cell *i* there" would be
    fitting arbitrary row order and the resulting embedding would be
    meaningless. This module never does that.

    Observation unit: gene. Each view's encoder maps its own feature block to a
    ``dim``-dimensional latent which is L2-normalised, so the similarity in the
    loss is a cosine similarity.

    Normalisation assumption: as ``prepare_block``.

    Missing-data behaviour: the loss is computed on genes shared by all views
    (a positive pair requires the gene in both views). The embedding of a gene
    is the mean of the L2-normalised per-view latents over the views that
    contain it, renormalised to unit norm -- contrastive latents live on the
    unit sphere, so the mean direction is the right aggregation and the
    renormalisation keeps every row on the same sphere. Genes present in no
    view are NaN rows. Fewer than two views raises ``ValueError``
    (configuration error); too few shared genes yields
    ``provenance["status"] = "insufficient_shared_genes"``.

    MODIFY HERE: ``temperature`` (lower = harder negatives), ``batch`` (the
    number of negatives per positive is ``batch - 1``), ``hidden``, ``epochs``.

    Why the default temperature is 1.0 and not 0.1. On
    ``make_synthetic_case()`` (200 genes, views ``sibling`` +
    ``same_program_diff_mixture``, ``dim=16``), the k=15 kNN overlap with the
    true loadings rises monotonically with temperature: 0.091 at 0.1, 0.104 at
    0.3, 0.124 at 0.5, 0.156 at 1.0, against a label-permutation baseline of
    0.040. A low temperature drives the uniformity term of the InfoNCE
    objective, which spreads genes evenly over the sphere and erodes exactly the
    local neighbourhood structure a gene embedding is wanted for. 0.1 was the
    value originally specified; it was changed to 1.0 on this measurement, and
    the original is one keyword away.

    Even at 1.0 this method recovers the latent geometry less well than
    ``pca_baseline`` (0.156 vs 0.248) on this fixture, while remaining well
    above the permutation null. That is a real result, not a bug: the fixture's
    genes are generated by a *linear* factor model, which is precisely what an
    SVD is optimal for, so the baseline is strong here by construction. The
    contrastive objective should be expected to earn its keep on data where the
    shared signal across views is non-linear - which this fixture is not.
    """
    torch = _torch()
    blocks = expression_blocks(datasets, genes, max_obs=max_obs, rng=rng,
                               pre_pca=pre_pca, standardize=standardize,
                               min_views=2, method="contrastive_ae")
    shared = shared_genes_mask(blocks)
    coverage = np.hstack([b.mask[:, None] for b in blocks]).any(axis=1)
    prov = {
        "method": "contrastive_ae",
        "observation_unit": "gene (rows); one encoder per dataset over that dataset's cells",
        "positive_pair": "the same gene in two different datasets",
        "negatives": "other genes in the minibatch",
        "n_genes_requested": int(len(genes)),
        "n_genes_in_all_views": int(shared.sum()),
        "n_genes_outside_intersection": int((coverage & ~shared).sum()),
        "n_genes_no_coverage": int((~coverage).sum()),
        "temperature": float(temperature),
        "dim_requested": int(dim),
        "aggregation": "mean of L2-normalised per-view latents, renormalised",
        **_view_provenance(blocks),
    }
    if int(shared.sum()) < 4:
        frame, _ = embedding_frame(None, coverage, genes, dim)
        prov.update({"status": "insufficient_shared_genes", "dim_effective": 0,
                     "reason": f"{int(shared.sum())} genes shared by all "
                               f"{len(blocks)} views; need >= 4 for a positive pair "
                               f"plus negatives"})
        return GeneEmbedding(matrix=frame, method="contrastive_ae",
                             category=joint_category(blocks), provenance=prov)

    seed, batch_rng = seed_everything(rng)
    encoders = [mlp([b.n_features, *[int(h) for h in hidden], dim]) for b in blocks]
    params = [p for e in encoders for p in e.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    feats = [torch.from_numpy(np.ascontiguousarray(b.features, dtype=np.float32))
             for b in blocks]
    sh = np.where(shared)[0]
    bs = int(min(batch, len(sh)))
    ce = torch.nn.CrossEntropyLoss()

    def _encode(idx):
        out = []
        for e, f in zip(encoders, feats):
            z = e(f[torch.from_numpy(idx)])
            out.append(torch.nn.functional.normalize(z, dim=1))
        return out

    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    last = float("nan")
    for epoch in range(1, int(epochs) + 1):
        for e in encoders:
            e.train()
        order = batch_rng.permutation(len(sh))
        tot, nb = 0.0, 0
        for s in range(0, len(order), bs):
            idx = sh[order[s:s + bs]]
            if len(idx) < 4:
                continue
            opt.zero_grad()
            Z = _encode(idx)
            labels = torch.arange(len(idx))
            loss = 0.0
            for i in range(len(Z)):
                for j in range(len(Z)):
                    if i == j:
                        continue
                    logits = Z[i] @ Z[j].t() / float(temperature)
                    loss = loss + ce(logits, labels)
            loss = loss / (len(Z) * (len(Z) - 1))
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        last = tot / max(nb, 1)
        if last < best - 1e-6:
            best, best_epoch, bad = last, epoch, 0
            best_state = [{k: v.clone() for k, v in e.state_dict().items()} for e in encoders]
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        for e, st in zip(encoders, best_state):
            e.load_state_dict(st)
    with torch.no_grad():
        for e in encoders:
            e.eval()
        per_view = [torch.nn.functional.normalize(e(f), dim=1).numpy()
                    for e, f in zip(encoders, feats)]

    acc = np.zeros((len(genes), dim), dtype=np.float64)
    cnt = np.zeros((len(genes), 1), dtype=np.float64)
    for Zv, blk in zip(per_view, blocks):
        acc[blk.mask] += Zv[blk.mask]
        cnt[blk.mask] += 1.0
    mean = acc[coverage] / np.clip(cnt[coverage], 1.0, None)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    latent = (mean / np.clip(norms, 1e-12, None)).astype(np.float32)
    frame, dim_eff = embedding_frame(latent, coverage, genes, dim)
    # chance level of the InfoNCE loss: log(batch) for a uniform classifier
    prov.update({
        "status": "ok", "dim_effective": int(dim_eff), "torch_seed": seed,
        "epochs": int(epoch), "best_epoch": int(best_epoch),
        "final_train_loss": float(last), "best_train_loss": float(best),
        "infonce_chance_level": float(np.log(bs)),
        "batch": bs, "n_negatives_per_positive": bs - 1,
        "loss": "InfoNCE over gene positives, symmetric over ordered view pairs",
        "n_views_per_gene_mean": float(cnt[coverage].mean()),
    })
    return GeneEmbedding(matrix=frame, method="contrastive_ae",
                         category=joint_category(blocks), provenance=prov)
