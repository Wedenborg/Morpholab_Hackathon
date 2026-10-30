"""Per-dataset gene embeddings: PCA baseline, autoencoder, variational autoencoder.

The observation unit (the thing that is easy to get wrong here)
--------------------------------------------------------------
The brief asks for a ``G x d`` gene embedding built from a ``G x N_k``
matrix, so in this module **the gene is the observation and the N_k cells are
the features**. The autoencoder trains on ``G`` samples of dimension ``N_k`` --
not on ``N_k`` samples of dimension ``G``, which is what a cell-embedding model
such as scVI does. Three consequences follow, and all three are implemented
rather than assumed away:

1. Two datasets have **different, non-comparable feature spaces**: the columns
   are different cells, and cell 7 of one experiment has nothing to do with
   cell 7 of another. A model fitted on dataset A therefore cannot be applied
   to dataset B, which is exactly why these embedders are ``mode="separate"``
   (the runner calls them once per dataset and keys the output
   ``"<embedder>::<dataset>"``). Methods that use several datasets at once must
   pair them by *gene*, which is what ``panelprep.embed.joint`` does.
2. **Sequencing depth would dominate the geometry.** A deeply sequenced cell
   has a larger value for nearly every gene, so the leading direction of
   variation across gene-rows is library size, not biology. Each cell-feature
   is therefore z-scored *across genes* (column-wise, ``standardize=True``).
   Note that this is a column operation on the ``G x N`` matrix, i.e. it
   standardises cells, and it is applied after the loader has already put the
   matrix on the ``log1p_cp10k`` contract.
3. ``N_k`` routinely exceeds ``G``. ``pre_pca`` (default
   ``min(256, N_k, G_present)``) projects the **cells** onto principal
   components before training. This is a *feature-space* reduction: it changes
   what the network reads, never the number of genes. The embedding still has
   exactly one row per requested gene, and the gene index is never touched.

Normalisation assumption
------------------------
Input is expected on ``schemas.CANONICAL_UNITS`` (``log1p_cp10k``). These
embedders are not units-sensitive in the way distributional scores are -- each
dataset is standardised independently and never compared on a raw scale -- but
a dataset left on raw counts would give the column z-score a heavy-tailed
input, so the loaders' normalisation is still assumed.

Missing-data behaviour
----------------------
``gene_by_cell`` returns a presence mask. A gene absent from the dataset has no
measurement here at all, so it is **excluded from the loss entirely** (never
learned from the fabricated zeros used to pad the matrix) and comes out as a
**NaN row** in the embedding. Filling such a gene in is the job of the joint
methods, which can borrow the datasets that do contain it. If no requested gene
is present, the embedder returns an all-NaN matrix with
``provenance["status"] = "no_genes_present"`` -- a data problem is a status,
not an exception.

MODIFY HERE
-----------
* ``DEFAULT_*`` below: architecture and optimisation defaults for every deep
  method in this module (``hidden``, ``epochs``, ``lr``, ``batch``, patience).
* ``prepare_block(standardize=..., pre_pca=...)``: turn off the cell z-score or
  change the feature-space reduction width. ``pre_pca=None`` trains on raw
  cells.
* ``_fit_autoencoder(val_frac=...)``: the gene hold-out used for early
  stopping. Genes, not cells, are held out -- they are the observations.
* To add a per-dataset embedder, write a function in
  ``panelprep/embed/custom.py`` with ``mode="separate"``; it can import
  ``prepare_block`` and ``_fit_autoencoder`` from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from ..schemas import CandidateDataset, GeneEmbedding
from . import EMBEDDERS, gene_by_cell

# --- MODIFY HERE: defaults shared by every deep method in this module ------
DEFAULT_PRE_PCA = 256       # feature-space (cell) reduction width
DEFAULT_HIDDEN = (256,)     # dense encoder widths; decoder mirrors it
DEFAULT_EPOCHS = 200
DEFAULT_LR = 1e-3
DEFAULT_BATCH = 128         # genes per minibatch
DEFAULT_PATIENCE = 20       # early-stopping patience in epochs
DEFAULT_VAL_FRAC = 0.1      # fraction of *genes* held out for early stopping
MIN_VAL_GENES = 10          # below this, early-stop on the train loss instead
LOGVAR_CLAMP = (-10.0, 10.0)


def embedding_columns(dim: int) -> list[str]:
    """Column names for a ``(G, d)`` embedding frame.

    Strings, not integers, because ``GeneEmbedding.to_parquet`` requires string
    column names. Zero-padded so lexical order is component order.
    """
    return [f"d{i:02d}" for i in range(dim)]


# ---------------------------------------------------------------------------
# feature preparation -- shared with panelprep.embed.joint
# ---------------------------------------------------------------------------

@dataclass
class FeatureBlock:
    """One dataset's gene-wise feature matrix, ready for a model.

    Observation unit: a **row is a gene**, a column is a cell-derived feature
    (a cell, or a principal component of cells when ``pre_pca`` is active).
    ``features`` has one row for every *requested* gene, in the requested
    order, with zeros in the rows of genes this dataset does not contain;
    ``mask`` says which rows are real. Consumers must use the mask -- the zeros
    are padding, not measurements.
    """

    name: str
    category: str
    features: np.ndarray            # (G, F), zero-filled at absent genes
    mask: np.ndarray                # (G,) True where the gene exists here
    genes: list
    n_obs: int                      # cells actually used (after max_obs cap)
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def n_present(self) -> int:
        return int(self.mask.sum())

    @property
    def frac_masked(self) -> float:
        """Fraction of requested genes absent from this dataset."""
        return float(1.0 - self.mask.mean()) if self.mask.size else 1.0

    def present_features(self) -> np.ndarray:
        """Just the rows that are real measurements."""
        return self.features[self.mask]


def prepare_block(
    dataset: CandidateDataset,
    genes: Sequence[str],
    *,
    max_obs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    pre_pca: Optional[int] = DEFAULT_PRE_PCA,
    standardize: bool = True,
    seed: int = 0,
) -> FeatureBlock:
    """Build the ``G x F`` feature matrix one model trains on.

    Observation unit: rows are genes (the samples), columns are cells or
    principal components of cells (the features). Nothing here transposes
    implicitly -- ``gene_by_cell`` does the one transpose, from the stored
    observations x genes matrix.

    Normalisation assumption: ``dataset.expression`` is on
    ``log1p_cp10k``. On top of that, ``standardize=True`` z-scores each
    cell-feature across the *present* genes only, which removes the library-size
    direction (see module docstring, point 2). Constant columns are set to zero
    rather than divided by zero.

    Missing-data behaviour: genes absent from the dataset are zero rows and
    ``mask=False``; all statistics (column mean/sd, the PCA fit, the global
    scale) are computed on present genes only, so padding never influences the
    geometry.

    After the optional ``pre_pca`` the whole block is divided by its global RMS,
    so blocks from datasets with different feature counts enter a joint model
    with comparable magnitude while the *relative* importance of components is
    preserved (per-component standardisation would inflate trailing noise
    components, so it is deliberately not done).

    MODIFY HERE: ``pre_pca`` (``None`` = train on raw cells), ``standardize``,
    and ``max_obs`` (cells subsampled per dataset).
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    frame, mask = gene_by_cell(dataset, genes, max_obs=max_obs, rng=rng)
    arr = frame.to_numpy(dtype=np.float32)
    n_obs = int(arr.shape[1])
    info: Dict[str, object] = {
        "n_obs_available": int(dataset.n_obs),
        "n_obs_used": n_obs,
        "n_genes_requested": int(len(genes)),
        "n_genes_present": int(mask.sum()),
        "frac_genes_masked": float(1.0 - mask.mean()) if mask.size else 1.0,
        "standardized": bool(standardize),
        "units": dataset.units,
        "obs_unit": dataset.obs_unit,
    }
    if not mask.any():
        info["status"] = "no_genes_present"
        return FeatureBlock(name=dataset.name, category=dataset.category,
                            features=np.zeros((len(genes), 0), dtype=np.float32),
                            mask=mask, genes=list(genes), n_obs=n_obs, info=info)

    present = arr[mask]                                   # (G_present, N)
    if standardize:
        mu = present.mean(axis=0, keepdims=True)
        sd = present.std(axis=0, keepdims=True)
        n_const = int((sd < 1e-8).sum())
        sd = np.where(sd < 1e-8, 1.0, sd)
        present = (present - mu) / sd
        info["n_constant_cells"] = n_const

    k = None
    if pre_pca:
        k = int(min(pre_pca, present.shape[1], present.shape[0]))
        if k >= 1 and k < present.shape[1]:
            pca = PCA(n_components=k, svd_solver="randomized",
                      random_state=int(rng.integers(0, 2**31 - 1)))
            present = pca.fit_transform(present).astype(np.float32)
            info["pre_pca_components"] = k
            info["pre_pca_explained_variance_ratio"] = float(
                pca.explained_variance_ratio_.sum())
        else:
            info["pre_pca_components"] = int(present.shape[1])
            info["pre_pca_explained_variance_ratio"] = 1.0
    else:
        info["pre_pca_components"] = int(present.shape[1])

    rms = float(np.sqrt(np.mean(present.astype(np.float64) ** 2)))
    if rms > 0:
        present = (present / rms).astype(np.float32)
    info["rms_before_scaling"] = rms

    features = np.zeros((len(genes), present.shape[1]), dtype=np.float32)
    features[mask] = present
    return FeatureBlock(name=dataset.name, category=dataset.category,
                        features=features, mask=mask, genes=list(genes),
                        n_obs=n_obs, info=info)


def single_dataset(datasets: Dict[str, CandidateDataset]) -> Tuple[str, CandidateDataset]:
    """Unwrap the one dataset a ``mode="separate"`` embedder was handed.

    ``run_embedders`` guarantees a single-entry dict for separate mode; calling
    one of these functions directly with several datasets is a *configuration*
    error (the model's features are that dataset's cells), so it raises.
    """
    usable = {k: v for k, v in datasets.items() if v.has_expression}
    if len(usable) != 1:
        raise ValueError(
            f"separate-mode embedders take exactly one dataset with an expression "
            f"matrix; got {len(usable)} ({sorted(usable)}). Cells are not comparable "
            f"across datasets, so use a joint embedder (panelprep.embed.joint) to "
            f"combine datasets -- it pairs them by gene, not by cell."
        )
    return next(iter(usable.items()))


def embedding_frame(latent: Optional[np.ndarray], mask: np.ndarray,
                    genes: Sequence[str], dim: int) -> Tuple[pd.DataFrame, int]:
    """Assemble the ``(G, d)`` frame: NaN rows for genes with no data.

    Returns ``(frame, dim_effective)``. If a method can only produce fewer than
    ``dim`` components (truncated SVD of a low-rank block), the extra columns
    are filled with ``0.0`` for present genes -- that component genuinely does
    not exist -- and ``dim_effective`` records how many are real. Absent genes
    stay NaN across *all* columns, which is how downstream code detects them.
    """
    out = np.full((len(genes), dim), np.nan, dtype=np.float32)
    dim_effective = 0
    if latent is not None and latent.size:
        dim_effective = int(min(dim, latent.shape[1]))
        out[mask, :dim_effective] = latent[:, :dim_effective]
        if dim_effective < dim:
            out[mask, dim_effective:] = 0.0
    return pd.DataFrame(out, index=list(genes), columns=embedding_columns(dim)), dim_effective


# ---------------------------------------------------------------------------
# torch plumbing
# ---------------------------------------------------------------------------

def _torch():
    """Import torch lazily so ``pca_baseline`` works without it installed."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        raise ImportError(
            "the autoencoder embedders need PyTorch (a CPU build is enough): "
            "`pip install torch`. pca_baseline runs without it."
        ) from exc
    return torch


def seed_everything(rng: np.random.Generator) -> Tuple[int, np.random.Generator]:
    """Derive a torch seed from the caller's numpy Generator.

    Determinism contract: the registry runner hands every embedder a seeded
    ``np.random.Generator``; we draw one integer from it, seed torch with it,
    and derive the minibatch-shuffling generator from the same integer. Same
    config + same seed -> bit-identical embeddings on CPU.
    """
    torch = _torch()
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)
    return seed, np.random.default_rng(seed)


def mlp(sizes: Sequence[int], *, final_activation: bool = False):
    """Dense ReLU stack ``sizes[0] -> ... -> sizes[-1]``.

    MODIFY HERE: swap ``ReLU`` for another activation, or add BatchNorm /
    Dropout layers, in this one function -- every model in this module and in
    ``joint.py`` is built from it.
    """
    torch = _torch()
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(int(sizes[i]), int(sizes[i + 1])))
        if i < len(sizes) - 2 or final_activation:
            layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def masked_mse(xhat, x, w):
    """Mean squared error over the *observed* entries only.

    ``w`` is a 0/1 weight with the same shape as ``x``: zero wherever the entry
    is padding for a gene the dataset does not contain. Dividing by ``w.sum()``
    rather than ``x.numel()`` keeps the loss comparable when the mask is sparse.
    """
    denom = w.sum().clamp_min(1.0)
    return ((xhat - x) ** 2 * w).sum() / denom


def _fit_autoencoder(
    X: np.ndarray,
    W: np.ndarray,
    dim: int,
    *,
    hidden: Sequence[int] = DEFAULT_HIDDEN,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    batch: int = DEFAULT_BATCH,
    patience: int = DEFAULT_PATIENCE,
    val_frac: float = DEFAULT_VAL_FRAC,
    rng: Optional[np.random.Generator] = None,
    variational: bool = False,
    beta: float = 1.0,
) -> Tuple[np.ndarray, dict]:
    """Fit a dense (variational) autoencoder over gene rows; return the latent.

    Observation unit: ``X`` is ``(n_genes, n_features)`` -- one row per gene.
    ``W`` is the same shape and holds 1 for an observed feature and 0 for
    padding, so masked genes (or masked per-dataset blocks in the joint model)
    contribute nothing to the gradient.

    Early stopping holds out ``val_frac`` of the **genes** (the observations);
    with fewer than ``MIN_VAL_GENES`` held out it falls back to monitoring the
    training loss, which is reported in the history either way. The returned
    latent is computed for *all* rows with the best checkpoint, so held-out
    genes are still embedded -- the hold-out selects the stopping epoch, it does
    not restrict the output.

    For ``variational=True`` the encoder emits ``(mu, logvar)``, the decoder
    reads a reparameterised sample during training, and the returned embedding
    is ``mu``: sampling would make the saved embedding non-deterministic, so
    the posterior mean is the embedding and the sampling noise stays inside
    training. The KL is divided by the number of features so ``beta`` means the
    same thing for datasets of different feature width.

    MODIFY HERE: ``hidden``, ``epochs``, ``lr``, ``batch``, ``patience``,
    ``val_frac``, and ``beta`` (VAE only).
    """
    torch = _torch()
    rng = np.random.default_rng(0) if rng is None else rng
    n, n_feat = X.shape
    hidden = tuple(int(h) for h in hidden)
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    Wt = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32))

    enc = mlp([n_feat, *hidden, dim * (2 if variational else 1)])
    dec = mlp([dim, *reversed(hidden), n_feat])
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    perm = rng.permutation(n)
    n_val = int(round(val_frac * n))
    use_val = n_val >= MIN_VAL_GENES and n - n_val >= max(2, dim)
    val_idx = perm[:n_val] if use_val else np.array([], dtype=int)
    tr_idx = perm[n_val:] if use_val else perm

    def _forward(idx, sample: bool):
        x, w = Xt[idx], Wt[idx]
        h = enc(x)
        if not variational:
            return masked_mse(dec(h), x, w), h, None, None
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(*LOGVAR_CLAMP)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        recon = masked_mse(dec(z), x, w)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean() / n_feat
        return recon + beta * kl, mu, recon, kl

    history: dict = {"epochs_requested": int(epochs), "batch": int(batch),
                     "lr": float(lr), "hidden": list(hidden),
                     "n_train_genes": int(len(tr_idx)), "n_val_genes": int(len(val_idx)),
                     "early_stopping_on": "val_loss" if use_val else "train_loss"}
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    last_train = np.nan
    for epoch in range(1, int(epochs) + 1):
        enc.train(); dec.train()
        order = rng.permutation(len(tr_idx))
        tot, nb = 0.0, 0
        for s in range(0, len(order), int(batch)):
            idx = tr_idx[order[s:s + int(batch)]]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss, _, _, _ = _forward(idx, sample=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        last_train = tot / max(nb, 1)
        enc.eval(); dec.eval()
        with torch.no_grad():
            monitor = float(_forward(val_idx, sample=False)[0]) if use_val else last_train
        if monitor < best - 1e-6:
            best, best_epoch, bad = monitor, epoch, 0
            best_state = ({k: v.clone() for k, v in enc.state_dict().items()},
                          {k: v.clone() for k, v in dec.state_dict().items()})
        else:
            bad += 1
            if bad >= int(patience):
                break
    if best_state is not None:
        enc.load_state_dict(best_state[0]); dec.load_state_dict(best_state[1])
    enc.eval(); dec.eval()
    with torch.no_grad():
        allidx = np.arange(n)
        total, latent, recon, kl = _forward(allidx, sample=False)
        latent = latent.numpy()
    history.update({
        "epochs_run": int(epoch), "best_epoch": int(best_epoch),
        "final_train_loss": float(last_train),
        "best_monitored_loss": float(best),
        "final_full_loss": float(total),
    })
    if variational:
        history.update({
            "beta": float(beta),
            "recon_mse_at_mean": float(recon),
            "kl_per_feature": float(kl),
            # Sign convention: ELBO = -(recon + beta*KL); evaluated at z = mu
            # (deterministic), so it is the ELBO of the mean, not a sampled estimate.
            "neg_elbo_at_mean": float(recon + beta * kl),
            "elbo_at_mean": float(-(recon + beta * kl)),
        })
    return latent, history


def _base_provenance(name: str, block: FeatureBlock, dim: int, dim_effective: int) -> dict:
    """The provenance every embedder in this module records."""
    return {
        "method": name,
        "dataset": block.name,
        "dataset_category": block.category,
        "n_obs_used": block.n_obs,
        "n_genes_requested": int(len(block.genes)),
        "n_genes_embedded": block.n_present,
        "frac_genes_masked_out": block.frac_masked,
        "dim_requested": int(dim),
        "dim_effective": int(dim_effective),
        "observation_unit": "gene (rows); features are cells / cell-PCs",
        **{f"block_{k}": v for k, v in block.info.items()},
    }


# ---------------------------------------------------------------------------
# (1) the baseline every deep method must beat
# ---------------------------------------------------------------------------

@EMBEDDERS.register(
    "pca_baseline",
    category="single_cell",      # default tag; the emitted category is the dataset's
    mode="separate",
    needs_expression=True,
    defaults=dict(pre_pca=None, standardize=True),
    description="truncated SVD of the G x N gene-by-cell matrix (genes are the "
                "observations); the linear baseline every deep method must beat",
)
def pca_baseline(datasets, *, genes, dim, rng, pre_pca: Optional[int] = None,
                 standardize: bool = True, max_obs: Optional[int] = None,
                 **_) -> GeneEmbedding:
    """Truncated SVD of the gene-by-cell matrix.

    Observation unit: gene. The SVD is taken of the ``G x N`` matrix, so the
    returned components are gene scores in a ``dim``-dimensional subspace of
    cell space.

    Normalisation assumption: ``log1p_cp10k`` input, then the column z-score of
    ``prepare_block`` (``standardize=True``) so that library size is not the
    first component. ``pre_pca`` defaults to ``None`` here because the SVD is
    itself the feature-space reduction.

    Missing-data behaviour: genes absent from the dataset are dropped before
    the decomposition and returned as NaN rows. If the block's rank is below
    ``dim`` the surplus columns are zeros and ``provenance["dim_effective"]``
    says so.

    MODIFY HERE: ``standardize``, ``max_obs``, or ``pre_pca`` if you want the
    SVD applied to cell-PCs instead of cells.
    """
    name, dataset = single_dataset(datasets)
    block = prepare_block(dataset, genes, max_obs=max_obs, rng=rng,
                          pre_pca=pre_pca, standardize=standardize)
    latent = None
    if block.n_present and block.n_features:
        k = int(min(dim, block.n_features, max(block.n_present - 1, 1)))
        svd = TruncatedSVD(n_components=k, random_state=int(rng.integers(0, 2**31 - 1)))
        latent = svd.fit_transform(block.present_features()).astype(np.float32)
        evr = float(svd.explained_variance_ratio_.sum())
    else:
        evr = float("nan")
    frame, dim_eff = embedding_frame(latent, block.mask, genes, dim)
    prov = _base_provenance("pca_baseline", block, dim, dim_eff)
    prov.update({"explained_variance_ratio": evr, "epochs": 0,
                 "final_train_loss": None, "deterministic": True,
                 "status": block.info.get("status", "ok")})
    return GeneEmbedding(matrix=frame, method="pca_baseline",
                         category=dataset.category, provenance=prov)


# ---------------------------------------------------------------------------
# (2) dense autoencoder
# ---------------------------------------------------------------------------

@EMBEDDERS.register(
    "ae_separate",
    category="single_cell",
    mode="separate",
    needs_expression=True,
    defaults=dict(pre_pca=DEFAULT_PRE_PCA, standardize=True, hidden=DEFAULT_HIDDEN,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch=DEFAULT_BATCH,
                  patience=DEFAULT_PATIENCE, val_frac=DEFAULT_VAL_FRAC),
    description="dense autoencoder on the G x N gene-by-cell matrix with masked-MSE "
                "reconstruction; genes are the samples, cells (or cell-PCs) the features",
)
def ae_separate(datasets, *, genes, dim, rng, pre_pca: Optional[int] = DEFAULT_PRE_PCA,
                standardize: bool = True, hidden: Sequence[int] = DEFAULT_HIDDEN,
                epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
                batch: int = DEFAULT_BATCH, patience: int = DEFAULT_PATIENCE,
                val_frac: float = DEFAULT_VAL_FRAC, max_obs: Optional[int] = None,
                **_) -> GeneEmbedding:
    """Autoencoder gene embedding for one dataset.

    Observation unit: gene. ``G`` samples of dimension ``min(pre_pca, N, G)``;
    the latent code of each gene row is its embedding.

    Normalisation assumption: ``log1p_cp10k`` input, cell-wise z-score, then the
    ``pre_pca`` feature reduction (see module docstring).

    Missing-data behaviour: absent genes are excluded from training via the
    mask -- they never appear as rows at all -- and are NaN rows in the output.
    ``provenance["frac_genes_masked_out"]`` records how many.

    MODIFY HERE: ``hidden`` (e.g. ``(512, 128)`` for a deeper encoder),
    ``epochs``, ``lr``, ``batch``, ``patience``, ``val_frac``, ``pre_pca``.
    """
    name, dataset = single_dataset(datasets)
    block = prepare_block(dataset, genes, max_obs=max_obs, rng=rng,
                          pre_pca=pre_pca, standardize=standardize)
    latent, history = None, {"status": "no_genes_present"}
    if block.n_present >= 2 and block.n_features:
        seed, batch_rng = seed_everything(rng)
        Xp = block.present_features()
        latent, history = _fit_autoencoder(
            Xp, np.ones_like(Xp), dim, hidden=hidden, epochs=epochs, lr=lr,
            batch=batch, patience=patience, val_frac=val_frac, rng=batch_rng,
            variational=False)
        history["torch_seed"] = seed
        history["status"] = "ok"
    frame, dim_eff = embedding_frame(latent, block.mask, genes, dim)
    prov = _base_provenance("ae_separate", block, dim, dim_eff)
    prov.update({"epochs": history.get("epochs_run", 0),
                 "final_train_loss": history.get("final_train_loss"),
                 "loss": "masked MSE over observed features", **history})
    return GeneEmbedding(matrix=frame, method="ae_separate",
                         category=dataset.category, provenance=prov)


# ---------------------------------------------------------------------------
# (3) variational autoencoder
# ---------------------------------------------------------------------------

@EMBEDDERS.register(
    "vae_separate",
    category="single_cell",
    mode="separate",
    needs_expression=True,
    defaults=dict(pre_pca=DEFAULT_PRE_PCA, standardize=True, hidden=DEFAULT_HIDDEN,
                  epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch=DEFAULT_BATCH,
                  patience=DEFAULT_PATIENCE, val_frac=DEFAULT_VAL_FRAC, beta=1.0),
    description="variational autoencoder with a Gaussian latent over gene rows; the "
                "embedding is the posterior MEAN (sampling would be non-deterministic)",
)
def vae_separate(datasets, *, genes, dim, rng, pre_pca: Optional[int] = DEFAULT_PRE_PCA,
                 standardize: bool = True, hidden: Sequence[int] = DEFAULT_HIDDEN,
                 epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR,
                 batch: int = DEFAULT_BATCH, patience: int = DEFAULT_PATIENCE,
                 val_frac: float = DEFAULT_VAL_FRAC, beta: float = 1.0,
                 max_obs: Optional[int] = None, **_) -> GeneEmbedding:
    """Variational autoencoder gene embedding for one dataset.

    Same architecture as ``ae_separate`` with a Gaussian latent: the encoder
    emits ``(mu, logvar)``, training reconstructs from a reparameterised sample,
    and the loss is ``masked_MSE + beta * KL`` with the KL divided by the number
    of features so ``beta`` is comparable across datasets.

    **The returned embedding is ``mu``, the posterior mean.** Returning a sample
    would make the saved ``(G, d)`` matrix non-deterministic -- re-running the
    same config would give a different file and every downstream kNN/clustering
    result would move -- so the stochasticity stays inside training. The final
    ELBO terms (``recon_mse_at_mean``, ``kl_per_feature``, ``elbo_at_mean``) are
    recorded in provenance; they are evaluated at ``z = mu``, not sampled.

    Observation unit, normalisation and missing-data behaviour: identical to
    ``ae_separate``.

    MODIFY HERE: ``beta`` (0 recovers a plain autoencoder with this loss), plus
    the architecture/optimisation arguments of ``ae_separate``.
    """
    name, dataset = single_dataset(datasets)
    block = prepare_block(dataset, genes, max_obs=max_obs, rng=rng,
                          pre_pca=pre_pca, standardize=standardize)
    latent, history = None, {"status": "no_genes_present"}
    if block.n_present >= 2 and block.n_features:
        seed, batch_rng = seed_everything(rng)
        Xp = block.present_features()
        latent, history = _fit_autoencoder(
            Xp, np.ones_like(Xp), dim, hidden=hidden, epochs=epochs, lr=lr,
            batch=batch, patience=patience, val_frac=val_frac, rng=batch_rng,
            variational=True, beta=beta)
        history["torch_seed"] = seed
        history["status"] = "ok"
    frame, dim_eff = embedding_frame(latent, block.mask, genes, dim)
    prov = _base_provenance("vae_separate", block, dim, dim_eff)
    prov.update({"epochs": history.get("epochs_run", 0),
                 "final_train_loss": history.get("final_train_loss"),
                 "embedding_is": "posterior mean (mu)",
                 "loss": "masked MSE + beta * KL (KL per feature)", **history})
    return GeneEmbedding(matrix=frame, method="vae_separate",
                         category=dataset.category, provenance=prov)
