"""Kernelisation and fusion of the prior feature maps (ITERPERT part 1, second half).

The paper (Huang et al., bioRxiv 2023.12.12.571389) turns each prior feature map
``phi_s`` into a gene-gene kernel ``K_s``, min-max normalises each kernel so one
source's scale cannot dominate, and fuses the kernels into a single ``K``. This
module is that pipeline, plus the two things the paper leaves unspecified and
that decide whether the result is usable:

* **Positive semi-definiteness** of the fused kernel, which every downstream
  kernel method assumes. ``fuse_kernels`` checks it and reports the smallest
  eigenvalue instead of silently returning an indefinite matrix.
* **Missing data.** Real priors do not cover the whole transcriptome (STRING has
  no node for every gene; Swiss-Prot has no reviewed sequence for every symbol),
  so a feature matrix has NaN rows and a naive kernel is NaN wherever either
  gene is uncovered. One NaN row poisons an entire row and column of the fused
  kernel. The ``missing=`` policy makes that choice explicit.

Observation unit
----------------
Throughout this module the **gene** is the observation: every kernel is
``genes x genes`` with an identical, identically ordered index. Kernels built on
different gene orders are a bug, and ``fuse_kernels`` refuses to align them
implicitly.

MODIFY HERE
-----------
* ``prior_kernel`` -- add a ``kind`` (e.g. a polynomial or Laplacian kernel) in
  the dispatch block; keep the "centre/scale decision" documentation habit.
* ``fuse_kernels`` -- add a combination rule in the ``how`` dispatch; if it is
  not a convex combination, say what it does to PSD.
* ``normalise_kernel`` -- add a normalisation (e.g. the kernel-correlation
  ``K_ij / sqrt(K_ii K_jj)``) in the ``how`` dispatch.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .sources import PriorFeatures

# ---------------------------------------------------------------------------
# feature map -> kernel
# ---------------------------------------------------------------------------


def _as_matrix(features) -> pd.DataFrame:
    if isinstance(features, PriorFeatures):
        return features.matrix
    return pd.DataFrame(features)


def prior_kernel(
    features,
    kind: str = "cosine",
    *,
    gamma: Optional[float] = None,
    normalise: Optional[str] = None,
) -> pd.DataFrame:
    """Turn a ``genes x f`` feature map into a ``genes x genes`` kernel.

    Parameters
    ----------
    features
        A :class:`~panelprep.priors.sources.PriorFeatures` or any genes x f
        DataFrame/array. Rows that are entirely NaN (genes the source does not
        cover) produce NaN rows and columns in the kernel, which is the honest
        representation: similarity to an unknown vector is unknown, not zero.
        Pass the result to :func:`fuse_kernels` with ``missing='mask'`` to handle
        it.
    kind
        ``'linear'``
            ``Z Z^T / f`` -- the inner-product kernel, scaled by the number of
            features. **For the z-scored co-expression feature map this is
            exactly the gene-gene Pearson correlation matrix** (each row has mean
            0 and sd 1, so ``sum_k z_ik z_jk / f = corr(i, j)``). That identity is
            the paper's reason for calling this feature map the co-expression
            prior, and the tests assert it against ``np.corrcoef`` to 1e-10.
        ``'cosine'``
            ``Z Z^T`` with rows L2-normalised: scale-free, in ``[-1, 1]``. The
            right default for sequence / text / node embeddings, whose vector
            norms carry model-specific magnitude rather than biology.
        ``'rbf'``
            ``exp(-gamma * ||z_i - z_j||^2)``; with ``gamma=None`` the bandwidth
            comes from the **median heuristic**
            ``gamma = 1 / (2 * median(pairwise squared distance))``, computed on
            covered rows only. Non-linear, strictly positive, and insensitive to
            the feature scale only through that heuristic.
    gamma
        RBF bandwidth. ``None`` -> median heuristic (recorded in
        ``K.attrs['gamma']``).
    normalise
        Convenience: apply :func:`normalise_kernel` with this ``how`` before
        returning (``None`` = leave as is).

    Why ``kind`` is a modelling decision, not formatting
    ---------------------------------------------------
    ``'linear'`` on the z-scored co-expression map is the correlation matrix;
    ``'cosine'`` on the *same* matrix divides each row by its own norm, which for
    a z-scored row is ``sqrt(f)`` -- so cosine and linear coincide there up to a
    constant, but they differ for every other feature map. ``'rbf'`` is a
    different hypothesis class entirely (local similarity, no negative entries).
    Choosing between them changes what "similar genes" means downstream.

    Observation unit: gene. Normalisation: see ``kind``. Missing data: NaN rows
    in, NaN rows and columns out.

    MODIFY HERE: add a ``kind`` to the dispatch below.
    """
    M = _as_matrix(features)
    genes = list(M.index)
    X = M.to_numpy(dtype=np.float64)
    covered = np.isfinite(X).any(axis=1) if X.shape[1] else np.zeros(len(genes), dtype=bool)
    Xc = X[covered]
    if Xc.size and np.isnan(Xc).any():
        Xc = np.nan_to_num(Xc, nan=0.0)        # partial NaN inside a covered row
    n, f = Xc.shape
    attrs: Dict[str, Any] = {"kind": kind, "n_covered": int(covered.sum()),
                             "n_genes": len(genes), "n_features": int(M.shape[1])}
    if n == 0:
        K = np.full((len(genes), len(genes)), np.nan)
        out = pd.DataFrame(K, index=genes, columns=genes)
        out.attrs.update(attrs)
        return out

    if kind == "linear":
        Ksub = Xc @ Xc.T / max(f, 1)
    elif kind == "cosine":
        nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        Z = Xc / nrm
        Ksub = Z @ Z.T
    elif kind == "rbf":
        sq = np.maximum(
            (Xc ** 2).sum(1)[:, None] + (Xc ** 2).sum(1)[None, :] - 2 * (Xc @ Xc.T), 0.0)
        if gamma is None:
            iu = np.triu_indices(n, k=1)
            med = float(np.median(sq[iu])) if iu[0].size else 1.0
            gamma = 1.0 / (2.0 * med) if med > 0 else 1.0
        Ksub = np.exp(-gamma * sq)
        attrs["gamma"] = float(gamma)
    else:
        raise ValueError("kind must be 'linear', 'cosine' or 'rbf'")

    K = np.full((len(genes), len(genes)), np.nan)
    idx = np.where(covered)[0]
    K[np.ix_(idx, idx)] = Ksub
    out = pd.DataFrame(K, index=genes, columns=genes)
    out.attrs.update(attrs)
    if normalise:
        out = normalise_kernel(out, how=normalise)
    return out


# ---------------------------------------------------------------------------
# per-kernel normalisation
# ---------------------------------------------------------------------------

def normalise_kernel(K, how: str = "minmax") -> pd.DataFrame:
    """Rescale one kernel so it cannot dominate the fusion by scale alone.

    The paper applies "a min-max scale normalization to each kernel ... to avoid
    one kernel with a large scale overriding the others". Here that is done over
    the **off-diagonal** entries only:

    * For a cosine kernel the diagonal is 1 by construction, and for an RBF
      kernel it is 1 exactly. Including it in the min/max statistics would pin
      the maximum at 1 for every source regardless of how its off-diagonal
      entries are distributed, which defeats the purpose -- a source whose gene
      pairs all sit near 0.02 would be left near 0.02 while a source with
      similarities near 0.9 stays near 0.9.
    * The diagonal is a gene's similarity to itself, never a comparison between
      two genes, so it carries no information about the source's scale.

    After scaling, the off-diagonal entries span ``[0, 1]`` and the diagonal is
    set to 1 (self-similarity is maximal, which keeps the matrix interpretable
    and -- for 'minmax' -- diagonally dominant in the common case).

    ``how``
        ``'minmax'``
            ``(K - min_offdiag) / (max_offdiag - min_offdiag)``. Monotone, so
            the ranking of gene pairs within the kernel is preserved exactly
            (asserted in the tests, rank correlation 1.0).
        ``'zscore'``
            standardise off-diagonal entries to mean 0, sd 1. Keeps negative
            similarities meaningful; does **not** bound the range.
        ``'none'``
            return a copy unchanged.

    Observation unit: gene pairs (the off-diagonal). Missing data: NaN entries
    are ignored in the statistics and stay NaN. PSD is **not** preserved by an
    affine shift of the off-diagonal (adding a constant to off-diagonal entries
    is adding ``c(J - I)``, which has a negative eigenvalue ``-c``); that is
    exactly why :func:`fuse_kernels` reports the smallest eigenvalue after
    fusing rather than assuming it.

    MODIFY HERE: add a ``how`` to the dispatch.
    """
    Kdf = pd.DataFrame(K)
    A = Kdf.to_numpy(dtype=np.float64).copy()
    n = A.shape[0]
    if n != A.shape[1]:
        raise ValueError(f"kernel must be square; got {A.shape}")
    off = ~np.eye(n, dtype=bool)
    vals = A[off]
    finite = vals[np.isfinite(vals)]
    attrs = dict(Kdf.attrs)
    if how == "none" or finite.size == 0:
        out = pd.DataFrame(A, index=Kdf.index, columns=Kdf.columns)
        out.attrs.update({**attrs, "normalisation": "none"})
        return out
    if how == "minmax":
        lo, hi = float(finite.min()), float(finite.max())
        if hi > lo:
            A[off] = (A[off] - lo) / (hi - lo)
        else:
            A[off] = 0.0
        stats = {"offdiag_min": lo, "offdiag_max": hi}
    elif how == "zscore":
        mu, sd = float(finite.mean()), float(finite.std())
        A[off] = (A[off] - mu) / (sd if sd > 0 else 1.0)
        stats = {"offdiag_mean": mu, "offdiag_sd": sd}
    else:
        raise ValueError("how must be 'minmax', 'zscore' or 'none'")
    # Self-similarity is set to 1 for genes the source covers, and left NaN for
    # genes it does not: writing 1.0 onto an uncovered gene's diagonal would
    # invent a data point and would make that gene look "covered" to
    # ``fuse_kernels``' per-entry source count.
    diag_known = np.isfinite(np.diag(Kdf.to_numpy(dtype=np.float64)))
    A[np.arange(n), np.arange(n)] = np.where(diag_known, 1.0, np.nan)
    out = pd.DataFrame(A, index=Kdf.index, columns=Kdf.columns)
    out.attrs.update({**attrs, "normalisation": how, **stats})
    return out


# ---------------------------------------------------------------------------
# PSD check
# ---------------------------------------------------------------------------

def check_psd(K, tol: float = 1e-8) -> tuple[bool, float]:
    """Return ``(is_psd, smallest_eigenvalue)`` of a symmetric kernel.

    NaN entries are treated as 0 for the eigen-decomposition **only** (an
    indefinite answer about an incomplete matrix is still useful, and
    ``numpy.linalg.eigvalsh`` cannot take NaN); the returned eigenvalue is
    therefore about the zero-filled matrix whenever the kernel has missing
    entries, and ``fuse_kernels(missing='mask')`` is the way to avoid that.

    ``tol`` absorbs floating-point noise: a kernel built as ``Z Z^T`` is PSD in
    exact arithmetic but can show eigenvalues of order ``-1e-16``.
    """
    A = np.asarray(pd.DataFrame(K).to_numpy(dtype=np.float64))
    A = np.nan_to_num(A, nan=0.0)
    A = 0.5 * (A + A.T)
    try:
        w = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        return False, float("nan")
    lam = float(w.min())
    return bool(lam >= -abs(tol)), lam


# ---------------------------------------------------------------------------
# fusion
# ---------------------------------------------------------------------------

def fuse_kernels(
    kernels: Mapping[str, Any] | Sequence[Any],
    how: str = "mean",
    *,
    weights: Optional[Mapping[str, float] | Sequence[float]] = None,
    missing: str = "mask",
    normalise: Optional[str] = "minmax",
    psd_tol: float = 1e-8,
    return_counts: bool = True,
):
    """Fuse several gene-gene kernels into one, with an explicit missing-data policy.

    Parameters
    ----------
    kernels
        ``{source_name: kernel}`` (or a sequence). All kernels must share the
        same gene index **in the same order**; this function will not reindex
        silently, because a quiet reorder is the kind of bug that produces a
        plausible-looking kernel of the wrong genes.
    how
        ``'mean'``   the paper's default: ``K = (1/S) sum_s K_s``.
        ``'weighted'`` ``K = sum_s w_s K_s`` with ``w`` from ``weights``
        (normalised to sum to 1 over the sources present for each entry).
        ``'max'``    element-wise maximum.
        ``'product'`` element-wise (Hadamard) product.
    weights
        Per-source weights for ``how='weighted'``.
    missing
        ``'mask'`` (default)
            Each fused entry is computed from only the sources that cover **both**
            genes. A gene covered by no source stays NaN. The companion matrix
            ``counts[i, j]`` records how many sources contributed to that entry,
            so a user can see that one similarity rests on one source and
            another on five.
        ``'drop'``
            Restrict to the genes covered by *every* source and fuse on that
            submatrix. Clean and comparable, but with six sources the surviving
            set can be much smaller than any single source's coverage.
        ``'zero'``
            Impute missing entries with 0 before fusing. **This silently asserts
            that an unmeasured pair is dissimilar**, which biases every
            downstream distance and is the wrong default; it exists only
            because it is what a naive ``nan_to_num`` pipeline does, and so that
            its effect can be measured against ``'mask'``.
    normalise
        ``how`` passed to :func:`normalise_kernel` for each input kernel before
        fusing (``None`` to skip). The paper normalises first; with
        heterogeneous sources, skipping it lets the widest-scaled kernel set the
        fused geometry.

    PSD, which is what makes the result usable downstream
    -----------------------------------------------------
    * ``'mean'`` and ``'weighted'`` with non-negative weights are **convex
      combinations**: for any ``v``, ``v^T (sum w_s K_s) v = sum w_s (v^T K_s v)
      >= 0``, so the fusion of PSD kernels is PSD. This is why the paper's
      default is safe.
    * ``'product'`` is the Hadamard product, which is PSD by the **Schur product
      theorem**. (Safe, but it multiplies small similarities into very small
      ones.)
    * ``'max'`` is **not** guaranteed PSD: the element-wise maximum of two PSD
      matrices can have a negative eigenvalue, and the tests exhibit a concrete
      2x2-block counterexample with its eigenvalue.
    * ``missing='mask'`` effectively uses a different weight vector per entry,
      which is *not* a single convex combination of the full kernels, so it also
      has no PSD guarantee.
    * ``normalise='minmax'`` -- the paper's own pre-fusion step -- is an **affine
      map of the off-diagonal**, i.e. it adds ``c(J - I)`` whose smallest
      eigenvalue is ``-c``. It is therefore not PSD-preserving either, and in
      practice it is usually what makes a mean-fused kernel indefinite (the
      tests measure both: PSD with ``normalise=None``, indefinite with
      ``'minmax'``, with the eigenvalues printed). Scale normalisation and
      PSD-preservation genuinely conflict; the honest resolution is to keep the
      normalisation, report ``min_eigenvalue``, and let the caller project the
      kernel if their downstream method needs PSD.

    In all cases the result is checked and, when it fails, a
    :class:`UserWarning` reports the smallest eigenvalue -- a warning rather than
    an exception, because an indefinite kernel is still informative and the
    caller may choose to project it (``K <- K - lambda_min I`` or clipping the
    negative spectrum).

    Returns
    -------
    (K, info)
        ``K`` is a DataFrame; ``info`` holds ``is_psd``, ``min_eigenvalue``, the
        per-source coverage, the policy, and -- with ``missing='mask'`` and
        ``return_counts=True`` -- ``counts``, the per-entry source-count matrix.

    MODIFY HERE: add a rule to the ``how`` dispatch, and document its PSD status.
    """
    items = dict(kernels) if isinstance(kernels, Mapping) else \
        {f"k{i}": k for i, k in enumerate(kernels)}
    if not items:
        raise ValueError("no kernels to fuse")
    frames = {k: pd.DataFrame(v) for k, v in items.items()}
    first = next(iter(frames.values()))
    genes = list(first.index)
    for name, Kdf in frames.items():
        if list(Kdf.index) != genes or list(Kdf.columns) != genes:
            raise ValueError(f"kernel {name!r} has a different gene index/order; align the "
                             "feature maps before kernelising (this is not fixed silently)")
    if normalise:
        frames = {k: normalise_kernel(v, how=normalise) for k, v in frames.items()}

    stack = np.stack([f.to_numpy(dtype=np.float64) for f in frames.values()])  # (S, G, G)
    names = list(frames)
    present = np.isfinite(stack)
    coverage = {n: float(np.isfinite(f.to_numpy(dtype=np.float64)).any(axis=1).mean())
                for n, f in frames.items()}

    if how == "weighted":
        if weights is None:
            raise ValueError("how='weighted' requires weights")
        w = np.array([float(dict(weights)[n]) for n in names]) if isinstance(weights, Mapping) \
            else np.asarray(list(weights), dtype=float)
        if w.shape[0] != len(names):
            raise ValueError(f"{w.shape[0]} weights for {len(names)} kernels")
        if (w < 0).any():
            warnings.warn("negative fusion weights break the PSD guarantee of a convex "
                          "combination", UserWarning, stacklevel=2)
    else:
        w = np.ones(len(names), dtype=float)

    keep = np.ones(len(genes), dtype=bool)
    if missing == "drop":
        # a gene is kept when EVERY source covers it (i.e. that gene's row has at
        # least one finite entry in each source) -- not when every *pair* is
        # finite, which a single uncovered gene anywhere would make impossible.
        keep = present.any(axis=2).all(axis=0)
        stack = stack[:, keep][:, :, keep]
        present = present[:, keep][:, :, keep]
    elif missing == "zero":
        stack = np.nan_to_num(stack, nan=0.0)
        present = np.ones_like(stack, dtype=bool)
    elif missing != "mask":
        raise ValueError("missing must be 'mask', 'drop' or 'zero'")

    counts = present.sum(axis=0)
    wb = w[:, None, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN slices
        if how in ("mean", "weighted"):
            num = np.nansum(np.where(present, stack, 0.0) * wb, axis=0)
            den = np.nansum(np.where(present, 1.0, 0.0) * wb, axis=0)
            K = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        elif how == "max":
            K = np.where(counts > 0, np.nanmax(np.where(present, stack, -np.inf), axis=0), np.nan)
            K[np.isneginf(K)] = np.nan
        elif how == "product":
            K = np.where(counts > 0, np.nanprod(np.where(present, stack, 1.0), axis=0), np.nan)
        else:
            raise ValueError("how must be 'mean', 'weighted', 'max' or 'product'")

    gkeep = [g for g, k in zip(genes, keep) if k]
    Kdf = pd.DataFrame(K, index=gkeep, columns=gkeep)
    is_psd, lam = check_psd(Kdf, tol=psd_tol)
    info: Dict[str, Any] = {
        "how": how, "missing": missing, "normalise": normalise, "sources": names,
        "weights": {n: float(x) for n, x in zip(names, w)} if how == "weighted" else None,
        "n_genes": len(gkeep), "n_genes_dropped": int((~keep).sum()),
        "coverage": coverage, "is_psd": bool(is_psd), "min_eigenvalue": lam,
        # A convex combination of PSD kernels is PSD -- but only if the inputs
        # are still PSD when they are combined. ``normalise='minmax'`` is an
        # affine map of the off-diagonal (adding c(J - I), eigenvalue -c), and
        # per-entry masking uses a different weight vector per entry; neither
        # preserves PSD, so the guarantee is claimed only without them.
        "psd_guaranteed": (
            (how in ("mean", "product") or (how == "weighted" and (w >= 0).all()))
            and normalise in (None, "none")
            and (missing != "mask" or bool(present.all()))
        ),
        "n_entries_without_any_source": int((counts == 0).sum()),
        "mean_sources_per_entry": float(counts.mean()),
    }
    if missing == "mask" and return_counts:
        info["counts"] = pd.DataFrame(counts, index=gkeep, columns=gkeep)
    if not is_psd:
        warnings.warn(
            f"fused kernel (how={how!r}, missing={missing!r}) is not PSD: smallest eigenvalue "
            f"{lam:.3e}. Convex combinations ('mean', non-negative 'weighted') and the Hadamard "
            "'product' are PSD-preserving; element-wise 'max' and per-entry masking are not. "
            "Project the kernel (clip negative eigenvalues) before using it in a method that "
            "requires PSD.", UserWarning, stacklevel=2)
    return Kdf, info


# ---------------------------------------------------------------------------
# agreement between priors
# ---------------------------------------------------------------------------

def kernel_alignment(K1, K2, *, centre: bool = False) -> float:
    """Kernel alignment: the Frobenius-inner-product cosine between two kernels.

    ``KA(K1, K2) = <K1, K2>_F / (||K1||_F ||K2||_F)`` (Cristianini et al. 2001),
    the diagnostic the paper uses to ask how much two priors agree. 1.0 means the
    two kernels induce the same similarity geometry up to scale; near 0 means
    they are (linearly) unrelated, i.e. the second source adds information the
    first does not have.

    Observation unit: gene pairs. Only entries finite in **both** kernels
    contribute, and the gene index must match -- comparing two kernels over
    different gene sets is comparing different objects. The number of shared
    entries matters for interpretation: an alignment computed on the 200 pairs
    where two sparse priors overlap is not comparable with one computed on
    4,000,000 pairs, so callers should read it together with the coverage column
    of :func:`fuse_report`.

    ``centre=True`` applies the centred alignment of Cortes et al. (2012),
    ``H K H`` with ``H = I - 11^T/n``, which removes the "everything is a bit
    similar to everything" offset that makes almost any two kernels look aligned
    at ~0.9. Off by default to match the paper's plain KA; turn it on when
    comparing sources whose kernels are all positive.
    """
    A = pd.DataFrame(K1)
    B = pd.DataFrame(K2)
    if list(A.index) != list(B.index):
        raise ValueError("kernel_alignment requires an identical gene index/order")
    X = A.to_numpy(dtype=np.float64)
    Y = B.to_numpy(dtype=np.float64)
    both = np.isfinite(X) & np.isfinite(Y)
    if not both.any():
        return float("nan")
    if centre:
        # H K H with H = I - 11^T/n, computed as K - rowmean - colmean + grandmean:
        # identical result in O(n^2) instead of two n x n matmuls, which matters
        # at transcriptome scale (n ~ 2e4 makes the matmul form unusable).
        def _centre(M):
            M = np.nan_to_num(M)
            r = M.mean(axis=1, keepdims=True)
            c = M.mean(axis=0, keepdims=True)
            return M - r - c + M.mean()

        X, Y = _centre(X), _centre(Y)
        both = np.ones_like(both)
    x, y = X[both], Y[both]
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return float("nan")
    return float(np.dot(x, y) / (nx * ny))


def fuse_report(
    kernels: Mapping[str, Any],
    *,
    out_path: Optional[str] = None,
    centre: bool = False,
    features: Optional[Mapping[str, PriorFeatures]] = None,
) -> pd.DataFrame:
    """Pairwise kernel alignments plus per-source coverage -- the comparison table.

    Rows and columns are the prior sources; the body is
    :func:`kernel_alignment`. Extra columns:

    ``coverage``
        fraction of genes with any finite entry in that kernel.
    ``n_covered``
        the count behind it.
    ``mean_offdiag``
        the kernel's own scale before normalisation, which is what makes
        min-max normalisation necessary in the first place.
    ``status`` / ``modality`` / ``n_features``
        taken from ``features[name].provenance`` when the feature maps are
        supplied, so a deferred source is visible in the same table rather than
        quietly absent.

    Adding a prior source therefore updates this table with no edit here: the
    table is built from whatever mapping it is given.

    Observation unit: one prior source per row. Missing data: a source with no
    finite entries gives NaN alignments and coverage 0, not an error.
    """
    names = list(kernels)
    frames = {n: pd.DataFrame(k) for n, k in kernels.items()}
    rows = []
    for a in names:
        row: Dict[str, Any] = {"source": a}
        for b in names:
            try:
                row[b] = kernel_alignment(frames[a], frames[b], centre=centre)
            except ValueError:
                row[b] = float("nan")
        A = frames[a].to_numpy(dtype=np.float64)
        cov = np.isfinite(A).any(axis=1)
        off = ~np.eye(A.shape[0], dtype=bool)
        vals = A[off]
        vals = vals[np.isfinite(vals)]
        row["n_covered"] = int(cov.sum())
        row["coverage"] = float(cov.mean()) if cov.size else 0.0
        row["mean_offdiag"] = float(vals.mean()) if vals.size else float("nan")
        if features and a in features:
            pf = features[a]
            row["status"] = pf.provenance.get("status")
            row["modality"] = pf.modality
            row["category"] = pf.category
            row["n_features"] = pf.dim
        rows.append(row)
    df = pd.DataFrame(rows).set_index("source")
    if out_path:
        df.to_csv(out_path)
    return df


__all__ = ["prior_kernel", "normalise_kernel", "fuse_kernels", "check_psd",
           "kernel_alignment", "fuse_report"]
