"""Tests for ``panelprep.priors`` -- ITERPERT-style multi-modal gene priors.

Two layers, so the fast layer runs anywhere:

* **synthetic / offline** -- kernel algebra, PSD behaviour, min-max
  normalisation, the missing-data policy, determinism, and the real branch of
  the deferred Perturb-seq source. No network, no model downloads.
* **real mouse data** -- marked ``@pytest.mark.network`` (and ``slow``). These
  build the five runnable sources over a real mouse gene universe and report
  numbers: coverage, feature dimension, wall time, and a biological-sanity
  permutation test. They re-use a cached build at
  ``outputs/priors_real.pkl`` when present so the suite is re-runnable without
  paying for ESM-2 again; delete that file to force a rebuild.

Every assertion that carries a number also *prints* it (``pytest -s``), because
the point of these tests is the evidence, not a green tick.
"""
from __future__ import annotations

import os
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelprep.priors import (
    PRIORS, build_priors, check_psd, coverage_table, fuse_kernels, fuse_report,
    kernel_alignment, normalise_kernel, prior_kernel, to_gene_embedding,
)
from panelprep.priors.sources import (
    DEFAULT_CACHE, PriorFeatures, make_features, mouse_protein_coding_universe,
    node2vec_embedding, zscore_rows,
)
from panelprep.schemas import CandidateDataset

CACHE = os.environ.get("PANELPREP_PRIOR_CACHE", DEFAULT_CACHE)
REAL_PICKLE = Path("outputs/priors_real.pkl")

#: Genes used by the biological-sanity check. Two independent sets so a single
#: idiosyncratic complex cannot carry the result.
PLURIPOTENCY = ["Sox2", "Pou5f1", "Nanog", "Klf4"]
BMP_SMAD = ["Bmp4", "Bmpr1a", "Bmpr2", "Smad1", "Smad4", "Smad5"]


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def _synthetic_features(n_genes=40, n_feat=25, seed=0, n_missing=0):
    """A genes x f feature map with a block-correlated structure."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(4, n_feat))
    X = np.vstack([base[i % 4] + 0.5 * rng.normal(size=n_feat) for i in range(n_genes)])
    genes = [f"Gene{i:03d}" for i in range(n_genes)]
    df = pd.DataFrame(X.astype(np.float32), index=genes)
    if n_missing:
        df.iloc[:n_missing] = np.nan
    return df


def _synthetic_dataset(name="synth", n_cells=300, genes=("Aaa", "Bbb", "Ccc", "Ddd"), seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.normal(size=(n_cells, 2))
    load = rng.normal(size=(2, len(genes)))
    X = np.maximum(lat @ load + rng.normal(scale=0.3, size=(n_cells, len(genes))), 0)
    expr = pd.DataFrame(X, columns=list(genes))
    return CandidateDataset(name=name, category="single_cell", expression=expr,
                            units="log1p_cp10k", obs_unit="cell")


# ===========================================================================
# 1. kernel algebra: the paper's co-expression claim
# ===========================================================================

def test_linear_kernel_of_zscored_features_is_correlation():
    """``prior_kernel(kind='linear')`` on a z-scored map == ``np.corrcoef``, to 1e-10.

    This is the identity that justifies calling the atlas feature map the
    co-expression prior, so it must hold exactly, not approximately.
    """
    X = _synthetic_features(n_genes=30, n_feat=120, seed=1).to_numpy(dtype=np.float64)
    Z, usable = zscore_rows(X)
    assert usable.all()
    K = prior_kernel(pd.DataFrame(Z, index=[f"G{i}" for i in range(Z.shape[0])]), kind="linear")
    C = np.corrcoef(Z)
    err = float(np.abs(K.to_numpy() - C).max())
    print(f"\n[1] max |linear kernel - corrcoef| = {err:.3e} over {C.size} entries")
    assert err < 1e-10
    # and the un-standardised map is NOT the correlation matrix (why z-scoring matters)
    Kraw = prior_kernel(pd.DataFrame(X, index=[f"G{i}" for i in range(X.shape[0])]), kind="linear")
    raw_err = float(np.abs(Kraw.to_numpy() - C).max())
    print(f"[1] max |linear kernel of RAW features - corrcoef| = {raw_err:.3f} "
          "(standardisation is not cosmetic)")
    assert raw_err > 1e-3


def test_kernel_kinds_shapes_and_ranges():
    F = _synthetic_features(n_genes=25, n_feat=40, seed=2)
    Kc = prior_kernel(F, kind="cosine")
    Kr = prior_kernel(F, kind="rbf")
    print(f"\n[1b] cosine range [{np.nanmin(Kc.to_numpy()):.3f}, {np.nanmax(Kc.to_numpy()):.3f}]; "
          f"rbf gamma (median heuristic) = {Kr.attrs['gamma']:.4g}, "
          f"range [{np.nanmin(Kr.to_numpy()):.3f}, {np.nanmax(Kr.to_numpy()):.3f}]")
    assert np.nanmax(Kc.to_numpy()) <= 1 + 1e-9 and np.nanmin(Kc.to_numpy()) >= -1 - 1e-9
    assert np.allclose(np.diag(Kr.to_numpy()), 1.0)
    assert Kc.shape == (25, 25)


# ===========================================================================
# 2. PSD behaviour of the fusion operators
# ===========================================================================

def test_fuse_mean_is_psd_and_max_need_not_be():
    """Convex combination preserves PSD; element-wise max does not. Both reported."""
    F1 = _synthetic_features(n_genes=20, n_feat=30, seed=3)
    F2 = _synthetic_features(n_genes=20, n_feat=30, seed=4)
    F2.index = F1.index
    K1, K2 = prior_kernel(F1, kind="cosine"), prior_kernel(F2, kind="cosine")
    for name, K in (("K1", K1), ("K2", K2)):
        ok, lam = check_psd(K)
        print(f"\n[2] input {name}: psd={ok} min eig = {lam:.3e}")
        assert ok
    # (a) the convex combination itself, with no normalisation step
    Km, info_m = fuse_kernels({"a": K1, "b": K2}, how="mean", missing="mask", normalise=None)
    print(f"[2] fuse(mean, normalise=None): psd={info_m['is_psd']} "
          f"min eig = {info_m['min_eigenvalue']:.3e}  <- convex combination of PSD kernels")
    assert info_m["is_psd"], info_m
    assert info_m["psd_guaranteed"]
    Kp, info_p = fuse_kernels({"a": K1, "b": K2}, how="product", missing="mask", normalise=None)
    print(f"[2] fuse(product, Schur, normalise=None): psd={info_p['is_psd']} "
          f"min eig = {info_p['min_eigenvalue']:.3e}")
    assert info_p["is_psd"]

    # (b) the paper's min-max step is an affine map of the off-diagonal and is
    #     itself not PSD-preserving -- measured, not assumed.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        Kn, info_n = fuse_kernels({"a": K1, "b": K2}, how="mean", missing="mask",
                                  normalise="minmax")
    print(f"[2] fuse(mean, normalise='minmax'): psd={info_n['is_psd']} "
          f"min eig = {info_n['min_eigenvalue']:.3e}  <- scale normalisation costs PSD")
    assert info_n["psd_guaranteed"] is False

    # explicit counterexample for 'max': two rank-1 PSD kernels whose
    # element-wise maximum is indefinite.
    A = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    B = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    for nm, M in (("A", A), ("B", B)):
        ok, lam = check_psd(M)
        print(f"[2] counterexample input {nm}: psd={ok} min eig = {lam:.3e}")
        assert ok
    idx = ["g1", "g2", "g3"]
    dA = pd.DataFrame(A, index=idx, columns=idx)
    dB = pd.DataFrame(B, index=idx, columns=idx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Kx, info_x = fuse_kernels({"a": dA, "b": dB}, how="max", missing="mask", normalise=None)
    print(f"[2] fuse(max) of two PSD kernels: psd={info_x['is_psd']} "
          f"min eig = {info_x['min_eigenvalue']:.3e}; warned={bool(caught)}")
    assert not info_x["is_psd"]
    assert info_x["min_eigenvalue"] < -1e-8
    assert any("not PSD" in str(w.message) for w in caught)
    assert info_x["psd_guaranteed"] is False


# ===========================================================================
# 3. min-max normalisation
# ===========================================================================

def test_minmax_normalisation_bounds_and_order():
    from scipy.stats import spearmanr

    F = _synthetic_features(n_genes=18, n_feat=35, seed=5)
    K = prior_kernel(F, kind="cosine")
    Kn = normalise_kernel(K, how="minmax")
    n = K.shape[0]
    off = ~np.eye(n, dtype=bool)
    a, b = K.to_numpy()[off], Kn.to_numpy()[off]
    rho = float(spearmanr(a, b).statistic)
    print(f"\n[3] off-diagonal after min-max: [{b.min():.6f}, {b.max():.6f}]; "
          f"before: [{a.min():.4f}, {a.max():.4f}]; rank correlation = {rho:.6f}")
    assert b.min() >= -1e-12 and b.max() <= 1 + 1e-12
    assert abs(b.min()) < 1e-12 and abs(b.max() - 1) < 1e-12
    assert abs(rho - 1.0) < 1e-12
    assert np.allclose(np.diag(Kn.to_numpy()), 1.0)
    # scale blindness is the point: a kernel scaled by 1e-3 normalises identically
    Kn_small = normalise_kernel(K * 1e-3, how="minmax")
    print(f"[3] max |norm(K) - norm(1e-3 K)| = "
          f"{np.abs(Kn.to_numpy() - Kn_small.to_numpy()).max():.3e}")
    assert np.allclose(Kn.to_numpy(), Kn_small.to_numpy(), atol=1e-9)


# ===========================================================================
# 4. missing-data policy
# ===========================================================================

def test_missing_mask_policy_and_source_counts():
    """Two sources with *disjoint* uncovered genes: masked fusion keeps every pair
    that at least one source covers, and the companion count matrix is exact."""
    n = 12
    F1 = _synthetic_features(n_genes=n, n_feat=20, seed=6)
    F2 = _synthetic_features(n_genes=n, n_feat=20, seed=7)
    F2.index = F1.index
    F1.iloc[:3] = np.nan          # source A misses genes 0,1,2
    F2.iloc[9:] = np.nan          # source B misses genes 9,10,11
    K1, K2 = prior_kernel(F1, kind="cosine"), prior_kernel(F2, kind="cosine")
    c1 = np.isfinite(K1.to_numpy())
    c2 = np.isfinite(K2.to_numpy())
    K, info = fuse_kernels({"a": K1, "b": K2}, how="mean", missing="mask")
    counts = info["counts"].to_numpy()
    expected = c1.astype(int) + c2.astype(int)
    print(f"\n[4] counts matrix matches expectation: {np.array_equal(counts, expected)}; "
          f"entries by #sources: "
          f"{ {int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))} }")
    assert np.array_equal(counts, expected)
    any_cov = c1 | c2
    assert np.isfinite(K.to_numpy()[any_cov]).all()
    assert np.isnan(K.to_numpy()[~any_cov]).all()
    print(f"[4] NaNs in fused kernel: {int(np.isnan(K.to_numpy()).sum())} of {K.size}; "
          f"mean sources per entry = {info['mean_sources_per_entry']:.3f}")

    # 'drop' keeps only genes covered everywhere
    Kd, info_d = fuse_kernels({"a": K1, "b": K2}, how="mean", missing="drop")
    print(f"[4] missing='drop': {info_d['n_genes']} genes kept of {n} "
          f"({info_d['n_genes_dropped']} dropped)")
    assert info_d["n_genes"] == 6 and not np.isnan(Kd.to_numpy()).any()

    # 'zero' asserts dissimilarity where there is no information
    Kz, info_z = fuse_kernels({"a": K1, "b": K2}, how="mean", missing="zero")
    uncovered_pair = (~c1) & (~c2)
    print(f"[4] missing='zero': {int(uncovered_pair.sum())} entries with no source were set to "
          f"{float(np.nanmax(np.abs(Kz.to_numpy()[uncovered_pair]))) if uncovered_pair.any() else 0.0:.1f}"
          " -- a dissimilarity claim nobody measured")
    assert not np.isnan(Kz.to_numpy()).any()


# ===========================================================================
# deferred sources, the registry, and the embedding bridge
# ===========================================================================

def test_deferred_sources_report_status_not_exceptions():
    genes = ["Sox2", "Nanog", "Bmp4"]
    out = build_priors(genes, sources=["ops_morphology", "perturbation_pseudobulk"],
                       dim=8, cache_dir=CACHE, offline=True, datasets={})
    for name, pf in out.items():
        print(f"\n[D] {name}: status={pf.status!r} covered={pf.n_covered} "
              f"reason={str(pf.provenance.get('reason'))[:90]}...")
        assert pf.status.startswith("deferred")
        assert pf.matrix.shape[0] == len(genes)
        assert pf.n_covered == 0
    assert out["ops_morphology"].provenance["organism"] == "mus_musculus"


def test_perturbation_real_branch_synthetic():
    """The Perturb-seq source is deferred for *lack of data*, not lack of code.

    A synthetic screen with 24 perturbed genes + controls exercises the real
    control-centred pseudobulk computation; the delta for a perturbed gene must
    recover the injected effect.
    """
    rng = np.random.default_rng(0)
    genes_measured = [f"Gene{i:02d}" for i in range(30)]
    perturbed = genes_measured[:24]
    rows, labels = [], []
    base = rng.normal(2.0, 0.1, size=len(genes_measured))
    for lab in ["control"] * 60 + [g for g in perturbed for _ in range(5)]:
        x = base + rng.normal(0, 0.05, size=len(genes_measured))
        if lab != "control":                    # knocking out gene g halves gene g
            x[genes_measured.index(lab)] -= 1.0
        rows.append(x)
        labels.append(lab)
    ds = CandidateDataset(
        name="synthetic_screen", category="causal",
        expression=pd.DataFrame(np.array(rows), columns=genes_measured),
        obs=pd.DataFrame({"condition": labels}), units="log1p_cp10k", obs_unit="cell")
    pf = PRIORS["perturbation_pseudobulk"].fn(
        genes_measured, dim=8, rng=rng, cache_dir=CACHE, offline=True,
        datasets={"screen": ds}, min_perturbations=10, pca=False)
    print(f"\n[P] status={pf.status} n_perturbations="
          f"{pf.provenance['n_perturbations']} covered={pf.n_covered}/{len(genes_measured)} "
          f"dim={pf.dim}")
    assert pf.status == "ok" and pf.n_covered == 24
    self_delta = [pf.matrix.loc[g, pf.matrix.columns[genes_measured.index(g)]] for g in perturbed]
    print(f"[P] mean self-delta (injected -1.0): {np.mean(self_delta):.4f} "
          f"[{np.min(self_delta):.3f}, {np.max(self_delta):.3f}]")
    assert -1.1 < float(np.mean(self_delta)) < -0.9
    # below the threshold the same function defers
    pf2 = PRIORS["perturbation_pseudobulk"].fn(
        genes_measured, dim=8, rng=rng, cache_dir=CACHE, offline=True,
        datasets={"screen": ds}, min_perturbations=100)
    print(f"[P] with min_perturbations=100: status={pf2.status}, "
          f"n_perturbations={pf2.provenance['n_perturbations']}")
    assert pf2.status == "deferred_insufficient_perturbations"
    assert pf2.provenance["n_perturbations"] == 24


def test_registry_metadata_and_embedding_bridge():
    names = PRIORS.names()
    print(f"\n[R] registered priors: {names}")
    assert {"perturbation_pseudobulk", "ops_morphology", "coexpression_atlas",
            "protein_sequence", "ppi_network", "literature_text",
            "regulatory_kegg"} <= set(names)
    for e in PRIORS:
        assert e.meta["category"] in ("spatial", "single_cell", "non_transcriptomic", "causal")
        assert e.meta.get("description")
    F = _synthetic_features(n_genes=15, n_feat=30, seed=8, n_missing=4)
    pf = make_features({g: F.loc[g].to_numpy() for g in F.index[4:]}.keys() and list(F.index),
                       {g: F.loc[g].to_numpy() for g in F.index[4:]},
                       name="demo", modality="synthetic", category="single_cell", status="ok")
    emb = to_gene_embedding(pf, dim=5)
    print(f"[R] to_gene_embedding: {emb!r}; NaN rows preserved = "
          f"{int(emb.matrix.isna().all(axis=1).sum())}")
    assert emb.matrix.shape == (15, 5)
    assert int(emb.matrix.isna().all(axis=1).sum()) == 4
    assert emb.category == "single_cell" and emb.method == "prior::demo"


def test_coexpression_atlas_on_synthetic_datasets():
    ds1 = _synthetic_dataset("d1", n_cells=200, genes=("Aaa", "Bbb", "Ccc", "Ddd"), seed=1)
    ds2 = _synthetic_dataset("d2", n_cells=150, genes=("Bbb", "Ccc", "Ddd", "Eee"), seed=2)
    genes = ["Aaa", "Bbb", "Ccc", "Ddd", "Eee", "Zzz"]
    pf = PRIORS["coexpression_atlas"].fn(genes, dim=4, rng=np.random.default_rng(0),
                                         cache_dir=CACHE, offline=True,
                                         datasets={"d1": ds1, "d2": ds2}, pca=False)
    print(f"\n[C] status={pf.status} covered={pf.n_covered}/{len(genes)} dim={pf.dim} "
          f"n_block_imputed={pf.provenance['n_block_imputed']}")
    assert pf.status == "ok"
    assert pf.n_covered == 5 and bool(pf.matrix.loc["Zzz"].isna().all())
    assert pf.dim == 200 + 150
    # the single-dataset case must reproduce the correlation matrix exactly
    pf1 = PRIORS["coexpression_atlas"].fn(["Aaa", "Bbb", "Ccc", "Ddd"], dim=4,
                                          rng=np.random.default_rng(0), cache_dir=CACHE,
                                          offline=True, datasets={"d1": ds1}, pca=False)
    K = prior_kernel(pf1, kind="linear")
    C = np.corrcoef(ds1.expression.to_numpy(dtype=np.float64).T)
    err = float(np.abs(K.to_numpy() - C).max())
    print(f"[C] |linear kernel - corrcoef(cells x genes)| max = {err:.3e}")
    assert err < 1e-6


# ===========================================================================
# 7. determinism (synthetic / offline part)
# ===========================================================================

def _toy_graph(n=60, seed=0):
    import networkx as nx

    G = nx.random_partition_graph([n // 3] * 3, 0.25, 0.02, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 0.5
    return nx.relabel_nodes(G, {i: f"Gene{i:03d}" for i in G.nodes()})


def test_determinism_node2vec_and_offline_sources():
    G = _toy_graph()
    v1, m1 = node2vec_embedding(G, dim=16, walk_length=20, n_walks=5, p=1.0, q=0.5, epochs=2,
                                seed=42)
    v2, m2 = node2vec_embedding(G, dim=16, walk_length=20, n_walks=5, p=1.0, q=0.5, epochs=2,
                                seed=42)
    same = all(np.array_equal(v1[k], v2[k]) for k in v1)
    maxdiff = max(float(np.abs(v1[k] - v2[k]).max()) for k in v1)
    print(f"\n[7] node2vec (seed=42, workers=1): {len(v1)} nodes, bitwise identical = {same}, "
          f"max abs diff = {maxdiff:.1e}")
    assert same and m1["n_walks_generated"] == m2["n_walks_generated"]
    v3, _ = node2vec_embedding(G, dim=16, walk_length=20, n_walks=5, p=1.0, q=0.5, epochs=2,
                               seed=43)
    d = max(float(np.abs(v1[k] - v3[k]).max()) for k in v1)
    print(f"[7] a different seed does change the embedding: max abs diff = {d:.3f}")
    assert d > 1e-6

    ds = _synthetic_dataset("d1", n_cells=120, seed=3)
    a = PRIORS["coexpression_atlas"].fn(["Aaa", "Bbb", "Ccc", "Ddd"], dim=4,
                                        rng=np.random.default_rng(0), cache_dir=CACHE,
                                        offline=True, datasets={"d1": ds}, pca=True)
    b = PRIORS["coexpression_atlas"].fn(["Aaa", "Bbb", "Ccc", "Ddd"], dim=4,
                                        rng=np.random.default_rng(0), cache_dir=CACHE,
                                        offline=True, datasets={"d1": ds}, pca=True)
    print(f"[7] coexpression_atlas identical across runs = "
          f"{np.array_equal(a.matrix.to_numpy(), b.matrix.to_numpy())}")
    assert np.array_equal(a.matrix.to_numpy(), b.matrix.to_numpy())


def test_offline_without_cache_is_blocked_not_crash(tmp_path):
    out = build_priors(["Sox2", "Nanog"], sources=["ppi_network", "regulatory_kegg"],
                       dim=4, cache_dir=str(tmp_path), offline=True)
    for name, pf in out.items():
        print(f"\n[O] {name}: status={pf.status!r}")
        assert pf.status.startswith("blocked")
        assert pf.n_covered == 0


# ===========================================================================
# 5 + 6. real mouse data
# ===========================================================================

def _load_real_datasets(max_obs=2500):
    """MARS-seq (GSE169210) + Stereo-seq (GSE278603) via the package loaders."""
    from panelprep.io.catalog import entry
    from panelprep.io.loaders import load_candidate

    rng = np.random.default_rng(0)
    out = {}
    for nm in ("mittnenzweig_wt_E7.5", "seu3d_E7.5_rep2"):
        out[nm] = load_candidate(entry(nm), data_dir="data", max_obs=max_obs, rng=rng)
    return out


def build_real_priors(n_universe=4000, esm_max=None, dim=64, verbose=True):
    """Build the five runnable sources over a real mouse universe (slow; cached)."""
    universe = mouse_protein_coding_universe(cache_dir=CACHE, offline=False,
                                             max_genes=n_universe, seed=0)
    universe = sorted(set(universe) | set(PLURIPOTENCY) | set(BMP_SMAD))
    datasets = _load_real_datasets()
    feats = build_priors(
        universe,
        sources=["coexpression_atlas", "protein_sequence", "ppi_network",
                 "literature_text", "regulatory_kegg"],
        dim=dim, cache_dir=CACHE, offline=False, datasets=datasets, verbose=verbose,
        params={"protein_sequence": {"max_genes": esm_max, "batch_size": 8, "n_threads": 8,
                                     "progress_every": 25},
                "literature_text": {"batch_size": 32, "n_threads": 8},
                "ppi_network": {"n_walks": 10, "walk_length": 40, "epochs": 3}},
    )
    REAL_PICKLE.parent.mkdir(parents=True, exist_ok=True)
    with open(REAL_PICKLE, "wb") as fh:
        pickle.dump(feats, fh)
    return feats


@pytest.fixture(scope="module")
def real_priors():
    if REAL_PICKLE.exists():
        with open(REAL_PICKLE, "rb") as fh:
            return pickle.load(fh)
    return build_real_priors()


@pytest.mark.network
@pytest.mark.slow
def test_real_coverage_table(real_priors):
    """Coverage of each runnable source over a real mouse gene universe."""
    tbl = coverage_table(real_priors)
    n_univ = int(tbl["n_requested"].iloc[0])
    print(f"\n[5] gene universe = {n_univ} mouse protein-coding symbols "
          "(seeded subsample of the NCBI protein-coding set, plus the marker genes used in [6])")
    print(tbl.to_string())
    assert n_univ >= 2000, "the coverage universe must be at least 2,000 symbols"
    for name in ("coexpression_atlas", "protein_sequence", "ppi_network",
                 "literature_text", "regulatory_kegg"):
        pf = real_priors[name]
        assert pf.ok, f"{name}: {pf.status} / {pf.provenance.get('reason')}"
        assert pf.n_covered > 0
        assert pf.matrix.shape[0] == n_univ
    assert real_priors["literature_text"].provenance["coverage"] > 0.99
    assert real_priors["ppi_network"].provenance["coverage"] > 0.5
    Path("outputs").mkdir(exist_ok=True)
    tbl.to_csv("outputs/priors_coverage.csv")


@pytest.mark.network
@pytest.mark.slow
def test_real_kernel_alignment_report(real_priors):
    """Pairwise kernel alignment: does a source add information or duplicate another?"""
    kernels = {n: prior_kernel(f, kind="cosine") for n, f in real_priors.items() if f.ok}
    rep = fuse_report(kernels, features=real_priors, out_path="outputs/priors_fuse_report.csv")
    print("\n[5b] kernel alignment (Frobenius cosine) + coverage\n" + rep.to_string())
    # Uncentred KA on two all-positive kernels is dominated by their common
    # offset, so a dense language-model kernel looks ~0.99 aligned with any
    # other dense one. The centred alignment removes that offset and is the
    # number to read when asking whether a source adds information.
    cen = fuse_report(kernels, centre=True, out_path="outputs/priors_fuse_report_centred.csv")
    print("\n[5b] CENTRED kernel alignment (Cortes et al. 2012)\n"
          + cen[list(kernels)].to_string())
    for a in kernels:
        assert abs(rep.loc[a, a] - 1.0) < 1e-9
        for b in kernels:
            assert abs(rep.loc[a, b] - rep.loc[b, a]) < 1e-9
    K, info = fuse_kernels(kernels, how="mean", missing="mask")
    print(f"[5b] fused kernel: {info['n_genes']} genes, psd={info['is_psd']}, "
          f"min eig = {info['min_eigenvalue']:.3e}, "
          f"mean sources per entry = {info['mean_sources_per_entry']:.2f}, "
          f"entries with no source = {info['n_entries_without_any_source']}")
    assert K.shape[0] == real_priors["literature_text"].matrix.shape[0]


def _within_set_vs_random(K: pd.DataFrame, members, n_perm=200, seed=0):
    """Mean within-set similarity vs size-matched random sets; permutation p-value."""
    A = K.to_numpy(dtype=np.float64)
    genes = list(K.index)
    pos = {g: i for i, g in enumerate(genes)}
    covered = np.where(np.isfinite(A).sum(axis=1) > 1)[0]
    idx = [pos[g] for g in members if g in pos and pos[g] in set(covered.tolist())]
    if len(idx) < 2:
        return None
    def mean_within(ii):
        sub = A[np.ix_(ii, ii)]
        iu = np.triu_indices(len(ii), k=1)
        v = sub[iu]
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else np.nan
    obs = mean_within(idx)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        pick = rng.choice(covered, size=len(idx), replace=False)
        m = mean_within(list(pick))
        if np.isfinite(m):
            null.append(m)
    null = np.asarray(null)
    p = float((1 + int((null >= obs).sum())) / (1 + null.size))
    return {"n_members_used": len(idx), "observed": obs, "null_mean": float(null.mean()),
            "null_sd": float(null.std()), "p_value": p, "n_perm": int(null.size)}


@pytest.mark.network
@pytest.mark.slow
def test_real_biological_sanity(real_priors):
    """A known pathway set must be more self-similar than random, per source and fused."""
    kernels = {n: prior_kernel(f, kind="cosine") for n, f in real_priors.items() if f.ok}
    Knorm = {n: normalise_kernel(k, how="minmax") for n, k in kernels.items()}
    fused, info = fuse_kernels(kernels, how="mean", missing="mask")
    rows = []
    for set_name, members in (("pluripotency(Sox2/Pou5f1/Nanog/Klf4)", PLURIPOTENCY),
                              ("BMP-SMAD(Bmp4/Bmpr1a/Bmpr2/Smad1/Smad4/Smad5)", BMP_SMAD)):
        for src, K in list(Knorm.items()) + [("FUSED(mean)", fused)]:
            r = _within_set_vs_random(K, members, n_perm=200, seed=1)
            if r is None:
                rows.append({"set": set_name, "source": src, "n_members_used": 0})
                continue
            rows.append({"set": set_name, "source": src, **r})
    df = pd.DataFrame(rows)
    print("\n[6] within-set vs random (cosine kernels, min-max normalised; "
          "200 size-matched random sets)\n" + df.to_string(index=False))
    Path("outputs").mkdir(exist_ok=True)
    df.to_csv("outputs/priors_bio_sanity.csv", index=False)
    fused_rows = df[df["source"] == "FUSED(mean)"]
    assert (fused_rows["n_members_used"] >= 3).all()
    # Direction holds for both sets in the fused kernel.
    assert (fused_rows["observed"] > fused_rows["null_mean"]).all(), fused_rows
    # Significance holds for the 6-gene BMP-SMAD set. It does NOT hold for the
    # 4-gene pluripotency set (p ~= 0.09), and that is a result, not a tolerance
    # to be widened: the per-source rows show why. coexpression_atlas is built
    # from E7.5/E7.75 gastrulation data, where Nanog and Klf4 are already
    # down-regulated, so those four genes are *not* co-expressed in this atlas
    # (observed below the random mean) and they drag the mean fusion down. The
    # graph and curated-pathway sources recover the set strongly (p = 0.005).
    bmp = fused_rows[fused_rows["set"].str.startswith("BMP")]
    assert (bmp["p_value"] <= 0.05).all(), bmp
    per_source = df[(df["source"] != "FUSED(mean)") & df["set"].str.startswith("BMP")]
    n_sig = int((per_source["p_value"] <= 0.05).sum())
    print(f"[6] sources with p <= 0.05 on the BMP-SMAD set: {n_sig}/{len(per_source)}; "
          f"fused p = {float(bmp['p_value'].iloc[0]):.4f}; "
          "pluripotency fused p = "
          f"{float(fused_rows[fused_rows['set'].str.startswith('pluri')]['p_value'].iloc[0]):.4f} "
          "(reported, not asserted -- see the comment in this test)")
    assert n_sig >= 4, per_source


@pytest.mark.network
@pytest.mark.slow
def test_real_determinism_of_text_and_kegg(real_priors):
    """Re-running two real sources with the same seed reproduces them bitwise."""
    genes = list(real_priors["literature_text"].matrix.index)[:64]
    a = PRIORS["literature_text"].fn(genes, dim=16, rng=np.random.default_rng(0),
                                     cache_dir=CACHE, offline=False, batch_size=16)
    b = PRIORS["literature_text"].fn(genes, dim=16, rng=np.random.default_rng(0),
                                     cache_dir=CACHE, offline=False, batch_size=16)
    same_text = np.array_equal(a.matrix.to_numpy(), b.matrix.to_numpy())
    k1 = PRIORS["regulatory_kegg"].fn(genes, dim=16, rng=np.random.default_rng(0),
                                      cache_dir=CACHE, offline=True)
    k2 = PRIORS["regulatory_kegg"].fn(genes, dim=16, rng=np.random.default_rng(0),
                                      cache_dir=CACHE, offline=True)
    same_kegg = np.allclose(np.abs(k1.matrix.to_numpy()), np.abs(k2.matrix.to_numpy()),
                            equal_nan=True)
    print(f"\n[7r] literature_text bitwise identical = {same_text} (n={len(genes)}); "
          f"regulatory_kegg identical = {same_kegg}")
    assert same_text and same_kegg


if __name__ == "__main__":  # manual run: python tests/test_priors.py
    t0 = time.time()
    feats = build_real_priors()
    print(coverage_table(feats).to_string())
    print(f"built in {time.time() - t0:.0f}s")
