"""Acceptance tests for the expression gene embedders (brief item 2).

Everything runs against ``panelprep.io.synthetic.make_synthetic_case()``, whose
generative model is known: gene ``g`` is produced from ``truth["loadings"][g]``,
a ``(G, K)`` matrix of latent-program loadings. Two genes with similar loadings
therefore have similar expression profiles in any candidate simulated from those
loadings, which is what makes the recovery check below a real acceptance test
rather than a smoke test.

What is asserted
----------------
1. ``test_shape_and_index``        -- ``(len(genes), dim)`` with the gene index preserved *in order*.
2. ``test_missing_genes_*``        -- absent genes are NaN rows for separate methods and
                                      are filled in from the other views by joint methods.
3. ``test_determinism``            -- same seed -> bit-identical matrices.
4. ``test_recovery_of_true_loadings`` -- the scientific acceptance test: the k=15 kNN graph of
                                      each embedding must agree with the kNN graph of the true
                                      loadings far better than a gene-label permutation of the
                                      same embedding.
5. ``test_pca_baseline_is_fast``   -- < 5 s on the default fixture.

Specificity, and why the recovery test is split by candidate
------------------------------------------------------------
Only ``sibling`` and ``same_program_diff_mixture`` are simulated from
``truth["loadings"]``. ``unrelated`` is simulated from a *different* loading
matrix, so an embedder that recovered ``truth["loadings"]`` from it would be
recovering an artefact of the metric, not biology; ``test_recovery_is_specific``
asserts that it does not. ``shuffled_structure`` permutes each gene
independently across cells, destroying co-expression while preserving each
gene's marginal, so a weak residual signal (mean expression level is still a
function of the loadings) is expected there and is reported, not asserted.

MODIFY HERE
-----------
``DIM`` (latent width used by the tests), ``K_NEIGHBOURS`` (neighbourhood size
for the recovery check), ``N_PERMUTATIONS``, and ``MIN_SIGMA_ABOVE_PERMUTATION``
(how far above the permutation baseline a method must land to pass). The
threshold is expressed in standard deviations of the permutation baseline so it
does not have to be retuned when the fixture size changes.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import NearestNeighbors

from panelprep.embed import EMBEDDERS, run_embedders
from panelprep.io.synthetic import make_synthetic_case
from panelprep.schemas import CandidateDataset, GeneEmbedding

# --- MODIFY HERE: test-wide knobs -----------------------------------------
DIM = 16
K_NEIGHBOURS = 15
N_PERMUTATIONS = 10
MIN_SIGMA_ABOVE_PERMUTATION = 5.0   # recovery must exceed perm_mean + 5 * perm_sd
MAX_RATIO_FOR_NEGATIVE_CONTROL = 2.0
N_DROPPED_GENES = 20                # genes removed to build the "holey" dataset

SEPARATE_METHODS = ["pca_baseline", "ae_separate", "vae_separate"]
JOINT_METHODS = ["joint_ae", "dcca", "contrastive_ae"]
ALL_METHODS = SEPARATE_METHODS + JOINT_METHODS
#: Candidates simulated from truth["loadings"]: recovery is expected here only.
ON_TRUTH = ["sibling", "same_program_diff_mixture"]
#: Simulated from a different loading matrix: recovery must NOT happen.
OFF_TRUTH = ["unrelated"]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def case():
    """The synthetic fixture at its documented defaults (200 genes, 800 cells)."""
    return make_synthetic_case()


@pytest.fixture(scope="module")
def genes(case) -> List[str]:
    return list(case.truth["genes"])


@pytest.fixture(scope="module")
def dropped(genes) -> List[str]:
    """Genes removed from the "holey" dataset to exercise the presence mask."""
    return list(genes[:N_DROPPED_GENES])


@pytest.fixture(scope="module")
def holey(case, dropped) -> CandidateDataset:
    """``sibling`` with ``N_DROPPED_GENES`` gene columns removed.

    The fixture's own candidates all carry every gene, so a dataset with a
    genuine gene gap has to be constructed here; without it the presence mask
    is never exercised.
    """
    return CandidateDataset(
        name="holey",
        category="single_cell",
        expression=case.candidates["sibling"].expression.drop(columns=dropped),
        units=case.candidates["sibling"].units,
        meta={"description": f"sibling with {N_DROPPED_GENES} genes removed"},
    )


@pytest.fixture(scope="module")
def no_matrix() -> CandidateDataset:
    """A candidate with no expression matrix (a PPI graph, in the real catalogue)."""
    return CandidateDataset(name="ppi_graph", category="non_transcriptomic",
                            expression=None, meta={"description": "no cells"})


def fit(method: str, datasets: Dict[str, CandidateDataset], genes: Sequence[str],
        *, dim: int = DIM, seed: int = 0, **overrides) -> GeneEmbedding:
    """Call one registered embedder directly with its registered defaults."""
    entry = EMBEDDERS[method]
    kw = dict(entry.meta.get("defaults", {}))
    kw.update(overrides)
    return entry.fn(datasets, genes=list(genes), dim=dim,
                    rng=np.random.default_rng(seed), **kw)


def datasets_for(method: str, case) -> Dict[str, CandidateDataset]:
    """One dataset for separate methods, the two on-truth datasets for joint ones."""
    if EMBEDDERS[method].meta["mode"] == "separate":
        return {"sibling": case.candidates["sibling"]}
    return {k: case.candidates[k] for k in ON_TRUTH}


# ---------------------------------------------------------------------------
# kNN recovery machinery
# ---------------------------------------------------------------------------

def knn_sets(X: np.ndarray, k: int) -> List[set]:
    """Index sets of the ``k`` nearest neighbours of each row, self excluded."""
    X = np.asarray(X, dtype=float)
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0])).fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    return [set([j for j in row if j != i][:k]) for i, row in enumerate(idx)]


def mean_jaccard(a: List[set], b: List[set]) -> float:
    """Mean per-row Jaccard overlap of two neighbour-set lists."""
    return float(np.mean([len(x & y) / max(len(x | y), 1) for x, y in zip(a, b)]))


def permutation_baseline(truth_sets: List[set], emb: np.ndarray, k: int,
                         n_perm: int = N_PERMUTATIONS, seed: int = 7) -> tuple:
    """Mean and sd of the Jaccard overlap under gene-label permutation.

    Permuting the *rows* of the embedding permutes which gene owns which vector,
    which is exactly a label permutation: the geometry is untouched, only the
    correspondence to the truth is destroyed.
    """
    rng = np.random.default_rng(seed)
    vals = [mean_jaccard(truth_sets, knn_sets(emb[rng.permutation(emb.shape[0])], k))
            for _ in range(n_perm)]
    return float(np.mean(vals)), float(np.std(vals))


def recovery(truth_sets: List[set], emb_matrix: pd.DataFrame, k: int = K_NEIGHBOURS) -> dict:
    """Jaccard overlap of an embedding's kNN graph with the truth's, vs permutation."""
    M = emb_matrix.to_numpy(dtype=float)
    finite = np.isfinite(M).all(axis=1)
    M = M[finite]
    sets = [s for s, f in zip(truth_sets, finite) if f]
    j = mean_jaccard(sets, knn_sets(M, k))
    pm, ps = permutation_baseline(sets, M, k)
    return {"jaccard": j, "perm_mean": pm, "perm_sd": ps,
            "ratio": j / pm if pm > 0 else float("inf"),
            "sigma_above": (j - pm) / ps if ps > 0 else float("inf"),
            "n_genes": int(finite.sum())}


# ---------------------------------------------------------------------------
# 1. shape, index order, category tag
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_shape_and_index(method, case, genes):
    emb = fit(method, datasets_for(method, case), genes)
    assert emb.matrix.shape == (len(genes), DIM), f"{method}: {emb.matrix.shape}"
    assert list(emb.matrix.index) == list(genes), f"{method}: gene index reordered"
    assert emb.category in ("spatial", "single_cell")
    assert emb.provenance["n_genes_requested"] == len(genes)
    print(f"[shape] {method:15s} {emb.matrix.shape} category={emb.category} "
          f"dim_effective={emb.provenance.get('dim_effective')}")


def test_separate_category_follows_the_dataset(case, genes):
    """The four-way tag on a separate embedding is the *dataset's* category."""
    for dname in ["sibling", "same_program_diff_mixture"]:
        emb = fit("pca_baseline", {dname: case.candidates[dname]}, genes)
        assert emb.category == case.candidates[dname].category, dname
    print("[category] pca_baseline::sibling -> spatial, "
          "pca_baseline::same_program_diff_mixture -> single_cell")


def test_joint_category_is_spatial_only_if_all_views_are(case, genes):
    both_spatial = {k: case.candidates[k] for k in ["sibling", "unrelated"]}
    mixed = {k: case.candidates[k] for k in ON_TRUTH}
    assert fit("joint_ae", both_spatial, genes).category == "spatial"
    assert fit("joint_ae", mixed, genes).category == "single_cell"
    print("[category] joint_ae: spatial+spatial -> spatial, spatial+single_cell -> single_cell")


# ---------------------------------------------------------------------------
# 2. the presence mask
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", SEPARATE_METHODS)
def test_missing_genes_are_nan_rows_separate(method, holey, genes, dropped):
    """A gene the dataset does not contain must not get a learned representation."""
    emb = fit(method, {"holey": holey}, genes)
    M = emb.matrix
    assert bool(M.loc[dropped].isna().all(axis=None)), f"{method}: dropped genes not NaN"
    kept = M.drop(index=dropped).to_numpy(dtype=float)
    assert np.isfinite(kept).all(), f"{method}: present genes have non-finite values"
    assert int(M.isna().all(axis=1).sum()) == len(dropped)
    assert emb.provenance["frac_genes_masked_out"] == pytest.approx(
        len(dropped) / len(genes))
    print(f"[mask] {method:15s} nan_rows={len(dropped)}/{len(genes)} "
          f"frac_masked={emb.provenance['frac_genes_masked_out']:.3f}")


@pytest.mark.parametrize("method", JOINT_METHODS)
def test_missing_genes_are_filled_by_joint_methods(method, holey, case, genes, dropped):
    """The point of a joint model: a gene missing from one view is learned from the other."""
    emb = fit(method, {"holey": holey, "sibling": case.candidates["sibling"]}, genes)
    M = emb.matrix
    assert np.isfinite(M.loc[dropped].to_numpy(dtype=float)).all(), \
        f"{method}: genes missing from one view were not filled in"
    assert int(M.isna().all(axis=1).sum()) == 0
    assert emb.provenance["n_genes_in_all_views"] == len(genes) - len(dropped)
    print(f"[mask] {method:15s} shared={emb.provenance['n_genes_in_all_views']} "
          f"outside_intersection={emb.provenance.get('n_genes_outside_intersection', 0)} "
          f"nan_rows=0")


def test_joint_methods_skip_datasets_without_a_matrix(case, genes, no_matrix):
    """non_transcriptomic / causal candidates carry no cells and must be ignored."""
    ds = {**{k: case.candidates[k] for k in ON_TRUTH}, "ppi_graph": no_matrix}
    emb = fit("joint_ae", ds, genes)
    assert emb.provenance["views"] == ON_TRUTH
    assert emb.matrix.shape == (len(genes), DIM)
    print(f"[views] joint_ae ignored ppi_graph (no matrix); views={emb.provenance['views']}")


# ---------------------------------------------------------------------------
# configuration errors (not data properties)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["dcca", "contrastive_ae"])
def test_multiview_methods_require_two_views(method, case, genes):
    with pytest.raises(ValueError, match="at least 2"):
        fit(method, {"sibling": case.candidates["sibling"]}, genes)
    print(f"[config] {method} with 1 view -> ValueError naming the 2-view requirement")


def test_joint_ae_works_with_one_view(case, genes):
    emb = fit("joint_ae", {"sibling": case.candidates["sibling"]}, genes)
    assert emb.matrix.shape == (len(genes), DIM)
    assert np.isfinite(emb.matrix.to_numpy(dtype=float)).all()
    print("[config] joint_ae with 1 view -> runs (degenerates to a single-view AE)")


def test_too_few_shared_genes_is_a_status_not_an_exception(case, genes):
    """Too little gene overlap is a property of the data, so it sets a status.

    The ``dcca`` guard is ``n_shared >= max(3, dim_effective + 1)`` -- the
    canonical correlation needs strictly more samples than latent dimensions --
    so at ``dim=32`` ten shared genes is not enough. ``contrastive_ae`` needs
    only four (a positive pair plus two negatives). Both return an all-NaN
    matrix of the right shape with the reason recorded, rather than raising.
    """
    expr = case.candidates["sibling"].expression

    def slice_ds(name, cols):
        return CandidateDataset(name=name, category="single_cell",
                                expression=expr.loc[:, cols],
                                meta={"description": "gene-sliced sibling"})

    pairs = {
        "dcca": (slice_ds("a", genes[:100]), slice_ds("b", genes[90:]), 10),
        "contrastive_ae": (slice_ds("a", genes[:100]), slice_ds("b", genes[98:]), 2),
    }
    for method, (a, b, n_shared) in pairs.items():
        emb = fit(method, {"a": a, "b": b}, genes, dim=32)
        assert emb.provenance["status"] == "insufficient_shared_genes", method
        assert emb.provenance["n_genes_in_all_views"] == n_shared, method
        assert emb.matrix.shape == (len(genes), 32), method
        assert bool(emb.matrix.isna().all(axis=None)), f"{method}: expected an all-NaN matrix"
        print(f"[status] {method:15s} {n_shared} shared genes at dim=32 -> "
              f"status={emb.provenance['status']!r} "
              f"reason={emb.provenance['reason']!r}")


def test_separate_methods_refuse_several_datasets(case, genes):
    """Cells are not comparable across datasets, so this is a configuration error."""
    with pytest.raises(ValueError, match="exactly one dataset"):
        fit("ae_separate", {k: case.candidates[k] for k in ON_TRUTH}, genes)
    print("[config] ae_separate with 2 datasets -> ValueError (cells are not paired)")


# ---------------------------------------------------------------------------
# 3. determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_determinism(method, case, genes):
    ds = datasets_for(method, case)
    a = fit(method, ds, genes, seed=42).matrix
    b = fit(method, ds, genes, seed=42).matrix
    assert list(a.index) == list(b.index)
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True), \
        f"{method}: same seed gave different matrices"
    print(f"[determinism] {method:15s} seed=42 twice -> bit-identical "
          f"({a.shape[0]}x{a.shape[1]})")


# ---------------------------------------------------------------------------
# 4. THE SCIENTIFIC ACCEPTANCE TEST: recovery of the true loadings
# ---------------------------------------------------------------------------

def test_recovery_of_true_loadings(case, genes, capsys):
    """kNN graph of each embedding vs kNN graph of ``truth["loadings"]``.

    Reports the mean Jaccard overlap at k=15 for every method together with the
    gene-label permutation baseline, and requires every method to land at least
    ``MIN_SIGMA_ABOVE_PERMUTATION`` permutation-standard-deviations above it.
    """
    truth_sets = knn_sets(case.truth["loadings"], K_NEIGHBOURS)
    rows = []
    for method in ALL_METHODS:
        t0 = time.time()
        emb = fit(method, datasets_for(method, case), genes)
        r = recovery(truth_sets, emb.matrix)
        r.update({"method": method, "mode": EMBEDDERS[method].meta["mode"],
                  "datasets": "+".join(datasets_for(method, case)),
                  "seconds": round(time.time() - t0, 2)})
        rows.append(r)
    table = pd.DataFrame(rows)[
        ["method", "mode", "datasets", "n_genes", "jaccard", "perm_mean",
         "perm_sd", "ratio", "sigma_above", "seconds"]]
    print("\n--- kNN recovery of truth['loadings'], k=%d ---" % K_NEIGHBOURS)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    failures = [r for r in rows if r["sigma_above"] < MIN_SIGMA_ABOVE_PERMUTATION]
    assert not failures, (
        "methods not beating the label-permutation baseline by "
        f"{MIN_SIGMA_ABOVE_PERMUTATION} sd: "
        + ", ".join(f"{r['method']} (jaccard {r['jaccard']:.4f} vs perm "
                    f"{r['perm_mean']:.4f} +/- {r['perm_sd']:.4f})" for r in failures))

    baseline = next(r for r in rows if r["method"] == "pca_baseline")
    deep = [r for r in rows if r["method"] != "pca_baseline"]
    print(f"\npca_baseline jaccard = {baseline['jaccard']:.4f}; deep methods "
          "above it: "
          + ", ".join(f"{r['method']}={r['jaccard']:.4f}" for r in deep
                      if r["jaccard"] > baseline["jaccard"])
          + " | below it: "
          + ", ".join(f"{r['method']}={r['jaccard']:.4f}" for r in deep
                      if r["jaccard"] <= baseline["jaccard"]))


def test_recovery_is_specific(case, genes):
    """A candidate built from *different* loadings must not recover ``truth["loadings"]``.

    This is what distinguishes a real recovery signal from a property of the
    metric. ``shuffled_structure`` is reported but not asserted on: permuting
    each gene across cells preserves its marginal, and the marginal is still a
    function of the loadings, so a weak residual overlap is expected there.
    """
    truth_sets = knn_sets(case.truth["loadings"], K_NEIGHBOURS)
    reported = {}
    for dname in OFF_TRUTH + ["shuffled_structure"]:
        emb = fit("pca_baseline", {dname: case.candidates[dname]}, genes)
        reported[dname] = recovery(truth_sets, emb.matrix)
        print(f"[specificity] pca_baseline::{dname:26s} jaccard="
              f"{reported[dname]['jaccard']:.4f} perm={reported[dname]['perm_mean']:.4f} "
              f"ratio={reported[dname]['ratio']:.2f}")
    for dname in OFF_TRUTH:
        assert reported[dname]["ratio"] < MAX_RATIO_FOR_NEGATIVE_CONTROL, (
            f"{dname} was built from different loadings yet scored "
            f"{reported[dname]['ratio']:.2f}x the permutation baseline")


# ---------------------------------------------------------------------------
# 5. runtime
# ---------------------------------------------------------------------------

def test_pca_baseline_is_fast(case, genes):
    t0 = time.time()
    emb = fit("pca_baseline", {"sibling": case.candidates["sibling"]}, genes)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"pca_baseline took {elapsed:.2f}s"
    assert emb.matrix.shape == (len(genes), DIM)
    print(f"[runtime] pca_baseline on {len(genes)} genes x "
          f"{case.candidates['sibling'].n_obs} cells: {elapsed:.3f}s "
          f"(evr={emb.provenance['explained_variance_ratio']:.3f})")


def test_runner_keys_and_runs_every_method(case, genes):
    """``run_embedders`` plumbing: separate methods are keyed per dataset."""
    got = run_embedders(case.candidates, embedder_names=ALL_METHODS,
                        genes=genes, dim=8, seed=0)
    expected_separate = {f"{m}::{d}" for m in SEPARATE_METHODS for d in case.candidates}
    assert expected_separate <= set(got)
    assert set(JOINT_METHODS) <= set(got)
    for k, v in got.items():
        assert v.matrix.shape == (len(genes), 8), k
    print(f"[runner] produced {len(got)} embeddings: "
          f"{len(expected_separate)} separate + {len(JOINT_METHODS)} joint")
