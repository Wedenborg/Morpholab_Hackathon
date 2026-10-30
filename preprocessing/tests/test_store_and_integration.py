"""Tests for the embedding store, the Morpholab adapter and the figure gallery.

What is actually being checked
------------------------------
* the store round-trips a ``GeneEmbedding`` without losing the matrix, the
  category or the provenance, and keeps NaN rows as NaN;
* ``ensure`` calls the build function **once** -- that property is the whole
  point of the cache, so it is asserted by counting invocations rather than by
  timing;
* ``corr_mae`` reproduces the Morpholab ``select_dataset`` score *exactly*
  (bit-for-bit against the formula written out longhand), which is the contract
  the merge depends on;
* ``factor_analysis`` honours this package's embedder contract (``G x d``
  indexed by gene symbol, absent genes NaN, one embedding per call);
* ``run_analysis`` refuses to recompute a ``CACHED_ONLY`` source and reports it
  as blocked instead of silently training for an hour;
* the gallery writes the figures it says it writes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from panelprep.config import Config
from panelprep.embed import EMBEDDERS, run_embedders
from panelprep.embed.store import GROUPS, EmbeddingStore
from panelprep.io.synthetic import make_synthetic_case
from panelprep.run import CACHED_ONLY, run_analysis
from panelprep.schemas import GeneEmbedding
from panelprep.scores import SCORES, run_scores

import panelprep.integrations.morpholab as morpholab  # registers corr_mae etc.


@pytest.fixture(scope="module")
def case():
    return make_synthetic_case(seed=0)


@pytest.fixture
def embedding():
    rng = np.random.default_rng(0)
    genes = [f"Gene{i:03d}" for i in range(40)]
    m = pd.DataFrame(rng.normal(size=(40, 8)).astype("float32"), index=genes)
    m.iloc[3] = np.nan                       # an uncovered gene
    return GeneEmbedding(matrix=m, method="demo", category="spatial",
                         provenance={"status": "implemented", "n_obs": 123,
                                     "note": "fixture"})


# ------------------------------------------------------------------- store
def test_store_round_trip_preserves_values_metadata_and_nan(tmp_path, embedding):
    store = EmbeddingStore(tmp_path / "store")
    path = store.put(embedding, "demo::ds1", group="expression")
    assert path.exists() and path.suffix == ".parquet"

    back = store.get("demo::ds1")
    assert back.matrix.shape == embedding.matrix.shape
    np.testing.assert_allclose(back.matrix.to_numpy(), embedding.matrix.to_numpy(),
                               equal_nan=True)
    assert list(back.matrix.index) == list(embedding.matrix.index)
    assert back.category == "spatial"
    assert back.method == "demo"
    assert back.provenance["n_obs"] == 123          # sidecar survived
    assert back.provenance["store_group"] == "expression"
    # the uncovered gene is still uncovered, not zero-filled
    assert bool(back.matrix.iloc[3].isna().all())


def test_store_reindexes_to_a_requested_gene_list(tmp_path, embedding):
    store = EmbeddingStore(tmp_path / "store")
    store.put(embedding, "demo", group="expression")
    wanted = ["Gene000", "NotAGene", "Gene001"]
    back = store.get("demo", genes=wanted)
    assert list(back.matrix.index) == wanted
    assert bool(back.matrix.loc["NotAGene"].isna().all())


def test_ensure_builds_once_then_loads(tmp_path, embedding):
    store = EmbeddingStore(tmp_path / "store")
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return embedding

    a = store.ensure("cached", build, group="external", verbose=False)
    b = store.ensure("cached", build, group="external", verbose=False)
    assert calls["n"] == 1, "ensure() must not rebuild what is already on disk"
    np.testing.assert_allclose(a.matrix.to_numpy(), b.matrix.to_numpy(),
                               equal_nan=True)

    store.ensure("cached", build, group="external", recompute=True, verbose=False)
    assert calls["n"] == 2, "recompute=True must bypass the cache"


def test_inventory_reports_shape_and_path(tmp_path, embedding):
    store = EmbeddingStore(tmp_path / "store")
    store.put(embedding, "a::x", group="expression")
    store.put(embedding, "b", group="prior")
    inv = store.inventory(write=True)
    assert len(inv) == 2
    assert set(inv["group"]) == {"expression", "prior"}
    assert (inv["shape"] == "40 x 8").all()
    assert (inv["n_nan_genes"] == 1).all()
    assert all(Pth and Pth.endswith(".parquet") for Pth in inv["path"])
    assert (tmp_path / "store" / "manifest.csv").exists()


def test_store_missing_entry_is_explicit(tmp_path):
    strict = EmbeddingStore(tmp_path / "s1")
    with pytest.raises(KeyError):
        strict.get("nope")
    lax = EmbeddingStore(tmp_path / "s2", strict=False)
    assert lax.get("nope") is None


def test_groups_are_real_directories(tmp_path):
    store = EmbeddingStore(tmp_path / "store")
    for g in GROUPS:
        assert (store.root / g).is_dir()


# --------------------------------------------------------- morpholab score
def test_corr_mae_equals_the_morpholab_formula_exactly(case):
    """The merge contract: our registry row must reproduce their number."""
    query, candidates = case.query, case.candidates
    panel = list(query.panel)
    res = run_scores(query, list(candidates.values()), score_names=["corr_mae"],
                     seed=0)
    iu = np.triu_indices(len(panel), k=1)
    A = query.expression[panel].corr().to_numpy()[iu]       # Pearson, as they do
    for name, cand in candidates.items():
        B = cand.expression[panel].corr().to_numpy()[iu]
        theirs = float(np.mean(np.abs(A - B)))
        ours = float(res[(res["dataset"] == name) &
                         (res["score"] == "corr_mae")]["value"].iloc[0])
        assert ours == pytest.approx(theirs, abs=1e-12), name


def test_corr_mae_is_registered_with_a_direction(case):
    assert "corr_mae" in SCORES
    assert SCORES["corr_mae"].meta["orientation"] == "lower_is_better"


def test_pearson_not_mae_is_what_inverts_the_known_order(case):
    """Documented finding, pinned as a test.

    The synthetic fixtures have a known ground-truth order. Both Pearson
    variants put ``same_program_diff_mixture`` ahead of ``sibling``; both
    Spearman variants get it right -- so the Pearson default in
    ``select_dataset`` is what drives the inversion, not the MAE reducer.
    """
    query, candidates = case.query, case.candidates
    panel = list(query.panel)
    iu = np.triu_indices(len(panel), k=1)

    def order(method, reducer):
        A = query.expression[panel].corr(method=method).to_numpy()[iu]
        vals = {}
        for name, cand in candidates.items():
            B = cand.expression[panel].corr(method=method).to_numpy()[iu]
            vals[name] = reducer(A - B)
        return sorted(vals, key=vals.get)

    mae = lambda d: float(np.mean(np.abs(d)))          # noqa: E731
    rms = lambda d: float(np.sqrt(np.mean(d ** 2)))    # noqa: E731
    for reducer in (mae, rms):
        assert order("pearson", reducer)[0] == "same_program_diff_mixture"
        assert order("spearman", reducer)[0] == "sibling"


# ------------------------------------------------------ morpholab embedder
def test_factor_analysis_follows_the_embedder_contract(case):
    genes = sorted(set(next(iter(case.candidates.values())).genes))[:60]
    out = run_embedders(case.candidates, embedder_names=["factor_analysis"],
                        genes=genes, dim=6, seed=0)
    assert out, "factor_analysis produced nothing"
    for key, emb in out.items():
        assert key.startswith("factor_analysis::")
        assert isinstance(emb, GeneEmbedding)
        assert emb.matrix.shape == (len(genes), 6)
        assert list(emb.matrix.index) == genes
        assert emb.provenance["observation_unit"].startswith("cell")
        assert emb.provenance["status"] == "implemented"


def test_factor_analysis_rejects_multi_dataset_calls(case):
    """A separate-mode embedder is called once per dataset, by contract."""
    entry = EMBEDDERS["factor_analysis"]
    assert entry.meta["mode"] == "separate"
    with pytest.raises(ValueError, match="exactly one"):
        entry.fn(case.candidates, genes=["Gene0000"], dim=2,
                 rng=np.random.default_rng(0))


def test_loadings_to_embedding_copies_without_touching_numbers():
    loadings = pd.DataFrame(np.arange(12, dtype="float32").reshape(4, 3),
                            index=["A", "B", "C", "D"],
                            columns=["factor_1", "factor_2", "factor_3"])
    result = {"loadings": loadings, "score": 0.0123, "gene_set_name": "tome:7.5",
              "reference": {"dataset": "tome", "timepoint": "7.5",
                            "layer": "logcounts"},
              "constant_genes": ["Z"]}
    emb = morpholab.loadings_to_embedding(result)
    np.testing.assert_array_equal(emb.matrix.to_numpy(), loadings.to_numpy())
    assert emb.provenance["selection_score_mean_abs_corr_diff"] == 0.0123
    assert emb.provenance["n_constant_genes_dropped"] == 1
    assert emb.provenance["observation_unit"] == "gene"


def test_reference_to_catalog_shape():
    entry = morpholab.reference_to_catalog(
        {"dataset": "tome", "timepoint": "7.5", "layer": "logcounts"})
    assert entry["name"] == "tome_7.5"
    assert entry["loader"] == "morpholab"
    assert entry["loader_kwargs"]["layer"] == "logcounts"
    spatial = morpholab.reference_to_catalog({"dataset": "seu-3d"})
    assert spatial["category"] == "spatial"      # name announces it


def test_gene_key_rule_matches_morpholab_character_for_character():
    """Both repos canonicalise identifiers the same way.

    Re-implements their ``_gene_key`` here rather than importing the repo, so
    the test runs without a checkout; :func:`check_gene_key_agreement` does the
    live comparison when one is present.
    """
    import re

    from panelprep.normalize import gene_key

    def theirs(name):
        name = str(name).upper()
        return re.sub(r"\.\d+$", "", name) if name.startswith("ENS") else name

    for s in ["Sox2", "sox2", "T", "ENSMUSG00000074637.4", "ENSMUSG00000062327",
              "Nkx1-2", "mt-Co1", "Hba-x"]:
        assert gene_key(s) == theirs(s), s


# ------------------------------------------------------------ run_analysis
def test_cached_only_sources_are_blocked_not_recomputed(tmp_path, case):
    """A missing expensive source must be reported, never silently trained."""
    assert "esm2_sequence" in CACHED_ONLY
    config = Config(run_name="t", outdir=str(tmp_path / "out"), seed=0,
                    scores=["corr_mae"], embedders=["pca_baseline", "esm2_sequence"],
                    embedding_dim=4, max_genes=40, annotation_sources=[],
                    offline=True)
    result = run_analysis(config, query=case.query, candidates=case.candidates,
                          stages=("embed",), verbose=False)
    assert "esm2_sequence" in result["blocked"]
    assert "build_embedding_store" in result["blocked"]["esm2_sequence"]
    assert any(k.startswith("pca_baseline::") for k in result["embeddings"])


def test_run_analysis_scores_and_writes_the_heatmaps(tmp_path, case):
    config = Config(run_name="t", outdir=str(tmp_path / "out"), seed=0,
                    scores=["corr_mae", "corr_frobenius"], embedders=[],
                    embedding_dim=4, max_genes=40, annotation_sources=[],
                    offline=True)
    result = run_analysis(config, query=case.query, candidates=case.candidates,
                          stages=("score",), verbose=False)
    table = result["score_table"]
    assert set(table.columns) == {"corr_mae", "corr_frobenius"}
    assert len(table) == len(case.candidates)
    from pathlib import Path
    for key in ("score_heatmap_agreement", "score_heatmap_raw"):
        assert Path(result["figures"][key]).exists()
    assert (Path(config.out) / "score_results.csv").exists()


def test_run_analysis_loads_a_store_entry_without_an_embedder(tmp_path, case,
                                                              embedding):
    """A prior source has no embedder entry; being in the store is enough."""
    store = EmbeddingStore(tmp_path / "store")
    genes = sorted(set(next(iter(case.candidates.values())).genes))[:40]
    m = pd.DataFrame(np.zeros((len(genes), 5), dtype="float32"), index=genes)
    store.put(GeneEmbedding(matrix=m, method="made_up_prior",
                            category="non_transcriptomic"),
              "made_up_prior", group="prior")
    config = Config(run_name="t", outdir=str(tmp_path / "out"), seed=0,
                    scores=[], embedders=["made_up_prior"], embedding_dim=4,
                    max_genes=40, annotation_sources=[], offline=True)
    result = run_analysis(config, query=case.query, candidates=case.candidates,
                          store=store, stages=("embed",), verbose=False)
    assert "made_up_prior" in result["embeddings"]
    assert not result["blocked"]


# ----------------------------------------------------------------- gallery
def test_visualize_all_writes_the_figures_it_reports(tmp_path, case):
    from panelprep.viz.gallery import visualize_all

    genes = sorted(set(next(iter(case.candidates.values())).genes))[:60]
    embs = run_embedders(case.candidates, embedder_names=["pca_baseline"],
                         genes=genes, dim=4, seed=0)
    manifest = visualize_all(embs, None, outdir=tmp_path / "figs",
                             method="pca", group_by=None, seed=0,
                             quality=False, verbose=False)
    from pathlib import Path
    assert not manifest.empty
    for p in manifest["path"]:
        assert Path(p).exists(), p
    assert (tmp_path / "figs" / "figure_manifest.csv").exists()


def test_gene_labels_are_drawn_and_counted():
    """``_draw_gene_labels`` adds text only, and thins colliding labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from panelprep.viz.embeddings import _draw_gene_labels

    coords = pd.DataFrame({0: [0.0, 1.0, 2.0, 2.001], 1: [0.0, 1.0, 2.0, 2.001]},
                          index=["Sox2", "T", "Krt8", "Krt8b"])
    fig, ax = plt.subplots()
    n = _draw_gene_labels(ax, coords, ["Sox2", "T"])
    assert n == 2
    n_skip = _draw_gene_labels(ax, coords, ["Sox2", "T"], skip=["Sox2"])
    assert n_skip == 1
    # Krt8 and Krt8b are nearly coincident: one of them is dropped.
    n_auto = _draw_gene_labels(ax, coords, n_auto=4, min_sep=0.1)
    assert n_auto < 4
    plt.close(fig)


# ------------------------------------------- host-repo compatibility
#: The 53 datasets registered in ``dataset.load_dataset.DATASET_NAMES`` as of the
#: merge check. Pinned here so the test runs without a repo checkout; if the host
#: repo adds entries, extend this list and re-check the categories.
MORPHOLAB_DATASET_NAMES = (
    "seu-3d", "mosta", "harland", "lohoff", "egastrulation", "wang-gastrulation",
    "peng-epiblast", "embryo-slide-seq", "spateo", "most", "embryo-visium",
    "dbit-seq", "mpsta", "gut-development", "endothelial",
    "embryo-spatial-atac-visium", "visium-hd-embryo", "xenium-neonatal", "tome",
    "pijuan-sala", "extended-mouse-atlas", "somite-atlas", "mittnenzweig", "omg",
    "moca", "mela", "nextcell", "gut-endoderm", "scialdone", "tyser-heart",
    "hand2", "mesp1", "embryo-multiome", "scnmt-gastrulation", "st-magic",
    "mohammed", "cheng", "xue", "mdcla", "foregut", "developing-brain",
    "cortical-diversification", "spinal-cord", "pancreatic-endocrinogenesis",
    "kidney-development", "tabula-muris", "tabula-muris-senis", "mouse-cell-atlas",
    "allen-brain", "starmap-plus", "osmfish", "slide-seq-v2", "liver-cell-atlas",
)

#: Spatially resolved entries in that list. A miss here is not cosmetic: a spatial
#: dataset tagged ``single_cell`` has its spatial scores silently skipped.
MORPHOLAB_SPATIAL = {
    "seu-3d", "mosta", "embryo-slide-seq", "spateo", "embryo-visium", "dbit-seq",
    "mpsta", "embryo-spatial-atac-visium", "visium-hd-embryo", "xenium-neonatal",
    "st-magic", "starmap-plus", "osmfish", "slide-seq-v2",
}


@pytest.mark.parametrize("name", sorted(MORPHOLAB_SPATIAL))
def test_spatial_morpholab_datasets_are_categorised_spatial(name):
    """Regression: seven of these were inferred ``single_cell`` before the fix."""
    assert morpholab._guess_category(name, None) == "spatial", name


def test_non_spatial_morpholab_datasets_are_not_called_spatial():
    """The inference must not over-reach: dissociated atlases stay single_cell."""
    wrong = [n for n in MORPHOLAB_DATASET_NAMES
             if n not in MORPHOLAB_SPATIAL
             and morpholab._guess_category(n, None) == "spatial"]
    assert not wrong, f"wrongly inferred spatial: {wrong}"


def test_coordinates_override_name_inference():
    """A dataset that carries coordinates is spatial whatever its name says."""
    class _Fake:
        obsm = {"spatial": np.zeros((3, 2))}
        obs = pd.DataFrame(index=range(3))
    assert morpholab._guess_category("tabula-muris", _Fake()) == "spatial"
    assert morpholab._guess_category("tabula-muris", None) == "single_cell"
