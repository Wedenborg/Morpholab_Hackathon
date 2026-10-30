"""Acceptance tests for the agreement scores.

These are not smoke tests. The synthetic fixture has a known correct answer by
construction (see panelprep/io/synthetic.py), so each assertion below is a claim
about scientific behaviour that would fail if a metric were implemented wrongly.
"""
from __future__ import annotations

import numpy as np
import pytest

from panelprep.io.synthetic import EXPECTED_ORDER, make_synthetic_case
from panelprep.schemas import CandidateDataset
from panelprep.scores import SCORES, run_scores

ALL_SCORES = ["avg_kl", "avg_js", "corr_frobenius", "corr_cosine", "ot_cluster_ari",
              "spatial_autocorr_frobenius", "spatial_corr_frobenius"]


@pytest.fixture(scope="module")
def case():
    return make_synthetic_case(seed=0)


@pytest.fixture(scope="module")
def table(case):
    res = run_scores(case.query, case.candidate_list, score_names=ALL_SCORES,
                     genes=case.query.panel, seed=0, max_obs=800)
    return res.pivot_table(index="dataset", columns="score", values="value", dropna=False)


def test_structure_scores_rank_the_sibling_first(table):
    """The dataset drawn from the same generative model must agree best."""
    assert table["corr_frobenius"].idxmin() == "sibling"
    assert table["corr_cosine"].idxmax() == "sibling"


def test_unrelated_dataset_ranks_last_on_structure(table):
    assert table["corr_frobenius"].idxmax() == "unrelated"


def test_distributional_scores_are_blind_to_destroyed_co_expression(table):
    """The central validity check.

    'shuffled_structure' permutes each gene independently *after* normalisation,
    so every gene's empirical distribution is preserved exactly while the
    gene-gene correlation structure is destroyed. A distribution-based score
    therefore cannot distinguish it from a genuine sibling, and a
    structure-based score must. If this test fails, one of the two score
    families is measuring something other than what its name says.
    """
    sib, shuf = "sibling", "shuffled_structure"
    for s in ("avg_kl", "avg_js"):
        marginal_ratio = table.loc[shuf, s] / table.loc[sib, s]
        assert marginal_ratio < 4.0, f"{s} separated the permuted candidate (ratio {marginal_ratio})"
    structure_ratio = table.loc[shuf, "corr_frobenius"] / table.loc[sib, "corr_frobenius"]
    assert structure_ratio > 5.0, f"corr_frobenius missed the permutation (ratio {structure_ratio})"
    assert table.loc[shuf, "corr_cosine"] < 0.5 < table.loc[sib, "corr_cosine"]


def test_ot_cluster_ari_separates_related_from_unrelated(table):
    assert table.loc["sibling", "ot_cluster_ari"] > table.loc["unrelated", "ot_cluster_ari"]
    assert table.loc["unrelated", "ot_cluster_ari"] < 0.3


def test_ot_ari_is_a_valid_same_set_comparison(case):
    """The soft contingency ARI must reduce to the standard ARI on a hard table."""
    from sklearn.metrics import adjusted_rand_score

    from panelprep.scores.alignment import soft_ari

    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=300)
    b = rng.integers(0, 4, size=300)
    tab = np.zeros((4, 4))
    for i, j in zip(a, b):
        tab[i, j] += 1
    assert soft_ari(tab) == pytest.approx(adjusted_rand_score(a, b), abs=1e-9)


def test_spatial_scores_are_not_applicable_without_coordinates(case):
    res = run_scores(case.query, case.candidate_list,
                     score_names=["spatial_autocorr_frobenius"], genes=case.query.panel, seed=0)
    no_coords = res[res.dataset.isin(["same_program_diff_mixture", "shuffled_structure"])]
    assert set(no_coords.status) == {"not_applicable"}
    assert no_coords.value.isna().all()


def test_missing_panel_genes_are_reported_not_imputed(case):
    """Dropping a panel gene from a candidate must reduce coverage, not invent data."""
    cand = case.candidates["sibling"]
    panel = case.query.panel
    trimmed = CandidateDataset(
        name="trimmed", category="single_cell",
        expression=cand.expression.drop(columns=[panel[0]]),
        obs=cand.obs, units=cand.units, meta={"description": "sibling minus one panel gene"})
    res = run_scores(case.query, [trimmed], score_names=["corr_frobenius", "avg_js"],
                     genes=panel, seed=0)
    assert set(res.n_genes_used) == {len(panel) - 1}
    assert all(panel[0] in g for g in res.genes_missing)


def test_embedding_only_sources_are_excluded_from_the_score_table(case):
    """A source with no cell x gene matrix is not a scoring candidate at all.

    An ESM-2 protein language model or a STRING graph cannot be compared to the
    panel's expression, so it must not appear in the agreement table -- not even
    as "not applicable", which would claim it had been considered as a reference
    and rejected. It belongs in the embedding inventory.
    """
    from panelprep.scores import split_scoreable

    nox = CandidateDataset(name="esm_like", category="non_transcriptomic", expression=None,
                           units="n/a", meta={"description": "protein language model"})
    scoreable, sources = split_scoreable([*case.candidate_list, nox])
    assert [c.name for c in sources] == ["esm_like"]
    assert len(scoreable) == len(case.candidate_list)

    res = run_scores(case.query, [*case.candidate_list, nox], score_names=ALL_SCORES,
                     genes=case.query.panel, seed=0, max_obs=400)
    assert "esm_like" not in set(res.dataset)
    assert set(res.dataset) == set(case.candidates)


def test_causal_datasets_with_a_matrix_are_scored(case):
    """A perturbation dataset has real expression and must be scored as such."""
    cand = case.candidates["sibling"]
    causal = CandidateDataset(
        name="perturbation", category="causal", expression=cand.expression,
        obs=cand.obs, units=cand.units, meta={"description": "knockout chimera arm"})
    res = run_scores(case.query, [causal], score_names=["avg_js", "corr_frobenius",
                                                       "ot_cluster_ari"],
                     genes=case.query.panel, seed=0, max_obs=400)
    assert set(res.status) == {"ok"}
    assert res.value.notna().all()


def test_units_guard_blocks_mismatched_normalisation(case):
    cand = case.candidates["sibling"]
    odd = CandidateDataset(name="fpkm_like", category="single_cell",
                           expression=cand.expression, units="fpkm",
                           meta={"description": "same numbers, different declared units"})
    res = run_scores(case.query, [odd], score_names=["avg_js", "corr_frobenius"],
                     genes=case.query.panel, seed=0)
    by = dict(zip(res.score, res.status))
    assert by["avg_js"] == "units_mismatch"      # distributional: refuses
    assert by["corr_frobenius"] == "ok"          # correlation: scale-free, proceeds
    allowed = run_scores(case.query, [odd], score_names=["avg_js"], genes=case.query.panel,
                        seed=0, allow_unit_mismatch=True)
    assert allowed.status.iloc[0] == "ok"


def test_scores_are_deterministic_for_a_fixed_seed(case):
    kw = dict(score_names=ALL_SCORES, genes=case.query.panel, seed=7, max_obs=400)
    a = run_scores(case.query, case.candidate_list, **kw)
    b = run_scores(case.query, case.candidate_list, **kw)
    np.testing.assert_allclose(a.value.to_numpy(dtype=float), b.value.to_numpy(dtype=float),
                               rtol=0, atol=0, equal_nan=True)


def test_every_registered_score_declares_its_metadata():
    for entry in SCORES:
        assert entry.meta.get("orientation") in ("higher_is_better", "lower_is_better"), entry.name
        assert entry.meta.get("description"), entry.name
        assert entry.meta.get("applies_to"), entry.name


def test_expected_order_is_respected_by_the_aggregate_ranking(table):
    """Averaging the agreement-normalised columns must recover the known order."""
    agree = {}
    for s in ("avg_js", "corr_frobenius", "ot_cluster_ari", "corr_cosine"):
        col = table[s].astype(float)
        if SCORES[s].meta["orientation"] == "lower_is_better":
            col = -col
        agree[s] = col.rank()
    ranked = list(sum(agree.values()).sort_values(ascending=False).index)
    assert ranked[0] == EXPECTED_ORDER[0]
    assert ranked[-1] == EXPECTED_ORDER[-1]
