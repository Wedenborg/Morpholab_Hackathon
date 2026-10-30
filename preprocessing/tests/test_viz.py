"""Acceptance tests for the visualisation layer and the generated report.

What these tests are actually asserting
--------------------------------------
1. the orientation correction really flips a ``lower_is_better`` column, so red
   means "agrees" in every column of the heatmap (the central correction);
2. a score registered *inside this test file* reaches the heatmap and the
   markdown report with **no edit** to ``panelprep/viz/*`` or
   ``panelprep/report.py`` -- the brief's "adding a metric must automatically
   update the comparison report";
3. a cell with no value is never painted as a number;
4. the report names every registry entry;
5. the figures exist on disk and are not blank.

Plus two end-to-end checks against ``panelprep.io.synthetic.make_synthetic_case``
(the fixture with a known correct answer) so the numbers in the report come from
a real run, and a masked-gene check that NaN gene rows are dropped and counted
rather than plotted at the origin.

Run with ``pytest -s tests/test_viz.py`` to see the numbers; every test prints
what it measured. ``NUMBERS`` collects them and the last test writes them to
``tests/_viz_numbers.json`` for the task report.

MODIFY HERE: the fake frame in ``fake_results`` is the minimal shape of
``panelprep.scores.run_scores`` output -- extend it to cover a new status.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelprep.config import Config
from panelprep.report import build_report, method_status_table, methods_table
from panelprep.schemas import CandidateDataset, GeneAnnotation, GeneEmbedding, PanelQuery
from panelprep.scores import SCORES
from panelprep.viz import (
    STATUS_COLOUR,
    STATUS_LABELS,
    agreement_frame,
    clean_matrix,
    embedding_panel,
    embedding_quality,
    embedding_scatter,
    score_heatmap,
    score_table_styled,
)

NUMBERS: dict = {}

DUMMY = "dummy_panel_overlap"


def _is_number(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


# --- a score registered from the TEST FILE: nothing in viz/report knows it ---
if DUMMY not in SCORES:
    @SCORES.register(
        DUMMY, orientation="higher_is_better",
        applies_to=("single_cell", "spatial"), needs_coords=False, units_sensitive=False,
        defaults=dict(),
        description="test-only score: fraction of requested panel genes present in the candidate",
    )
    def _dummy_panel_overlap(query, candidate, *, genes, rng, **_):
        from panelprep.schemas import ScoreResult
        from panelprep.scores import prepare_panel

        _, _, used, missing = prepare_panel(query, candidate, genes, rng=rng)
        return ScoreResult(score_name=DUMMY, dataset_name=candidate.name,
                           value=float(len(used)) / max(len(genes), 1),
                           orientation="higher_is_better", status="ok",
                           n_genes_used=len(used), n_genes_requested=len(genes),
                           genes_missing=tuple(missing))


@pytest.fixture
def fake_results() -> pd.DataFrame:
    """Minimal long-format frame: one divergence, one agreement, one dead cell.

    ``avg_js`` is ``lower_is_better`` and ``ds_a`` is the best dataset by it
    (0.05 < 0.20 < 0.60); ``ot_cluster_ari`` is ``higher_is_better`` and ``ds_a``
    is also best by it (0.80 > 0.40). So after the direction correction both
    columns must rank ``ds_a`` top -- that is the flip being tested.
    """
    rows = [
        ("ds_a", "avg_js", 0.05, "lower_is_better", "ok", 4, 4, ""),
        ("ds_b", "avg_js", 0.20, "lower_is_better", "ok", 4, 4, ""),
        ("ds_c", "avg_js", 0.60, "lower_is_better", "ok", 3, 4, "Gene0007"),
        ("ds_a", "ot_cluster_ari", 0.80, "higher_is_better", "ok", 4, 4, ""),
        ("ds_b", "ot_cluster_ari", 0.40, "higher_is_better", "ok", 4, 4, ""),
        ("ds_c", "ot_cluster_ari", None, "higher_is_better", "not_applicable", 0, 4, ""),
    ]
    return pd.DataFrame(rows, columns=["dataset", "score", "value", "orientation", "status",
                                       "n_genes_used", "n_genes_requested", "genes_missing"])


# ---------------------------------------------------------------------------
# 1. the agreement mapping flips the lower_is_better column
# ---------------------------------------------------------------------------

def test_agreement_mapping_flips_lower_is_better(fake_results):
    tab = agreement_frame(fake_results, normalise="rank")

    raw_js = tab.raw["avg_js"]
    agr_js = tab.agreement["avg_js"]
    # raw: smallest is best; agreement: largest is best -> the ORDER must invert
    raw_order = list(raw_js.sort_values().index)
    agr_order = list(agr_js.sort_values(ascending=False).index)
    assert raw_order == agr_order, (raw_order, agr_order)
    assert agr_js["ds_a"] > agr_js["ds_b"] > agr_js["ds_c"]
    assert raw_js["ds_a"] < raw_js["ds_b"] < raw_js["ds_c"]

    # the higher_is_better column must NOT be flipped
    agr_ari = tab.agreement["ot_cluster_ari"]
    assert agr_ari["ds_a"] > agr_ari["ds_b"]
    assert tab.raw["ot_cluster_ari"]["ds_a"] > tab.raw["ot_cluster_ari"]["ds_b"]

    # and the two columns must now agree on who is best
    assert agr_js.idxmax() == agr_ari.idxmax() == "ds_a"
    assert tab.ranking.iloc[0]["dataset"] == "ds_a"

    # minmax and zscore must preserve the same ordering
    orders = {}
    for norm in ("rank", "minmax", "zscore"):
        t = agreement_frame(fake_results, normalise=norm)
        orders[norm] = list(t.agreement["avg_js"].sort_values(ascending=False).index)
    assert all(v == ["ds_a", "ds_b", "ds_c"] for v in orders.values()), orders

    NUMBERS["flip"] = {
        "raw_avg_js": raw_js.round(4).to_dict(),
        "agreement_avg_js": agr_js.round(4).to_dict(),
        "raw_ot_cluster_ari": tab.raw["ot_cluster_ari"].round(4).to_dict(),
        "agreement_ot_cluster_ari": agr_ari.round(4).to_dict(),
        "ranking": tab.ranking.round(4).to_dict("records"),
        "orders_by_normaliser": orders,
    }
    print("\n[1] raw avg_js              :", raw_js.round(3).to_dict())
    print("[1] agreement avg_js        :", agr_js.round(3).to_dict(), "(order inverted)")
    print("[1] agreement ot_cluster_ari:", agr_ari.round(3).to_dict(), "(order preserved)")
    print("[1] mean-agreement ranking  :", tab.ranking.round(3).to_dict("records"))


# ---------------------------------------------------------------------------
# 2. a score registered in the test reaches the heatmap and the report
# ---------------------------------------------------------------------------

def _tiny_case():
    """Toy 4-gene query + three candidates, enough for any registered score."""
    rng = np.random.default_rng(0)
    genes = ["Gata1", "Sox2", "Pou5f1", "Tbx5"]
    q = pd.DataFrame(np.abs(rng.normal(1.0, 0.6, size=(120, 4))), columns=genes)
    a = pd.DataFrame(np.abs(rng.normal(1.0, 0.6, size=(100, 4))), columns=genes)
    b = pd.DataFrame(np.abs(rng.normal(2.0, 1.1, size=(100, 3))), columns=genes[:3])
    query = PanelQuery(expression=q, coords=rng.uniform(0, 10, size=(120, 2)),
                       descriptor="toy mouse embryo panel", name="toy_query")
    cands = [
        CandidateDataset(name="cand_sc", category="single_cell", expression=a,
                         meta={"description": "toy single cell"}),
        CandidateDataset(name="cand_partial", category="spatial", expression=b,
                         coords=rng.uniform(0, 10, size=(100, 2)),
                         meta={"description": "toy spatial, missing one panel gene"}),
        CandidateDataset(name="cand_ppi", category="non_transcriptomic",
                         meta={"description": "toy PPI graph, no expression"}),
    ]
    return query, cands


def test_new_score_appears_without_editing_viz_or_report(tmp_path):
    from panelprep.scores import run_scores

    query, cands = _tiny_case()
    names = sorted(SCORES.names())
    assert DUMMY in names
    res = run_scores(query, cands, score_names=names, genes=query.panel, seed=0)

    fp = tmp_path / "heatmap_with_dummy.png"
    tab = score_heatmap(res, out_path=str(fp), title="dummy-score check", close=False)
    assert DUMMY in tab.raw.columns, tab.raw.columns
    assert tab.orientation[DUMMY] == "higher_is_better"

    # the dummy's value is the panel-coverage fraction, known by construction
    assert tab.raw.loc["cand_sc", DUMMY] == pytest.approx(1.0)
    assert tab.raw.loc["cand_partial", DUMMY] == pytest.approx(0.75)

    cfg = Config(run_name="dummy_check", scores=names, embedders=[], outdir=str(tmp_path))
    md = build_report(cfg, match_table=None, score_results=res, embeddings={},
                      annotation=None, figures={"heatmap": str(fp)},
                      out_path=str(tmp_path / "report.md"))
    assert DUMMY in md
    assert "fraction of requested panel genes" in md       # its registry description
    mt = methods_table()
    assert DUMMY in set(mt["name"])

    NUMBERS["new_score"] = {
        "registry_scores": names,
        "heatmap_columns": list(tab.raw.columns),
        "dummy_values": tab.raw[DUMMY].round(4).to_dict(),
        "dummy_in_report": bool(DUMMY in md),
        "report_chars": len(md),
    }
    print(f"\n[2] registry scores      : {names}")
    print(f"[2] heatmap columns      : {list(tab.raw.columns)}")
    print(f"[2] dummy values         : {tab.raw[DUMMY].round(3).to_dict()}")
    print(f"[2] dummy in report      : {DUMMY in md}  (report {len(md)} chars, "
          f"no edit to viz/ or report.py)")


# ---------------------------------------------------------------------------
# 3. value-less cells are not coloured as numbers
# ---------------------------------------------------------------------------

def test_status_cells_are_not_numbers(fake_results, tmp_path):
    import matplotlib.text as mtext

    tab = agreement_frame(fake_results)
    i = list(tab.raw.index).index("ds_c")
    j = list(tab.raw.columns).index("ot_cluster_ari")
    assert tab.status.iat[i, j] == "not_applicable"
    assert not np.isfinite(tab.colour.iat[i, j])          # masked -> grey, not a colour
    assert not np.isfinite(tab.agreement.iat[i, j])       # excluded from the normalisation
    assert np.isnan(tab.raw.iat[i, j])                    # never 0

    # excluded from the ranking, and that exclusion is visible as a count
    row = tab.ranking.set_index("dataset").loc["ds_c"]
    assert int(row["n_scores_ok"]) == 1 and int(row["n_scores_total"]) == 2

    fp = tmp_path / "status_cells.png"
    t2 = score_heatmap(fake_results, out_path=str(fp), close=False)
    ax = t2.fig.axes[0]
    # text drawn at the (j, i) data position IS the cell annotation; the colourbar
    # ticks live on another axes and must not be mistaken for cell contents
    def _cell_texts(col: int, rowi: int) -> list[str]:
        # the value line of a partial-coverage cell is nudged off centre, so
        # accept anything within half a cell of the centre
        return [x.get_text() for x in ax.texts
                if x.get_position()[0] == col and abs(x.get_position()[1] - rowi) < 0.4]

    in_cell = _cell_texts(j, i)
    assert in_cell == [STATUS_LABELS["not_applicable"]], in_cell
    assert not any(_is_number(t) for t in in_cell), in_cell
    ok_cell = _cell_texts(list(tab.raw.columns).index("avg_js"), i)
    assert any(_is_number(t) for t in ok_cell), ok_cell     # an ok cell DOES print a number
    assert isinstance(t2.fig.findobj(mtext.Text), list)

    pytest.importorskip("jinja2", reason="pandas Styler.to_html needs jinja2")
    styled = score_table_styled(fake_results)
    html = styled.to_html()
    assert STATUS_COLOUR.lower() in html.lower()
    assert ">n/a<" in html

    NUMBERS["status_cells"] = {
        "status": str(tab.status.iat[i, j]),
        "raw_is_nan": bool(np.isnan(tab.raw.iat[i, j])),
        "colour_is_masked": bool(not np.isfinite(tab.colour.iat[i, j])),
        "ds_c_n_scores_ok": int(row["n_scores_ok"]),
        "grey_cells": int((tab.status != "ok").to_numpy().sum()),
        "status_label_in_figure": True,
    }
    print(f"\n[3] ds_c/ot_cluster_ari  : status={tab.status.iat[i, j]}, raw=NaN, "
          f"colour=masked(grey {STATUS_COLOUR}), label '{STATUS_LABELS['not_applicable']}' "
          f"drawn in the cell")
    print(f"[3] ds_c contributes     : {int(row['n_scores_ok'])}/{int(row['n_scores_total'])} "
          f"scores to its mean agreement")


# ---------------------------------------------------------------------------
# 4. the report names every registry entry
# ---------------------------------------------------------------------------

def test_report_contains_every_registry_entry(tmp_path, fake_results):
    from panelprep.embed import EMBEDDERS

    cfg = Config(run_name="registry_coverage", scores=sorted(SCORES.names()),
                 embedders=sorted(EMBEDDERS.names()), outdir=str(tmp_path))
    md = build_report(cfg, match_table=None, score_results=fake_results, embeddings={},
                      annotation=None, figures={},
                      out_path=str(tmp_path / "registry_report.md"),
                      manual_status={"esm2_protein_lm": {"status": "deferred",
                                                         "reason": "owned by another worker"}})
    expected = sorted(SCORES.names()) + sorted(EMBEDDERS.names())
    missing = [n for n in expected if n not in md]
    assert not missing, missing
    assert "deferred" in md and "owned by another worker" in md

    st = method_status_table(cfg, score_results=fake_results, embeddings={},
                             manual_status={"esm2_protein_lm": "deferred: not my file"})
    assert set(st["status"]) <= {"implemented", "evaluated", "blocked", "deferred"}
    ev = set(st.loc[st["status"] == "evaluated", "method"])
    assert ev == {"avg_js", "ot_cluster_ari"}, ev      # only these ran in fake_results

    NUMBERS["report_coverage"] = {
        "n_registry_entries": len(expected),
        "entries": expected,
        "missing_from_report": missing,
        "report_chars": len(md),
        "status_counts": st["status"].value_counts().to_dict(),
        "evaluated": sorted(ev),
    }
    print(f"\n[4] registry entries     : {len(expected)} -> all present in the report "
          f"({len(md)} chars)")
    print(f"[4] status counts        : {st['status'].value_counts().to_dict()}")
    print(f"[4] evaluated this run   : {sorted(ev)}")


# ---------------------------------------------------------------------------
# 5. figures are written and non-trivial
# ---------------------------------------------------------------------------

def _toy_embeddings(n_genes=60, seed=0):
    """Three gene embeddings with known structure + a shared annotation.

    ``clustered`` encodes the pathway labels, ``noise`` does not, ``partial``
    is ``clustered`` with 10 genes given no embedding (all-NaN rows) -- the
    missing-data path.
    """
    rng = np.random.default_rng(seed)
    genes = [f"Gene{i:04d}" for i in range(n_genes)]
    path = np.array([f"pathway_{i % 4}" for i in range(n_genes)])
    centres = rng.normal(0, 6, size=(4, 8))
    X = np.stack([centres[int(p.split("_")[1])] for p in path]) \
        + rng.normal(0, 0.8, size=(n_genes, 8))
    emb_c = GeneEmbedding(pd.DataFrame(X, index=genes), method="clustered",
                          category="single_cell", provenance={"toy": True})
    emb_n = GeneEmbedding(pd.DataFrame(rng.normal(0, 1, size=(n_genes, 8)), index=genes),
                          method="noise", category="single_cell", provenance={"toy": True})
    Xp = X.copy()
    Xp[:10] = np.nan
    emb_p = GeneEmbedding(pd.DataFrame(Xp, index=genes), method="partial",
                          category="spatial", provenance={"toy": True, "masked_genes": 10})
    # same geometry seen through heavy noise: a PARTIALLY agreeing method, so the
    # pairwise metrics are tested somewhere between "identical" and "unrelated"
    emb_s = GeneEmbedding(pd.DataFrame(X + rng.normal(0, 5.0, size=(n_genes, 8)), index=genes),
                          method="clustered_noisy", category="single_cell",
                          provenance={"toy": True, "noise_sd": 5.0})
    ann = GeneAnnotation(pd.DataFrame({
        "is_ligand": [i % 7 == 0 for i in range(n_genes)],
        "is_receptor": [i % 11 == 0 for i in range(n_genes)],
        "is_tf": [i % 13 == 0 for i in range(n_genes)],
        "pathways": [f"{p};extra" for p in path],
        "top_pathway": path,
    }, index=genes), sources={"toy": "synthetic labels"})
    return {"clustered": emb_c, "noise": emb_n, "partial": emb_p,
            "clustered_noisy": emb_s}, ann


def test_figures_written_and_nontrivial(tmp_path):
    embs, ann = _toy_embeddings()
    fp_scatter = tmp_path / "scatter.png"
    sc = embedding_scatter(embs["clustered"], ann, method="pca", out_path=str(fp_scatter),
                           role_overlay=True, top_n=12, seed=0,
                           quality_kwargs={"n_permutations": 20})
    fp_panel = tmp_path / "panel.png"
    q = embedding_panel(embs, ann, str(fp_panel), method="pca", seed=0)

    rows = [("ds_a", "avg_js", 0.1, "lower_is_better", "ok", 4, 4, "")]
    fp_heat = tmp_path / "heat.png"
    score_heatmap(pd.DataFrame(rows, columns=["dataset", "score", "value", "orientation",
                                              "status", "n_genes_used", "n_genes_requested",
                                              "genes_missing"]), out_path=str(fp_heat))

    sizes = {p.name: p.stat().st_size for p in (fp_scatter, fp_panel, fp_heat)}
    for name, size in sizes.items():
        assert size > 20_000, (name, size)

    # the numbers that must accompany the pictures
    qi = q.set_index("embedding")
    assert qi.loc["clustered", "silhouette"] > qi.loc["noise", "silhouette"]
    assert qi.loc["clustered", "silhouette_p"] <= 0.05
    assert qi.loc["noise", "silhouette_p"] > 0.05
    assert int(qi.loc["partial", "n_genes_plotted"]) == 50
    assert int(qi.loc["partial", "n_genes_total"]) == 60

    NUMBERS["figures"] = {
        "sizes_bytes": sizes,
        "panel_quality": q[["embedding", "silhouette", "silhouette_p", "knn_agreement",
                            "knn_p", "n_genes_plotted", "n_genes_total"]].to_dict("records"),
        "scatter_quality": {k: sc.quality.get(k) for k in
                            ("silhouette", "silhouette_p", "knn_agreement", "knn_p",
                             "n_genes_scored", "n_classes")},
    }
    print(f"\n[5] figure sizes (bytes) : {sizes}")
    print("[5] panel quality        :")
    for r in NUMBERS["figures"]["panel_quality"]:
        print("    ", r)


# ---------------------------------------------------------------------------
# masked genes: NaN rows are dropped and counted, never plotted at the origin
# ---------------------------------------------------------------------------

def test_masked_genes_are_dropped_and_counted():
    from panelprep.embed import gene_by_cell
    from panelprep.normalize import resolve_genes

    rng = np.random.default_rng(1)
    present = [f"Gene{i:04d}" for i in range(8)]
    asked = present[:5] + ["Absent1", "Absent2"]
    ds = CandidateDataset(name="d", category="single_cell",
                          expression=pd.DataFrame(np.abs(rng.normal(1, 0.4, size=(40, 8))),
                                                  columns=present))
    names, missing = resolve_genes(asked, ds.expression.columns)
    assert len(names) == len(asked)                       # positional alignment (interface fix)
    assert names[-2:] == [None, None] and missing == ["Absent1", "Absent2"]

    gxc, mask = gene_by_cell(ds, asked)
    assert gxc.shape == (len(asked), 40) and mask.tolist() == [True] * 5 + [False] * 2

    M = gxc.iloc[:, :6].copy().astype(float)
    M.loc[~mask] = np.nan                                 # an embedder's honest "no embedding"
    emb = GeneEmbedding(M, method="masked_toy", category="single_cell")
    kept, counts = clean_matrix(emb)
    assert counts == {"n_total": 7, "n_kept": 5, "n_all_nan": 2, "n_partial_nan": 0, "dim": 6}
    assert list(kept.index) == asked[:5]

    q = embedding_quality(emb, None)
    assert q["status"] == "no_annotation" and q["silhouette"] is None

    NUMBERS["masked_genes"] = {"resolve_len": len(names), "mask": mask.tolist(),
                               "clean_counts": counts}
    print(f"\n[masked] resolve_genes len={len(names)} (== {len(asked)} requested), "
          f"mask={mask.tolist()}")
    print(f"[masked] clean_matrix    : {counts}")


# ---------------------------------------------------------------------------
# end to end on the synthetic fixture with a KNOWN correct answer
# ---------------------------------------------------------------------------

def test_synthetic_end_to_end(tmp_path):
    from panelprep.io.synthetic import make_synthetic_case
    from panelprep.scores import run_scores

    case = make_synthetic_case(n_genes=200, n_query=800, n_candidate=800, seed=0)
    names = sorted(SCORES.names())
    res = run_scores(case.query, case.candidate_list, score_names=names,
                     genes=case.query.panel, seed=0, max_obs=800)
    fp = tmp_path / "synthetic_heatmap.png"
    tab = score_heatmap(res, out_path=str(fp),
                        title=f"Panel-gene agreement with {case.query.name}")

    # the fixture's structure-vs-marginal split must be visible in the table
    js = tab.raw["avg_js"]
    frob = tab.raw["corr_frobenius"]
    assert js["shuffled_structure"] < js["unrelated"]          # marginals preserved
    assert frob["shuffled_structure"] > frob["sibling"]        # structure destroyed
    assert tab.ranking.iloc[0]["dataset"] == "sibling"
    assert fp.stat().st_size > 20_000

    cfg = Config(run_name="synthetic", scores=names, embedders=[], outdir=str(tmp_path))
    md = build_report(cfg, match_table=None, score_results=res, embeddings={},
                      annotation=None, figures={"agreement heatmap": str(fp)},
                      out_path=str(tmp_path / "comparison_report.md"))
    assert "sibling" in md and "## Status of every requested method" in md

    NUMBERS["synthetic"] = {
        "raw": tab.raw.round(4).to_dict(),
        "agreement": tab.agreement.round(4).to_dict(),
        "status": tab.status.to_dict(),
        "ranking": tab.ranking.round(4).to_dict("records"),
        "expected_order": case.truth["expected_order"],
        "heatmap_bytes": fp.stat().st_size,
        "report_chars": len(md),
    }
    print("\n[synthetic] raw values:\n", tab.raw.round(4))
    print("[synthetic] agreement (rank-normalised):\n", tab.agreement.round(3))
    print("[synthetic] ranking   :", tab.ranking.round(3).to_dict("records"))
    print("[synthetic] expected  :", case.truth["expected_order"])


# ---------------------------------------------------------------------------
# pairwise comparison BETWEEN embedding methods
# ---------------------------------------------------------------------------

def test_pairwise_embedding_comparisons(tmp_path):
    """clustered/partial share their geometry; noise shares nothing with either."""
    from panelprep.viz import (
        embedding_agreement_matrix,
        embedding_pair_grid,
        embedding_pair_scatter,
    )

    embs, ann = _toy_embeddings()
    out = {}
    for metric in ("knn_jaccard", "cka", "mantel"):
        pa = embedding_agreement_matrix(embs, metric=metric, k=10, n_permutations=10,
                                        out_path=str(tmp_path / f"agree_{metric}.png"), seed=0)
        M = pa.matrix
        assert np.allclose(np.diag(M.to_numpy()), 1.0)                  # self-agreement
        assert np.allclose(M.to_numpy(), M.to_numpy().T, equal_nan=True)  # symmetric
        # A graded expectation, not a binary one: 'partial' is 'clustered' with
        # 10 genes masked (identical geometry on the shared genes),
        # 'clustered_noisy' is the same geometry under heavy noise (partial
        # agreement), 'noise' is independent (none).
        assert (M.loc["clustered", "partial"]
                > M.loc["clustered", "clustered_noisy"]
                > M.loc["clustered", "noise"]), M.loc["clustered"].to_dict()
        assert M.loc["clustered", "clustered_noisy"] > pa.null.loc["clustered",
                                                                   "clustered_noisy"]
        assert M.loc["clustered", "partial"] > pa.null.loc["clustered", "partial"]
        # the shared-gene count is the honest denominator, not the full gene set
        assert int(pa.n_shared.loc["clustered", "partial"]) == 50
        assert int(pa.n_shared.loc["clustered", "noise"]) == 60
        row = pa.long.set_index(["method_a", "method_b"]).loc[("clustered", "partial")]
        assert row["p"] <= 0.1
        assert (tmp_path / f"agree_{metric}.png").stat().st_size > 20_000
        out[metric] = {
            "clustered_vs_partial": float(M.loc["clustered", "partial"]),
            "clustered_vs_clustered_noisy": float(M.loc["clustered", "clustered_noisy"]),
            "clustered_vs_noise": float(M.loc["clustered", "noise"]),
            "null_clustered_vs_partial": float(pa.null.loc["clustered", "partial"]),
            "null_clustered_vs_clustered_noisy": float(pa.null.loc["clustered",
                                                                   "clustered_noisy"]),
            "null_clustered_vs_noise": float(pa.null.loc["clustered", "noise"]),
            "p_clustered_vs_partial": float(row["p"]),
            "n_shared_clustered_partial": int(pa.n_shared.loc["clustered", "partial"]),
        }

    ps = embedding_pair_scatter(embs["clustered"], embs["partial"], ann, method="pca",
                                out_path=str(tmp_path / "pair_scatter.png"), n_arrows=15,
                                seed=0)
    assert ps.n_shared == 50
    assert 0.0 <= ps.disparity <= 2.0
    d = ps.displacement["displacement"].to_numpy()
    assert (np.diff(d) <= 1e-12).all()                   # sorted worst-first
    assert len(ps.displacement) == 50
    assert (tmp_path / "pair_scatter.png").stat().st_size > 20_000

    ps_bad = embedding_pair_scatter(embs["clustered"], embs["noise"], ann, method="pca",
                                    out_path=str(tmp_path / "pair_scatter_noise.png"), seed=0)
    assert ps_bad.disparity > ps.disparity                # unrelated pair disagrees more

    grid = embedding_pair_grid(embs, ann, out_path=str(tmp_path / "pair_grid.png"), seed=0)
    n_m = len(embs)
    assert len(grid) == n_m * (n_m - 1) // 2              # every unordered pair
    assert (tmp_path / "pair_grid.png").stat().st_size > 20_000
    assert set(grid.columns) >= {"method_a", "method_b", "abs_spearman_coord1",
                                 "knn_jaccard", "n_shared"}

    NUMBERS["pairwise"] = {
        "agreement_by_metric": out,
        "procrustes_clustered_vs_partial": round(ps.disparity, 4),
        "procrustes_clustered_vs_noise": round(ps_bad.disparity, 4),
        "top_movers": ps.displacement.head(5).round(4).to_dict("records"),
        "grid": grid.to_dict("records"),
    }
    print("\n[6] pairwise agreement with 'clustered' (observed, shuffled-label null):")
    for m, v in out.items():
        print(f"    {m:12s} partial {v['clustered_vs_partial']:+.3f} "
              f"(null {v['null_clustered_vs_partial']:+.3f}, "
              f"p={v['p_clustered_vs_partial']:.3f}, n={v['n_shared_clustered_partial']}) > "
              f"clustered_noisy {v['clustered_vs_clustered_noisy']:+.3f} "
              f"(null {v['null_clustered_vs_clustered_noisy']:+.3f}) > "
              f"noise {v['clustered_vs_noise']:+.3f} "
              f"(null {v['null_clustered_vs_noise']:+.3f})")
    print(f"[6] Procrustes disparity : clustered~partial {ps.disparity:.4f} < "
          f"clustered~noise {ps_bad.disparity:.4f}")
    print(f"[6] pair grid rows       : {len(grid)}")


def test_zz_write_numbers():
    """Last test: dump everything measured, so the numbers are inspectable."""
    out = Path(__file__).parent / "_viz_numbers.json"
    out.write_text(json.dumps(NUMBERS, indent=1, default=str))
    print(f"\n[numbers] wrote {out} with keys {sorted(NUMBERS)}")
    assert out.stat().st_size > 100
