"""Tests for the eval-team benchmark: region scores, held-out gene score, cell-type
proportions, the measurement oracle and experiment log, and the dummy method.

Run with:  python -m unittest test_region_benchmark -v     (or: python -m pytest)

Synthetic tests need nothing downloaded. Tests on the real E7.5 embryos are skipped when
the data or the reference tables (prepare_references.py) are missing.
"""
import pathlib
import tempfile
import unittest
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from evaluate import Evaluator
from region_benchmark import (REF_DIR, RegionBenchmark, assign_regions, canonical_celltype,
                              group_means, peng_region_table, peng_sample_regions, rowwise_spearman)
from spatial_scores import (celltype_proportion_score, heldout_gene_score, neighbourhood_bins,
                            sparsity_summary)
from submission import MeasurementOracle, check_log, load_log, measured_mask

warnings.filterwarnings("ignore")
DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "GSE278603_RAW"
REGIONS8 = ["ecto_A", "ecto_AL", "ecto_PL", "ecto_P", "meso_A", "meso_P", "endo_A", "endo_P"]
PENG_SAMPLES = {"ecto_A": "4A", "ecto_AL": "4L1", "ecto_PL": "4R2", "ecto_P": "4P",
                "meso_A": "5MA", "meso_P": "5MP", "endo_A": "5EA", "endo_P": "5EP"}


def synthetic_embryo(n_per_ring=400, n_sections=6, n_genes=40, rotation_deg=0.0, seed=0):
    """Three concentric rings per section (ectoderm inside, then mesoderm, then endoderm).
    Anterior is -x before rotation. Each gene's mean depends only on the cell's region.
    Returns (AnnData, true region per cell, pattern table genes x 8 regions in log2 units)."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sections):
        for layer, radius in [("ectoderm", 1.0), ("mesoderm", 1.5), ("endoderm", 2.0)]:
            theta = rng.uniform(0, 360, n_per_ring)            # 0 deg = anterior (-x)
            ang = np.minimum(theta, 360 - theta)                 # 0..180 from anterior
            if layer == "ectoderm":
                ct = np.where(ang < 30, "Rostral neurectoderm", np.where(ang >= 150, "Primitive Streak", "Epiblast"))
                sector = np.select([ang < 30, ang < 90, ang < 150], ["A", "AL", "PL"], "P")
            else:
                ct = np.full(n_per_ring, "Nascent mesoderm" if layer == "mesoderm" else "Gut")
                sector = np.where(ang < 90, "A", "P")
            x = -radius * np.cos(np.radians(theta))
            y = radius * np.sin(np.radians(theta))
            for i in range(n_per_ring):
                rows.append((x[i], y[i], 10.0 * s, layer, ct[i], f"{layer[:4]}_{sector[i]}"))
    obs = pd.DataFrame(rows, columns=["x", "y", "z", "germ_layer", "celltype", "true_region"])
    rot = np.radians(rotation_deg)
    xy = obs[["x", "y"]].to_numpy() @ np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
    obs[["x", "y"]] = xy * 100  # arbitrary units
    # Extra-embryonic cells far proximal; they must be ignored.
    extra = pd.DataFrame({"x": rng.normal(0, 50, 200), "y": rng.normal(0, 50, 200), "z": 200.0,
                          "germ_layer": "endoderm", "celltype": "ExE endoderm", "true_region": None})
    obs = pd.concat([obs, extra], ignore_index=True)
    obs.index = [f"cell{i}" for i in range(len(obs))]

    pattern = pd.DataFrame(rng.uniform(0, 5, (n_genes, len(REGIONS8))), columns=REGIONS8,
                           index=[f"g{i}" for i in range(n_genes)])
    pattern.iloc[:, 0] += 3.5  # every gene's range across regions is then > 3 log2 units,
    pattern = pattern.sub(pattern.min(1), axis=0)  # so it passes the 8-fold Peng filter
    rate = np.zeros((len(obs), n_genes))
    known = obs["true_region"].notna().to_numpy()
    rate[known] = 0.1 * 2 ** pattern.loc[:, obs.loc[known, "true_region"]].to_numpy().T
    X = np.log1p(rng.poisson(rate)).astype(float)
    a = ad.AnnData(X=sp.csr_matrix(X), obs=obs.drop(columns="true_region"),
                   var=pd.DataFrame(index=pattern.index))
    return a, obs["true_region"], pattern


def write_fake_refs(ref_dir, truth, pattern):
    peng = pd.DataFrame({PENG_SAMPLES[r]: pattern[r] for r in REGIONS8})
    (ref_dir / "peng_2019").mkdir(parents=True)
    peng.to_csv(ref_dir / "peng_2019" / "peng_E7.5_log2fpkm.csv.gz")
    # Atlas: per-cell-type means of the synthetic truth itself, on a log2 scale.
    ct = truth.obs["celltype"].to_numpy()
    types = [t for t in pd.unique(ct) if t != "ExE endoderm"]
    X = np.expm1(truth.X.toarray())
    atlas = pd.DataFrame({t: np.log2(X[ct == t].mean(0) + 1) * 3 for t in types}, index=truth.var_names)
    (ref_dir / "pijuan_sala_2019").mkdir(parents=True)
    atlas.to_csv(ref_dir / "pijuan_sala_2019" / "atlas_E7.5_mean_log2norm.csv.gz")
    pd.Series({t: 500 for t in types}, name="n_cells").to_csv(ref_dir / "pijuan_sala_2019" / "atlas_E7.5_ncells.csv")


class TestPengParsing(unittest.TestCase):
    def test_sample_names(self):
        self.assertEqual(peng_sample_regions("3L1"), ["ecto_AL_distal"])
        self.assertEqual(peng_sample_regions("5R2"), ["ecto_PL_mid"])
        self.assertEqual(peng_sample_regions("7A"), ["ecto_A_proximal"])
        self.assertEqual(peng_sample_regions("10MP"), ["meso_P_proximal"])
        self.assertEqual(peng_sample_regions("1EAP"), ["endo_A_distal", "endo_P_distal"])
        self.assertEqual(peng_sample_regions("2L"), [])
        with self.assertRaises(ValueError):
            peng_sample_regions("3X")

    def test_region_table_merges_levels(self):
        peng = pd.DataFrame({"2A": [1.0], "8A": [3.0], "3MP": [5.0]}, index=["g"])
        self.assertEqual(peng_region_table(peng).loc["g", "ecto_A"], 2.0)
        lv = peng_region_table(peng, levels=True)
        self.assertEqual(lv.loc["g", "ecto_A_distal"], 1.0)
        self.assertEqual(lv.loc["g", "ecto_A_proximal"], 3.0)
        self.assertEqual(lv.loc["g", "meso_P_distal"], 5.0)


class TestHelpers(unittest.TestCase):
    def test_spearman(self):
        A = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 1, 1]], float)
        B = np.array([[2, 4, 6, 8], [4, 3, 2, 1], [1, 2, 3, 4]], float)
        np.testing.assert_allclose(rowwise_spearman(A, B), [1, -1, 0])
        # invariant to a monotone per-gene transform
        np.testing.assert_allclose(rowwise_spearman(np.exp(A[:2]), B[:2]), rowwise_spearman(A[:2], B[:2]))

    def test_group_means(self):
        X = np.array([[1, 0], [3, 0], [0, 5]], float)
        out = group_means(sp.csr_matrix(X), np.array(["a", "a", "b"]), ["b", "a", "missing"])
        np.testing.assert_allclose(out, [[0, 5], [2, 0], [0, 0]])


class TestRegionAssignment(unittest.TestCase):
    def test_synthetic_regions_recovered_under_rotation(self):
        for rot in [0, 37, 200]:
            a, true_region, _ = synthetic_embryo(rotation_deg=rot)
            reg = assign_regions(a.obs)
            got = reg["region"].dropna().str.rsplit("_", n=1).str[0]
            want = true_region.loc[got.index]
            self.assertGreater((got == want).mean(), 0.97, f"rotation {rot}")
            # extra-embryonic cells are excluded
            self.assertTrue(reg.loc[a.obs["celltype"] == "ExE endoderm", "region"].isna().all())

    def test_levels_follow_z(self):
        a, _, _ = synthetic_embryo()
        reg = assign_regions(a.obs).dropna(subset=["region"])
        z = a.obs.loc[reg.index, "z"]
        self.assertTrue((reg.loc[z == z.min(), "level"] == "distal").all())
        self.assertTrue((reg.loc[z == z.max(), "level"] == "proximal").all())


class TestBenchmarkSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.truth, _, cls.pattern = synthetic_embryo()
        write_fake_refs(pathlib.Path(cls.tmp.name), cls.truth, cls.pattern)
        cls.bench = RegionBenchmark(cls.truth, ref_dir=cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_true_data_matches_reference(self):
        s, per_gene = self.bench.score(self.truth)
        self.assertGreater(s["peng_spearman"], 0.9)
        self.assertGreater(s["atlas_spearman"], 0.9)
        self.assertAlmostEqual(s["truth_region_spearman"], 1.0)
        self.assertAlmostEqual(s["peng_relative"], 1.0)
        self.assertEqual(s["peng_n_groups"], 8)
        self.assertEqual(len(per_gene), len(self.truth.var_names))

    def test_shuffled_cells_score_near_zero(self):
        perm = np.random.default_rng(1).permutation(self.truth.n_obs)
        rec = ad.AnnData(X=self.truth.X[perm], obs=pd.DataFrame(index=self.truth.obs_names),
                         var=pd.DataFrame(index=self.truth.var_names))
        s, _ = self.bench.score(rec)
        self.assertLess(abs(s["peng_spearman"]), 0.2)
        self.assertLess(abs(s["truth_region_spearman"]), 0.2)

    def test_monotone_rescaling_and_dataframe_input(self):
        X = self.truth.X.toarray()
        df = pd.DataFrame(2 * X + 1, index=self.truth.obs_names, columns=self.truth.var_names)
        s1, _ = self.bench.score(self.truth)
        s2, _ = self.bench.score(df)
        self.assertAlmostEqual(s1["peng_spearman"], s2["peng_spearman"], places=10)

    def test_cell_order_does_not_matter(self):
        perm = np.random.default_rng(2).permutation(self.truth.n_obs)
        rec = self.truth[perm].copy()
        s1, _ = self.bench.score(self.truth)
        s2, _ = self.bench.score(rec)
        self.assertAlmostEqual(s1["peng_spearman"], s2["peng_spearman"], places=10)

    def test_missing_genes_score_zero(self):
        keep = self.truth.var_names[:20]
        s, per_gene = self.bench.score(self.truth[:, keep].copy())
        self.assertEqual(s["peng_missing_genes"], len(self.truth.var_names) - 20)
        dropped = per_gene.index.difference(keep)
        self.assertTrue((per_gene.loc[dropped, "peng_spearman"] == 0).all())

    def test_missing_cells_raise(self):
        with self.assertRaises(ValueError):
            self.bench.score(self.truth[:100].copy())


def shuffled_cells(a, seed=1):
    perm = np.random.default_rng(seed).permutation(a.n_obs)
    return ad.AnnData(X=sp.csr_matrix(a.X)[perm], obs=pd.DataFrame(index=a.obs_names),
                      var=pd.DataFrame(index=a.var_names))


class TestSubmission(unittest.TestCase):
    def setUp(self):
        self.truth, _, _ = synthetic_embryo(n_per_ring=60, n_sections=3, n_genes=10)
        self.oracle = MeasurementOracle(self.truth)

    def test_oracle_returns_true_values_and_logs(self):
        z = self.oracle.sections[1]
        m = self.oracle.measure(["g1", "g2"], section=z)
        cells = self.truth.obs_names[self.truth.obs["z"] == z]
        self.assertEqual(list(m.index), list(cells))
        np.testing.assert_allclose(m.to_numpy(), self.truth[cells, ["g1", "g2"]].X.toarray())
        sub = list(cells[:5])
        self.oracle.measure(["g3"], cells=sub)
        log = self.oracle.log()
        self.assertEqual(list(log["genes"]), ["g1;g2", "g3"])
        self.assertEqual(log.loc[0, "cells"], "")
        self.assertEqual(log.loc[1, "cells"].split(";"), sub)

    def test_oracle_hides_labels_and_enforces_limits(self):
        self.assertEqual(list(self.oracle.cells.columns), ["x", "y", "z"])
        with self.assertRaises(ValueError):
            self.oracle.measure(["g1", "g2", "g3", "g4", "g5"], section=0.0)
        with self.assertRaises(ValueError):
            self.oracle.measure(["not_a_gene"], section=0.0)
        with self.assertRaises(ValueError):
            self.oracle.measure(["g1"])
        with self.assertRaises(ValueError):
            self.oracle.measure(["g1"], section=12345.0)
        self.assertEqual(len(self.oracle.log()), 0)

    def test_measured_mask_and_csv_roundtrip(self):
        z0 = self.oracle.sections[0]
        self.oracle.measure(["g1", "g2"], section=z0)
        cells = list(self.truth.obs_names[self.truth.obs["z"] == self.oracle.sections[2]][:7])
        self.oracle.measure(["g3"], cells=cells)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "experiments.csv"
            self.oracle.log().to_csv(path, index=False)
            log = load_log(path)
        mask = measured_mask(log, self.truth.obs, self.truth.var_names).toarray()
        in_z0 = (self.truth.obs["z"] == z0).to_numpy()
        g = list(self.truth.var_names)
        self.assertTrue(mask[in_z0][:, [g.index("g1"), g.index("g2")]].all())
        self.assertFalse(mask[~in_z0][:, g.index("g1")].any())
        rows = self.truth.obs_names.get_indexer(cells)
        self.assertTrue(mask[rows, g.index("g3")].all())
        self.assertEqual(mask.sum(), 2 * in_z0.sum() + 7)

    def test_check_log(self):
        log = pd.DataFrame({"step": [0], "stage": ["E7.5"], "section_z": [0.0],
                            "genes": ["a;b;c;d;e"], "cells": [""]})
        self.assertEqual(len(check_log(log)), 1)
        self.assertEqual(check_log(log.assign(genes="a;b")), [])


class TestSpatialScoresSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, _, _ = synthetic_embryo()
        cls.bins = neighbourhood_bins(cls.truth.obs)
        cls.genes = cls.truth.var_names

    def test_bins_stay_within_sections(self):
        z = pd.Series(self.truth.obs["z"].to_numpy()).groupby(self.bins).nunique()
        self.assertTrue((z == 1).all())
        sizes = pd.Series(self.bins).value_counts()
        self.assertTrue(15 <= sizes.median() <= 40)

    def test_true_data_scores_one_and_shuffled_near_zero(self):
        r = heldout_gene_score(self.truth, self.truth, self.bins, self.genes)
        np.testing.assert_allclose(r["corr"].dropna(), 1.0)
        r0 = heldout_gene_score(shuffled_cells(self.truth), self.truth, self.bins, self.genes)
        self.assertLess(abs(r0["corr"].mean()), 0.1)

    def test_measured_neighbourhoods_are_excluded(self):
        oracle = MeasurementOracle(self.truth)
        for z in oracle.sections:          # g0 measured everywhere, g1 in the first section only
            oracle.measure(["g0"], section=z)
        oracle.measure(["g1"], section=oracle.sections[0])
        mask = measured_mask(oracle.log(), self.truth.obs, self.genes)
        # A reconstruction that is right only where it measured and flat elsewhere.
        flat = ad.AnnData(X=np.zeros(self.truth.shape), obs=pd.DataFrame(index=self.truth.obs_names),
                          var=pd.DataFrame(index=self.genes))
        r = heldout_gene_score(flat, self.truth, self.bins, self.genes, mask=mask)
        self.assertTrue(np.isnan(r.loc["g0", "corr"]))          # nothing left to score
        n_all = heldout_gene_score(flat, self.truth, self.bins, self.genes).loc["g1", "n_bins"]
        self.assertLess(r.loc["g1", "n_bins"], n_all)          # first section's bins dropped
        self.assertEqual(r.loc["g2", "corr"], 0.0)             # flat prediction scores 0

    def test_sparsity_summary_counts_every_scored_gene(self):
        r = heldout_gene_score(self.truth, self.truth, self.bins, self.genes)
        self.assertEqual(sparsity_summary(r)["n_genes"].sum(), r["corr"].notna().sum())

    def test_proportions_true_data_beats_shuffled(self):
        labels = canonical_celltype(self.truth.obs["celltype"])
        good = celltype_proportion_score(self.truth, self.truth, labels, self.genes)
        bad = celltype_proportion_score(shuffled_cells(self.truth), self.truth, labels, self.genes)
        self.assertLess(good["celltype_tv"], bad["celltype_tv"])
        self.assertGreater(good["celltype_balanced_accuracy"], bad["celltype_balanced_accuracy"])


class TestEvaluatorSynthetic(unittest.TestCase):
    def test_end_to_end_with_log(self):
        truth, _, pattern = synthetic_embryo()
        with tempfile.TemporaryDirectory() as d:
            write_fake_refs(pathlib.Path(d), truth, pattern)
            ev = Evaluator(truth, ref_dir=d)
            oracle = MeasurementOracle(truth)
            for i, z in enumerate(oracle.sections):
                oracle.measure([f"g{i}", f"g{i + 10}"], section=z)
            s, tables = ev.evaluate(truth, oracle.log())
        self.assertEqual(s["n_experiments"], len(oracle.sections))
        self.assertEqual(s["log_problems"], [])
        self.assertAlmostEqual(s["heldout_corr"], 1.0)
        self.assertGreater(s["peng_spearman"], 0.9)
        self.assertLess(s["celltype_tv"], 0.2)
        self.assertEqual(set(tables), {"region_per_gene", "heldout_per_gene", "heldout_by_sparsity"})


def _real_available():
    return (DATA.exists() and any(DATA.glob("*E7.5_stereo_rep1.h5ad"))
            and (REF_DIR / "peng_2019" / "peng_E7.5_log2fpkm.csv.gz").exists()
            and (REF_DIR / "pijuan_sala_2019" / "atlas_E7.5_mean_log2norm.csv.gz").exists())


@unittest.skipUnless(_real_available(), "E7.5 data or reference tables not found")
class TestRealE75(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():  # the h5ad files were written by an old anndata
            warnings.simplefilter("ignore")
            cls.emb = {r: ad.read_h5ad(next(DATA.glob(f"*_Embryo_E7.5_stereo_rep{r}.h5ad"))) for r in (1, 2)}

    def test_orientation_puts_markers_in_the_right_place(self):
        for rep, a in self.emb.items():
            reg = assign_regions(a.obs)
            r = reg["region"].str.rsplit("_", n=1).str[0]
            def mean(gene, region):
                return float(a[(r == region).to_numpy(), gene].X.mean())
            # T (Brachyury) marks the primitive streak, posterior.
            self.assertGreater(mean("T", "ecto_P"), 5 * mean("T", "ecto_A"), f"rep{rep}")
            # Otx2 and Sox2 mark the anterior epiblast / neurectoderm.
            self.assertGreater(mean("Otx2", "ecto_A"), 3 * mean("Otx2", "ecto_P"), f"rep{rep}")
            self.assertGreater(mean("Sox2", "ecto_A"), mean("Sox2", "ecto_P"), f"rep{rep}")

    def test_scores_separate_true_data_from_scrambled(self):
        for rep, a in self.emb.items():
            bench = RegionBenchmark(a)
            s, _ = bench.score(a)
            self.assertGreater(s["peng_spearman"], 0.4, f"rep{rep}")
            self.assertGreater(s["atlas_spearman"], 0.4, f"rep{rep}")
            perm = np.random.default_rng(0).permutation(a.n_obs)
            shuffled = ad.AnnData(X=sp.csr_matrix(a.X)[perm], obs=pd.DataFrame(index=a.obs_names),
                                  var=pd.DataFrame(index=a.var_names))
            s0, _ = bench.score(shuffled)
            for k in ["peng_spearman", "atlas_spearman", "truth_region_spearman"]:
                self.assertLess(abs(s0[k]), 0.1, f"rep{rep} {k}")

    def test_shallower_data_scores_lower(self):
        from sanity_checks import thinned
        a = self.emb[2]
        bench = RegionBenchmark(a)
        scores = [bench.score(a)[0]] + [bench.score(thinned(a, p))[0] for p in (0.5, 0.2, 0.05)]
        for k in ["peng_spearman", "atlas_spearman", "truth_region_spearman"]:
            vals = [s[k] for s in scores]
            self.assertTrue(all(x > y for x, y in zip(vals, vals[1:])), f"{k}: {vals}")


    def test_dummy_method_submission(self):
        import dummy_method
        hidden, visible = self.emb[2], self.emb[1]
        oracle = MeasurementOracle(hidden)
        recon, log = dummy_method.run(oracle, visible, n_sets=4, how="markers")
        self.assertEqual(check_log(log), [])
        self.assertEqual(len(log), hidden.obs["z"].nunique())       # one experiment per section
        self.assertEqual(log["genes"].str.split(";").explode().nunique(), 16)
        self.assertEqual(list(recon.obs_names), list(hidden.obs_names))
        self.assertEqual(list(recon.var_names), list(hidden.var_names))
        # measured values are passed through unchanged
        e = log.iloc[3]
        cells = hidden.obs_names[hidden.obs["z"] == e["section_z"]]
        genes = e["genes"].split(";")
        np.testing.assert_allclose(recon[cells, genes].X.toarray(), hidden[cells, genes].X.toarray(), atol=1e-5)
        s, _ = Evaluator(hidden).evaluate(recon, log)
        self.assertGreater(s["heldout_corr"], 0.02)
        self.assertLess(s["heldout_corr"], 0.5)
        self.assertGreater(s["peng_spearman"], 0.3)


if __name__ == "__main__":
    unittest.main()
