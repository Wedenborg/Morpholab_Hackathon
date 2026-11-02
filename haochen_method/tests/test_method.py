import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from haochen_method.method import HaochenMethod
from haochen_method.mlp import MLPImputer
from haochen_method.run import parse_args, resolve_reference, load_training_reference
from haochen_method.runner import run_experiment
from haochen_method.selection import HVGSelector


class TestSelection(unittest.TestCase):
    def test_strict_threshold_preserves_rank_and_never_refills(self):
        reference = ad.AnnData(sparse.csr_matrix([[1., 1, 1, 1], [1, 1, 1, 1],
                                                 [0, 1, 1, 1], [0, 0, 1, 1]]))
        reference.var_names = ["g0", "g1", "g2", "outside_pool"]
        table = pd.DataFrame({"sc_detected_cells": [2, 3, 4, 4]}, index=reference.var_names)
        with patch("haochen_method.selection.rank_sc_hvgs", return_value=(np.array([1, 0, 2]), table)):
            selector = HVGSelector(reference, top_k=3, min_cells=1, threshold_pct=50)
            self.assertEqual(selector.candidates.tolist(), ["g1", "g2"])
            self.assertEqual(selector.select(["g1"], 1), ["g2"])
            with self.assertRaises(ValueError):
                selector.select(["g1"], 2)
            random_a = HVGSelector(reference, strategy="random_hvg", seed=9)
            random_b = HVGSelector(reference, strategy="random_hvg", seed=9)
            self.assertEqual(random_a.select([], 2), random_b.select([], 2))


class TestMLP(unittest.TestCase):
    def test_reference_regression_and_zero_measurements(self):
        inputs = np.random.default_rng(4).uniform(0, 3, (100, 2))
        reference = np.column_stack([inputs[:, 1], 2 * inputs[:, 0] + inputs[:, 1], inputs[:, 0]])
        model = MLPImputer(num_layers=1, hidden_dim=16, epochs=100, learning_rate=.02, seed=7)
        model.fit(sparse.csr_matrix(reference), [2, 0])
        observed = np.array([[0., 0.], [1., 2.], [2., 1.]])
        prediction = model.predict(observed)
        np.testing.assert_array_equal(prediction[:, [2, 0]], observed)
        self.assertLess(np.mean((prediction[:, 1] - (2 * observed[:, 0] + observed[:, 1])) ** 2), .05)


class FakeOracle:
    k_max = 4

    def __init__(self, truth):
        self.truth = truth
        self.cells = truth.obs[["x", "y", "z"]].copy()
        self.genes = truth.var_names.copy()
        self.sections = [0., 1.]
        self.records = []

    def measure(self, genes, section):
        if not 1 <= len(genes) <= self.k_max:
            raise ValueError("Oracle budget exceeded")
        chosen = self.truth[self.truth.obs.z == section, genes]
        self.records.append({"step": len(self.records), "stage": "E7.5", "section_z": section,
                             "genes": ";".join(genes), "cells": ""})
        return pd.DataFrame(chosen.X.toarray(), index=chosen.obs_names, columns=genes)

    def log(self):
        return pd.DataFrame(self.records, columns=["step", "stage", "section_z", "genes", "cells"])


class TestPipeline(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.reference = ad.AnnData(sparse.csr_matrix(rng.uniform(.1, 3, (24, 10))))
        self.reference.var_names = [f"g{i}" for i in range(10)]
        self.reference.uns["log1p"] = {"base": None}
        self.library = ["unsupported", "g5", "g0", "g2", "g3", "g9", "g1", "g8", "g7", "g4", "g6"]
        # Explicit zeros test measured/unknown disambiguation.
        matrix = rng.uniform(0, 3, (6, len(self.library)))
        matrix[0] = 0
        obs = pd.DataFrame({"x": np.arange(6), "y": 0., "z": [0., 0., 0., 1., 1., 1.]},
                           index=[f"cell{i}" for i in range(6)])
        self.truth = ad.AnnData(sparse.csr_matrix(matrix), obs=obs,
                                var=pd.DataFrame(index=self.library))

    def make_method(self):
        # Training reference order deliberately differs from target library order.
        return HaochenMethod(self.reference[:, ::-1], self.library, top_k=10, min_cells=1,
                             mlp_options={"epochs": 2, "num_layers": 1, "hidden_dim": 8, "batch_size": 3})

    def test_round_timing_budget_alignment_and_hidden_truth_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            method = self.make_method()
            oracle = FakeOracle(self.truth)
            observed_panels = []

            def score(path, log):
                recon = ad.read_h5ad(path)
                observed_panels.append(method.panel.tolist())
                self.assertEqual(recon.shape, self.truth.shape)
                self.assertEqual(recon.obs_names.tolist(), self.truth.obs_names.tolist())
                np.testing.assert_array_equal(recon[:, method.panel].X,
                                              self.truth[:, method.panel].X.toarray())
                return {"panel_used_for_fit": len(method.panel), "n_experiments": len(log)}, {}

            history = run_experiment(method, oracle, Path(tmp) / "first", initial_genes=["g4", "g0"],
                                      rounds=2, n_select=2, batch_size=2, evaluator=score,
                                      diagnostic_truth=self.truth.X)
            self.assertEqual(history.n_sequenced.tolist(), [2, 4, 6])
            self.assertEqual(history.n_experiments.tolist(), [2, 4, 6])
            self.assertEqual(history.benchmark_panel_used_for_fit.tolist(), [2, 4, 6])
            self.assertEqual(len(set(observed_panels[-1])), 6)
            first = ad.read_h5ad(Path(tmp) / "first/reconstruction.h5ad")
            np.testing.assert_array_equal(first[:, ["unsupported"]].X, np.zeros((6, 1)))
            summary = json.loads((Path(tmp) / "first/summary.json").read_text())
            self.assertEqual(summary["round"], 2)
            # Alter only unmeasured truth. Selection and reconstruction must not change.
            changed = self.truth.copy()
            unknown = ~changed.var_names.isin(observed_panels[-1])
            values = changed.X.toarray()
            values[:, unknown] += 100
            changed.X = sparse.csr_matrix(values)
            run_experiment(self.make_method(), FakeOracle(changed), Path(tmp) / "second",
                           initial_genes=["g4", "g0"], rounds=2, n_select=2, batch_size=2)
            second = ad.read_h5ad(Path(tmp) / "second/reconstruction.h5ad")
            np.testing.assert_array_equal(first.X, second.X)

    def test_pool_shortage_fails_before_measurement_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            method, oracle = self.make_method(), FakeOracle(self.truth)
            with self.assertRaises(ValueError):
                run_experiment(method, oracle, Path(tmp) / "new", initial_genes=["g0"],
                               rounds=10, n_select=4)
            self.assertEqual(len(oracle.log()), 0)
            self.assertFalse((Path(tmp) / "new").exists())
            sentinel = Path(tmp) / "keep.txt"
            sentinel.write_text("existing experiment")
            with self.assertRaises(FileExistsError):
                run_experiment(method, oracle, tmp, initial_genes=["g0"], rounds=0)
            self.assertEqual(sentinel.read_text(), "existing experiment")
            self.assertEqual(len(oracle.log()), 0)


class TestConfiguration(unittest.TestCase):
    def test_defaults_and_sc_precedence(self):
        with patch.dict("os.environ", {}, clear=True):
            args = parse_args(["--output", "unused"])
        self.assertEqual((args.rounds, args.n_select, args.top_k), (5, 4, 2000))
        self.assertEqual(args.initial_genes, ["Sox2", "Hoxa1", "Hhex", "Pdx1"])
        args.reference = Path("explicit_sc.h5ad")
        self.assertEqual(resolve_reference(args, "/unused"), ("sc", Path("explicit_sc.h5ad")))
        args.reference_source, args.reference = "sc", None
        with self.assertRaises(ValueError):
            resolve_reference(args, "/unused")
        with tempfile.TemporaryDirectory() as tmp:
            rep1 = Path(tmp) / "GSM_Embryo_E7.5_stereo_rep1.h5ad"
            rep1.touch()
            args.reference_source = "auto"
            self.assertEqual(resolve_reference(args, tmp), ("rep1", rep1))

    def test_config_overrides_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text(json.dumps({"rounds": 2, "device": "cpu"}))
            args = parse_args(["--config", str(config), "--rounds", "3", "--output", "unused"])
            self.assertEqual(args.rounds, 3)
            config.write_text(json.dumps({"typo": 1}))
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(["--config", str(config), "--output", "unused"])

    def test_sc_loading_preserves_log_count_scale_and_rep1_uses_x(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = np.array([[0., 1.], [2., 0.], [3., 4.]], dtype=np.float32)
            reference = ad.AnnData(sparse.csr_matrix(np.log1p(values)))
            reference.layers["counts"] = sparse.csr_matrix(values)
            path = Path(tmp) / "reference.h5ad"
            reference.write_h5ad(path)
            args = argparse.Namespace(reference_cache=None, annotation=None,
                                      reference_scale="auto", reference_counts_layer="counts")
            sc_reference = load_training_reference(args, "sc", path)
            rep1_reference = load_training_reference(args, "rep1", path)
            for actual in [sc_reference, rep1_reference]:
                np.testing.assert_allclose(actual.X.toarray(), np.log1p(values))
            self.assertEqual(sc_reference.uns["preprocessing"]["normalization"], "none")


if __name__ == "__main__":
    unittest.main()
