"""Optional test against the actual eval_team package, without real-data downloads."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import anndata as ad
import numpy as np
from scipy import sparse

from haochen_method.run import main


@unittest.skipUnless(sys.version_info >= (3, 10) and importlib.util.find_spec("eval_team") is not None,
                     "requires the target repository's eval_team and Python >=3.10")
class TestUpstreamBenchmark(unittest.TestCase):
    def test_cli_submission_and_real_evaluator(self):
        # Reuse upstream's own small synthetic embryo and mock-reference fixtures.
        from eval_team.test_region_benchmark import synthetic_embryo, write_fake_refs
        from eval_team.submission import check_log, load_log
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            references = root / "references"
            references.mkdir()
            truth, _, pattern = synthetic_embryo(n_per_ring=20, n_sections=6, n_genes=30, seed=4)
            visible, _, _ = synthetic_embryo(n_per_ring=20, n_sections=6, n_genes=30, seed=8)
            # Positive reference baseline ensures this tiny fixture has >50% detection.
            visible.X = sparse.csr_matrix(np.log1p(np.expm1(visible.X.toarray()) + .05))
            visible.uns["log1p"] = {"base": None}
            visible.write_h5ad(data / "synthetic_Embryo_E7.5_stereo_rep1.h5ad")
            truth.write_h5ad(data / "synthetic_Embryo_E7.5_stereo_rep2.h5ad")
            write_fake_refs(references, truth, pattern)
            output = root / "result"
            main(["--data-dir", str(data), "--eval-reference-dir", str(references),
                  "--reference-source", "rep1", "--initial-genes", "g0", "g1", "g2", "g3",
                  "--rounds", "1", "--top-k", "20", "--hvg-min-cells", "1",
                  "--mlp-epochs", "1", "--mlp-layers", "1", "--mlp-hidden-dim", "8",
                  "--batch-size", "32", "--output", str(output)])
            reconstruction = ad.read_h5ad(output / "reconstruction.h5ad")
            self.assertEqual(reconstruction.shape, truth.shape)
            self.assertTrue(reconstruction.obs_names.equals(truth.obs_names))
            self.assertTrue(reconstruction.var_names.equals(truth.var_names))
            log = load_log(output / "experiments.csv")
            self.assertEqual(check_log(log), [])
            self.assertEqual(len(log), 2 * truth.obs.z.nunique())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["round"], 1)
            self.assertEqual(summary["n_sequenced"], 8)
            self.assertEqual(summary["benchmark_log_problems"], [])
            for name in ["heldout_corr", "peng_spearman", "atlas_spearman", "celltype_tv"]:
                self.assertIn(f"benchmark_{name}", summary)
            self.assertTrue((output / "round_00/heldout_per_gene.csv").exists())
            self.assertTrue((output / "round_01/heldout_per_gene.csv").exists())


if __name__ == "__main__":
    unittest.main()
