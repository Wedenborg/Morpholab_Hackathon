import itertools
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from vae_panel import (
    LatentDecoder,
    PanelScorer,
    decode_panel,
    log_normalize,
    pca_embedding,
    select_panel,
    train_vae,
)


def modular_cells(n_cells: int = 1500, n_modules: int = 4, genes_per_module: int = 25, seed: int = 0) -> torch.Tensor:
    """Independent gene modules of decreasing size, with noise, z-scored."""
    generator = torch.Generator().manual_seed(seed)
    factors = torch.randn(n_cells, n_modules, generator=generator)
    columns = []
    for module in range(n_modules):
        size = genes_per_module - 4 * module
        columns.append(factors[:, [module]] + 0.5 * torch.randn(n_cells, size, generator=generator))
    cells = torch.cat(columns, dim=1)
    return (cells - cells.mean(0)) / cells.std(0)


def module_of(n_modules: int = 4, genes_per_module: int = 25) -> torch.Tensor:
    return torch.cat([torch.full((genes_per_module - 4 * m,), m) for m in range(n_modules)])


def pca_scorer(cells: torch.Tensor, latent: int = 4, **kwargs) -> PanelScorer:
    half = len(cells) // 2
    latents, decoder = pca_embedding(cells[:half], latent)
    return PanelScorer(decoder, cells[:half], latents, cells[half:], cells[half:], **kwargs)


class LogNormalizeTest(unittest.TestCase):
    def test_removes_depth(self):
        counts = sp.csr_matrix(np.array([[1.0, 3.0], [10.0, 30.0]]))
        normalized = log_normalize(counts, target_depth=4).toarray()
        np.testing.assert_allclose(normalized, np.log1p([[1.0, 3.0], [1.0, 3.0]]), rtol=1e-6)


class PanelScorerTest(unittest.TestCase):
    def test_scan_matches_decoding_each_panel_directly(self):
        cells = modular_cells(600)
        half = len(cells) // 2
        latents = cells[:half, :6]
        torch.manual_seed(0)
        body = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.GELU())
        weight, bias = torch.randn(cells.shape[1], 8), torch.randn(cells.shape[1])
        decoder = LatentDecoder(body, weight, bias)
        scorer = PanelScorer(decoder, cells[:half], latents, cells[half:], cells[half:])
        for chosen in ([], [3], [3, 40, 70]):
            measured = torch.zeros(cells.shape[1], dtype=torch.bool)
            measured[chosen] = True
            errors = scorer.scan(measured)
            for gene in (0, 25, 60):
                panel = chosen + [gene]
                predicted = decode_panel(decoder, cells[:half][:, panel], latents, cells[half:][:, panel])
                direct = (predicted - cells[half:]).square().mean().item()
                self.assertAlmostEqual(errors[gene].item(), direct, places=3)
            self.assertTrue(errors[measured].isinf().all())

    def test_eligible_restricts_candidates(self):
        cells = modular_cells()
        eligible = torch.ones(cells.shape[1], dtype=torch.bool)
        eligible[:25] = False
        scorer = pca_scorer(cells, eligible=eligible)
        picks, _ = select_panel(scorer, torch.zeros(cells.shape[1], dtype=torch.bool), 4)
        self.assertTrue((picks >= 25).all())


class SelectPanelTest(unittest.TestCase):
    def setUp(self):
        self.cells = modular_cells()
        self.none_measured = torch.zeros(self.cells.shape[1], dtype=torch.bool)

    def test_covers_each_module_rather_than_the_largest(self):
        picks, _ = select_panel(pca_scorer(self.cells), self.none_measured, 4)
        self.assertEqual(sorted(module_of()[picks].tolist()), [0, 1, 2, 3])

    def test_next_round_avoids_measured_genes_and_lowers_error(self):
        scorer = pca_scorer(self.cells)
        first, first_error = select_panel(scorer, self.none_measured, 4)
        measured = self.none_measured.clone()
        measured[first] = True
        second, second_error = select_panel(scorer, measured, 4)
        self.assertFalse(measured[second].any())
        self.assertEqual(len(second.unique()), 4)
        self.assertLess(second_error, first_error)

    def test_swaps_never_lose_to_greedy_and_reported_error_is_real(self):
        scorer = pca_scorer(self.cells)
        _, greedy_error = select_panel(scorer, self.none_measured, 3, joint=False)
        picks, joint_error = select_panel(scorer, self.none_measured, 3)
        self.assertLessEqual(joint_error, greedy_error)
        measured = self.none_measured.clone()
        measured[picks] = True
        self.assertAlmostEqual(scorer.error(measured), joint_error, places=5)

    def test_joint_search_beats_greedy_when_genes_only_work_together(self):
        # c = a + b: greedy starts from c, but a and b together explain everything.
        generator = torch.Generator().manual_seed(0)
        a, b = torch.randn(2000, 1, generator=generator), torch.randn(2000, 1, generator=generator)
        c = (a + b) / 2 ** 0.5
        noise = 0.05 * torch.randn(2000, 12, generator=generator)
        cells = torch.cat([a.expand(-1, 3), b.expand(-1, 3), c.expand(-1, 6)], dim=1) + noise
        scorer = pca_scorer(cells, latent=2)
        none = torch.zeros(12, dtype=torch.bool)
        _, greedy_error = select_panel(scorer, none, 2, joint=False)
        picks, joint_error = select_panel(scorer, none, 2)
        self.assertLess(joint_error, greedy_error)
        best = min(
            scorer.error(torch.isin(torch.arange(12), torch.tensor(pair)))
            for pair in itertools.combinations(range(12), 2)
        )
        self.assertAlmostEqual(joint_error, best, places=4)
        self.assertEqual(len(picks.unique()), 2)


class TrainVaeTest(unittest.TestCase):
    def test_vae_supports_selection(self):
        cells = modular_cells()
        split = int(0.8 * len(cells))
        model = train_vae(cells[:split], cells[split:], latent=4, hidden=32, steps=600, warmup=200, log_every=600)
        scorer = PanelScorer(model.latent_decoder(), cells[:split], model.encode(cells[:split]),
                             cells[split:], cells[split:])
        none = torch.zeros(cells.shape[1], dtype=torch.bool)
        picks, error = select_panel(scorer, none, 4)
        self.assertEqual(sorted(module_of()[picks].tolist()), [0, 1, 2, 3])
        self.assertLess(error, scorer.error(none))


if __name__ == "__main__":
    unittest.main()
