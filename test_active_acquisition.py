"""Synthetic checks of the rules the acquisition experiment depends on.

python -m unittest test_active_acquisition -v
"""
import unittest

import numpy as np
import torch

from active_acquisition import (
    AcquisitionNet,
    LogisticNormalPolicy,
    MaskedImputer,
    RevealSimulator,
    TissueSummary,
    draw_panels,
    policy_episode_loss,
    propose_genes,
    sample_panels,
    simulate_examples,
    spatial_smoother,
    summarize_tissue,
    tissue_state,
    train_imputer,
    train_policy,
    within_state_rank_correlation,
)


def low_rank_tissue(n_cells=1500, n_targets=60, n_factors=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    factors = torch.randn(n_cells, n_factors, generator=generator)
    loadings = torch.randn(n_factors, n_targets, generator=generator)
    values = factors @ loadings + 0.1 * torch.randn(n_cells, n_targets, generator=generator)
    return (values - values.mean(0)) / values.std(0)


class HiddenValuesStayHidden(unittest.TestCase):
    def test_unmeasured_values_cannot_change_a_recommendation(self):
        torch.manual_seed(0)
        imputer = MaskedImputer(10, 20).eval()
        net = AcquisitionNet(2 * 128, 10).eval()
        measured = torch.zeros(10, dtype=torch.bool)
        measured[:3] = True
        cells = torch.randn(50, 10)
        altered = cells.clone()
        altered[:, 3:] = 1000.0
        with torch.no_grad():
            first = propose_genes(net, tissue_state(imputer, cells, measured), measured, 4)
            second = propose_genes(net, tissue_state(imputer, altered, measured), measured, 4)
            self.assertTrue(torch.equal(imputer(cells, measured.expand(50, -1)), imputer(altered, measured.expand(50, -1))))
        self.assertTrue(torch.equal(first, second))

    def test_measured_zero_differs_from_missing(self):
        torch.manual_seed(0)
        imputer = MaskedImputer(5, 8).eval()
        cells = torch.zeros(1, 5)
        missing = torch.zeros(1, 5, dtype=torch.bool)
        measured = missing.clone()
        measured[0, 0] = True
        with torch.no_grad():
            self.assertFalse(torch.equal(imputer(cells, missing), imputer(cells, measured)))


class Proposals(unittest.TestCase):
    def test_proposals_are_distinct_and_unmeasured(self):
        torch.manual_seed(0)
        net = AcquisitionNet(4, 12).eval()
        measured = torch.zeros(12, dtype=torch.bool)
        measured[[1, 5, 7]] = True
        for use_pending in (True, False):
            with torch.no_grad():
                picks = propose_genes(net, torch.zeros(4), measured, 4, use_pending=use_pending)
            self.assertEqual(len(picks.unique()), 4)
            self.assertFalse(measured[picks].any())

    def test_pending_genes_can_change_later_picks(self):
        class Avoids(torch.nn.Module):
            """Gene 0 is best, but gene 1 is worthless once gene 0 is proposed."""
            def forward(self, state, measured, pending):
                scores = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
                return scores - 5 * pending[:, [0]] * torch.tensor([[0.0, 1.0, 0.0, 0.0]])

        measured = torch.zeros(4, dtype=torch.bool)
        sequential = propose_genes(Avoids(), torch.zeros(1), measured, 2)
        independent = propose_genes(Avoids(), torch.zeros(1), measured, 2, use_pending=False)
        self.assertEqual(sequential.tolist(), [0, 2])
        self.assertEqual(independent.tolist(), [0, 1])


class RevealLabels(unittest.TestCase):
    def setUp(self):
        self.values = low_rank_tissue()
        self.candidates = torch.arange(0, 30)
        self.weights = torch.zeros(30)
        self.imputer = train_imputer(
            self.values[:1200], self.values[1200:], self.candidates, self.weights,
            steps=400, batch_size=64, max_measured=12, log_every=400,
        )
        self.simulator = RevealSimulator(self.imputer, self.candidates, n_scored=60)

    def test_imputer_improves_as_measurements_arrive(self):
        cells = self.values[1200:]
        empty = torch.zeros((1, 30), dtype=torch.bool)
        panel = empty.clone()
        panel[0, :8] = True
        losses = self.simulator.loss(cells, torch.cat([empty, panel]))
        self.assertLess(losses[1], 0.6 * losses[0])

    def test_gains_telescope_to_the_total_fall_in_error(self):
        cells = self.values[1200:]
        base = torch.zeros(30, dtype=torch.bool)
        mask = base.clone()
        total = 0.0
        for gene in [2, 11, 17, 25]:
            total += self.simulator.improvements(cells, mask, torch.tensor([gene]))[0].item()
            mask[gene] = True
        losses = self.simulator.loss(cells, torch.stack([base, mask]))
        self.assertAlmostEqual(total, (losses[0] - losses[1]).item(), places=4)

    def test_measured_genes_add_no_error(self):
        cells = self.values[1200:]
        mask = torch.zeros(30, dtype=torch.bool)
        mask[:30] = True
        scored_candidates = self.simulator.scored_candidates
        self.assertEqual(len(scored_candidates), 30)
        unmeasured_only = self.simulator.loss(cells, mask[None])
        predictions = self.imputer(cells[:, self.candidates], mask.expand(len(cells), -1))
        hidden_targets = torch.arange(30, 60)
        expected = (predictions[:, hidden_targets] - cells[:, hidden_targets]).square().sum() / (len(cells) * 60)
        self.assertAlmostEqual(unmeasured_only.item(), expected.item(), places=4)

    def test_examples_never_reveal_measured_or_pending_genes(self):
        examples = simulate_examples(
            self.simulator, [self.values], self.weights, 20,
            n_tried=5, n_cells=32, n_state_cells=64, max_rounds=4, seed=1,
        )
        self.assertFalse((examples.measured & examples.pending).any())
        self.assertFalse(examples.measured.gather(1, examples.tried).any())
        self.assertFalse(examples.pending.gather(1, examples.tried).any())
        self.assertLessEqual(examples.pending.sum(1).max().item(), 3)
        self.assertTrue(all(size % 4 == 0 for size in examples.measured.sum(1).tolist()))


class LogisticNormal(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.n_candidates = 12
        self.imputer = MaskedImputer(self.n_candidates, 20).eval().requires_grad_(False)
        self.policy = LogisticNormalPolicy(2 * 128, self.n_candidates)
        self.index = torch.arange(self.n_candidates)
        self.cells = torch.randn(40, self.n_candidates)
        self.correlation = torch.eye(self.n_candidates)
        self.measured = torch.zeros(self.n_candidates, dtype=torch.bool)
        self.measured[[1, 5]] = True

    def summary(self, cells=None):
        return summarize_tissue(self.imputer, self.cells if cells is None else cells, self.measured, self.index)

    def test_panels_are_distinct_and_unmeasured(self):
        generator = torch.Generator().manual_seed(1)
        for draw_from in (generator, None):
            panels, _ = draw_panels(self.policy, self.summary(), self.measured, self.correlation, 4, 8, draw_from)
            self.assertEqual(panels.shape, (8, 4))
            for panel in panels:
                self.assertEqual(len(panel.unique()), 4)
                self.assertFalse(self.measured[panel].any())

    def test_unmeasured_values_cannot_change_a_panel(self):
        altered = self.cells.clone()
        altered[:, ~self.measured] = 1000.0
        torch.manual_seed(3)
        first = draw_panels(self.policy, self.summary(), self.measured, self.correlation, 4)[0]
        torch.manual_seed(3)
        second = draw_panels(self.policy, self.summary(altered), self.measured, self.correlation, 4)[0]
        self.assertTrue(torch.equal(first, second))

    def test_log_probability_is_sampling_without_replacement(self):
        class Fixed(LogisticNormalPolicy):
            def means(self, summary, coverage):
                return torch.tensor([[2.0, 1.0, 0.0, -1.0]]).expand(len(coverage), -1)

        policy = Fixed(4, 4)
        summary = TissueSummary(torch.zeros(4), torch.zeros(4, 2))
        panels, log_prob = draw_panels(policy, summary, torch.zeros(4, dtype=torch.bool), torch.eye(4), 2)
        self.assertEqual(panels.tolist(), [[0, 1]])
        logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
        expected = logits.log_softmax(0)[0] + logits[1:].log_softmax(0)[0]
        self.assertAlmostEqual(log_prob.item(), expected.item(), places=5)

    def test_proposed_genes_lower_the_score_of_similar_genes(self):
        class AvoidsCovered(LogisticNormalPolicy):
            def means(self, summary, coverage):
                return torch.tensor([3.0, 2.0, 1.0, 0.0]) - 5 * coverage

        correlation = torch.eye(4)
        correlation[0, 1] = correlation[1, 0] = 0.9
        policy = AvoidsCovered(4, 4)
        summary = TissueSummary(torch.zeros(4), torch.zeros(4, 2))
        panels, _ = draw_panels(policy, summary, torch.zeros(4, dtype=torch.bool), correlation, 2)
        self.assertEqual(panels.tolist(), [[0, 2]])

    def test_gradients_are_finite_and_reach_the_scorer(self):
        generator = torch.Generator().manual_seed(1)
        _, log_probs = draw_panels(self.policy, self.summary(), self.measured, self.correlation, 4, 4, generator)
        log_probs.sum().backward()
        for parameter in self.policy.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.policy.scorer[-1].weight.grad.abs().sum().item(), 0)

    def test_training_finds_the_informative_genes(self):
        # Only the first six candidates carry signal; the rest are pure noise.
        generator = torch.Generator().manual_seed(0)
        factors = torch.randn(1500, 3, generator=generator)
        signal = factors @ torch.randn(3, 66, generator=generator) + 0.1 * torch.randn(1500, 66, generator=generator)
        noise = torch.randn(1500, 24, generator=generator)
        values = torch.cat([signal[:, :6], noise, signal[:, 6:]], dim=1)
        values = (values - values.mean(0)) / values.std(0)
        index = torch.arange(30)
        imputer = train_imputer(values[:1200], values[1200:], index, torch.zeros(30),
                                steps=500, batch_size=64, max_measured=8, log_every=500)
        simulator = RevealSimulator(imputer, index, n_scored=60)
        state_cells, label_cells = values[1200:1328, :30], values[1328:]

        def informative_share_and_error(policy):
            summary = summarize_tissue(imputer, state_cells, torch.zeros(30, dtype=torch.bool), index)
            with torch.no_grad():
                panel = draw_panels(policy, summary, torch.zeros(30, dtype=torch.bool), torch.eye(30), 4)[0][0]
            error = policy_episode_loss(simulator, policy, state_cells, label_cells, torch.eye(30), 1, 4)
            return (panel < 6).float().mean().item(), error

        torch.manual_seed(0)
        untrained_share, untrained_error = informative_share_and_error(LogisticNormalPolicy(2 * 128, 30))
        policy, report = train_policy(
            simulator, [values[:1200]], values[1200:], torch.eye(30),
            steps=150, n_samples=8, n_cells=64, n_state_cells=128, max_rounds=1, log_every=50,
        )
        trained_share, trained_error = informative_share_and_error(policy)
        print(f"  informative share {untrained_share:.2f} -> {trained_share:.2f}, "
              f"error {untrained_error:.4f} -> {trained_error:.4f}", flush=True)
        self.assertEqual(trained_share, 1.0)
        self.assertLess(trained_error, untrained_error)
        self.assertLess(report["val_episode_error"], report["val_random_panel_error"])


class Helpers(unittest.TestCase):
    def test_smoother_stays_within_a_section(self):
        locations = np.array([[0, 0, 1], [1, 0, 1], [2, 0, 1], [0, 0, 2], [1, 0, 2]], dtype=float)
        matrix = spatial_smoother(locations, n_neighbors=2).toarray()
        np.testing.assert_allclose(matrix.sum(1), 1)
        self.assertEqual(matrix[:3, 3:].sum(), 0)
        self.assertEqual(matrix[3:, :3].sum(), 0)

    def test_panels_have_requested_shape_and_size_limit(self):
        panels = sample_panels(200, 8, torch.zeros(20), torch.Generator().manual_seed(0))
        self.assertEqual(panels.shape, (200, 20))
        self.assertLessEqual(panels.sum(1).max().item(), 8)
        self.assertGreater(panels.sum(1).float().std().item(), 0)

    def test_rank_correlation_extremes(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        self.assertAlmostEqual(within_state_rank_correlation(values, values), 1.0, places=5)
        self.assertAlmostEqual(within_state_rank_correlation(values, -values), -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
