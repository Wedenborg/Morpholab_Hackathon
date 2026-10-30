"""Contract tests for the EDDI, PPCA, and linear benchmark methods.

Run from the repo root:  python -m unittest discover -s eval_team -t . -v
"""
import unittest

import torch

from methods import EDDIMethod, LinearImputationMethod, PPCAMethod


def synthetic_reference(n_cells=64, n_genes=24, seed=0):
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(n_cells, 3, generator=generator)
    loadings = torch.randn(n_genes, 3, generator=generator)
    reference = latent @ loadings.T
    reference = reference - reference.mean(dim=0)
    reference = reference / reference.std(dim=0).clamp_min(1e-6)
    reference = (reference + 1.5).clamp_min(0.0)
    reference = reference * (torch.rand(reference.shape, generator=generator) < 0.7)
    return reference


def synthetic_locations(n_cells):
    rows = torch.arange(n_cells, dtype=torch.float32)
    return torch.stack([rows % 8, rows // 8, rows % 4], dim=1)


class MethodContractTest(unittest.TestCase):
    n_genes = 24
    n_hidden = 16

    def check_rounds(self, method, n_rounds=3, n_per_round=4):
        reference = synthetic_reference(n_genes=self.n_genes)
        hidden = reference[:self.n_hidden]
        measured = torch.zeros_like(hidden)
        locations = synthetic_locations(self.n_hidden)
        selected = []

        for _ in range(n_rounds):
            choices, imputed = method(
                n_per_round,
                torch.zeros(self.n_genes, 1),
                torch.zeros(0),
                locations,
                measured,
                [reference],
            )
            chosen = [int(i) for i in torch.as_tensor(choices).flatten()]
            self.assertEqual(len(chosen), n_per_round)
            self.assertEqual(len(set(chosen)), n_per_round)
            self.assertFalse(set(chosen) & set(selected))
            self.assertEqual(tuple(imputed.shape), tuple(hidden.shape))
            if selected:
                torch.testing.assert_close(imputed[:, selected], measured[:, selected])
            measured[:, chosen] = hidden[:, chosen]
            selected.extend(chosen)


class TestPPCA(MethodContractTest):
    def test_contract(self):
        self.check_rounds(
            PPCAMethod(
                seed=0,
                latent_dim=4,
                max_train_cells=64,
                max_imputation_cells=16,
                verbose=False,
                device="cpu",
            )
        )


class TestEDDI(MethodContractTest):
    def test_contract(self):
        self.check_rounds(
            EDDIMethod(
                seed=0,
                latent_dim=4,
                gene_dim=8,
                hidden_dim=16,
                decoder_dim=16,
                epochs=1,
                batch_size=32,
                max_train_cells=64,
                max_acquisition_cells=16,
                max_imputation_cells=16,
                spatial_k=2,
                verbose=False,
                device="cpu",
            )
        )


class FixedSelector:
    def __init__(self):
        self.round = 0

    def __call__(self, n_genes_per_round, *args):
        start = 8 if self.round else 0
        self.round += 1
        choices = torch.arange(start, start + n_genes_per_round)
        return choices, torch.full_like(args[3], 999.0)


class TestLinearImputation(MethodContractTest):
    def test_contract_with_ppca_selector(self):
        self.check_rounds(
            LinearImputationMethod(
                selector=PPCAMethod(
                    seed=0,
                    latent_dim=4,
                    max_train_cells=64,
                    max_imputation_cells=16,
                    verbose=False,
                    device="cpu",
                ),
                max_impute_cells=16,
                verbose=False,
                device="cpu",
            )
        )

    def test_uses_only_previously_measured_genes(self):
        reference = synthetic_reference(n_genes=self.n_genes)
        hidden = reference[:self.n_hidden]
        measured = torch.zeros_like(hidden)
        locations = synthetic_locations(self.n_hidden)
        method = LinearImputationMethod(
            selector=FixedSelector(),
            max_impute_cells=16,
            verbose=False,
            device="cpu",
        )

        choices, imputed = method(
            4, torch.zeros(self.n_genes, 1), torch.zeros(0), locations, measured, [reference]
        )
        expected = reference.mean(dim=0).expand_as(imputed)
        torch.testing.assert_close(imputed, expected)
        measured[:, choices] = hidden[:, choices]

        choices, imputed = method(
            4, torch.zeros(self.n_genes, 1), torch.zeros(0), locations, measured, [reference]
        )
        torch.testing.assert_close(imputed[:, :4], measured[:, :4])
        self.assertFalse(torch.allclose(imputed, torch.full_like(imputed, 999.0)))


if __name__ == "__main__":
    unittest.main()
