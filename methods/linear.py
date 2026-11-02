"""Reuse a gene selector but replace its imputation with ridge regression.

The wrapped selector decides which genes to measure each round.  This wrapper ignores the
selector's imputed expression and instead fits one ridge regression from the genes measured
in previous rounds to every gene in the visible reference.  The current round's choices are
not used for imputation because they have not been measured yet when the method returns.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class LinearImputationConfig:
    alpha: float = 1.0
    max_train_cells: int = 8000
    max_impute_cells: int = 2048
    seed: int = 0
    verbose: bool = True
    device: str | None = None


class LinearImputationMethod:
    """Wrap a selector and impute all genes with ridge regression."""

    def __init__(self, selector, **kwargs):
        self.selector = selector
        self.config = LinearImputationConfig(**kwargs)
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reference: torch.Tensor | None = None
        self.reference_mean: torch.Tensor | None = None
        self.measured: torch.Tensor | None = None
        self.n_genes: int | None = None

    def reset(self) -> None:
        if hasattr(self.selector, "reset"):
            self.selector.reset()
        self.reference = None
        self.reference_mean = None
        self.measured = None
        self.n_genes = None

    def _prepare_reference(self, references: list[torch.Tensor]) -> None:
        if not references:
            raise ValueError("Linear imputation needs at least one reference expression matrix.")
        reference = torch.cat(
            [torch.as_tensor(r, dtype=torch.float32) for r in references], dim=0
        )
        reference = torch.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        if reference.ndim != 2:
            raise ValueError("Reference expression must be cells x genes.")

        n_cells = reference.shape[0]
        if n_cells > self.config.max_train_cells:
            generator = torch.Generator().manual_seed(self.config.seed)
            keep = torch.randperm(n_cells, generator=generator)[: self.config.max_train_cells]
            reference = reference[keep]

        self.reference = reference.to(self.device)
        self.reference_mean = self.reference.mean(dim=0)

    def _impute(self, gene_log_counts: torch.Tensor, observed_idx: torch.Tensor) -> torch.Tensor:
        n_cells, n_genes = gene_log_counts.shape
        if observed_idx.numel() == 0:
            return self.reference_mean.unsqueeze(0).expand(n_cells, n_genes).cpu().clone()

        observed_cpu = observed_idx.detach().cpu()
        reference = self.reference
        x_reference = reference[:, observed_idx]
        x_mean = x_reference.mean(dim=0)
        x_scale = x_reference.std(dim=0, unbiased=False).clamp_min(1e-6)
        x_centered = (x_reference - x_mean) / x_scale

        xtx = x_centered.T @ x_centered
        penalty = max(float(self.config.alpha), 0.0)
        xtx = xtx + torch.eye(
            xtx.shape[0], dtype=xtx.dtype, device=xtx.device
        ) * max(penalty, 1e-6)
        xty = (
            x_centered.T @ reference
            - x_centered.sum(dim=0).unsqueeze(1) * self.reference_mean.unsqueeze(0)
        )
        coefficients = torch.linalg.solve(xtx, xty)

        output = torch.empty(n_cells, n_genes, dtype=torch.float32)
        with torch.no_grad():
            for start in range(0, n_cells, self.config.max_impute_cells):
                end = min(start + self.config.max_impute_cells, n_cells)
                raw = gene_log_counts[start:end]
                x_hidden = (raw[:, observed_cpu].to(self.device) - x_mean) / x_scale
                predicted = x_hidden @ coefficients + self.reference_mean
                predicted = predicted.clamp_min(0.0)
                predicted[:, observed_idx] = raw[:, observed_cpu].to(self.device)
                output[start:end] = predicted.cpu()
        return output

    def __call__(
        self,
        n_genes_per_round: int,
        gene_embeddings: torch.Tensor,
        gene_similarity: torch.Tensor,
        locations: torch.Tensor,
        gene_log_counts: torch.Tensor,
        sc_gene_log_counts: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gene_log_counts = torch.as_tensor(gene_log_counts, dtype=torch.float32).cpu()
        n_genes = gene_log_counts.shape[1]
        if self.measured is None or self.n_genes != n_genes:
            self.measured = torch.zeros(n_genes, dtype=torch.bool)
            self.n_genes = n_genes
        if self.reference is None:
            self._prepare_reference(sc_gene_log_counts)

        choices, _ = self.selector(
            n_genes_per_round,
            gene_embeddings,
            gene_similarity,
            locations,
            gene_log_counts,
            sc_gene_log_counts,
        )

        observed_idx = self.measured.to(self.device).nonzero(as_tuple=True)[0]
        started = time.time()
        imputed = self._impute(gene_log_counts, observed_idx)
        if self.config.verbose:
            print(
                f"Linear: regressed from {observed_idx.numel()} measured genes "
                f"onto {self.n_genes} genes in {time.time() - started:.1f}s",
                flush=True,
            )

        choice_idx = torch.as_tensor(choices, dtype=torch.long).flatten().cpu()
        if choice_idx.numel():
            self.measured[choice_idx] = True
        return choices, imputed
