"""PPCA-based adaptive gene selection and expression reconstruction.

This is a linear-Gaussian variant of the EDDI/SING acquisition idea, kept separate from the
nonlinear Partial VAE in ``methods/eddi.py``.  It is *not* the EDDI model; it replaces the
Partial VAE with probabilistic PCA (PPCA):

    z ~ N(0, I)
    x_g = mean_g + scale * (W_g . z) + noise_g

Genes are selected greedily by their expected information gain about the latent ``z``.
Given the genes measured in earlier rounds, the Gaussian posterior of ``z`` is computed in
closed form and used to reconstruct every gene.  Because the model is linear, it does not
suffer from the posterior collapse seen in the current Partial VAE on sparse expression.

The implementation is PyTorch-only and follows the benchmark method contract used by
``eval_team.harness.run_and_evaluate``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class PPCAConfig:
    latent_dim: int = 12
    max_train_cells: int = 8000
    min_detect: float = 0.005
    noise: str = "gene"          # "gene" or "scalar"
    noise_floor: float = 1e-4
    max_imputation_cells: int = 512
    seed: int = 0
    verbose: bool = True
    device: str | None = None


class PPCAMethod:
    """Stateful callable implementing PPCA + information-gain acquisition."""

    def __init__(self, **kwargs):
        self.config = PPCAConfig(**kwargs)
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gene_mean: torch.Tensor | None = None
        self.gene_scale: torch.Tensor | None = None
        self.loadings: torch.Tensor | None = None
        self.noise_var: torch.Tensor | None = None
        self.informative: torch.Tensor | None = None
        self.selected: torch.Tensor | None = None
        self.n_genes: int | None = None

    def reset(self) -> None:
        self.gene_mean = None
        self.gene_scale = None
        self.loadings = None
        self.noise_var = None
        self.informative = None
        self.selected = None
        self.n_genes = None

    def _fit(self, references: list[torch.Tensor]) -> None:
        if not references:
            raise ValueError("PPCA needs at least one reference expression matrix.")
        reference = torch.cat(
            [torch.as_tensor(r, dtype=torch.float32) for r in references], dim=0
        )
        reference = torch.nan_to_num(reference, nan=0.0, posinf=0.0, neginf=0.0)
        if reference.ndim != 2:
            raise ValueError("Reference expression must be cells x genes.")

        n_genes = reference.shape[1]
        if self.n_genes is not None and self.n_genes != n_genes:
            self.reset()
        self.n_genes = n_genes

        n_cells = reference.shape[0]
        if n_cells > self.config.max_train_cells:
            generator = torch.Generator().manual_seed(self.config.seed)
            keep = torch.randperm(n_cells, generator=generator)[: self.config.max_train_cells]
            reference = reference[keep]
            n_cells = reference.shape[0]

        detection = (reference > 0).float().mean(dim=0)
        reference = reference.to(self.device)
        self.gene_mean = reference.mean(dim=0)
        self.gene_scale = reference.std().clamp_min(1e-6)
        x = (reference - self.gene_mean) / self.gene_scale

        latent_dim = min(self.config.latent_dim, n_cells, n_genes)
        if latent_dim < 1:
            raise ValueError("PPCA needs at least one cell and one gene.")

        if self.config.verbose:
            print(
                f"PPCA: fitting {latent_dim} latent components on {n_cells} reference cells, "
                f"{n_genes} genes, device={self.device}",
                flush=True,
            )
        started = time.time()
        _, singular, components = torch.pca_lowrank(x, q=latent_dim, center=False)
        # x ~= U diag(S) V^T.  With unit-variance latitudes, x ~= z @ loadings^T.
        self.loadings = (
            components * (singular / (n_cells ** 0.5)).unsqueeze(0)
        ).contiguous()

        total_var = x.var(dim=0, unbiased=False)
        residual = total_var - self.loadings.pow(2).sum(dim=1)
        floor = max(self.config.noise_floor, 0.01 * float(total_var.mean()))
        if self.config.noise == "scalar":
            value = residual.mean().clamp_min(floor)
            self.noise_var = torch.full_like(residual, float(value))
        elif self.config.noise == "gene":
            self.noise_var = residual.clamp_min(floor)
        else:
            raise ValueError("PPCA noise must be 'gene' or 'scalar'.")

        self.informative = detection >= self.config.min_detect
        self.selected = torch.zeros(n_genes, dtype=torch.bool)
        if self.config.verbose:
            print(f"PPCA: fit finished in {time.time() - started:.1f}s", flush=True)

    def _acquire(self, n_select: int) -> list[int]:
        if n_select <= 0:
            return []
        available = (~self.selected) & self.informative
        if int(available.sum()) < n_select:
            available = ~self.selected
        candidates = available.nonzero(as_tuple=True)[0].to(self.device)
        if candidates.numel() == 0:
            return []

        latent_dim = self.loadings.shape[1]
        precision = torch.eye(
            latent_dim, dtype=self.loadings.dtype, device=self.device
        )
        chosen: list[int] = []
        for _ in range(min(n_select, int(candidates.numel()))):
            weights = self.loadings[candidates]
            gains = ((weights @ precision) * weights).sum(dim=1)
            gains = gains / self.noise_var[candidates]
            best_pos = int(torch.argmax(gains))
            best = int(candidates[best_pos])
            chosen.append(best)

            vector = self.loadings[best]
            precision = precision + torch.outer(vector, vector) / self.noise_var[best]
            keep = torch.ones(candidates.numel(), dtype=torch.bool, device=self.device)
            keep[best_pos] = False
            candidates = candidates[keep]
        return chosen

    def _impute(
        self,
        gene_log_counts: torch.Tensor,
        observed_idx: torch.Tensor,
    ) -> torch.Tensor:
        n_cells = gene_log_counts.shape[0]
        output = torch.empty(n_cells, self.n_genes, dtype=torch.float32)

        if observed_idx.numel() == 0:
            prior = self.gene_mean.clamp_min(0.0).cpu()
            output[:] = prior
            return output

        weights = self.loadings[observed_idx]
        noise_var = self.noise_var[observed_idx]
        precision = torch.eye(
            self.loadings.shape[1], dtype=self.loadings.dtype, device=self.device
        )
        precision = precision + (weights / noise_var.unsqueeze(1)).T @ weights
        observed_mean = self.gene_mean[observed_idx]

        with torch.no_grad():
            for start in range(0, n_cells, self.config.max_imputation_cells):
                end = min(start + self.config.max_imputation_cells, n_cells)
                raw = gene_log_counts[start:end].to(self.device)
                values = (raw[:, observed_idx] - observed_mean) / self.gene_scale
                rhs = (values / noise_var) @ weights
                latent = torch.linalg.solve(precision, rhs.T).T
                recon = (latent @ self.loadings.T) * self.gene_scale + self.gene_mean
                recon = recon.clamp_min(0.0)
                recon[:, observed_idx] = raw[:, observed_idx]
                output[start:end] = recon.cpu()
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
        del gene_embeddings, gene_similarity, locations
        gene_log_counts = torch.as_tensor(gene_log_counts, dtype=torch.float32).cpu()
        if self.loadings is None:
            self._fit(sc_gene_log_counts)
        if gene_log_counts.shape[1] != self.n_genes:
            raise ValueError(
                f"gene_log_counts has {gene_log_counts.shape[1]} genes, expected {self.n_genes}."
            )

        observed = self.selected.clone()
        observed_idx = observed.nonzero(as_tuple=True)[0].to(self.device)

        chosen = self._acquire(int(n_genes_per_round))
        new_selected = torch.zeros(self.n_genes, dtype=torch.bool)
        if chosen:
            new_selected[torch.tensor(chosen, dtype=torch.long)] = True

        imputed = self._impute(gene_log_counts, observed_idx)
        self.selected = self.selected | new_selected
        choices = torch.tensor(chosen, dtype=torch.long)
        return choices, imputed


def ppca_method(**kwargs) -> PPCAMethod:
    """Create a fresh PPCA method instance for one benchmark run."""
    return PPCAMethod(**kwargs)
