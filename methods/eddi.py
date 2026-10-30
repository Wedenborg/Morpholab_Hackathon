"""EDDI-style adaptive gene selection and expression reconstruction.

This adapts the EDDI idea to the E7.5 benchmark: a Partial VAE is trained on the visible
embryo, then each benchmark round selects genes by their expected information gain about
the latent state.  The observed set is updated after every measurement, so the 4-gene
panel may change from one round to the next.  Reconstruction uses the same conditional
model to impute all genes from the genes measured so far.

The implementation is PyTorch-only and follows the benchmark method contract used by
``eval_team.harness.run_and_evaluate``.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn


@dataclass
class EDDIConfig:
    latent_dim: int = 12
    gene_dim: int = 16
    hidden_dim: int = 64
    decoder_dim: int = 128
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-3
    beta: float = 0.05
    warmup_epochs: int = 5
    max_train_cells: int = 8000
    max_acquisition_cells: int = 128
    max_imputation_cells: int = 512
    mask_min_genes: int = 1
    mask_max_genes: int = 16
    mask_dense_prob: float = 0.2
    mask_dense_max_genes: int = 256
    detection_weight: float = 1.0
    expression_weight: float = 1.0
    observed_expression_weight: float = 0.1
    detection_pos_weight: float = 5.0
    output_gate: bool = True
    linear_skip: bool = True
    min_detect: float = 0.002
    spatial_k: int = 6
    spatial_alpha: float = 0.4
    expression_spatial_alpha: float = 0.5
    seed: int = 0
    verbose: bool = True
    device: str | None = None


class PartialVAE(nn.Module):
    def __init__(self, n_genes: int, config: EDDIConfig):
        super().__init__()
        self.n_genes = n_genes
        self.config = config

        self.gene_embedding = nn.Embedding(n_genes, config.gene_dim)
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        self.value_encoder = nn.Sequential(
            nn.Linear(2, config.gene_dim),
            nn.SiLU(),
            nn.Linear(config.gene_dim, config.gene_dim),
        )
        self.empty_token = nn.Parameter(torch.zeros(config.gene_dim))

        self.encoder = nn.Sequential(
            nn.Linear(config.gene_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim + 1, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.decoder_dim),
            nn.SiLU(),
        )
        self.fc_expr = nn.Linear(config.decoder_dim, n_genes)
        self.fc_detect = nn.Linear(config.decoder_dim, n_genes)
        self.linear_out = nn.Linear(config.latent_dim, n_genes) if config.linear_skip else None
        if self.linear_out is not None:
            nn.init.zeros_(self.fc_expr.weight)
            nn.init.zeros_(self.fc_expr.bias)
            nn.init.zeros_(self.linear_out.bias)

    def _tokenize(
        self, values: torch.Tensor, gene_idx: torch.Tensor, detected: torch.Tensor
    ) -> torch.Tensor:
        features = torch.stack([values, detected], dim=-1)
        return self.value_encoder(features) + self.gene_embedding(gene_idx)

    def _encode_pooled(
        self, pooled: torch.Tensor, library: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([pooled, library.unsqueeze(1)], dim=1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-8.0, 8.0)
        return mu, logvar

    def encode_subset(
        self,
        values: torch.Tensor,
        gene_idx: torch.Tensor,
        detected: torch.Tensor,
        library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gene_idx.numel() == 0:
            n = values.shape[0]
            return (values.new_zeros(n, self.config.latent_dim),
                    values.new_zeros(n, self.config.latent_dim))
        pooled = self._tokenize(values, gene_idx, detected).mean(dim=1)
        return self._encode_pooled(pooled, library)

    def encode_full(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        detected: torch.Tensor,
        library: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.stack([values, detected], dim=-1)
        tokens = self.value_encoder(features) + self.gene_embedding.weight.unsqueeze(0)
        counts = mask.sum(dim=1, keepdim=True)
        pooled = (tokens * mask.unsqueeze(-1)).sum(dim=1) / counts.clamp(min=1)
        empty = counts.squeeze(1) == 0
        if empty.any():
            pooled = pooled.clone()
            pooled[empty] = self.empty_token
        return self._encode_pooled(pooled, library)

    def token_sum(
        self, values: torch.Tensor, gene_idx: torch.Tensor, detected: torch.Tensor
    ) -> torch.Tensor:
        return self._tokenize(values, gene_idx, detected).sum(dim=1)

    def encode_sum(
        self, token_sum: torch.Tensor, n_observed: int, library: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_observed == 0:
            n = token_sum.shape[0]
            return (token_sum.new_zeros(n, self.config.latent_dim),
                    token_sum.new_zeros(n, self.config.latent_dim))
        return self._encode_pooled(token_sum / n_observed, library)

    def decode(self, z: torch.Tensor, library: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(torch.cat([z, library.unsqueeze(1)], dim=1))
        expr = self.fc_expr(h)
        if self.linear_out is not None:
            expr = expr + self.linear_out(z)
        return expr, self.fc_detect(h)


class EDDIMethod:
    """Stateful callable implementing EDDI for ``eval_team.harness.run_and_evaluate``."""

    def __init__(self, **kwargs):
        self.config = EDDIConfig(**kwargs)
        self.device = torch.device(
            self.config.device
            if self.config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: PartialVAE | None = None
        self.gene_mean: torch.Tensor | None = None
        self.gene_scale: torch.Tensor | None = None
        self.library_mean: float = 0.0
        self.library_std: float = 1.0
        self.reference_log: torch.Tensor | None = None
        self.reference_lib_log: torch.Tensor | None = None
        self.informative: torch.Tensor | None = None
        self.selected: torch.Tensor | None = None
        self.n_genes: int | None = None
        self._spatial_key = None
        self._spatial_matrix = None

    def reset(self) -> None:
        self.model = None
        self.gene_mean = None
        self.gene_scale = None
        self.library_mean = 0.0
        self.library_std = 1.0
        self.reference_log = None
        self.reference_lib_log = None
        self.informative = None
        self.selected = None
        self.n_genes = None
        self._spatial_key = None
        self._spatial_matrix = None

    def _standardize(
        self, values: torch.Tensor, gene_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        if gene_idx is None:
            mean = self.gene_mean
        else:
            mean = self.gene_mean[gene_idx]
        return ((values - mean) / self.gene_scale).clamp(-10.0, 10.0)

    def _library_for_cells(
        self, gene_log_counts: torch.Tensor, observed_idx: torch.Tensor
    ) -> torch.Tensor:
        n_cells = gene_log_counts.shape[0]
        if observed_idx.numel() == 0:
            return torch.zeros(n_cells, dtype=torch.float32, device=self.device)
        observed_cpu = observed_idx.detach().cpu()
        raw_cpu = gene_log_counts.detach().cpu()
        observed_sum = torch.expm1(raw_cpu[:, observed_cpu].clamp_min(0.0)).sum(dim=1)
        reference_sum = torch.expm1(
            self.reference_log[:, observed_cpu].clamp_min(0.0)
        ).sum(dim=1)
        reference_total = torch.expm1(self.reference_lib_log).clamp_min(1.0)
        fraction = (reference_sum / reference_total).mean().clamp_min(1e-4)
        median_total = reference_total.median()
        estimate = torch.where(observed_sum > 0, observed_sum / fraction, median_total)
        library = torch.log1p(estimate.clamp_min(0.0))
        return ((library - self.library_mean) / self.library_std).to(self.device)

    def _fit(self, references: list[torch.Tensor]) -> None:
        if not references:
            raise ValueError("EDDI needs at least one reference expression matrix.")
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
        self.selected = torch.zeros(n_genes, dtype=torch.bool)

        n_cells = reference.shape[0]
        if n_cells > self.config.max_train_cells:
            generator = torch.Generator().manual_seed(self.config.seed)
            keep = torch.randperm(n_cells, generator=generator)[: self.config.max_train_cells]
            reference = reference[keep]

        counts = torch.expm1(reference.clamp_min(0.0))
        total_counts = counts.sum(dim=1)
        library_log = torch.log1p(total_counts)
        self.reference_log = reference.contiguous()
        self.reference_lib_log = library_log.contiguous()
        self.library_mean = float(library_log.mean())
        self.library_std = max(float(library_log.std(unbiased=False)), 1e-3)

        reference_gpu = reference.to(self.device)
        self.gene_mean = reference_gpu.mean(dim=0)
        self.gene_scale = reference_gpu.std(unbiased=False).clamp_min(1e-3)
        detected = (reference_gpu > 0).float()
        library = ((library_log - self.library_mean) / self.library_std).to(self.device)
        x = self._standardize(reference_gpu)
        detection_rate = detected.mean(dim=0)
        self.informative = (detection_rate >= self.config.min_detect).cpu()
        del reference_gpu

        self.model = PartialVAE(n_genes, self.config).to(self.device)
        if self.model.linear_out is not None:
            q = min(self.config.latent_dim, x.shape[0], n_genes)
            _, singular, components = torch.pca_lowrank(x, q=q, center=False)
            loadings = components * (
                singular / math.sqrt(max(1, x.shape[0]))
            ).unsqueeze(0)
            self.model.linear_out.weight.data.zero_()
            self.model.linear_out.weight.data[:, :q] = loadings
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        if self.config.verbose:
            print(
                f"EDDI v2: training Partial VAE on {x.shape[0]} reference cells, "
                f"{n_genes} genes, device={self.device}",
                flush=True,
            )
        started = time.time()
        self.model.train()
        pos_weight = torch.tensor(
            self.config.detection_pos_weight, device=self.device, dtype=x.dtype
        )
        for epoch in range(self.config.epochs):
            perm = torch.randperm(x.shape[0], device=x.device)
            beta = self.config.beta * min(
                1.0, (epoch + 1) / max(1, self.config.warmup_epochs)
            )
            running_loss = 0.0
            running_det = 0.0
            running_expr = 0.0
            n_batches = 0
            for start in range(0, x.shape[0], self.config.batch_size):
                idx = perm[start:start + self.config.batch_size]
                xb = x[idx]
                if xb.shape[0] == 0:
                    continue
                detected_b = detected[idx]
                library_b = library[idx]

                dense = torch.rand(1).item() < self.config.mask_dense_prob
                if dense:
                    lo = self.config.mask_max_genes / n_genes
                    hi = self.config.mask_dense_max_genes / n_genes
                else:
                    lo = self.config.mask_min_genes / n_genes
                    hi = self.config.mask_max_genes / n_genes
                lo = max(lo, 1.0 / n_genes)
                hi = min(max(hi, lo), 1.0)
                p = math.exp(torch.empty(1).uniform_(math.log(lo), math.log(hi)).item())
                mask = torch.rand(xb.shape, device=xb.device) < p

                empty = ~mask.any(dim=1)
                if empty.any():
                    rows = empty.nonzero(as_tuple=True)[0]
                    cols = torch.randint(n_genes, (rows.numel(),), device=xb.device)
                    mask[rows, cols] = True
                full = mask.all(dim=1)
                if full.any():
                    rows = full.nonzero(as_tuple=True)[0]
                    cols = torch.randint(n_genes, (rows.numel(),), device=xb.device)
                    mask[rows, cols] = False

                mu, logvar = self.model.encode_full(xb, mask, detected_b, library_b)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                expr, logits = self.model.decode(z, library_b)

                hidden = ~mask
                det_loss = F.binary_cross_entropy_with_logits(
                    logits[hidden], detected_b[hidden], pos_weight=pos_weight
                )
                positive = hidden & (detected_b > 0.5)
                if positive.any():
                    expr_loss = torch.mean((expr - xb)[positive] ** 2)
                else:
                    expr_loss = expr.new_zeros(())
                observed_positive = mask & (detected_b > 0.5)
                if observed_positive.any():
                    observed_loss = torch.mean((expr - xb)[observed_positive] ** 2)
                else:
                    observed_loss = expr.new_zeros(())
                kl = 0.5 * (
                    mu.pow(2) + logvar.exp() - 1.0 - logvar
                ).sum(dim=1).mean()
                loss = (
                    self.config.detection_weight * det_loss
                    + self.config.expression_weight * expr_loss
                    + self.config.observed_expression_weight * observed_loss
                    + beta * kl
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                running_loss += float(loss.detach())
                running_det += float(det_loss.detach())
                running_expr += float(expr_loss.detach())
                n_batches += 1

            if self.config.verbose and (epoch + 1) % max(1, self.config.epochs // 5) == 0:
                denom = max(1, n_batches)
                print(
                    f"EDDI v2: epoch {epoch + 1}/{self.config.epochs}, "
                    f"loss={running_loss / denom:.4f} "
                    f"det={running_det / denom:.4f} "
                    f"expr={running_expr / denom:.4f}",
                    flush=True,
                )
        self.model.eval()
        if self.config.verbose:
            print(f"EDDI v2: training finished in {time.time() - started:.1f}s", flush=True)

    def _get_spatial_matrix(self, locations: torch.Tensor, dtype, device):
        k = self.config.spatial_k
        if k <= 0 or locations is None:
            return None
        key = (locations.data_ptr(), tuple(locations.shape), str(device), str(dtype))
        if self._spatial_key == key:
            return self._spatial_matrix

        loc = locations.detach().cpu().numpy()
        n_cells = loc.shape[0]
        rows, cols, vals = [], [], []
        sections = np.unique(loc[:, 2])
        for section in sections:
            idx = np.where(loc[:, 2] == section)[0]
            if len(idx) < 2:
                for global_i in idx:
                    rows.append(int(global_i))
                    cols.append(int(global_i))
                    vals.append(1.0)
                continue
            kk = min(k, len(idx) - 1)
            tree = cKDTree(loc[idx, :2])
            dist, neighbours = tree.query(loc[idx, :2], k=kk + 1)
            dist = np.atleast_2d(dist)
            neighbours = np.atleast_2d(neighbours)
            for local_i, global_i in enumerate(idx):
                d = dist[local_i, 1:]
                nbr = idx[neighbours[local_i, 1:]]
                w = 1.0 / (d + 1e-6)
                w = w / w.sum()
                rows.extend([int(global_i)] * len(nbr))
                cols.extend(nbr.tolist())
                vals.extend(w.tolist())

        if not rows:
            return None
        matrix = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))
        coo = matrix.tocoo()
        indices = torch.from_numpy(np.vstack((coo.row, coo.col))).long()
        values = torch.from_numpy(coo.data).to(device=device, dtype=dtype)
        matrix_t = torch.sparse_coo_tensor(
            indices, values, (n_cells, n_cells), device=device, dtype=dtype,
            check_invariants=False,
        ).coalesce()
        self._spatial_key = key
        self._spatial_matrix = matrix_t
        return matrix_t

    def _smooth_latent(self, latent: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
        alpha = self.config.spatial_alpha
        if alpha <= 0:
            return latent
        matrix = self._get_spatial_matrix(locations, latent.dtype, latent.device)
        if matrix is None:
            return latent
        return (1.0 - alpha) * latent + alpha * torch.sparse.mm(matrix, latent)

    def _candidate_scores(
        self,
        token_sum: torch.Tensor,
        n_observed: int,
        predicted_expr: torch.Tensor,
        predicted_detect: torch.Tensor,
        candidates: torch.Tensor,
        mu_obs: torch.Tensor,
        logvar_obs: torch.Tensor,
        library: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.empty(candidates.numel(), device=self.device)
        chunk_size = 2048
        with torch.no_grad():
            for start in range(0, candidates.numel(), chunk_size):
                chunk = candidates[start:start + chunk_size]
                values = predicted_expr[:, chunk]
                detected = predicted_detect[:, chunk]
                tokens = self.model._tokenize(values, chunk, detected)
                if n_observed:
                    pooled = (token_sum.unsqueeze(1) + tokens) / (n_observed + 1)
                else:
                    pooled = tokens
                n_cells, n_candidates, gene_dim = pooled.shape
                mu_c, logvar_c = self.model._encode_pooled(
                    pooled.reshape(-1, gene_dim),
                    library.repeat_interleave(n_candidates),
                )
                mu_c = mu_c.reshape(n_cells, n_candidates, -1)
                logvar_c = logvar_c.reshape(n_cells, n_candidates, -1)
                kl = 0.5 * (
                    (mu_c - mu_obs.unsqueeze(1)) ** 2 * torch.exp(-logvar_obs).unsqueeze(1)
                    + torch.exp(logvar_c - logvar_obs.unsqueeze(1))
                    - 1.0
                    + logvar_obs.unsqueeze(1)
                    - logvar_c
                ).sum(dim=-1).mean(dim=0)
                scores[start:start + chunk.numel()] = kl.clamp_min(0)
        return scores

    def _acquire(
        self,
        observed_values: torch.Tensor,
        observed_idx: torch.Tensor,
        observed_detected: torch.Tensor,
        library: torch.Tensor,
        n_select: int,
    ) -> list[int]:
        if n_select <= 0:
            return []
        n_cells = observed_values.shape[0]
        generator = torch.Generator().manual_seed(self.config.seed)
        if n_cells > self.config.max_acquisition_cells:
            acq_rows = torch.randperm(
                n_cells, generator=generator
            )[: self.config.max_acquisition_cells]
        else:
            acq_rows = torch.arange(n_cells)
        acq_rows = acq_rows.to(self.device)
        values = observed_values[acq_rows]
        detected = observed_detected[acq_rows]
        lib = library[acq_rows]
        n_observed = int(observed_idx.numel())

        available = ~self.selected.to(self.device)
        if self.informative is not None:
            informative = self.informative.to(self.device)
            if int((available & informative).sum()) >= n_select:
                available = available & informative

        chosen: list[int] = []
        with torch.no_grad():
            if n_observed:
                token_sum = self.model.token_sum(values, observed_idx, detected)
            else:
                token_sum = values.new_zeros((values.shape[0], self.config.gene_dim))

            for _ in range(min(n_select, int(available.sum()))):
                mu_obs, logvar_obs = self.model.encode_sum(token_sum, n_observed, lib)
                predicted_expr, predicted_logits = self.model.decode(mu_obs, lib)
                predicted_detect = (predicted_logits.sigmoid() > 0.5).float()
                candidates = available.nonzero(as_tuple=True)[0]
                if candidates.numel() == 0:
                    break
                scores = self._candidate_scores(
                    token_sum,
                    n_observed,
                    predicted_expr,
                    predicted_detect,
                    candidates,
                    mu_obs,
                    logvar_obs,
                    lib,
                )
                best_pos = int(torch.argmax(scores))
                best = int(candidates[best_pos])
                chosen.append(best)

                available[best] = False
                new_token = self.model._tokenize(
                    predicted_expr[:, best].unsqueeze(1),
                    torch.tensor([best], device=self.device),
                    predicted_detect[:, best].unsqueeze(1),
                ).squeeze(1)
                if n_observed == 0:
                    token_sum = new_token
                else:
                    token_sum = token_sum + new_token
                n_observed += 1
        return chosen

    def _impute(
        self,
        gene_log_counts: torch.Tensor,
        observed: torch.Tensor,
        locations: torch.Tensor,
    ) -> torch.Tensor:
        n_cells = gene_log_counts.shape[0]
        observed_idx = observed.to(self.device).nonzero(as_tuple=True)[0]
        latent = torch.zeros(
            (n_cells, self.config.latent_dim), dtype=torch.float32, device=self.device
        )
        library = torch.zeros(n_cells, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for start in range(0, n_cells, self.config.max_imputation_cells):
                end = min(start + self.config.max_imputation_cells, n_cells)
                raw = gene_log_counts[start:end].to(self.device)
                if observed_idx.numel():
                    observed_values = self._standardize(raw[:, observed_idx], observed_idx)
                    observed_detected = (raw[:, observed_idx] > 0).float()
                else:
                    observed_values = raw.new_zeros((end - start, 0))
                    observed_detected = raw.new_zeros((end - start, 0))
                lib = self._library_for_cells(raw, observed_idx)
                mu, _ = self.model.encode_subset(
                    observed_values, observed_idx, observed_detected, lib
                )
                latent[start:end] = mu
                library[start:end] = lib

            latent = self._smooth_latent(latent, locations)
            output = torch.empty(n_cells, self.n_genes, dtype=torch.float32)
            for start in range(0, n_cells, self.config.max_imputation_cells):
                end = min(start + self.config.max_imputation_cells, n_cells)
                expr, logits = self.model.decode(latent[start:end], library[start:end])
                recon = expr * self.gene_scale + self.gene_mean
                if self.config.output_gate:
                    recon = torch.sigmoid(logits) * recon.clamp_min(0.0)
                else:
                    recon = recon.clamp_min(0.0)
                raw = gene_log_counts[start:end]
                if observed_idx.numel():
                    recon[:, observed_idx] = raw[:, observed_idx.cpu()].to(self.device)
                output[start:end] = recon.cpu()

        alpha = self.config.expression_spatial_alpha
        if alpha > 0:
            matrix = self._get_spatial_matrix(locations, torch.float32, self.device)
            if matrix is not None:
                smoothed = output.to(self.device)
                smoothed = (1.0 - alpha) * smoothed + alpha * torch.sparse.mm(matrix, smoothed)
                output = smoothed.cpu()
                if observed_idx.numel():
                    output[:, observed_idx.cpu()] = gene_log_counts[:, observed_idx.cpu()]
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
        del gene_embeddings, gene_similarity
        gene_log_counts = torch.as_tensor(gene_log_counts, dtype=torch.float32).cpu()
        if self.model is None:
            self._fit(sc_gene_log_counts)
        if gene_log_counts.shape[1] != self.n_genes:
            raise ValueError(
                f"gene_log_counts has {gene_log_counts.shape[1]} genes, expected {self.n_genes}."
            )

        observed = self.selected.clone()
        observed_idx = observed.to(self.device).nonzero(as_tuple=True)[0]
        if observed_idx.numel():
            observed_values = self._standardize(
                gene_log_counts[:, observed_idx.cpu()].to(self.device), observed_idx
            )
            observed_detected = (
                gene_log_counts[:, observed_idx.cpu()].to(self.device) > 0
            ).float()
        else:
            observed_values = gene_log_counts.new_zeros(
                (gene_log_counts.shape[0], 0), device=self.device
            )
            observed_detected = gene_log_counts.new_zeros(
                (gene_log_counts.shape[0], 0), device=self.device
            )
        library = self._library_for_cells(gene_log_counts, observed_idx)

        chosen = self._acquire(
            observed_values, observed_idx, observed_detected, library, int(n_genes_per_round)
        )
        new_selected = torch.zeros(self.n_genes, dtype=torch.bool)
        if chosen:
            new_selected[torch.tensor(chosen, dtype=torch.long)] = True

        imputed = self._impute(gene_log_counts, observed, locations)
        self.selected = self.selected | new_selected
        choices = torch.tensor(chosen, dtype=torch.long)
        return choices, imputed


def eddi_method(**kwargs) -> EDDIMethod:
    """Create a fresh EDDI method instance for one benchmark run."""
    return EDDIMethod(**kwargs)
