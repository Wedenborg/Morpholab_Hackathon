"""Choose gene panels that best reconstruct a single-cell reference through a VAE.

A VAE learns a low-dimensional embedding of z-scored log expression. A panel is scored
by what it would recover: ridge regression estimates each cell's latent code from the
panel's genes, the decoder turns that estimate into all genes, and the score is the
reconstruction error on cells held out from the ridge fit.

Each round picks its genes jointly. Greedy steps give a starting panel, then each of its
genes is swapped for the best replacement until no swap lowers the error. Earlier rounds
stay fixed, so every batch is optimised given what has already been measured.

The decoder ends in a linear layer, so the error of a whole batch of candidate panels
needs only (hidden, hidden) matrices rather than a pass through every gene. That makes
it cheap to score every candidate at every step. Selection does not read any values from
the tissue being measured, so the panels depend only on the reference and on which genes
were measured earlier.
"""

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn


def log_normalize(counts: sp.spmatrix, target_depth: float | None = None) -> sp.csr_matrix:
    """log1p of counts scaled to a common depth, by default the median cell's."""
    counts = sp.csr_matrix(counts, dtype=np.float32)
    depth = np.asarray(counts.sum(1)).ravel()
    target_depth = float(np.median(depth)) if target_depth is None else target_depth
    scaled = sp.diags(target_depth / np.maximum(depth, 1)) @ counts
    return sp.csr_matrix((np.log1p(scaled.data), scaled.indices, scaled.indptr), shape=scaled.shape)


@dataclass
class LatentDecoder:
    """Maps latent codes to genes as body(z) @ weight.T + bias."""
    body: Callable[[torch.Tensor], torch.Tensor]
    weight: torch.Tensor  # (G, hidden)
    bias: torch.Tensor    # (G,)

    def __call__(self, latents: torch.Tensor) -> torch.Tensor:
        return self.body(latents) @ self.weight.T + self.bias


class ExpressionVAE(nn.Module):
    def __init__(self, n_genes: int, latent: int = 32, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_genes, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.mean = nn.Linear(hidden, latent)
        self.log_variance = nn.Linear(hidden, latent)
        self.decoder_body = nn.Sequential(
            nn.Linear(latent, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.decoder_head = nn.Linear(hidden, n_genes)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        """Posterior means, the embedding used for selection."""
        return self.mean(self.encoder(values))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder_head(self.decoder_body(latents))

    def latent_decoder(self) -> LatentDecoder:
        return LatentDecoder(self.decoder_body, self.decoder_head.weight, self.decoder_head.bias)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder(values)
        mean, log_variance = self.mean(hidden), self.log_variance(hidden)
        latents = mean + (0.5 * log_variance).exp() * torch.randn_like(mean)
        return self.decode(latents), mean, log_variance


def reconstruction_r2(predicted: torch.Tensor, truth: torch.Tensor, baseline: torch.Tensor) -> float:
    """1 - error / error of predicting `baseline` for every cell."""
    return 1 - ((predicted - truth).square().sum() / (baseline - truth).square().sum()).item()


def train_vae(
    train: torch.Tensor,
    val: torch.Tensor,
    *,
    latent: int = 32,
    hidden: int = 256,
    steps: int = 4000,
    warmup: int = 1000,
    batch_size: int = 256,
    beta: float = 1.0,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 500,
) -> ExpressionVAE:
    """Fit on z-scored profiles with unit-variance Gaussian noise, so the reconstruction term
    is the squared error selection optimises. Keeps the best validation ELBO after warmup."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ExpressionVAE(train.shape[1], latent, hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    best_loss, best_state = math.inf, None
    for step in range(1, steps + 1):
        model.train()
        cells = train[torch.randint(len(train), (batch_size,), generator=generator)]
        predictions, mean, log_variance = model(cells)
        reconstruction = 0.5 * (predictions - cells).square().sum(1).mean()
        kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(1).mean()
        loss = reconstruction + beta * min(1.0, step / warmup) * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                predictions, mean, log_variance = model(val)
                val_reconstruction = 0.5 * (predictions - val).square().sum(1).mean()
                val_kl = 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(1).mean()
                val_loss = (val_reconstruction + beta * val_kl).item()
                r2 = reconstruction_r2(model.decode(mean), val, torch.zeros_like(val))
            print(f"  vae step {step}: train {loss.item():.1f} val {val_loss:.1f} "
                  f"kl {val_kl.item():.1f} val r2 {r2:.4f}", flush=True)
            if step >= warmup and val_loss < best_loss:
                best_loss = val_loss
                best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False)


def pca_embedding(train: torch.Tensor, latent: int = 32) -> tuple[torch.Tensor, LatentDecoder]:
    """The linear counterpart of the VAE: scores (N, latent) and their decoder."""
    means = train.mean(0)
    _, _, components = torch.pca_lowrank(train - means, q=latent, center=False, niter=4)
    return (train - means) @ components, LatentDecoder(nn.Identity(), components, means)


class PanelScorer:
    """Scores panels with exact ridge predictions of the latent code, decoded to all genes.

    Candidates are the columns of the predictor matrices. Those not eligible are never
    proposed. Latent codes are fitted on one set of cells and error is scored on another.
    """

    def __init__(
        self,
        decoder: LatentDecoder,
        fit_predictors: torch.Tensor,
        fit_latents: torch.Tensor,
        score_predictors: torch.Tensor,
        score_targets: torch.Tensor,
        eligible: torch.Tensor | None = None,
        ridge_penalty: float = 10.0,
        chunk: int = 64,
    ):
        n_candidates = fit_predictors.shape[1]
        self.body = decoder.body
        self.chunk = chunk
        self.eligible = torch.ones(n_candidates, dtype=torch.bool) if eligible is None else eligible
        self.predictor_means = fit_predictors.mean(0)
        self.latent_means = fit_latents.mean(0)
        centered = fit_predictors - self.predictor_means
        self.gram = centered.T @ centered
        self.gram.diagonal().add_(ridge_penalty)
        self.cross = centered.T @ (fit_latents - self.latent_means)
        self.score_predictors = score_predictors - self.predictor_means
        # Squared error is ||y||^2 - 2 sum(h * y W) + sum(h W'W * h) for hidden activations h.
        residual_targets = score_targets - decoder.bias
        self.target_energy = residual_targets.square().sum()
        self.target_projection = residual_targets @ decoder.weight
        self.head_gram = decoder.weight.T @ decoder.weight
        self.n_terms = score_targets.numel()

    def _latent_updates(self, measured: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Code predicted from `measured`, (N, d), and how adding each gene changes it:
        residual (N, C) times step (C, d) is the exact ridge update, by block inversion."""
        chosen = measured.nonzero(as_tuple=True)[0]
        if len(chosen) == 0:
            base = torch.zeros(len(self.score_predictors), self.cross.shape[1])
            return base, self.score_predictors, self.cross / self.gram.diagonal()[:, None]
        chosen_gram = self.gram[chosen][:, chosen]
        solved = torch.linalg.solve(chosen_gram, self.gram[chosen])
        chosen_cross = self.cross[chosen]
        base = self.score_predictors[:, chosen] @ torch.linalg.solve(chosen_gram, chosen_cross)
        residual = self.score_predictors - self.score_predictors[:, chosen] @ solved
        schur = (self.gram.diagonal() - (self.gram[chosen] * solved).sum(0)).clamp_min(1e-6)
        return base, residual, (self.cross - solved.T @ chosen_cross) / schur[:, None]

    def _error(self, latents: torch.Tensor) -> torch.Tensor:
        """Mean squared error per panel for codes of shape (P, N, d)."""
        hidden = self.body(latents)
        quadratic = ((hidden @ self.head_gram) * hidden).sum((1, 2))
        linear = (hidden * self.target_projection).sum((1, 2))
        return (self.target_energy + quadratic - 2 * linear) / self.n_terms

    @torch.no_grad()
    def error(self, measured: torch.Tensor) -> float:
        base, _, _ = self._latent_updates(measured)
        return self._error((base + self.latent_means)[None]).item()

    @torch.no_grad()
    def scan(self, measured: torch.Tensor) -> torch.Tensor:
        """Error after adding each candidate, (C,). Infinite where it cannot be added."""
        base, residual, step = self._latent_updates(measured)
        errors = torch.full((len(measured),), math.inf)
        open_genes = (self.eligible & ~measured).nonzero(as_tuple=True)[0]
        for genes in open_genes.split(self.chunk):
            updates = residual[:, genes].T[..., None] * step[genes][:, None]
            errors[genes] = self._error(base + self.latent_means + updates)
        return errors


def select_panel(
    scorer: PanelScorer,
    measured: torch.Tensor,
    k: int,
    joint: bool = True,
    max_passes: int = 5,
) -> tuple[torch.Tensor, float]:
    """The next k candidates given those measured, and the error they reach."""
    mask = measured.clone()
    picks = []
    best = math.inf
    for _ in range(k):
        errors = scorer.scan(mask)
        pick = int(errors.argmin())
        best = errors[pick].item()
        picks.append(pick)
        mask[pick] = True
    for _ in range(max_passes if joint and k > 1 else 0):
        changed = False
        for position in range(k):
            mask[picks[position]] = False
            errors = scorer.scan(mask)
            pick = int(errors.argmin())
            if pick != picks[position] and errors[pick].item() < best:
                picks[position] = pick
                best = errors[pick].item()
                changed = True
            mask[picks[position]] = True
        if not changed:
            break
    return torch.tensor(picks), best


@torch.no_grad()
def decode_panel(
    decoder: LatentDecoder,
    fit_predictors: torch.Tensor,
    fit_latents: torch.Tensor,
    predictors: torch.Tensor,
    ridge_penalty: float = 10.0,
) -> torch.Tensor:
    """All genes predicted from panel columns, through ridge on the latent code."""
    predictor_means, latent_means = fit_predictors.mean(0), fit_latents.mean(0)
    centered = fit_predictors - predictor_means
    gram = centered.T @ centered
    gram.diagonal().add_(ridge_penalty)
    coefficients = torch.linalg.solve(gram, centered.T @ (fit_latents - latent_means))
    return decoder((predictors - predictor_means) @ coefficients + latent_means)
