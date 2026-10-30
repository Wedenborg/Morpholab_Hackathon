"""Active gene acquisition: learn which genes to measure next from simulated experiments.

Two models work on neighbourhood-averaged, z-scored log expression:

- MaskedImputer predicts every target gene from any measured panel. It sees the values
  and the mask, so a measured low value differs from a gene that was never measured.
- AcquisitionNet predicts how much reconstruction error falls if a gene is revealed.
  It sees a tissue summary, the measured mask, and the genes already proposed for the
  current round, so it can avoid four genes that are useful only individually.

Acquisition labels come from fully measured training tissues: hide most genes, reveal a
candidate, and record the realized fall in error. Truth is only used for those labels.
Deployment is one forward pass per proposed gene.

Genes have two roles. Targets are the genes the imputer predicts; candidates are the
targets that may be measured, and they are the imputer's only inputs.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn


@dataclass
class GeneSpace:
    target_genes: pd.Index       # (T,)
    candidate_index: torch.Tensor  # Positions of candidates among targets, (C,).
    mean: torch.Tensor           # Log-expression mean per target, (T,).
    std: torch.Tensor            # Log-expression standard deviation per target, (T,).
    reference_sample: torch.Tensor  # Log profiles that defined the candidates, (R, T).

    @property
    def candidate_genes(self) -> pd.Index:
        return self.target_genes[self.candidate_index.numpy()]

    def standardize_candidates(self, log_values: torch.Tensor) -> torch.Tensor:
        return (log_values - self.mean[self.candidate_index]) / self.std[self.candidate_index]

    def to_log(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std + self.mean


def spatial_smoother(locations: np.ndarray, n_neighbors: int = 25) -> sp.csr_matrix:
    """(N, N) matrix that averages each cell over its nearest cells in the same section."""
    rows, columns, weights = [], [], []
    for section in np.unique(locations[:, 2]):
        members = np.where(locations[:, 2] == section)[0]
        k = min(n_neighbors, len(members))
        xy = locations[members, :2]
        nearest = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(xy, return_distance=False)
        rows.append(np.repeat(members, k))
        columns.append(members[nearest].ravel())
        weights.append(np.full(len(members) * k, 1 / k, dtype=np.float32))
    shape = (len(locations), len(locations))
    return sp.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))), shape=shape)


def aligned_expression(adata, genes: pd.Index) -> sp.csr_matrix:
    """adata.X with columns in `genes` order; genes the embryo lacks are zero columns."""
    columns = pd.Index(adata.var_names).get_indexer(genes)
    present = columns >= 0
    selector = sp.csr_matrix(
        (np.ones(present.sum(), dtype=np.float32), (columns[present], np.where(present)[0])),
        shape=(adata.n_vars, len(genes)),
    )
    return (sp.csr_matrix(adata.X) @ selector).tocsr()


def smoothed_expression(adata, genes: pd.Index, n_neighbors: int = 25) -> torch.Tensor:
    locations = adata.obs[["x", "y", "z"]].to_numpy(dtype=float)
    smoothed = spatial_smoother(locations, n_neighbors) @ aligned_expression(adata, genes)
    return torch.from_numpy(np.asarray(smoothed.todense(), dtype=np.float32))


def build_gene_space(
    embryos: list,
    library: pd.Index,
    n_candidates: int = 1000,
    min_detection: float = 0.02,
    n_neighbors: int = 25,
    n_reference_cells: int = 4000,
    seed: int = 0,
) -> tuple[GeneSpace, list[torch.Tensor]]:
    """Choose genes from training embryos only and return their log neighbourhood profiles.

    Candidates are the most variable targets in reference_sample, the same rule
    gene_recommender applies, so both methods choose from one pool.
    """
    detected = np.zeros(len(library))
    n_cells = 0
    for adata in embryos:
        detected += np.asarray((aligned_expression(adata, library) > 0).sum(0)).ravel()
        n_cells += adata.n_obs
    target_genes = library[detected / n_cells >= min_detection]
    profiles = [smoothed_expression(adata, target_genes, n_neighbors) for adata in embryos]
    pooled = torch.cat(profiles)
    mean, std = pooled.mean(0), pooled.std(0).clamp_min(1e-3)
    generator = torch.Generator().manual_seed(seed)
    reference_sample = pooled[torch.randperm(len(pooled), generator=generator)[:n_reference_cells]]
    candidate_index = reference_sample.var(0).topk(n_candidates).indices.sort().values
    return GeneSpace(target_genes, candidate_index, mean, std, reference_sample), profiles


class MaskedImputer(nn.Module):
    def __init__(self, n_candidates: int, n_targets: int, hidden: int = 512, latent: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 * n_candidates, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, latent), nn.GELU(),
        )
        self.decoder = nn.Linear(latent, n_targets)

    def encode(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Unmeasured values are zeroed here, so they cannot influence any prediction."""
        mask = mask.to(values.dtype)
        return self.encoder(torch.cat([values * mask, mask], dim=-1))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(values, mask))


class AcquisitionNet(nn.Module):
    def __init__(self, state_size: int, n_candidates: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size + 2 * n_candidates, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_candidates),
        )

    def forward(self, state: torch.Tensor, measured: torch.Tensor, pending: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat([state, measured.to(state.dtype), pending.to(state.dtype)], dim=-1))


def sample_panels(
    n_panels: int,
    max_measured: int,
    log_weights: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Random Boolean panels, (n_panels, C): half uniform, half favouring variable genes."""
    n_candidates = len(log_weights)
    sizes = torch.randint(0, max_measured + 1, (n_panels,), generator=generator)
    gumbel = -torch.log(-torch.log(torch.rand(n_panels, n_candidates, generator=generator).clamp_min(1e-9)))
    favour_variable = torch.rand(n_panels, 1, generator=generator) < 0.5
    keys = gumbel + favour_variable * log_weights
    ranks = keys.argsort(1, descending=True).argsort(1)
    return ranks < sizes[:, None]


def train_imputer(
    train: torch.Tensor,
    val: torch.Tensor,
    candidate_index: torch.Tensor,
    log_weights: torch.Tensor,
    *,
    steps: int = 6000,
    batch_size: int = 256,
    max_measured: int = 40,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 500,
) -> MaskedImputer:
    """Fit on z-scored profiles by hiding random panels; keep the best validation weights."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = MaskedImputer(len(candidate_index), train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    val_cells = val[torch.randperm(len(val), generator=generator)[:2048]]
    val_panels = sample_panels(len(val_cells), max_measured, log_weights, generator)
    best_loss, best_state = math.inf, None
    for step in range(1, steps + 1):
        model.train()
        cells = train[torch.randint(len(train), (batch_size,), generator=generator)]
        panels = sample_panels(batch_size, max_measured, log_weights, generator)
        hidden = torch.ones_like(cells)
        hidden[:, candidate_index] = (~panels).to(cells.dtype)
        predictions = model(cells[:, candidate_index], panels)
        loss = ((predictions - cells).square() * hidden).sum() / hidden.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                val_loss = (model(val_cells[:, candidate_index], val_panels) - val_cells).square().mean().item()
            print(f"  imputer step {step}: train {loss.item():.4f} val {val_loss:.4f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {name: value.clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval().requires_grad_(False)


def ridge_fit_predict(
    profiles: torch.Tensor,
    features: torch.Tensor,
    observed: torch.Tensor,
    ridge_penalty: float = 10.0,
) -> torch.Tensor:
    """Ridge from the observed columns of `profiles` to all columns, applied to `features`."""
    means = profiles.mean(0)
    if not observed.any():
        return means.expand(len(features), -1).clone()
    predictors = profiles[:, observed] - means[observed]
    gram = predictors.T @ predictors
    gram.diagonal().add_(ridge_penalty)
    coefficients = torch.linalg.solve(gram, predictors.T @ (profiles - means))
    return (features - means[observed]) @ coefficients + means


def tissue_state(imputer: MaskedImputer, candidates: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
    """Summarise a tissue as the mean and spread of its cells' latent codes."""
    latent = imputer.encode(candidates, measured.expand(len(candidates), -1))
    return torch.cat([latent.mean(0), latent.std(0)])


def propose_genes(
    net: AcquisitionNet,
    state: torch.Tensor,
    measured: torch.Tensor,
    k: int,
    use_pending: bool = True,
) -> torch.Tensor:
    """Pick k candidate positions one at a time, conditioning on those already proposed."""
    pending = torch.zeros_like(measured)
    chosen = []
    for _ in range(k):
        scores = net(state[None], measured[None], pending[None])[0].masked_fill(measured | pending, -math.inf)
        if not use_pending:
            return scores.topk(k).indices
        position = scores.argmax()
        chosen.append(position)
        pending[position] = True
    return torch.stack(chosen)


class RevealSimulator:
    """Computes realized error reductions on a fully measured training tissue."""

    def __init__(self, imputer: MaskedImputer, candidate_index: torch.Tensor, n_scored: int = 1500, seed: int = 0):
        self.imputer = imputer
        self.candidate_index = candidate_index
        n_targets = imputer.decoder.out_features
        generator = torch.Generator().manual_seed(seed)
        self.scored = torch.randperm(n_targets, generator=generator)[:n_scored].sort().values
        membership = torch.isin(candidate_index, self.scored)
        self.scored_candidates = membership.nonzero(as_tuple=True)[0]
        positions = torch.full((n_targets,), -1, dtype=torch.long)
        positions[self.scored] = torch.arange(len(self.scored))
        self.candidate_in_scored = positions[candidate_index[membership]]

    @torch.no_grad()
    def loss(self, cells: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Reconstruction error per panel. Measured genes keep their true values, so they
        contribute zero, and every panel is scored on the same genes and cells."""
        candidates = cells[:, self.candidate_index]
        latent = self.imputer.encode(candidates[None].expand(len(masks), -1, -1), masks[:, None].expand(-1, len(cells), -1))
        decoder = self.imputer.decoder
        predictions = latent @ decoder.weight[self.scored].T + decoder.bias[self.scored]
        errors = (predictions - cells[:, self.scored]).square()
        panel, position = masks[:, self.scored_candidates].nonzero(as_tuple=True)
        errors[panel, :, self.candidate_in_scored[position]] = 0
        return errors.mean((1, 2))

    def improvements(self, cells: torch.Tensor, base: torch.Tensor, genes: torch.Tensor) -> torch.Tensor:
        masks = base.expand(len(genes) + 1, -1).clone()
        masks[torch.arange(1, len(genes) + 1), genes] = True
        losses = self.loss(cells, masks)
        return losses[0] - losses[1:]


@dataclass
class Examples:
    tissue: torch.Tensor    # (S,)
    state: torch.Tensor     # (S, state_size)
    measured: torch.Tensor  # (S, C)
    pending: torch.Tensor   # (S, C)
    tried: torch.Tensor     # (S, n_tried) candidate positions.
    gains: torch.Tensor     # (S, n_tried) realized fall in reconstruction error.

    def concat(self, other: "Examples") -> "Examples":
        fields = ("tissue", "state", "measured", "pending", "tried", "gains")
        return Examples(*[torch.cat([getattr(self, name), getattr(other, name)]) for name in fields])


def _rollout(
    imputer: MaskedImputer,
    net: AcquisitionNet,
    candidates: torch.Tensor,
    n_measured: int,
    k: int,
    epsilon: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Measured panel reached by following the policy, with random exploration."""
    measured = torch.zeros(candidates.shape[1], dtype=torch.bool)
    for _ in range(n_measured // k):
        state = tissue_state(imputer, candidates, measured)
        picks = propose_genes(net, state, measured, k)
        explore = torch.rand(k, generator=generator) < epsilon
        unmeasured = (~measured).nonzero(as_tuple=True)[0]
        random_picks = unmeasured[torch.randperm(len(unmeasured), generator=generator)[:k]]
        picks = torch.where(explore, random_picks, picks)
        measured[picks] = True
    return measured


@torch.no_grad()
def simulate_examples(
    simulator: RevealSimulator,
    tissues: list[torch.Tensor],
    log_weights: torch.Tensor,
    n_states: int,
    *,
    policy: AcquisitionNet | None = None,
    n_tried: int = 24,
    n_cells: int = 128,
    n_state_cells: int = 1024,
    max_rounds: int = 9,
    genes_per_round: int = 4,
    epsilon: float = 0.3,
    seed: int = 0,
    log_every: int = 500,
) -> Examples:
    """Hide a tissue, optionally propose some genes, then reveal candidates and record gains.

    Pending genes stay unseen in the state. Their true values enter only the label, since
    the gain of a gene depends on what its batch mates will turn out to show.
    """
    imputer = simulator.imputer
    candidate_index = simulator.candidate_index
    n_candidates = len(candidate_index)
    generator = torch.Generator().manual_seed(seed)
    records = []
    for step in range(n_states):
        tissue = int(torch.randint(len(tissues), (1,), generator=generator))
        values = tissues[tissue]
        state_cells = values[torch.randperm(len(values), generator=generator)[:n_state_cells]][:, candidate_index]
        n_measured = int(torch.randint(max_rounds + 1, (1,), generator=generator)) * genes_per_round
        if policy is None:
            ranks = (
                torch.rand(n_candidates, generator=generator).log() / log_weights.exp()
            ).argsort(descending=True).argsort()
            measured = ranks < n_measured
        else:
            measured = _rollout(imputer, policy, state_cells, n_measured, genes_per_round, epsilon, generator)
        state = tissue_state(imputer, state_cells, measured)

        unmeasured = (~measured).nonzero(as_tuple=True)[0]
        n_pending = int(torch.randint(genes_per_round, (1,), generator=generator))
        if policy is None or n_pending == 0:
            pending_genes = unmeasured[torch.randperm(len(unmeasured), generator=generator)[:n_pending]]
        else:
            pending_genes = propose_genes(policy, state, measured, n_pending)
        pending = torch.zeros_like(measured)
        pending[pending_genes] = True

        available = (~(measured | pending)).nonzero(as_tuple=True)[0]
        tried = available[torch.randperm(len(available), generator=generator)[:n_tried]]
        if policy is not None:
            # Spend half the labels on the genes the policy currently prefers.
            scores = policy(state[None], measured[None], pending[None])[0].masked_fill(measured | pending, -math.inf)
            preferred = scores.topk(n_tried // 2).indices
            remaining = available[~torch.isin(available, preferred)]
            others = remaining[torch.randperm(len(remaining), generator=generator)[:n_tried - len(preferred)]]
            tried = torch.cat([preferred, others])

        label_cells = values[torch.randperm(len(values), generator=generator)[:n_cells]]
        gains = simulator.improvements(label_cells, measured | pending, tried)
        records.append((tissue, state, measured, pending, tried, gains))
        if (step + 1) % log_every == 0:
            print(f"  simulated {step + 1}/{n_states} states", flush=True)
    columns = list(zip(*records))
    return Examples(
        torch.tensor(columns[0]), *[torch.stack(column) for column in columns[1:]]
    )


def reference_correlation(space: GeneSpace) -> torch.Tensor:
    """Absolute correlation between candidates in the reference, (C, C)."""
    return space.reference_sample[:, space.candidate_index].T.corrcoef().nan_to_num().abs()


@dataclass
class TissueSummary:
    state: torch.Tensor          # Whole-tissue summary, (state_size,).
    gene_features: torch.Tensor  # Predicted expression and uncertainty per candidate, (C, 2).


@torch.no_grad()
def summarize_tissue(
    imputer: MaskedImputer,
    candidates: torch.Tensor,
    measured: torch.Tensor,
    candidate_index: torch.Tensor,
    n_passes: int = 3,
    n_uncertainty_cells: int = 128,
) -> TissueSummary:
    """Features the policy may see. Uncertainty is the spread of dropout predictions."""
    weights, bias = imputer.decoder.weight[candidate_index], imputer.decoder.bias[candidate_index]
    predicted = imputer.encode(candidates, measured.expand(len(candidates), -1)) @ weights.T + bias
    few = candidates[:n_uncertainty_cells]
    imputer.train()
    passes = torch.stack([imputer.encode(few, measured.expand(len(few), -1)) @ weights.T + bias for _ in range(n_passes)])
    imputer.eval()
    features = torch.stack([predicted.mean(0), passes.std(0).mean(0)], dim=1)
    features = (features - features.mean(0)) / features.std(0).clamp_min(1e-6)
    return TissueSummary(tissue_state(imputer, candidates, measured), features)


class LogisticNormalPolicy(nn.Module):
    """z ~ N(m(D), tau^2 I), p = softmax(z), then a panel drawn without replacement.

    One small network scores every gene from its features, so m(D) is shared across genes.
    """

    def __init__(self, state_size: int, n_candidates: int, embedding: int = 16, context: int = 32,
                 hidden: int = 64, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        self.embedding = nn.Embedding(n_candidates, embedding)
        self.context = nn.Sequential(nn.Linear(state_size, context), nn.GELU())
        self.scorer = nn.Sequential(
            nn.Linear(embedding + context + 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def means(self, summary: TissueSummary, coverage: torch.Tensor) -> torch.Tensor:
        """Mean logits, (S, C). coverage holds each gene's similarity to the chosen genes."""
        n_samples, n_candidates = coverage.shape
        genes = torch.cat([
            self.embedding.weight,
            summary.gene_features,
            self.context(summary.state).expand(n_candidates, -1),
        ], dim=1)
        inputs = torch.cat([genes.expand(n_samples, -1, -1), coverage[..., None]], dim=-1)
        return self.scorer(inputs).squeeze(-1)


def draw_panels(
    policy: LogisticNormalPolicy,
    summary: TissueSummary,
    measured: torch.Tensor,
    correlation: torch.Tensor,
    k: int,
    n_samples: int = 1,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Panels (S, k) and their log probabilities (S,). Without a generator, takes the most
    probable gene at each step and adds no noise.

    The noise is one draw per panel. With it held fixed, the log probability of the panel
    differentiates through m, which is far less noisy than scoring the full Gaussian draw.
    """
    n_candidates = len(measured)
    noise = 0.0
    if generator is not None:
        noise = policy.tau * torch.randn(n_samples, n_candidates, generator=generator)
    blocked = measured.expand(n_samples, -1).clone()
    coverage = torch.zeros(n_samples, n_candidates)
    if measured.any():
        coverage = correlation[:, measured].amax(1).expand(n_samples, -1).clone()
    rows = torch.arange(n_samples)
    picks, log_probs = [], 0.0
    for _ in range(k):
        logits = (policy.means(summary, coverage) + noise).masked_fill(blocked, -math.inf)
        log_p = logits.log_softmax(1)
        if generator is None:
            pick = logits.argmax(1)
        else:
            pick = torch.multinomial(log_p.detach().exp(), 1, generator=generator).squeeze(1)
        log_probs = log_probs + log_p[rows, pick]
        picks.append(pick)
        blocked = blocked.clone()
        blocked[rows, pick] = True
        coverage = torch.maximum(coverage, correlation[pick])
    return torch.stack(picks, dim=1), log_probs


@torch.no_grad()
def policy_episode_loss(
    simulator: RevealSimulator,
    policy: LogisticNormalPolicy,
    state_cells: torch.Tensor,
    label_cells: torch.Tensor,
    correlation: torch.Tensor,
    rounds: int,
    k: int,
) -> float:
    """Reconstruction error after following the policy's most probable panels."""
    measured = torch.zeros(state_cells.shape[1], dtype=torch.bool)
    for _ in range(rounds):
        summary = summarize_tissue(simulator.imputer, state_cells, measured, simulator.candidate_index)
        panels, _ = draw_panels(policy, summary, measured, correlation, k)
        measured[panels[0]] = True
    return simulator.loss(label_cells, measured[None]).item()


def train_policy(
    simulator: RevealSimulator,
    tissues: list[torch.Tensor],
    val: torch.Tensor,
    correlation: torch.Tensor,
    *,
    steps: int = 1500,
    n_samples: int = 16,
    n_cells: int = 128,
    n_state_cells: int = 512,
    max_rounds: int = 8,
    genes_per_round: int = 4,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 100,
) -> tuple[LogisticNormalPolicy, dict]:
    """Maximize the realized fall in error from revealing a sampled panel.

    Half the states follow the current policy and half are random panels. Rewards are
    compared within a state, so a state's overall difficulty does not move the gradient.
    The saved weights give the lowest validation error after a full episode.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    imputer = simulator.imputer
    index = simulator.candidate_index
    n_candidates = len(index)
    policy = LogisticNormalPolicy(2 * imputer.decoder.in_features, n_candidates)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    picked = torch.randperm(len(val), generator=generator)
    val_state = val[picked[:n_state_cells]][:, index]
    val_labels = val[picked[n_state_cells:n_state_cells + 1024]]
    random_masks = torch.zeros(20, n_candidates, dtype=torch.bool)
    for mask in random_masks:
        mask[torch.randperm(n_candidates, generator=generator)[:max_rounds * genes_per_round]] = True
    random_loss = simulator.loss(val_labels, random_masks).mean().item()

    best, best_state = math.inf, None
    for step in range(1, steps + 1):
        values = tissues[int(torch.randint(len(tissues), (1,), generator=generator))]
        order = torch.randperm(len(values), generator=generator)
        state_cells = values[order[:n_state_cells]][:, index]
        label_cells = values[order[n_state_cells:n_state_cells + n_cells]]
        rounds = int(torch.randint(max_rounds, (1,), generator=generator))
        measured = torch.zeros(n_candidates, dtype=torch.bool)
        if torch.rand(1, generator=generator) < 0.5:
            measured[torch.randperm(n_candidates, generator=generator)[:rounds * genes_per_round]] = True
        else:
            for _ in range(rounds):
                summary = summarize_tissue(imputer, state_cells, measured, index)
                with torch.no_grad():
                    panels, _ = draw_panels(policy, summary, measured, correlation, genes_per_round, 1, generator)
                measured[panels[0]] = True

        summary = summarize_tissue(imputer, state_cells, measured, index)
        panels, log_probs = draw_panels(policy, summary, measured, correlation, genes_per_round, n_samples, generator)
        masks = measured.expand(n_samples + 1, -1).clone()
        masks[torch.arange(1, n_samples + 1)[:, None], panels] = True
        losses = simulator.loss(label_cells, masks)
        rewards = losses[0] - losses[1:]
        advantages = (rewards - rewards.mean()) / rewards.std().clamp_min(1e-9)
        loss = -(advantages * log_probs).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()

        if step % log_every == 0 or step == steps:
            episode = policy_episode_loss(
                simulator, policy, val_state, val_labels, correlation, max_rounds, genes_per_round)
            print(f"  policy step {step}: val error {episode:.4f} (random panels {random_loss:.4f})", flush=True)
            if episode < best:
                best = episode
                best_state = {name: value.clone() for name, value in policy.state_dict().items()}
    policy.load_state_dict(best_state)
    report = {"val_episode_error": best, "val_random_panel_error": random_loss}
    return policy.eval().requires_grad_(False), report


def within_state_rank_correlation(predicted: torch.Tensor, realized: torch.Tensor) -> float:
    """Mean Spearman correlation across candidates, computed separately in each state."""
    def ranks(values: torch.Tensor) -> torch.Tensor:
        return values.argsort(1).argsort(1).to(torch.float32)

    a, b = ranks(predicted), ranks(realized)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-9)).mean().item()


def gene_prior(examples: Examples, n_candidates: int) -> torch.Tensor:
    """Average gain per gene, ignoring the state: the best any fixed ranking can learn."""
    totals = torch.zeros(n_candidates).index_add_(0, examples.tried.ravel(), examples.gains.ravel())
    counts = torch.zeros(n_candidates).index_add_(0, examples.tried.ravel(), torch.ones(examples.gains.numel()))
    return totals / counts.clamp_min(1)


def train_acquisition(
    train: Examples,
    val: Examples,
    n_candidates: int,
    *,
    steps: int = 4000,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 500,
) -> tuple[AcquisitionNet, dict]:
    """Regress realized gains on tried genes; keep the best validation rank correlation."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    scale = train.gains.std().clamp_min(1e-9)
    net = AcquisitionNet(train.state.shape[1], n_candidates)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)

    def predict(examples: Examples, net: AcquisitionNet) -> torch.Tensor:
        scores = net(examples.state, examples.measured, examples.pending)
        return scores.gather(1, examples.tried)

    prior = gene_prior(train, n_candidates)
    prior_correlation = within_state_rank_correlation(prior[val.tried], val.gains)
    best = -math.inf
    best_state = None
    for step in range(1, steps + 1):
        net.train()
        batch = torch.randint(len(train.gains), (batch_size,), generator=generator)
        scores = net(train.state[batch], train.measured[batch], train.pending[batch])
        loss = (scores.gather(1, train.tried[batch]) - train.gains[batch] / scale).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % log_every == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                correlation = within_state_rank_correlation(predict(val, net), val.gains)
            print(f"  acquisition step {step}: loss {loss.item():.4f} val rank corr {correlation:.3f}", flush=True)
            if correlation > best:
                best = correlation
                best_state = {name: value.clone() for name, value in net.state_dict().items()}
    net.load_state_dict(best_state)
    report = {"val_rank_correlation": best, "val_gene_prior_rank_correlation": prior_correlation}
    print(f"  val rank corr: net {best:.3f}, fixed gene ranking {prior_correlation:.3f}", flush=True)
    return net.eval().requires_grad_(False), report
