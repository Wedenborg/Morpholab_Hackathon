"""Fast gene panel selection and ridge imputation using only PyTorch.

All expression matrices must use aligned gene columns and comparable log-count
normalization.

Selection is greedy. Each step picks the gene that most reduces the reference
variance not yet explained by the measured panel, with every target gene weighted
equally. Optional extras:

- Spatial: with reference_locations, selection and the ridge fit use averages over
  small reference neighbourhoods. spatial_neighbors smooths measured values over
  nearby spatial cells before prediction.
- Cell types: reference_labels with label_weight adds cell-type composition to the
  selection targets.
- Reweighting: reweight_mix moves reference weight towards cells that resemble the
  measured tissue, so measured values can change the next recommendation. It lowered
  held-out scores on the E7.5 to E8.0 embryos, so it is off by default.

An (N, G) observed_mask allows different panels per cell. Rotating panels this way
also lowered scores on those embryos, so measuring one shared panel is preferred.

Without reweighting, selection depends on which genes were measured, not on their
values, so measuring in rounds gives the same panel as one larger batch.

Imputation is a ridge regression from the measured genes to all genes, fitted on
the reference and applied to the spatial cells. Embeddings and similarity are
accepted for API compatibility but are not used.
"""

from dataclasses import dataclass
import math

import torch


@dataclass
class SelectionResult:
    choices: torch.Tensor          # Boolean mask, (G,).
    imputed: torch.Tensor          # Ridge predictions, (N, G); measured values kept.
    initial_scores: torch.Tensor   # First-step scores, (G,); NaN for genes not scored.
    selection_order: torch.Tensor  # Gene indices, (n_genes_per_round,).
    marginal_gains: torch.Tensor   # Share of total target variance, in selection order.


def _spatial_bins(locations: torch.Tensor, cells_per_bin: int) -> torch.Tensor:
    """Assign each cell to an equal-count neighbourhood, never crossing a section."""
    n_cells = len(locations)
    sections = locations[:, -1] if locations.shape[1] >= 3 else torch.zeros_like(locations[:, 0])
    values, section_ids = sections.unique(return_inverse=True)
    if len(values) > n_cells / cells_per_bin:  # Continuous third axis: not discrete sections.
        values, section_ids = values[:1], torch.zeros_like(section_ids)
    bins = torch.zeros(n_cells, dtype=torch.long, device=locations.device)
    next_bin = 0
    for section in range(len(values)):
        members = (section_ids == section).nonzero(as_tuple=True)[0]
        n_bins = max(1, len(members) // cells_per_bin)
        n_strips = max(1, round(math.sqrt(n_bins)))
        by_x = members[locations[members, 0].argsort()]
        for strip in by_x.tensor_split(n_strips):
            by_y = strip[locations[strip, 1].argsort()]
            for part in by_y.tensor_split(max(1, len(strip) // cells_per_bin)):
                bins[part] = next_bin
                next_bin += 1
    return bins


def _bin_average(values: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    n_bins = int(bins.max()) + 1
    totals = torch.zeros(n_bins, *values.shape[1:], device=values.device, dtype=values.dtype)
    totals.index_add_(0, bins, values)
    counts = torch.zeros(n_bins, device=values.device, dtype=values.dtype)
    counts.index_add_(0, bins, torch.ones_like(bins, dtype=values.dtype))
    return totals / counts.reshape(-1, *[1] * (values.ndim - 1))


def _nearest(query: torch.Tensor, points: torch.Tensor, k: int, chunk: int = 2048) -> torch.Tensor:
    parts = [
        torch.cdist(query[start:start + chunk], points).topk(k, dim=1, largest=False).indices
        for start in range(0, len(query), chunk)
    ]
    return torch.cat(parts)


def _measured_features(
    data: torch.Tensor,
    measured_cells: torch.Tensor,
    observed: torch.Tensor,
    locations: torch.Tensor,
    spatial_neighbors: int,
) -> torch.Tensor:
    """Each measured gene in every cell, averaged over nearby cells that measured it."""
    measured = measured_cells[:, observed]
    features = data[:, observed].clone()
    patterns, pattern_ids = measured.unique(dim=1, return_inverse=True)
    for index in range(patterns.shape[1]):
        pattern = patterns[:, index]
        if spatial_neighbors == 1 and pattern.all():
            continue
        columns = (pattern_ids == index).nonzero(as_tuple=True)[0]
        sources = pattern.nonzero(as_tuple=True)[0]
        neighbors = _nearest(locations, locations[sources], min(spatial_neighbors, len(sources)))
        features[:, columns] = data[sources][:, observed][:, columns][neighbors].mean(1)
    return features


def _prepare_references(
    sc_gene_log_counts: list[torch.Tensor],
    reference_locations: list[torch.Tensor] | None,
    reference_labels: list[torch.Tensor] | None,
    cells_per_bin: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return raw cells, the profiles to fit on, and label composition of those profiles."""
    cells = torch.cat([r.to(device=device, dtype=dtype) for r in sc_gene_log_counts])
    labels = None
    if reference_labels is not None:
        ids = torch.cat([l.to(device=device) for l in reference_labels])
        labels = torch.nn.functional.one_hot(ids).to(dtype=dtype)
    if reference_locations is None:
        return cells, cells, labels
    bins, offset = [], 0
    for coordinates in reference_locations:
        ids = _spatial_bins(coordinates.to(device=device, dtype=dtype), cells_per_bin)
        bins.append(ids + offset)
        offset += int(ids.max()) + 1
    bins = torch.cat(bins)
    return cells, _bin_average(cells, bins), None if labels is None else _bin_average(labels, bins)


def _match_weights(
    profiles: torch.Tensor,
    features: torch.Tensor,
    observed: torch.Tensor,
    noise_variance: torch.Tensor,
    mix: float,
    generator: torch.Generator,
    max_cells: int = 512,
) -> torch.Tensor | None:
    """Reference weights, moved towards profiles that resemble the measured cells."""
    if mix == 0 or not observed.any():
        return None
    sampled = torch.randperm(len(features), device=features.device, generator=generator)[:max_cells]
    difference = features[sampled][:, None] - profiles[:, observed]
    log_weights = -0.5 * (difference.square() / noise_variance[observed]).sum(-1)
    matched = torch.softmax(log_weights, -1).mean(0)
    return (1 - mix) / len(profiles) + mix * matched


def _select_panel(
    profiles: torch.Tensor,
    observed: torch.Tensor,
    n_genes: int,
    noise_variance: torch.Tensor,
    max_candidates: int,
    target_genes: torch.Tensor,
    labels: torch.Tensor | None,
    label_weight: float,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Greedily reduce unexplained reference variance, starting from the observed panel."""
    if weights is None:
        weights = torch.full((len(profiles),), 1 / len(profiles), device=profiles.device, dtype=profiles.dtype)
    scale = weights.sqrt()[:, None]
    centered = (profiles - (weights[:, None] * profiles).sum(0)) * scale
    std = centered.square().sum(0).sqrt()
    targets = target_genes & (std > 1e-6)
    n_targets = int(targets.sum())
    if n_targets == 0:
        raise ValueError("No reference gene passes min_detection.")
    target_matrix = centered[:, targets] / std[targets]
    total_weight = n_targets
    if labels is not None and label_weight > 0:
        label_columns = (labels - (weights[:, None] * labels).sum(0)) * scale
        label_std = label_columns.square().sum(0).sqrt()
        label_columns = label_columns[:, label_std > 1e-6] / label_std[label_std > 1e-6]
        target_matrix = torch.cat(
            [target_matrix, label_columns * math.sqrt(label_weight * n_targets / label_columns.shape[1])], dim=1
        )
        total_weight = n_targets * (1 + label_weight)

    # Observed genes come first so the search can condition on them.
    n_candidates = min(max(max_candidates, n_genes), int((~observed).sum()))
    eligible = std.square().masked_fill(observed, -math.inf).topk(n_candidates).indices
    observed_indices = observed.nonzero(as_tuple=True)[0]
    candidates = torch.cat([observed_indices, eligible])
    n_observed = len(observed_indices)
    predictors = centered[:, candidates]
    cross = predictors.T @ target_matrix
    gram = predictors.T @ predictors
    noise = noise_variance[candidates]

    def condition_on(position: int) -> None:
        nonlocal cross, gram
        factor = gram[:, position] / (gram[position, position].clamp_min(0) + noise[position])
        cross = cross - factor[:, None] * cross[position]
        gram = gram - factor[:, None] * gram[position]

    for position in range(n_observed):
        condition_on(position)

    available = torch.ones(len(candidates), dtype=torch.bool, device=profiles.device)
    available[:n_observed] = False
    initial_scores = torch.full(
        (profiles.shape[1],), math.nan, device=profiles.device, dtype=profiles.dtype
    )
    order, gains = [], []
    for step in range(n_genes):
        scores = cross.square().sum(1) / (gram.diagonal().clamp_min(0) + noise) / total_weight
        if step == 0:
            initial_scores[candidates[available]] = scores[available]
        scores[~available] = -math.inf
        position = int(scores.argmax())
        order.append(candidates[position])
        gains.append(scores[position])
        available[position] = False
        condition_on(position)
    return torch.stack(order), torch.stack(gains), initial_scores


def _ridge_predict(
    profiles: torch.Tensor,
    features: torch.Tensor,
    observed: torch.Tensor,
    ridge_penalty: float,
) -> torch.Tensor:
    gene_means = profiles.mean(0)
    observed_means = gene_means[observed]
    predictors = profiles[:, observed] - observed_means
    gram = predictors.T @ predictors
    gram.diagonal().add_(ridge_penalty)
    coefficients = torch.linalg.solve(gram, predictors.T @ (profiles - gene_means))
    return (features - observed_means) @ coefficients + gene_means


def _complete_observed_mask(observed_mask: torch.Tensor, n_cells: int) -> torch.Tensor:
    return observed_mask.expand(n_cells, -1) if observed_mask.ndim == 1 else observed_mask


@torch.no_grad()
def ridge_impute(
    sc_gene_log_counts: list[torch.Tensor],
    data: torch.Tensor,
    observed_mask: torch.Tensor,
    locations: torch.Tensor,
    *,
    ridge_penalty: float = 10.0,
    reference_locations: list[torch.Tensor] | None = None,
    spatial_neighbors: int = 1,
    cells_per_bin: int = 25,
) -> torch.Tensor:
    """Predict every gene from the measured genes; measured values stay as measured.

    observed_mask is (G,) when every cell measured the same genes, or (N, G) when
    panels differ between cells. Genes measured elsewhere are filled from nearby cells.
    """
    dtype = data.dtype
    _, profiles, _ = _prepare_references(
        sc_gene_log_counts, reference_locations, None, cells_per_bin, data.device, dtype
    )
    measured_cells = _complete_observed_mask(observed_mask.to(data.device), len(data))
    observed = measured_cells.any(0)
    if not observed.any():
        return profiles.mean(0).expand(len(data), -1).clone()
    features = _measured_features(
        data, measured_cells, observed, locations.to(data.device, dtype), spatial_neighbors
    )
    imputed = _ridge_predict(profiles, features, observed, ridge_penalty)
    imputed[:, observed] = features
    imputed[measured_cells] = data[measured_cells]
    return imputed


@torch.no_grad()
def recommend_genes(
    n_genes_per_round: int,
    gene_embeddings: torch.Tensor,
    gene_similarity: torch.Tensor,
    locations: torch.Tensor,
    gene_log_counts: torch.Tensor,
    sc_gene_log_counts: list[torch.Tensor],
    *,
    observed_mask: torch.Tensor | None = None,
    noise_std: float | torch.Tensor = 0.5,
    ridge_penalty: float = 10.0,
    max_candidates: int = 3000,
    max_reference_cells: int = 4000,
    min_detection: float = 0.02,
    reference_locations: list[torch.Tensor] | None = None,
    spatial_neighbors: int = 1,
    cells_per_bin: int = 25,
    reference_labels: list[torch.Tensor] | None = None,
    label_weight: float = 0.0,
    reweight_mix: float = 0.0,
    seed: int = 0,
) -> SelectionResult:
    """Recommend the next genes to measure and impute the unmeasured ones.

    observed_mask is a Boolean (G,) mask, or (N, G) when panels differ between cells.
    If omitted, unmeasured columns must be entirely NaN. Zero-filled unmeasured
    columns require an explicit mask.

    noise_std (scalar or (G,)) sets how much each measurement is trusted during
    selection. Candidates are the max_candidates most variable unmeasured genes.
    Target genes are those detected in at least min_detection of reference cells.

    Without reference_locations, selection uses at most max_reference_cells randomly
    sampled reference cells. Neighbourhood averages use every reference cell.
    reference_labels are integer cell-type ids, one tensor per reference.
    seed controls cell sampling.
    """
    if gene_log_counts.ndim != 2 or min(gene_log_counts.shape) == 0:
        raise ValueError("gene_log_counts must have nonempty shape (N, G).")
    n_cells, n_genes = gene_log_counts.shape
    if locations.ndim != 2 or locations.shape[0] != n_cells:
        raise ValueError("locations must have shape (N, D).")
    for name, value in {
        "n_genes_per_round": n_genes_per_round,
        "max_candidates": max_candidates,
        "max_reference_cells": max_reference_cells,
        "spatial_neighbors": spatial_neighbors,
        "cells_per_bin": cells_per_bin,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not ridge_penalty > 0 or not math.isfinite(ridge_penalty):
        raise ValueError("ridge_penalty must be positive and finite.")
    if not 0 <= min_detection <= 1:
        raise ValueError("min_detection must be between 0 and 1.")
    if label_weight < 0 or not 0 <= reweight_mix <= 1:
        raise ValueError("label_weight must be nonnegative and reweight_mix between 0 and 1.")

    device = gene_log_counts.device
    dtype = torch.float64 if gene_log_counts.dtype == torch.float64 else torch.float32
    data = gene_log_counts.to(dtype=dtype)
    coordinates = locations.to(device=device, dtype=dtype)

    if observed_mask is None:
        missing = torch.isnan(data)
        partial = missing.any(0) & ~missing.all(0)
        if partial.any():
            raise ValueError("Each gene must be measured in all cells or missing in all cells.")
        measured_cells = ~missing
    else:
        if observed_mask.dtype != torch.bool or observed_mask.shape not in ((n_genes,), (n_cells, n_genes)):
            raise ValueError("observed_mask must be a Boolean tensor of shape (G,) or (N, G).")
        measured_cells = _complete_observed_mask(observed_mask.to(device=device), n_cells)
    observed = measured_cells.any(0)
    if not torch.isfinite(data[measured_cells]).all():
        raise ValueError("Observed expression values must be finite.")
    if n_genes_per_round > int((~observed).sum()):
        raise ValueError(
            "Not enough unmeasured genes. For zero-filled inputs, supply observed_mask."
        )

    if not sc_gene_log_counts:
        raise ValueError("Provide at least one single-cell reference.")
    for reference in sc_gene_log_counts:
        if reference.ndim != 2 or reference.shape[0] == 0 or reference.shape[1] != n_genes:
            raise ValueError("Every reference must have nonempty shape (N_reference, G).")
        if not torch.isfinite(reference).all():
            raise ValueError("Reference expression must be finite.")
    for name, extras in {"reference_locations": reference_locations, "reference_labels": reference_labels}.items():
        if extras is not None and [len(e) for e in extras] != [len(r) for r in sc_gene_log_counts]:
            raise ValueError(f"{name} must have one entry per reference cell.")
    if reference_labels is not None and any((l < 0).any() for l in reference_labels):
        raise ValueError("reference_labels must be nonnegative integer ids.")
    if sum(len(r) for r in sc_gene_log_counts) < 2:
        raise ValueError("Provide at least two reference cells.")

    noise = torch.as_tensor(noise_std, device=device, dtype=dtype)
    if noise.ndim == 0:
        noise = noise.expand(n_genes)
    if noise.shape != (n_genes,) or not torch.isfinite(noise).all() or (noise <= 0).any():
        raise ValueError("noise_std must be positive and finite: scalar or shape (G,).")
    noise_variance = noise.square()
    if reference_locations is not None:
        noise_variance = noise_variance / cells_per_bin

    cells, profiles, labels = _prepare_references(
        sc_gene_log_counts, reference_locations, reference_labels, cells_per_bin, device, dtype
    )
    target_genes = (cells > 0).to(dtype).mean(0) >= min_detection
    generator = torch.Generator(device=device).manual_seed(seed)
    selection_profiles, selection_labels = profiles, labels
    if reference_locations is None and len(profiles) > max_reference_cells:
        indices = torch.randperm(len(profiles), device=device, generator=generator)[:max_reference_cells]
        selection_profiles = profiles[indices]
        selection_labels = None if labels is None else labels[indices]

    features = None
    weights = None
    if observed.any():
        features = _measured_features(data, measured_cells, observed, coordinates, spatial_neighbors)
        weights = _match_weights(
            selection_profiles, features, observed, noise_variance, reweight_mix, generator
        )

    selection_order, marginal_gains, initial_scores = _select_panel(
        selection_profiles, observed, n_genes_per_round, noise_variance, max_candidates,
        target_genes, selection_labels, label_weight, weights,
    )
    if features is None:
        imputed = profiles.mean(0).expand(n_cells, -1).clone()
    else:
        imputed = _ridge_predict(profiles, features, observed, ridge_penalty)
        imputed[:, observed] = features
        imputed[measured_cells] = data[measured_cells]
    choices = torch.zeros(n_genes, device=device, dtype=torch.bool)
    choices[selection_order] = True
    return SelectionResult(choices, imputed, initial_scores, selection_order, marginal_gains)


def ridge_method(
    n_genes_per_round: int,
    gene_embeddings: torch.Tensor,
    gene_similarity: torch.Tensor,
    locations: torch.Tensor,
    gene_log_counts: torch.Tensor,
    sc_gene_log_counts: list[torch.Tensor],
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adapter matching mouse_embryo.random_method: returns (gene indices, imputation).

    Like random_method, it treats all-zero columns as unmeasured unless observed_mask is given.
    """
    kwargs.setdefault("observed_mask", gene_log_counts.ne(0).any(0))
    result = recommend_genes(
        n_genes_per_round, gene_embeddings, gene_similarity, locations,
        gene_log_counts, sc_gene_log_counts, **kwargs,
    )
    return result.selection_order, result.imputed
