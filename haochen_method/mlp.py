"""Supervised reference-only regression from measured genes to missing genes."""

import numpy as np
from scipy import sparse
import torch
from torch import nn

from ._validation import _gene_indices


def _matrix(value, name):
    value = value.tocsr() if sparse.issparse(value) else np.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    data = value.data if sparse.issparse(value) else value
    if not np.isfinite(data).all():
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _dense(value):
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


class MLPImputer:
    """Fit on reference cells, then predict ST using only its measured columns.

    ``num_layers`` counts hidden Linear/ReLU layers (the output layer is extra).
    ``hidden_dim`` is one width repeated across layers, or a sequence containing
    one width per hidden layer. Zero hidden layers gives linear regression.
    Inputs and targets are standardized using SC statistics only. Training and
    prediction densify one cell batch at a time. Every acquisition round needs
    a new fit because its measured input panel has changed.
    """

    def __init__(
        self, num_layers=2, hidden_dim=128, epochs=100, batch_size=256,
        learning_rate=1e-3, weight_decay=0.0, seed=42, device="cpu",
    ):
        if not isinstance(num_layers, (int, np.integer)) or num_layers < 0:
            raise ValueError("num_layers must be a nonnegative integer.")
        if isinstance(hidden_dim, (int, np.integer)):
            if hidden_dim < 1:
                raise ValueError("hidden_dim must be positive.")
            widths = (int(hidden_dim),) * num_layers
        else:
            widths = tuple(hidden_dim)
            if len(widths) != num_layers or any(
                not isinstance(w, (int, np.integer)) or w < 1 for w in widths
            ):
                raise ValueError("hidden_dim must contain one positive width per hidden layer.")
        for name, value in [("epochs", epochs), ("batch_size", batch_size)]:
            if not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative.")
        self.hidden_dims = widths
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = torch.device(device)

    def fit(self, reference, sequenced_genes):
        """Train only on reference expression; no spatial targets are accepted."""
        reference = _matrix(reference, "reference")
        if reference.shape[0] < 2:
            raise ValueError("MLP requires at least two reference cells.")
        genes = _gene_indices(sequenced_genes, reference.shape[1])
        if genes.size == 0:
            raise ValueError("MLP requires at least one measured gene.")
        self.sequenced_genes_ = genes.copy()
        self.n_genes_ = reference.shape[1]
        self.missing_genes_ = np.setdiff1d(np.arange(self.n_genes_), genes)
        self.loss_history_ = []
        self.model_ = None
        if self.missing_genes_.size == 0:
            return self

        mean = np.asarray(reference.mean(axis=0)).ravel()
        second = reference.multiply(reference) if sparse.issparse(reference) else reference ** 2
        variance = np.asarray(second.mean(axis=0)).ravel() - mean ** 2
        std = np.sqrt(np.maximum(variance, 0))
        std[std < 1e-6] = 1.0
        self.input_mean_ = mean[genes].astype(np.float32)
        self.input_std_ = std[genes].astype(np.float32)
        self.target_mean_ = mean[self.missing_genes_].astype(np.float32)
        self.target_std_ = std[self.missing_genes_].astype(np.float32)

        # Local RNGs keep fitting reproducible without changing acquisition RNGs.
        cuda_devices = ([self.device.index if self.device.index is not None else torch.cuda.current_device()]
                        if self.device.type == "cuda" else [])
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            layers = []
            in_features = genes.size
            for width in self.hidden_dims:
                layers.extend([nn.Linear(in_features, width), nn.ReLU()])
                in_features = width
            layers.append(nn.Linear(in_features, self.missing_genes_.size))
            self.model_ = nn.Sequential(*layers).to(self.device)

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        rng = np.random.default_rng(self.seed)
        self.model_.train()
        for _ in range(self.epochs):
            order = rng.permutation(reference.shape[0])
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = reference[order[start:start + self.batch_size]]
                inputs = (_dense(batch[:, genes]).astype(np.float32) - self.input_mean_) / self.input_std_
                targets = (_dense(batch[:, self.missing_genes_]).astype(np.float32) - self.target_mean_) / self.target_std_
                inputs = torch.as_tensor(inputs, device=self.device)
                targets = torch.as_tensor(targets, device=self.device)
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model_(inputs), targets)
                if not torch.isfinite(loss):
                    raise ValueError("MLP training produced a nonfinite loss.")
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.shape[0]
            self.loss_history_.append(epoch_loss / reference.shape[0])
        self.model_.eval()
        return self

    def predict(self, sequenced_expression):
        """Return dense N x G expression, copying measured values exactly."""
        if not hasattr(self, "n_genes_"):
            raise RuntimeError("Call fit before predict.")
        measured = _matrix(sequenced_expression, "sequenced_expression")
        if measured.shape[1] != self.sequenced_genes_.size:
            raise ValueError("sequenced_expression columns must match the fitted panel order.")
        result = np.empty((measured.shape[0], self.n_genes_), dtype=float)
        with torch.no_grad():
            for start in range(0, measured.shape[0], self.batch_size):
                stop = min(start + self.batch_size, measured.shape[0])
                batch = _dense(measured[start:stop])
                if self.missing_genes_.size:
                    inputs = (batch.astype(np.float32) - self.input_mean_) / self.input_std_
                    predicted = self.model_(torch.as_tensor(inputs, device=self.device)).cpu().numpy()
                    result[start:stop, self.missing_genes_] = predicted * self.target_std_ + self.target_mean_
                result[start:stop, self.sequenced_genes_] = batch
        return result


def mlp_imputation(reference, sequenced_expression, sequenced_genes, **kwargs):
    """Fit a fresh MLP and return its predictions; kwargs configure MLPImputer."""
    return MLPImputer(**kwargs).fit(reference, sequenced_genes).predict(sequenced_expression)
