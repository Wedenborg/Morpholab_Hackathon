import argparse
import pathlib
from collections.abc import Callable
from typing import Self

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset


class MouseEmbryoDataset(Dataset):

    data: torch.Tensor
    n_samples: int
    one_hot_choices: torch.Tensor

    def __init__(
        self: Self,
        fname: pathlib.Path,
        min_group_size: int,
        max_group_size: int,
        n_cells: int,
        n_samples: int,
    ) -> None:
        self.n_samples = n_samples
        adata = ad.io.read_h5ad(fname)
        # convert back to counts with expm1
        adata = ad.AnnData(np.expm1(adata.X.toarray()[:n_cells]).astype("int"))  # convert to counts
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", filter_unexpressed_genes=True)
        # there are only 2000 highly variable ones
        self.data = torch.Tensor(adata.X[:, adata.var["highly_variable"].to_numpy()])

        self.data = torch.Tensor(adata.X)  # works for scipy sparse matrix
        _, n_genes = self.data.shape
        group_sizes = torch.randint(min_group_size, max_group_size, (n_samples,))

        one_hot_choices_ls = []
        for group_size in group_sizes:
            one_hot_choices = torch.zeros(n_genes).bool()
            choices = torch.randperm(n_genes)[:group_size]
            one_hot_choices[choices] = True
            one_hot_choices_ls.append(one_hot_choices)

        self.one_hot_choices = torch.stack(one_hot_choices_ls)

    def __len__(self: Self) -> int:
        return self.n_samples

    def __getitem__(self: Self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        choices_full = torch.tensor(
            np.full(self.data.shape, self.one_hot_choices[idx].numpy()[None])
        )

        chosen_data = torch.where(choices_full, self.data, 0)
        return chosen_data.ravel(), self.data.ravel()


def get_data(
    root: pathlib.Path,
    n_genes_per_round: int,
    n_rounds: int,
    n_cells: int,
    n_train: int,
    batch_size: int,
) -> tuple[int, DataLoader[Dataset]]:
    train_dataset = MouseEmbryoDataset(
        min(root.glob("*.h5ad")), n_genes_per_round, n_genes_per_round * n_rounds, n_cells, n_train
    )

    _, n_genes = train_dataset.data.shape

    train_dl = DataLoader(train_dataset, batch_size, shuffle=True)

    return n_cells * n_genes, train_dl


class SimpleMLP(torch.nn.Sequential):

    def __init__(
        self: Self,
        n_in: int,
        n_out: int,
        n_layers: int,
        hidden_width: int,
        activation: Callable = torch.nn.GELU,
    ) -> None:

        widths = [n_in] + [hidden_width] * n_layers + [n_out]
        layers = []
        for i in range(len(widths) - 2):
            layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
            layers.append(activation())

        layers.append(torch.nn.Linear(widths[-2], widths[-1]))

        super().__init__(*layers)


def train_one_epoch(train_dl: DataLoader, epoch_index: int, loss_fn: Callable, optimizer):
    running_loss = 0.0
    last_loss = 0.0

    for data in iter(train_dl):
        # Every data instance is an input + label pair
        x, y = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(x)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        print(running_loss)

    return last_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run random_method on the real E7.5 embryos and score it with the eval-team benchmark",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--verbose", action="store_true", help="show dataset selection progress")
    args = parser.parse_args()

    folder = pathlib.Path("../GSE278603_RAW")

    n_genes_per_round = 4
    hidden_dim = 128
    n_train = 10
    n_rounds = 3
    batch_size = 16
    n_cells = 5000

    n_in, train_dl = get_data(folder, n_genes_per_round, n_rounds, n_cells, n_train, batch_size)

    model = SimpleMLP(n_in, n_in, 3, hidden_dim)
    print(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    train_one_epoch(train_dl, 0, loss_fn, optimizer)
