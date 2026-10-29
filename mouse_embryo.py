import pathlib

import anndata as ad
import torch


def explore_data(data_dir: pathlib.Path) -> None:
    # data location
    for fname in data_dir.iterdir():
        adata = ad.io.read_h5ad(fname)
        # print(adata.X)  # a compressed sparse matrix (n_obs x n_vars)
        # In this case, the observations are cells and the vars are genes.
        # For example, the first one is 20,935 x 18,455
        # These are all pandas dataframes, which can be converted to numpy arrays with .to_numpy()
        # This is the counts data after applying the log1p transform.

        # The observation and variable names
        # print(adata.obs_names)
        # print(adata.var_names)

        # Arrays with 1 value for each observation
        # print(adata.obs["orig.ident"])
        # print(adata.obs["nCount_RNA"])
        # print(adata.obs["nFeature_RNA"])
        # print(adata.obs["x"])
        # print(adata.obs["y"])
        # print(adata.obs["cell_name"])
        # print(adata.obs["chip"])
        # print(adata.obs["id"])
        # print(adata.obs["nCount_SCT"])
        # print(adata.obs["nFeature_SCT"])
        # print(adata.obs["image_x"])
        # print(adata.obs["image_y"])
        # print(adata.obs["germ_layer"])
        # print(adata.obs["celltype"])
        # print(adata.obs["z"])
        # print(adata.obs["predicted.id"])

        # Arrays with 1 value for each variable
        # print(adata.var["in_scaled_data"])
        # print(adata.var["highly_variable"])

        # These appear to be numpy arrays already? Shape (n_obs,2)
        # print(adata.obsm["X_image"].shape)
        # print(adata.obsm["X_sagittal"].shape)
        # print(adata.obsm["X_spatial"].shape)

        # These appear to be all nan
        # print(adata.varm["image"].shape)
        # print(adata.varm["sagittal"].shape)
        # print(adata.varm["spatial"].shape)

        # Each of these is a dict with a single entry 'variance' with a (2,) array
        # print(adata.uns["image"])
        # print(adata.uns["sagittal"])
        # print(adata.uns["spatial"])

        # The original counts data, a compressed sparse matrix
        # print(adata.layers["counts"])

        print(adata)


def get_locations(adata: ad.AnnData) -> torch.Tensor:
    """
    The x, y, z coordinates of every cell.

    args:
        adata: one embryo, observations are cells

    returns:
        array of shape (N, 3), row i is cell i of adata
    """
    return torch.tensor(adata.obs[["x", "y", "z"]].to_numpy(), dtype=torch.float32)


def random_method(
    n_genes_per_round: int,
    gene_embeddings: torch.Tensor,
    gene_similarity: torch.Tensor,
    locations: torch.Tensor,
    gene_log_counts: torch.Tensor,
    sc_gene_log_counts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Our stupidest selection method, select genes randomly.

    args:
        gene_embeddings: shape (G,d)
        gene_similarity: shape (G,G)
        locations: shape (N,D)
        gene_log_counts: shape (N,G) Only has g genes filled out.
        sc_gene_log_counts: list of tensors of shape (N',G)

    returns:
        tuple of array one hot encoding (G,) of genes to select array of (N,G)
        imputation of gene log counts
    """
    n_genes = len(gene_embeddings)

    # Find the one hot encoding of what genes are already selected
    selected_genes = gene_log_counts.sum(dim=0).bool()  # (G,)

    # select n_genes_per_round of new genes
    one_hot_choices = torch.zeros(n_genes).bool()
    choices = torch.randperm(n_genes)[:n_genes_per_round]
    one_hot_choices[choices] = True

    # For the imputed data, get the mean gene expression counts for the single cell data
    # sum over number of sc datasets and N' number of cells
    mean_sc = torch.stack(sc_gene_log_counts, dim=0).float().mean(dim=(0, 1))

    # first select the genes that have already been selected, then the ones that we
    # selected in this round.
    imputed = torch.where(
        selected_genes,
        gene_log_counts,
        torch.where(one_hot_choices, mean_sc, torch.zeros_like(mean_sc)),
    )

    return choices, imputed


if __name__ == "__main__":
    folder = pathlib.Path("../GSE278603_RAW")

    n_genes_per_round = 4
    G = 10
    d = 128
    N_prime = 25
    n_sc_datasets = 3
    n_rounds = 2

    adata = ad.io.read_h5ad(sorted(folder.glob("*.h5ad"))[0])
    locations = get_locations(adata)
    N = len(locations)

    gene_embeddings = torch.ones(G, d)
    gene_similarity = gene_embeddings.T @ gene_embeddings
    gene_log_counts = torch.zeros(N, G)
    sc_gene_log_counts = [
        torch.arange(N_prime * G).reshape(N_prime, G) for _ in range(n_sc_datasets)
    ]

    for _ in range(n_rounds):
        choice, gene_log_counts = random_method(
            n_genes_per_round,
            gene_embeddings,
            gene_similarity,
            locations,
            gene_log_counts,
            sc_gene_log_counts,
        )

        print(gene_log_counts)
