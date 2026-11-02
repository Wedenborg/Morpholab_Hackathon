import argparse
import pathlib
import sys

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from methods import EDDIMethod, LinearImputationMethod, PPCAMethod

# Benchmark and data helpers live in eval_team/ (see eval_team/README.md):
#   load_e75_inputs, measure, evaluate_reconstruction, run_and_evaluate, ...
from eval_team import run_and_evaluate
from preprocessing.dataset_selection import select_dataset


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


def scanpy_hvg(
    n_genes_per_round: int,
    gene_embeddings: torch.Tensor,
    gene_similarity: torch.Tensor,
    locations: torch.Tensor,
    gene_log_counts: torch.Tensor,
    sc_gene_log_counts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Method using scanpy's highly variable genes method to select the genes.

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

    # Find the one hot encoding of what genes are already selected
    selected_genes = gene_log_counts.sum(dim=0).bool()  # (G,)

    # for now, just do the ranks based on the first data set
    # convert back to counts with expm1
    adata = ad.AnnData(np.expm1(sc_gene_log_counts[0].numpy()).astype("int"))
    # annotates in var 'highly variable' genes
    # ranks them in 'highly_variable_rank'
    # sc.pp.filter_cells(adata, min_genes=3) # doesn't remove any, could up this though
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", filter_unexpressed_genes=True)
    ranks = adata.var["highly_variable_rank"].to_numpy()

    # remove any ranks where we have already selected the gene
    new_ranks = np.where(selected_genes, np.full(len(selected_genes), np.nan), ranks)

    top_ranks = np.argpartition(new_ranks, 4)[:4]
    one_hot_choices = torch.zeros_like(selected_genes).bool()
    one_hot_choices[top_ranks] = True

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

    return torch.tensor(top_ranks), imputed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="run the selected method on the real E7.5 embryos and score it with the eval-team benchmark",
    )
    parser.add_argument("--method", choices=["eddi", "ppca", "linear", "scanpy", "random"], default="eddi",
                        help="selection/reconstruction method (default: eddi)")
    parser.add_argument("--linear-selection", choices=["ppca", "eddi", "random"], default="ppca",
                        help="gene selector used by --method linear (default: ppca)")
    parser.add_argument("--linear-alpha", type=float, default=1.0)
    parser.add_argument("--linear-train-cells", type=int, default=8000)
    parser.add_argument("--linear-imputation-cells", type=int, default=2048)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eddi-epochs", type=int, default=30)
    parser.add_argument("--eddi-train-cells", type=int, default=8000)
    parser.add_argument("--eddi-acquisition-cells", type=int, default=128)
    parser.add_argument("--eddi-imputation-cells", type=int, default=512)
    parser.add_argument("--eddi-mask-min", type=int, default=1)
    parser.add_argument("--eddi-mask-max", type=int, default=16)
    parser.add_argument("--eddi-spatial-k", type=int, default=6)
    parser.add_argument("--eddi-spatial-alpha", type=float, default=0.4)
    parser.add_argument("--eddi-expression-alpha", type=float, default=0.5)
    parser.add_argument("--eddi-detection-weight", type=float, default=1.0)
    parser.add_argument("--eddi-expression-weight", type=float, default=1.0)
    parser.add_argument("--eddi-observed-weight", type=float, default=0.1)
    parser.add_argument("--eddi-no-output-gate", action="store_false", dest="eddi_output_gate")
    parser.add_argument("--eddi-no-linear-skip", action="store_false", dest="eddi_linear_skip")
    parser.add_argument("--eddi-min-detect", type=float, default=0.002)
    parser.add_argument("--ppca-latent-dim", type=int, default=12)
    parser.add_argument("--ppca-train-cells", type=int, default=8000)
    parser.add_argument("--ppca-min-detect", type=float, default=0.005)
    parser.add_argument("--ppca-noise", choices=["gene", "scalar"], default="gene")
    parser.add_argument("--ppca-imputation-cells", type=int, default=512)
    parser.add_argument("--verbose", action="store_true", help="show progress")
    args = parser.parse_args()

    def make_selector(selection):
        if selection == "eddi":
            return EDDIMethod(
                seed=args.seed,
                epochs=args.eddi_epochs,
                max_train_cells=args.eddi_train_cells,
                max_acquisition_cells=args.eddi_acquisition_cells,
                max_imputation_cells=args.eddi_imputation_cells,
                mask_min_genes=args.eddi_mask_min,
                mask_max_genes=args.eddi_mask_max,
                spatial_k=args.eddi_spatial_k,
                spatial_alpha=args.eddi_spatial_alpha,
                expression_spatial_alpha=args.eddi_expression_alpha,
                detection_weight=args.eddi_detection_weight,
                expression_weight=args.eddi_expression_weight,
                observed_expression_weight=args.eddi_observed_weight,
                output_gate=args.eddi_output_gate,
                linear_skip=args.eddi_linear_skip,
                min_detect=args.eddi_min_detect,
                verbose=args.verbose,
            )
        if selection == "ppca":
            return PPCAMethod(
                seed=args.seed,
                latent_dim=args.ppca_latent_dim,
                max_train_cells=args.ppca_train_cells,
                min_detect=args.ppca_min_detect,
                noise=args.ppca_noise,
                max_imputation_cells=args.ppca_imputation_cells,
                verbose=args.verbose,
            )
        if selection == "scanpy":
            return scanpy_hvg
        return random_method

    def make_method():
        if args.method == "linear":
            return LinearImputationMethod(
                selector=make_selector(args.linear_selection),
                alpha=args.linear_alpha,
                max_train_cells=args.linear_train_cells,
                max_impute_cells=args.linear_imputation_cells,
                seed=args.seed,
                verbose=args.verbose,
            )
        return make_selector(args.method)

    if args.evaluate:
        summary = run_and_evaluate(
            make_method(), n_rounds=args.rounds, seed=args.seed, verbose=args.verbose
        )
        keys = ["n_experiments", "n_genes_measured", "peng_spearman", "atlas_spearman",
                "celltype_tv", "celltype_balanced_accuracy", "heldout_corr"]
        for k in keys:
            if k in summary:
                print(f"{k:28s} {summary[k]}")
        for k in ["peng_spearman", "atlas_spearman", "celltype_tv", "celltype_balanced_accuracy"]:
            if k + "_true_data" in summary:
                print(f"{k + ' (true data)':28s} {summary[k + '_true_data']}")
        sys.exit(0)

    folder = pathlib.Path("../GSE278603_RAW")

    n_genes_per_round = 4
    d = 128

    adata = ad.io.read_h5ad(sorted(folder.glob("*.h5ad"))[0])
    gene_names = list(adata.var_names)
    locations = get_locations(adata)
    N = len(locations)

    dataset = select_dataset(
        adata,
        genes="random",
        reference_genes=gene_names,
        d=d,
        top_n_var_genes=None,
        n_random_genes=4,
        seed=42,
        verbose=args.verbose,
    )  # This is assuming we do not change dataset each iteration. Currently no searching only uses TOME dataset at tau = 7.5
    sc_gene_embeddings = torch.tensor(dataset["loadings"].to_numpy(), dtype=torch.float32)
    sc_gene_similarity = torch.tensor(dataset["gene_similarity"].to_numpy(), dtype=torch.float32)
    sc_gene_log_counts = [torch.tensor(dataset["gene_log_counts"].to_numpy(), dtype=torch.float32)]
    gene_names = dataset["common_gene_names"]
    G = len(gene_names)
    gene_log_counts = torch.zeros(N, G)
    method = make_method()
    for _ in range(args.rounds):
        choice, gene_log_counts = method(
            n_genes_per_round,
            sc_gene_embeddings,
            sc_gene_similarity,
            locations,
            gene_log_counts,
            sc_gene_log_counts,
        )

        nonzero_genes = (gene_log_counts != 0).any(dim=0).nonzero(as_tuple=True)[0]
        print("Nonzero genes:", [gene_names[i] for i in nonzero_genes.tolist()])
        print(gene_log_counts)
