import argparse
import pathlib
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

REPO = pathlib.Path(__file__).resolve().parent
DATA_DIR = REPO.parent / "GSE278603_RAW"
EVAL_DIR = REPO / "eval-team"


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


# --------------------------------------------------------------------------------------
# Real data and evaluation (benchmark code lives in eval-team/, see eval-team/README.md)
# --------------------------------------------------------------------------------------
def _eval_modules():
    """Import the benchmark modules from eval-team/."""
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import evaluate
    import submission

    return evaluate, submission


def load_embryo(rep: int, stage: str = "E7.5", data_dir: pathlib.Path = DATA_DIR) -> ad.AnnData:
    """Load one Stereo-seq embryo, e.g. load_embryo(2) is E7.5 rep2."""
    return ad.read_h5ad(next(pathlib.Path(data_dir).glob(f"*_Embryo_{stage}_stereo_rep{rep}.h5ad")))


def load_e75_inputs(hidden_rep: int = 2, visible_rep: int = 1, data_dir: pathlib.Path = DATA_DIR) -> dict:
    """Real E7.5 inputs in the shapes random_method expects.

    The hidden embryo is the one to reconstruct; the method may only touch it through
    `oracle` (positions + measurements of at most 4 genes per experiment). The visible
    embryo is free training data, given as the single-cell reference.

    Returns a dict with
        hidden               AnnData of the hidden embryo (for measuring and evaluating only)
        oracle               measurement interface for the hidden embryo
        genes                gene names, the order of the G axis everywhere
        locations            (N, 3) tensor of x, y, z
        gene_log_counts      (N, G) zeros, to be filled by measure()
        sc_gene_log_counts   [ (N', G) visible embryo, genes aligned to `genes` ]
    """
    _, submission = _eval_modules()
    hidden = load_embryo(hidden_rep, data_dir=data_dir)
    visible = load_embryo(visible_rep, data_dir=data_dir)
    genes = pd.Index(hidden.var_names)
    cols = pd.Index(visible.var_names).get_indexer(genes)
    vis = np.zeros((visible.n_obs, len(genes)), dtype=np.float32)
    vis[:, cols >= 0] = sp.csr_matrix(visible.X)[:, cols[cols >= 0]].toarray()
    return dict(
        hidden=hidden,
        oracle=submission.MeasurementOracle(hidden),
        genes=genes,
        locations=get_locations(hidden),
        gene_log_counts=torch.zeros(hidden.n_obs, len(genes)),
        sc_gene_log_counts=[torch.from_numpy(vis)],
    )


def measure(oracle, genes: pd.Index, gene_idx, gene_log_counts: torch.Tensor) -> torch.Tensor:
    """Measure the chosen genes on every section of the hidden embryo and write the true
    values into a copy of gene_log_counts. Each section is one experiment of at most 4
    genes, so measuring k genes everywhere costs ceil(k / 4) * n_sections experiments,
    all recorded in oracle.log()."""
    out = gene_log_counts.clone()
    idx = [int(i) for i in torch.as_tensor(gene_idx).flatten()]
    row = {c: i for i, c in enumerate(oracle.cells.index)}
    for start in range(0, len(idx), oracle.k_max):
        chunk = idx[start:start + oracle.k_max]
        for z in oracle.sections:
            values = oracle.measure([genes[i] for i in chunk], section=z)
            rows = torch.tensor([row[c] for c in values.index])
            out[rows[:, None], torch.tensor(chunk)[None, :]] = torch.tensor(values.to_numpy(), dtype=out.dtype)
    return out


def evaluate_reconstruction(imputed, hidden: ad.AnnData, log: pd.DataFrame | None = None, evaluator=None):
    """Score a reconstructed (N, G) tensor of the hidden embryo with the eval-team benchmark.

    imputed: torch tensor or numpy array, rows in hidden.obs_names order, columns in
        hidden.var_names order, values on the log1p scale of the data.
    log: the oracle's experiment log, if any. Without it the held-out gene score also
        counts measured values; benchmarks 1 and 3 do not use it.
    evaluator: reuse an evaluate.Evaluator(hidden) across calls to save ~2 s each time.

    Returns (summary dict, dict of per-gene tables). The keys for benchmark 1 are
    peng_spearman and atlas_spearman, for benchmark 3 celltype_tv and
    celltype_balanced_accuracy; see eval-team/README.md.
    """
    evaluate, _ = _eval_modules()
    X = imputed.detach().cpu().numpy() if hasattr(imputed, "detach") else np.asarray(imputed)
    if X.shape != hidden.shape:
        raise ValueError(f"reconstruction has shape {X.shape}, hidden embryo is {hidden.shape}")
    recon = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=hidden.obs_names),
                       var=pd.DataFrame(index=hidden.var_names))
    evaluator = evaluator or evaluate.Evaluator(hidden)
    return evaluator.evaluate(recon, log)


def run_and_evaluate(method=None, n_rounds: int = 3, n_genes_per_round: int = 4, seed: int = 0) -> dict:
    """Run a selection method on the real E7.5 data for n_rounds and score the result.

    Each round the method picks genes and returns its imputation; the picked genes are then
    measured on every section. The final reconstruction is the last imputation with all
    measured values filled in."""
    method = method or random_method
    torch.manual_seed(seed)
    inp = load_e75_inputs()
    G = len(inp["genes"])
    gene_embeddings = torch.zeros(G, 1)        # placeholders: random_method only uses len()
    gene_similarity = torch.zeros(0)
    measured = inp["gene_log_counts"]
    measured_cols = torch.zeros(G, dtype=torch.bool)
    imputed = measured
    for r in range(n_rounds):
        choices, imputed = method(n_genes_per_round, gene_embeddings, gene_similarity,
                                  inp["locations"], measured, inp["sc_gene_log_counts"])
        measured = measure(inp["oracle"], inp["genes"], choices, measured)
        measured_cols[torch.as_tensor(choices).flatten()] = True
        print(f"round {r}: measured {[inp['genes'][int(i)] for i in torch.as_tensor(choices).flatten()]}")
    recon = torch.where(measured_cols, measured, imputed)
    log = inp["oracle"].log()
    summary, _ = evaluate_reconstruction(recon, inp["hidden"], log)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true",
                        help="run random_method on the real E7.5 embryos and score it with the eval-team benchmark")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.evaluate:
        summary = run_and_evaluate(n_rounds=args.rounds)
        keys = ["n_experiments", "n_genes_measured", "peng_spearman", "atlas_spearman",
                "celltype_tv", "celltype_balanced_accuracy", "heldout_corr"]
        for k in keys:
            print(f"{k:28s} {summary[k]}")
        for k in ["peng_spearman", "atlas_spearman", "celltype_tv", "celltype_balanced_accuracy"]:
            print(f"{k + ' (true data)':28s} {summary[k + '_true_data']}")
        sys.exit(0)

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
