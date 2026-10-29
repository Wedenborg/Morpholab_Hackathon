"""Glue between a selection method (see mouse_embryo.py) and the benchmark.

load_embryo           read one Stereo-seq embryo
load_e75_inputs       real E7.5 inputs in the tensor shapes the methods use
measure               measure chosen genes on every section of the hidden embryo (logged)
evaluate_reconstruction  score an (N, G) tensor of the hidden embryo
run_and_evaluate      run a method for a few rounds, measure its picks, score the result

torch is imported only inside the functions that build or fill tensors, so the rest of
the package works without it.
"""
import pathlib

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .paths import DATA_DIR
from .submission import MeasurementOracle


def load_embryo(rep: int, stage: str = "E7.5", data_dir: pathlib.Path = DATA_DIR) -> ad.AnnData:
    """Load one Stereo-seq embryo, e.g. load_embryo(2) is E7.5 rep2."""
    return ad.read_h5ad(next(pathlib.Path(data_dir).glob(f"*_Embryo_{stage}_stereo_rep{rep}.h5ad")))


def load_e75_inputs(hidden_rep: int = 2, visible_rep: int = 1, data_dir: pathlib.Path = DATA_DIR) -> dict:
    """Real E7.5 inputs in the shapes random_method expects.

    The hidden embryo is the one to reconstruct; a method may only touch it through
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
    import torch

    hidden = load_embryo(hidden_rep, data_dir=data_dir)
    visible = load_embryo(visible_rep, data_dir=data_dir)
    genes = pd.Index(hidden.var_names)
    cols = pd.Index(visible.var_names).get_indexer(genes)
    vis = np.zeros((visible.n_obs, len(genes)), dtype=np.float32)
    vis[:, cols >= 0] = sp.csr_matrix(visible.X)[:, cols[cols >= 0]].toarray()
    return dict(
        hidden=hidden,
        oracle=MeasurementOracle(hidden),
        genes=genes,
        locations=torch.tensor(hidden.obs[["x", "y", "z"]].to_numpy(), dtype=torch.float32),
        gene_log_counts=torch.zeros(hidden.n_obs, len(genes)),
        sc_gene_log_counts=[torch.from_numpy(vis)],
    )


def measure(oracle, genes: pd.Index, gene_idx, gene_log_counts):
    """Measure the chosen genes on every section of the hidden embryo and write the true
    values into a copy of gene_log_counts (an (N, G) torch tensor). Each section is one
    experiment of at most 4 genes, so measuring k genes everywhere costs
    ceil(k / 4) * n_sections experiments, all recorded in oracle.log()."""
    import torch

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
    """Score a reconstructed (N, G) tensor of the hidden embryo.

    imputed: torch tensor or numpy array, rows in hidden.obs_names order, columns in
        hidden.var_names order, values on the log1p scale of the data.
    log: the oracle's experiment log, if any. Without it the held-out gene score also
        counts measured values; benchmarks 1 and 3 do not use it.
    evaluator: reuse an Evaluator(hidden) across calls to save ~2 s each time.

    Returns (summary dict, dict of per-gene tables). Benchmark 1 keys: peng_spearman,
    atlas_spearman; benchmark 3: celltype_tv, celltype_balanced_accuracy. See README.md.
    """
    from .evaluate import Evaluator

    X = imputed.detach().cpu().numpy() if hasattr(imputed, "detach") else np.asarray(imputed)
    if X.shape != hidden.shape:
        raise ValueError(f"reconstruction has shape {X.shape}, hidden embryo is {hidden.shape}")
    recon = ad.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=hidden.obs_names),
                       var=pd.DataFrame(index=hidden.var_names))
    evaluator = evaluator or Evaluator(hidden)
    return evaluator.evaluate(recon, log)


def run_and_evaluate(method, n_rounds: int = 3, n_genes_per_round: int = 4, seed: int = 0,
                     verbose: bool = True) -> dict:
    """Run a selection method on the real E7.5 data for n_rounds and score the result.

    `method` has the signature of mouse_embryo.random_method:
        method(n_genes_per_round, gene_embeddings, gene_similarity, locations,
               gene_log_counts, sc_gene_log_counts) -> (chosen gene indices, (N, G) imputation)
    Each round the method picks genes and returns its imputation; the picked genes are then
    measured on every section. The final reconstruction is the last imputation with all
    measured values filled in. Returns the summary dict."""
    import torch

    torch.manual_seed(seed)
    inp = load_e75_inputs()
    G = len(inp["genes"])
    gene_embeddings = torch.zeros(G, 1)        # placeholders until real embeddings exist
    gene_similarity = torch.zeros(0)
    measured = inp["gene_log_counts"]
    measured_cols = torch.zeros(G, dtype=torch.bool)
    imputed = measured
    for r in range(n_rounds):
        choices, imputed = method(n_genes_per_round, gene_embeddings, gene_similarity,
                                  inp["locations"], measured, inp["sc_gene_log_counts"])
        measured = measure(inp["oracle"], inp["genes"], choices, measured)
        measured_cols[torch.as_tensor(choices).flatten()] = True
        if verbose:
            print(f"round {r}: measured {[inp['genes'][int(i)] for i in torch.as_tensor(choices).flatten()]}")
    recon = torch.where(measured_cols, measured, imputed)
    summary, _ = evaluate_reconstruction(recon, inp["hidden"], inp["oracle"].log())
    return summary
