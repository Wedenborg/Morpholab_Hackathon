"""Select gene panels on TOME E7.5 scRNA-seq through a VAE and test them two ways.

1. Reference: how well does each panel reconstruct scRNA-seq cells that took no part in
   training or selection? This is the objective the panels were chosen for.
2. Transfer: measure the same panels on a hidden Stereo-seq embryo through the logged
   MeasurementOracle. Every policy shares one ridge imputer fitted on the other Stereo-seq
   embryos, so differences come from the panels.

Policies share the candidate pool of evaluate_active_acquisition, restricted for the
scRNA-based ones to genes detected in TOME. Only the hidden embryo's gene names are read
before scoring.

Usage:
    python dataset/convert_seurat_rds.py dataset/data/tome/seurat_object_E7.5.rds
    python evaluate_vae_panel.py --hidden-rep 1 --out-dir data/vae_panel/rep1
"""
import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from active_acquisition import aligned_expression, build_gene_space, ridge_fit_predict, spatial_smoother
from dataset.load_dataset import load_dataset
from eval_team import Evaluator, load_embryo
from evaluate_active_acquisition import (
    SMOOTHING_NEIGHBORS,
    STAGES_OTHER_THAN_E75,
    greedy_policy,
    paired_gene_bootstrap,
    random_policy,
    run_policy,
    variance_policy,
)
from vae_panel import (
    ExpressionVAE,
    LatentDecoder,
    PanelScorer,
    decode_panel,
    log_normalize,
    pca_embedding,
    reconstruction_r2,
    select_panel,
    train_vae,
)

MIN_DETECTION = 0.02
PANEL_SIZES = (4, 8, 16, 32)


def load_tome_counts(stage: str = "7.5"):
    """TOME cells at one stage with gene symbols. Depth-normalised here because the file's
    logcounts slot holds raw counts and its cells are far deeper than Stereo-seq's."""
    return next(iter(load_dataset("tome", timepoint=stage).values()))


def split_cells(n_cells: int, seed: int, val_share: float = 0.1, test_share: float = 0.1):
    order = torch.randperm(n_cells, generator=torch.Generator().manual_seed(seed))
    n_val, n_test = int(n_cells * val_share), int(n_cells * test_share)
    return order[n_val + n_test:], order[:n_val], order[n_val:n_val + n_test]


def scrna_policy(scorer: PanelScorer, joint: bool = True, log: list | None = None):
    def choose(episode, k: int) -> torch.Tensor:
        picks, error = select_panel(scorer, episode.measured, k, joint=joint)
        if log is not None:
            log.append(error)
        return picks
    return choose


def celltype_means(values: torch.Tensor, labels: np.ndarray) -> torch.Tensor:
    return torch.stack([values[torch.from_numpy(labels == label)].mean(0) for label in np.unique(labels)])


def reference_scores(
    name: str,
    positions: list[int],
    predictors: torch.Tensor,
    targets: torch.Tensor,
    splits: tuple,
    pool_in_targets: torch.Tensor,
    eligible: torch.Tensor,
    labels: np.ndarray,
    decoders: dict[str, tuple[LatentDecoder, torch.Tensor]],
) -> list[dict]:
    """R² on the untouched test cells, over genes the panel did not measure."""
    train, _, test = splits
    baseline = targets[train].mean(0)
    rows = []
    for size in PANEL_SIZES:
        if size > len(positions):
            continue
        # A gene TOME does not detect tells these cells nothing, so it is dropped from the panel.
        panel = torch.tensor([position for position in positions[:size] if eligible[position]])
        observed = torch.zeros(targets.shape[1], dtype=torch.bool)
        observed[pool_in_targets[panel]] = True
        predictions = {"ridge": ridge_fit_predict(targets[train], targets[test][:, observed], observed)}
        for decoder_name, (decoder, latents) in decoders.items():
            predictions[decoder_name] = decode_panel(
                decoder, predictors[train][:, panel], latents, predictors[test][:, panel])
        for model, predicted in predictions.items():
            unmeasured = ~observed
            means_true = celltype_means(targets[test][:, unmeasured], labels[test.numpy()])
            means_predicted = celltype_means(predicted[:, unmeasured], labels[test.numpy()])
            rows.append(dict(
                policy=name, n_genes=size, reconstruction=model,
                cell_r2=reconstruction_r2(predicted[:, unmeasured], targets[test][:, unmeasured], baseline[unmeasured]),
                celltype_r2=reconstruction_r2(means_predicted, means_true, baseline[unmeasured].expand_as(means_true)),
            ))
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-rep", type=int, choices=(1, 2), default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--genes-per-round", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--vae-steps", type=int, default=6000)
    parser.add_argument("--vae-seeds", type=int, default=3)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    hidden = load_embryo(args.hidden_rep)
    train_embryos = [load_embryo(3 - args.hidden_rep)] + [load_embryo(rep, stage=stage) for stage, rep in STAGES_OTHER_THAN_E75]
    space, train_profiles = build_gene_space(train_embryos, hidden.var_names, args.n_candidates, seed=args.seed)
    del train_embryos
    pool_genes = space.candidate_genes

    tome = load_tome_counts()
    tome_counts = sp.csr_matrix(tome.X)
    labels = tome.obs["cell_type"].astype(str).to_numpy()
    detected = np.asarray((tome_counts > 0).mean(0)).ravel() >= MIN_DETECTION
    tome_genes = pd.Index(tome.var_names)[detected]
    genes = pd.Index(space.target_genes).intersection(tome_genes, sort=False)
    eligible = torch.from_numpy(pool_genes.isin(genes))
    print(f"{tome.n_obs} TOME cells, {len(space.target_genes)} Stereo-seq targets, {len(genes)} shared genes modelled, "
          f"{int(eligible.sum())}/{len(pool_genes)} candidates in TOME ({time.perf_counter() - started:.0f} s)", flush=True)

    normalized = log_normalize(tome_counts)[:, pd.Index(tome.var_names).get_indexer(genes)]
    log_values = torch.from_numpy(np.asarray(normalized.todense(), dtype=np.float32))
    splits = split_cells(len(log_values), args.seed)
    train, val, test = splits
    means, stds = log_values[train].mean(0), log_values[train].std(0).clamp_min(1e-3)
    targets = ((log_values - means) / stds).clamp(-10, 10)

    # Predictors are candidates on the pool's axis; candidates TOME lacks stay zero and ineligible.
    pool_in_targets = torch.from_numpy(genes.get_indexer(pool_genes))
    predictors = torch.zeros(len(targets), len(pool_genes))
    predictors[:, eligible] = targets[:, pool_in_targets[eligible]]

    pca_latents, pca_decoder = pca_embedding(targets[train], args.latent)
    fit_args = dict(fit_predictors=predictors[train], score_predictors=predictors[val], score_targets=targets[val],
                    eligible=eligible)
    pca_scorer = PanelScorer(pca_decoder, fit_latents=pca_latents, **fit_args)

    embedding_rows = [dict(
        model="pca", seed=args.seed,
        test_r2=reconstruction_r2(pca_decoder((targets[test] - targets[train].mean(0)) @ pca_decoder.weight),
                                  targets[test], targets[train].mean(0)),
    )]
    vaes, decoders = {}, {"pca_decode": (pca_decoder, pca_latents)}
    for seed in range(args.seed, args.seed + args.vae_seeds):
        path = out / f"vae_seed{seed}.pt"
        tick = time.perf_counter()
        if path.exists():
            print(f"Reusing saved VAE seed {seed}", flush=True)
            model = ExpressionVAE(targets.shape[1], args.latent)
            model.load_state_dict(torch.load(path))
            model = model.eval().requires_grad_(False)
        else:
            print(f"Training VAE seed {seed}", flush=True)
            model = train_vae(targets[train], targets[val], latent=args.latent, steps=args.vae_steps,
                              warmup=args.vae_steps // 4, seed=seed)
            torch.save(model.state_dict(), path)
        with torch.no_grad():
            latents = model.encode(targets[train])
            test_latents = model.encode(targets[test])
            test_r2 = reconstruction_r2(model.decode(test_latents), targets[test], targets[train].mean(0))
        active = int((latents.std(0) > 0.1).sum())
        embedding_rows.append(dict(model="vae", seed=seed, test_r2=test_r2, active_latents=active))
        print(f"  seed {seed}: held-out R² {test_r2:.4f}, {active} active latents ({time.perf_counter() - tick:.0f} s)", flush=True)
        vaes[seed] = (model, latents)
        if seed == args.seed:
            decoders["vae_decode"] = (model.latent_decoder(), latents)
    embeddings = pd.DataFrame(embedding_rows)
    embeddings.to_csv(out / "embedding_scores.csv", index=False)
    print(embeddings.round(4).to_string(index=False), flush=True)

    policies = [("random", seed, random_policy(seed)) for seed in range(args.random_seeds)]
    policies += [
        ("variance", 0, variance_policy(space)),
        ("greedy_ridge_stereo", 0, greedy_policy(space, torch.from_numpy(hidden.obs[["x", "y", "z"]].to_numpy(dtype=float)).float())),
        ("pca_joint", 0, scrna_policy(pca_scorer)),
        ("pca_greedy", 0, scrna_policy(pca_scorer, joint=False)),
    ]
    selection_errors: dict[str, list] = {}
    for seed, (model, latents) in vaes.items():
        scorer = PanelScorer(model.latent_decoder(), fit_latents=latents, **fit_args)
        selection_errors[f"vae_joint_{seed}"] = []
        selection_errors[f"vae_greedy_{seed}"] = []
        policies.append(("vae_joint", seed, scrna_policy(scorer, log=selection_errors[f"vae_joint_{seed}"])))
        policies.append(("vae_greedy", seed, scrna_policy(scorer, joint=False, log=selection_errors[f"vae_greedy_{seed}"])))

    # The stereo imputer and truth: the hidden embryo's values are only read for scoring.
    generator = torch.Generator().manual_seed(args.seed)
    pooled_log = torch.cat(train_profiles)
    del train_profiles
    ridge_profiles = pooled_log[torch.randperm(len(pooled_log), generator=generator)[:20000]].clone()
    del pooled_log
    locations = hidden.obs[["x", "y", "z"]].to_numpy(dtype=float)
    smoother = spatial_smoother(locations, SMOOTHING_NEIGHBORS)
    truth_log = np.asarray((smoother @ aligned_expression(hidden, space.target_genes)).todense(), dtype=np.float32)
    evaluator = Evaluator(hidden)
    imputers = {"ridge": (None, ridge_profiles)}
    score_rounds = {1, 2, args.rounds // 2, args.rounds}

    rows, finals, panels, reference_rows = [], {}, {}, []
    for name, seed, choose in policies:
        policy_rows, final, episode = run_policy(
            name, choose, seed, hidden, space, smoother, evaluator, imputers, truth_log,
            args.rounds, args.genes_per_round, score_rounds)
        rows.extend(policy_rows)
        panels[f"{name}_{seed}"] = episode.chosen
        finals[f"{name}_{seed}"] = final["ridge"]["per_gene"]
        episode.oracle.log().to_csv(out / f"experiments_{name}_{seed}.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "policy_scores.csv", index=False)
        (out / "panels.json").write_text(json.dumps(panels, indent=2))

        positions = pool_genes.get_indexer(episode.chosen).tolist()
        for row in reference_scores(name, positions, predictors, targets, splits, pool_in_targets, eligible, labels, decoders):
            reference_rows.append(dict(seed=seed, **row))
        pd.DataFrame(reference_rows).to_csv(out / "reference_scores.csv", index=False)

    (out / "selection_errors.json").write_text(json.dumps(selection_errors, indent=2))
    reference_table = pd.DataFrame(reference_rows).groupby(
        ["reconstruction", "policy", "n_genes"])[["cell_r2", "celltype_r2"]].mean().round(4)
    print("\nReconstructing held-out TOME cells\n" + reference_table.to_string(), flush=True)
    scores = pd.DataFrame(rows)
    table = scores.groupby(["policy", "n_genes"])[["heldout_corr", "neighbourhood_r2"]].agg(["mean", "std"]).round(4)
    print("\nHidden Stereo-seq embryo\n" + table.to_string(), flush=True)

    comparisons = []
    for candidate in [name for name in finals if name.startswith(("vae_joint", "pca_joint"))]:
        for other in finals:
            if other != candidate:
                comparisons.append(dict(policy=candidate, against=other,
                                        **paired_gene_bootstrap(finals[candidate], finals[other])))
    pd.DataFrame(comparisons).to_csv(out / "vae_vs_others.csv", index=False)
    print("\n" + pd.DataFrame(comparisons).round(4).to_string(index=False), flush=True)

    first_rounds = {name: panel[:args.genes_per_round] for name, panel in panels.items()}
    print("\nFirst four genes\n" + "\n".join(f"  {name:24s} {genes_}" for name, genes_ in first_rounds.items()), flush=True)
    print(f"Finished in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
