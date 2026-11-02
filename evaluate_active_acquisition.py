"""Compare gene selection policies on a held-out E7.5 embryo.

The hidden embryo is used for gene names and positions, measurements through the logged
MeasurementOracle, and final scoring. Models train on the other embryos, split by embryo.
Only the "hindsight" policy reads the truth: it is an upper bound and is labelled as one.

Usage:
    python evaluate_active_acquisition.py --hidden-rep 2 --out-dir data/active_acquisition/rep2
    python evaluate_active_acquisition.py --hidden-rep 2 --out-dir ... --imputer-only
"""
import argparse
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
import torch

from active_acquisition import (
    GeneSpace,
    MaskedImputer,
    RevealSimulator,
    AcquisitionNet,
    LogisticNormalPolicy,
    aligned_expression,
    build_gene_space,
    draw_panels,
    propose_genes,
    reference_correlation,
    ridge_fit_predict,
    simulate_examples,
    smoothed_expression,
    spatial_smoother,
    summarize_tissue,
    tissue_state,
    train_acquisition,
    train_imputer,
    train_policy,
)
from eval_team import Evaluator, MeasurementOracle, load_embryo
from eval_team.spatial_scores import heldout_gene_score, sparsity_summary
from eval_team.submission import measured_mask
from gene_recommender import recommend_genes

STAGES_OTHER_THAN_E75 = (("E7.75", 1), ("E7.75", 2), ("E8.0", 2))
VALIDATION_EMBRYO = ("E8.0", 1)
SMOOTHING_NEIGHBORS = 25
GREEDY_NOISE_STD = 0.5 / SMOOTHING_NEIGHBORS ** 0.5  # gene_recommender's default for neighbourhood averages.
N_STATE_CELLS = 1024


def imputation_report(
    imputer: MaskedImputer,
    ridge_profiles: torch.Tensor,
    space: GeneSpace,
    val: torch.Tensor,
    panel_sizes: tuple[int, ...] = (0, 4, 8, 16, 32),
    n_panels: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """R² on unmeasured genes of the validation embryo, for random panels of each size."""
    generator = torch.Generator().manual_seed(seed)
    cells = val[torch.randperm(len(val), generator=generator)[:4096]]
    candidates = space.candidate_index
    rows = []
    for size in panel_sizes:
        for panel in range(n_panels):
            positions = torch.randperm(len(candidates), generator=generator)[:size]
            mask = torch.zeros(len(candidates), dtype=torch.bool)
            mask[positions] = True
            hidden = torch.ones(cells.shape[1], dtype=torch.bool)
            hidden[candidates[positions]] = False
            baseline = (cells[:, hidden] - space_baseline(ridge_profiles, space)[hidden]).square().mean()
            neural = imputer(cells[:, candidates], mask.expand(len(cells), -1))
            observed = torch.zeros(cells.shape[1], dtype=torch.bool)
            observed[candidates[positions]] = True
            features = (cells[:, observed] * space.std[observed] + space.mean[observed])
            ridge = ridge_fit_predict(ridge_profiles, features, observed)
            ridge = (ridge - space.mean) / space.std
            rows.append(dict(
                panel_size=size, panel=panel,
                neural_r2=1 - ((neural[:, hidden] - cells[:, hidden]).square().mean() / baseline).item(),
                ridge_r2=1 - ((ridge[:, hidden] - cells[:, hidden]).square().mean() / baseline).item(),
            ))
    return pd.DataFrame(rows)


def space_baseline(ridge_profiles: torch.Tensor, space: GeneSpace) -> torch.Tensor:
    """The reference mean in z-scored units: the prediction made with no measurements."""
    return (ridge_profiles.mean(0) - space.mean) / space.std


class Episode:
    """One measurement campaign on the hidden embryo. Holds only what has been measured."""

    def __init__(self, hidden, space: GeneSpace, smoother):
        self.oracle = MeasurementOracle(hidden)
        self.space = space
        self.smoother = smoother
        n_cells, n_candidates = hidden.n_obs, len(space.candidate_index)
        self.row_of = {cell: row for row, cell in enumerate(hidden.obs_names)}
        self.measured = torch.zeros(n_candidates, dtype=torch.bool)
        self.raw = np.zeros((n_cells, n_candidates), dtype=np.float32)
        self.smoothed = torch.zeros(n_cells, n_candidates)
        self.chosen: list[str] = []

    @property
    def standardized(self) -> torch.Tensor:
        return self.space.standardize_candidates(self.smoothed)

    def measure(self, positions: torch.Tensor) -> None:
        positions = positions.tolist()
        genes = self.space.candidate_genes[positions].tolist()
        for section in self.oracle.sections:
            values = self.oracle.measure(genes, section=section)
            rows = [self.row_of[cell] for cell in values.index]
            self.raw[np.ix_(rows, positions)] = values.to_numpy()
        self.smoothed[:, positions] = torch.from_numpy(np.asarray(self.smoother @ self.raw[:, positions], dtype=np.float32))
        self.measured[positions] = True
        self.chosen.extend(genes)


def reconstruct(episode: Episode, imputer: MaskedImputer | None, ridge_profiles: torch.Tensor) -> np.ndarray:
    """Predicted log expression (N, T); measured genes keep their measured cell values."""
    space = episode.space
    candidates = space.candidate_index
    if imputer is None:
        observed = torch.zeros(len(space.target_genes), dtype=torch.bool)
        observed[candidates[episode.measured]] = True
        predicted = ridge_fit_predict(ridge_profiles, episode.smoothed[:, episode.measured], observed)
    else:
        mask = episode.measured.expand(len(episode.smoothed), -1)
        predicted = space.to_log(imputer(episode.standardized, mask))
    predicted = predicted.numpy().copy()
    columns = candidates[episode.measured].numpy()
    predicted[:, columns] = episode.raw[:, episode.measured.numpy()]
    return predicted


def score_round(
    episode: Episode,
    evaluator: Evaluator,
    hidden,
    predicted: np.ndarray,
    truth_log: np.ndarray,
) -> tuple[dict, pd.Series]:
    space = episode.space
    frame = pd.DataFrame(predicted, index=hidden.obs_names, columns=space.target_genes)
    mask = measured_mask(episode.oracle.log(), hidden.obs, evaluator.genes)
    per_gene = heldout_gene_score(frame, hidden, evaluator.bins, evaluator.genes, mask=mask)
    unmeasured = torch.ones(len(space.target_genes), dtype=torch.bool)
    unmeasured[space.candidate_index[episode.measured]] = False
    unmeasured = unmeasured.numpy()
    std = space.std.numpy()[unmeasured]
    error = ((predicted[:, unmeasured] - truth_log[:, unmeasured]) / std) ** 2
    baseline = ((space.mean.numpy()[unmeasured] - truth_log[:, unmeasured]) / std) ** 2
    by_sparsity = sparsity_summary(per_gene)["mean"].to_dict()
    return dict(
        heldout_corr=float(per_gene["corr"].mean()),
        neighbourhood_r2=float(1 - error.mean() / baseline.mean()),
        **{f"heldout | {label}": float(value) for label, value in by_sparsity.items()},
    ), per_gene["corr"]


def random_policy(seed: int):
    generator = torch.Generator().manual_seed(seed)

    def choose(episode: Episode, k: int) -> torch.Tensor:
        available = (~episode.measured).nonzero(as_tuple=True)[0]
        return available[torch.randperm(len(available), generator=generator)[:k]]
    return choose


def variance_policy(space: GeneSpace):
    variances = space.reference_sample[:, space.candidate_index].var(0)

    def choose(episode: Episode, k: int) -> torch.Tensor:
        return variances.masked_fill(episode.measured, -np.inf).topk(k).indices
    return choose


def greedy_policy(space: GeneSpace, locations: torch.Tensor, reweight_mix: float = 0.0):
    """The existing gene_recommender method, on the same reference and candidate pool."""
    n_targets = len(space.target_genes)
    position_of = torch.full((n_targets,), -1, dtype=torch.long)
    position_of[space.candidate_index] = torch.arange(len(space.candidate_index))

    def choose(episode: Episode, k: int) -> torch.Tensor:
        observed = torch.zeros(n_targets, dtype=torch.bool)
        observed[space.candidate_index[episode.measured]] = True
        data = torch.zeros(len(episode.smoothed), n_targets)
        data[:, space.candidate_index] = episode.smoothed
        n_unmeasured = int((~episode.measured).sum())
        result = recommend_genes(
            k, torch.zeros(n_targets, 1), torch.zeros(0), locations, data, [space.reference_sample],
            observed_mask=observed, noise_std=GREEDY_NOISE_STD, max_candidates=n_unmeasured,
            max_reference_cells=len(space.reference_sample), min_detection=0.0, reweight_mix=reweight_mix,
        )
        positions = position_of[result.selection_order]
        if (positions < 0).any():
            raise RuntimeError("gene_recommender chose a gene outside the shared candidate pool.")
        return positions
    return choose


def learned_policy(imputer: MaskedImputer, net: AcquisitionNet, use_values: bool = True,
                   use_pending: bool = True, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def choose(episode: Episode, k: int) -> torch.Tensor:
        cells = torch.randperm(len(episode.smoothed), generator=generator)[:N_STATE_CELLS]
        values = episode.standardized[cells]
        if not use_values:
            values = torch.zeros_like(values)
        state = tissue_state(imputer, values, episode.measured)
        return propose_genes(net, state, episode.measured, k, use_pending=use_pending)
    return choose


def logistic_normal_policy(imputer: MaskedImputer, policy: LogisticNormalPolicy, correlation: torch.Tensor,
                           use_values: bool = True, seed: int = 0):
    """Deployed without noise: the most probable gene at each step of the panel."""
    generator = torch.Generator().manual_seed(seed)

    def choose(episode: Episode, k: int) -> torch.Tensor:
        cells = torch.randperm(len(episode.smoothed), generator=generator)[:N_STATE_CELLS]
        values = episode.standardized[cells]
        if not use_values:
            values = torch.zeros_like(values)
        summary = summarize_tissue(imputer, values, episode.measured, episode.space.candidate_index)
        return draw_panels(policy, summary, episode.measured, correlation, k)[0][0]
    return choose


def hindsight_policy(simulator: RevealSimulator, truth: torch.Tensor, n_cells: int = 256, seed: int = 0, chunk: int = 64):
    """Greedily reveals the gene that most reduces true error. Reads the truth, so it is
    an upper bound on one-step selection and not a method anyone could deploy."""
    generator = torch.Generator().manual_seed(seed)
    cells = truth[torch.randperm(len(truth), generator=generator)[:n_cells]]

    def choose(episode: Episode, k: int) -> torch.Tensor:
        mask = episode.measured.clone()
        picks = []
        for _ in range(k):
            available = (~mask).nonzero(as_tuple=True)[0]
            gains = torch.cat([simulator.improvements(cells, mask, part) for part in available.split(chunk)])
            pick = available[gains.argmax()]
            picks.append(pick)
            mask[pick] = True
        return torch.stack(picks)
    return choose


def run_policy(name, choose, seed, hidden, space, smoother, evaluator, imputers, truth_log,
               rounds, genes_per_round, score_rounds):
    episode = Episode(hidden, space, smoother)
    rows, final = [], {}
    started = time.perf_counter()
    choosing = 0.0
    for round_index in range(1, rounds + 1):
        tick = time.perf_counter()
        with torch.no_grad():
            picks = choose(episode, genes_per_round)
        choosing += time.perf_counter() - tick
        if len(picks.unique()) != genes_per_round or episode.measured[picks].any():
            raise RuntimeError(f"{name} proposed repeated or already measured genes.")
        episode.measure(picks)
        if round_index in score_rounds:
            for imputer_name, (imputer, ridge_profiles) in imputers.items():
                with torch.no_grad():
                    predicted = reconstruct(episode, imputer, ridge_profiles)
                scores, per_gene = score_round(episode, evaluator, hidden, predicted, truth_log)
                rows.append(dict(policy=name, seed=seed, imputer=imputer_name, round=round_index,
                                 n_genes=round_index * genes_per_round, **scores))
                if round_index == rounds:
                    final[imputer_name] = dict(predicted=predicted, per_gene=per_gene)
    print(f"  {name} seed {seed}: {time.perf_counter() - started:.0f} s "
          f"({choosing / rounds:.2f} s per round choosing), {rows[-1]['imputer']} final "
          f"heldout {rows[-1]['heldout_corr']:.4f}", flush=True)
    return rows, final, episode


def paired_gene_bootstrap(a: pd.Series, b: pd.Series, n: int = 2000, seed: int = 0) -> dict:
    differences = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = differences[rng.integers(len(differences), size=(n, len(differences)))].mean(1)
    low, high = np.percentile(means, [2.5, 97.5])
    return dict(mean_difference=float(differences.mean()), ci_low=float(low), ci_high=float(high), n_genes=len(differences))


def main() -> None:
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden-rep", type=int, choices=(1, 2), default=2)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--genes-per-round", type=int, default=4)
    parser.add_argument("--n-candidates", type=int, default=1000)
    parser.add_argument("--imputer-steps", type=int, default=6000)
    parser.add_argument("--states", type=int, default=4000, help="simulated training states per pass")
    parser.add_argument("--val-states", type=int, default=400)
    parser.add_argument("--acquisition-steps", type=int, default=4000)
    parser.add_argument("--policy-steps", type=int, default=1500)
    parser.add_argument("--random-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imputer-only", action="store_true", help="stop after validating the imputer")
    args = parser.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    hidden = load_embryo(args.hidden_rep)
    library = hidden.var_names
    visible_rep = 3 - args.hidden_rep
    train_embryos = [load_embryo(visible_rep)] + [load_embryo(rep, stage=stage) for stage, rep in STAGES_OTHER_THAN_E75]
    val_embryo = load_embryo(VALIDATION_EMBRYO[1], stage=VALIDATION_EMBRYO[0])

    started = time.perf_counter()
    space, train_profiles = build_gene_space(train_embryos, library, args.n_candidates, seed=args.seed)
    val_log = smoothed_expression(val_embryo, space.target_genes)
    print(f"{len(space.target_genes)} targets, {len(space.candidate_index)} candidates, "
          f"{sum(len(p) for p in train_profiles)} training cells ({time.perf_counter() - started:.0f} s)", flush=True)
    del train_embryos, val_embryo

    generator = torch.Generator().manual_seed(args.seed)
    pooled_log = torch.cat(train_profiles)
    ridge_profiles = pooled_log[torch.randperm(len(pooled_log), generator=generator)[:20000]].clone()
    del pooled_log
    tissues = [(profile - space.mean) / space.std for profile in train_profiles]
    del train_profiles
    val = (val_log - space.mean) / space.std
    del val_log
    log_weights = space.std[space.candidate_index].square().log()

    imputer_path = out / "imputer.pt"
    if imputer_path.exists():
        imputer = MaskedImputer(len(space.candidate_index), len(space.target_genes))
        imputer.load_state_dict(torch.load(imputer_path))
        imputer = imputer.eval().requires_grad_(False)
        print("Reusing saved imputer", flush=True)
    else:
        imputer = train_imputer(torch.cat(tissues), val, space.candidate_index, log_weights,
                                steps=args.imputer_steps, seed=args.seed)
        torch.save(imputer.state_dict(), imputer_path)

    with torch.no_grad():
        report = imputation_report(imputer, ridge_profiles, space, val, seed=args.seed)
    report.to_csv(out / "imputer_validation.csv", index=False)
    print(report.groupby("panel_size")[["neural_r2", "ridge_r2"]].agg(["mean", "std"]).round(4).to_string(), flush=True)
    if args.imputer_only:
        return

    simulator = RevealSimulator(imputer, space.candidate_index, seed=args.seed)
    acquisition_path = out / "acquisition.pt"
    state_size = 2 * imputer.decoder.in_features
    acquisition_report = {}
    if acquisition_path.exists():
        net = AcquisitionNet(state_size, len(space.candidate_index))
        net.load_state_dict(torch.load(acquisition_path))
        net = net.eval().requires_grad_(False)
        print("Reusing saved acquisition network", flush=True)
    else:
        print("Simulating reveals from random panels", flush=True)
        train_examples = simulate_examples(simulator, tissues, log_weights, args.states, seed=args.seed)
        val_examples = simulate_examples(simulator, [val], log_weights, args.val_states, seed=args.seed + 1)
        net, acquisition_report["random_panels"] = train_acquisition(
            train_examples, val_examples, len(space.candidate_index), steps=args.acquisition_steps, seed=args.seed)
        print("Simulating reveals along the policy's own path", flush=True)
        train_examples = train_examples.concat(simulate_examples(
            simulator, tissues, log_weights, args.states, policy=net, seed=args.seed + 2))
        val_examples = val_examples.concat(simulate_examples(
            simulator, [val], log_weights, args.val_states, policy=net, seed=args.seed + 3))
        net, acquisition_report["with_policy_states"] = train_acquisition(
            train_examples, val_examples, len(space.candidate_index), steps=args.acquisition_steps, seed=args.seed)
        torch.save(net.state_dict(), acquisition_path)
        (out / "acquisition_validation.json").write_text(json.dumps(acquisition_report, indent=2))

    correlation = reference_correlation(space)
    policy_path = out / "logistic_normal.pt"
    if policy_path.exists():
        logistic_normal = LogisticNormalPolicy(state_size, len(space.candidate_index))
        logistic_normal.load_state_dict(torch.load(policy_path))
        logistic_normal = logistic_normal.eval().requires_grad_(False)
        print("Reusing saved logistic-normal policy", flush=True)
    else:
        print("Training the logistic-normal policy on realized panel rewards", flush=True)
        logistic_normal, policy_report = train_policy(
            simulator, tissues, val, correlation, steps=args.policy_steps,
            max_rounds=args.rounds, genes_per_round=args.genes_per_round, seed=args.seed)
        torch.save(logistic_normal.state_dict(), policy_path)
        (out / "logistic_normal_validation.json").write_text(json.dumps(policy_report, indent=2))
    del tissues, val

    locations = hidden.obs[["x", "y", "z"]].to_numpy(dtype=float)
    smoother = spatial_smoother(locations, SMOOTHING_NEIGHBORS)
    truth_log_sparse = smoother @ aligned_expression(hidden, space.target_genes)  # For scoring and hindsight only.
    truth_log = np.asarray(truth_log_sparse.todense(), dtype=np.float32)
    truth = (torch.from_numpy(truth_log) - space.mean) / space.std
    evaluator = Evaluator(hidden)
    location_tensor = torch.from_numpy(locations).float()

    policies = [(f"random", seed, random_policy(seed)) for seed in range(args.random_seeds)]
    policies += [
        ("variance", 0, variance_policy(space)),
        ("greedy_ridge", 0, greedy_policy(space, location_tensor)),
        ("greedy_reweight", 0, greedy_policy(space, location_tensor, reweight_mix=0.5)),
        ("learned", 0, learned_policy(imputer, net, seed=args.seed)),
        ("learned_no_values", 0, learned_policy(imputer, net, use_values=False, seed=args.seed)),
        ("learned_no_pending", 0, learned_policy(imputer, net, use_pending=False, seed=args.seed)),
        ("logistic_normal", 0, logistic_normal_policy(imputer, logistic_normal, correlation, seed=args.seed)),
        ("logistic_normal_no_values", 0, logistic_normal_policy(
            imputer, logistic_normal, correlation, use_values=False, seed=args.seed)),
        ("hindsight", 0, hindsight_policy(simulator, truth, seed=args.seed)),
    ]
    imputers = {"neural": (imputer, ridge_profiles), "ridge": (None, ridge_profiles)}
    score_rounds = {1, 2, args.rounds // 2, args.rounds}
    rows, finals, panels = [], {}, {}
    for name, seed, choose in policies:
        policy_rows, final, episode = run_policy(
            name, choose, seed, hidden, space, smoother, evaluator, imputers, truth_log,
            args.rounds, args.genes_per_round, score_rounds)
        rows.extend(policy_rows)
        panels[f"{name}_{seed}"] = episode.chosen
        episode.oracle.log().to_csv(out / f"experiments_{name}_{seed}.csv", index=False)
        if seed == 0:
            finals[name] = final
            # The full benchmark on the final reconstruction, at the same experiment budget.
            for imputer_name, result in final.items():
                frame = pd.DataFrame(result["predicted"], index=hidden.obs_names, columns=space.target_genes)
                summary, _ = evaluator.evaluate(frame, episode.oracle.log())
                (out / f"benchmark_{name}_{imputer_name}.json").write_text(json.dumps(summary, indent=2))
        pd.DataFrame(rows).to_csv(out / "policy_scores.csv", index=False)
    (out / "panels.json").write_text(json.dumps(panels, indent=2))

    comparisons = []
    for imputer_name in imputers:
        for candidate in ("learned", "logistic_normal"):
            reference = finals[candidate][imputer_name]["per_gene"]
            for other in finals:
                if other != candidate:
                    comparisons.append(dict(policy=candidate, imputer=imputer_name, against=other, **paired_gene_bootstrap(
                        reference, finals[other][imputer_name]["per_gene"])))
    pd.DataFrame(comparisons).to_csv(out / "learned_vs_others.csv", index=False)
    print(pd.DataFrame(comparisons).round(4).to_string(index=False), flush=True)

    scores = pd.DataFrame(rows)
    table = scores.groupby(["imputer", "policy", "n_genes"])[["heldout_corr", "neighbourhood_r2"]].mean().round(4)
    print(table.to_string(), flush=True)


if __name__ == "__main__":
    main()
