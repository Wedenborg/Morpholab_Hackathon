"""Offline CLI for the Morpholab eval_team benchmark."""

import argparse
import json
import os
from pathlib import Path
import platform
import sys


def parse_args(argv=None):
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    known, _ = config_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__, parents=[config_parser])
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("MORPHO_DATA_DIR"))
    parser.add_argument("--eval-reference-dir", type=Path, default=os.environ.get("MORPHO_REF_DIR"))
    parser.add_argument("--hidden", type=Path, help="Hidden E7.5 rep2 .h5ad; defaults to rep2 in data-dir")
    parser.add_argument("--reference-source", choices=["auto", "sc", "rep1"], default="auto")
    parser.add_argument("--reference", type=Path, default=os.environ.get("MORPHO_SC_REFERENCE"),
                        help="SC .h5ad/.rds; auto prefers this over rep1")
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--annotation", type=Path, help="MGI MRK_ENSEMBL.rpt for mouse Ensembl SC IDs")
    parser.add_argument("--reference-scale", choices=["auto", "counts", "log1p"], default="auto")
    parser.add_argument("--reference-counts-layer", default="counts")
    parser.add_argument("--selection", choices=["detected_hvg", "hvg", "random_hvg"], default="detected_hvg")
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--detection-threshold-pct", type=float, default=50.0)
    parser.add_argument("--hvg-min-cells", type=int, default=10)
    parser.add_argument("--initial-genes", nargs="+", default=["Sox2", "Hoxa1", "Hhex", "Pdx1"])
    parser.add_argument("--rounds", type=int, default=5, help="Acquisition rounds after baseline round 0")
    parser.add_argument("--n-select", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--mlp-hidden-dim", type=int, nargs="+", default=[128])
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-lr", type=float, default=0.001)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:INDEX")
    parser.add_argument("--skip-evaluation", action="store_true", help="Produce a submission without benchmark scores")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing directories are refused")
    if known.config:
        defaults = json.loads(known.config.read_text())
        if not isinstance(defaults, dict):
            parser.error("Config must be a JSON object.")
        allowed = {action.dest for action in parser._actions} - {"help", "config", "output"}
        unknown = set(defaults) - allowed
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    for action in parser._actions:
        if action.choices is not None and getattr(args, action.dest) not in action.choices:
            parser.error(f"Invalid {action.dest}: {getattr(args, action.dest)}")
    return args


def resolve_reference(args, data_dir):
    """An explicitly supplied SC is preferred; invalid SC never silently falls back."""
    if args.reference_source == "sc" and args.reference is None:
        raise ValueError("--reference-source sc requires --reference or MORPHO_SC_REFERENCE.")
    if args.reference_source == "rep1":
        return "rep1", embryo_path(data_dir, 1)
    if args.reference is not None:
        return "sc", Path(args.reference)
    return "rep1", embryo_path(data_dir, 1)


def embryo_path(data_dir, rep):
    paths = sorted(Path(data_dir).glob(f"*_Embryo_E7.5_stereo_rep{rep}.h5ad"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one E7.5 rep{rep} file in {data_dir}, found {len(paths)}.")
    return paths[0]


def load_training_reference(args, source, path):
    import anndata as ad
    from .reference import read_reference, map_reference_gene_symbols, prepare_expression

    if source == "rep1":
        # The benchmark contract explicitly defines file X as natural-log1p counts.
        return prepare_expression(ad.read_h5ad(path), scale="log1p", normalize_total=False)
    reference = read_reference(path, cache_dir=args.reference_cache)
    scale = args.reference_scale
    if reference.var_names.str.startswith("ENSMUSG").any():
        if args.annotation is None:
            raise ValueError("Ensembl SC IDs require --annotation with an MGI report.")
        if scale == "counts" and args.reference_counts_layer not in reference.layers:
            reference.uns["expression_source"] = "counts"
        reference = map_reference_gene_symbols(reference, args.annotation, args.reference_counts_layer)
        scale = "counts"
    # Preserve the benchmark's log1p-count scale. Never normalize hidden libraries.
    return prepare_expression(reference, scale=scale, counts_layer=args.reference_counts_layer,
                              normalize_total=False)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise ValueError(f"Output already exists: {args.output}; choose a fresh directory.")
    if sys.version_info < (3, 10):
        raise RuntimeError("The upstream eval_team requires Python >=3.10; use Python 3.11+ for the full runner. "
                           "The current Python 3.9 LUNA environment can run the standalone core tests only.")
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import torch
    if torch.device(args.device).type == "cuda":
        probe = torch.tensor([-1., 2.], device=args.device, requires_grad=True)
        probe.relu().sum().backward()
        torch.cuda.synchronize(torch.device(args.device))
        del probe
    try:
        from eval_team import MeasurementOracle, Evaluator
        from eval_team.paths import DATA_DIR, REF_DIR
    except ModuleNotFoundError as exc:
        if exc.name in {"eval_team", "eval_team.paths"}:
            raise RuntimeError("Run from the Morpholab_Hackathon root with eval_team available.") from exc
        raise
    from .method import HaochenMethod
    from .runner import run_experiment

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    source, reference_path = resolve_reference(args, data_dir)
    hidden_path = args.hidden if args.hidden is not None else embryo_path(data_dir, 2)
    if Path(reference_path).resolve() == Path(hidden_path).resolve():
        raise ValueError("Training reference and hidden target must be different files.")
    reference = load_training_reference(args, source, reference_path)
    hidden = ad.read_h5ad(hidden_path)
    # Hidden expression remains evaluator/oracle-owned and is never passed to the method.
    oracle = MeasurementOracle(hidden)
    widths = args.mlp_hidden_dim[0] if len(args.mlp_hidden_dim) == 1 else args.mlp_hidden_dim
    method = HaochenMethod(reference, oracle.genes, selection=args.selection,
                           top_k=args.top_k, threshold_pct=args.detection_threshold_pct,
                           min_cells=args.hvg_min_cells, seed=args.seed,
                           mlp_options={"num_layers": args.mlp_layers, "hidden_dim": widths,
                                        "epochs": args.mlp_epochs, "batch_size": args.batch_size,
                                        "learning_rate": args.mlp_lr, "device": args.device})
    evaluator = None
    if not args.skip_evaluation:
        benchmark = Evaluator(hidden, ref_dir=args.eval_reference_dir or REF_DIR)

        def evaluator(path, log):
            # Upstream scoring may materialize host matrices; method prediction is batched.
            return benchmark.evaluate(ad.read_h5ad(path), log)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    from . import __version__
    config.update({"method_version": __version__,
                   "resolved_reference_source": source, "resolved_reference_path": str(reference_path),
                   "resolved_hidden_path": str(hidden_path),
                   "reference_preprocessing": reference.uns["preprocessing"],
                   "output_scale": "natural log1p counts; no additional library normalization",
                   "versions": {"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "anndata": ad.__version__, "scanpy": sc.__version__}})
    print(f"Reference: {source} {reference.shape}; hidden target: {hidden.shape}", flush=True)
    run_experiment(method, oracle, args.output, initial_genes=args.initial_genes,
                   rounds=args.rounds, n_select=args.n_select, batch_size=args.batch_size,
                   evaluator=evaluator, diagnostic_truth=None if args.skip_evaluation else hidden.X,
                   config=config)
    print(f"Saved submission and round metrics: {args.output}", flush=True)


if __name__ == "__main__":
    main()
