"""Pre-run the selectable methods on the local E7.5 preset for the webpage's comparison view.

For every method and panel size it simulates up to --rounds rounds of experimenting: round 0 is
the four previously measured genes, and each further round adds `ng` genes. After every round the
embryo is reconstructed from all genes measured so far and scored with `eval_team.Evaluator`, the
same benchmark the live webpage uses. Results are retrospective on the local E7.5 data.

    python backend/scripts/prerun_methods.py --rounds 100              # everything, 8 processes
    python backend/scripts/prerun_methods.py --rounds 5 --ng 4 --methods dummy_markers --out test.json

Each (method, ng) result is cached in --cache so an interrupted run resumes; the cache is merged
into --out (default frontend/public/prerun/e75_prerun.json) at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Keep each worker to a couple of threads; the workers share the machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

DATA_DIR = Path(os.getenv("MORPHO_DATA_DIR", ROOT / "data" / "GSE278603_RAW"))
SPATIAL = DATA_DIR / "GSM9046243_Embryo_E7.5_stereo_rep1.h5ad"
REFERENCE = DATA_DIR / "GSM9046244_Embryo_E7.5_stereo_rep2.h5ad"
PREVIOUS = ["Fth1", "Rpl8", "Eif5a", "Cox7c"]
SEED = 42
METHODS = {
    "gaussian_visible": "Gaussian information gain · visible embryo",
    "rotating_visible": "Rotating markers · visible embryo",
    "dummy_markers": "eval_team dummy · cell-type markers",
    "dummy_variance": "eval_team dummy · variable genes",
    "dummy_random": "eval_team dummy · random expressed genes",
}
DUMMY_HOW = {"dummy_markers": "markers", "dummy_variance": "variance", "dummy_random": "random"}
DEFAULT_NG = (4, 8, 16)
SCORE_KEYS = ("truth_region_spearman", "heldout_corr", "celltype_tv", "celltype_balanced_accuracy")


SCHEDULE = (0, 1, 2, 5, 10, 25, 50, 100)
MAX_SCORED_GENES = 420  # scoring one round costs roughly 30 s + 0.23 s per measured gene


def _scored_rounds(ng: int, rounds: int, max_genes: int) -> list[int]:
    return [r for r in SCHEDULE if r <= rounds and len(PREVIOUS) + r * ng <= max_genes]


def _scorer(ng: int, scored: list[int], reconstruct=None):
    from app.evaluation import mouse_embryo_round_scorer

    base = mouse_embryo_round_scorer(SPATIAL, REFERENCE, reconstruct=reconstruct)
    reference_values: dict[str, float] = {}
    labels: dict[str, str] = {}

    def score(genes, *args):
        # Scoring dominates the run time, so only scheduled rounds are evaluated.
        round_number = (len(genes) - len(PREVIOUS)) // ng
        if round_number not in scored:
            return {"metrics": []}
        result = base(genes, *args)
        for metric in result["metrics"]:
            labels[metric["id"]] = metric["label"]
            if metric["true_data"] is not None:
                reference_values[metric["id"]] = metric["true_data"]
        return result

    return score, reference_values, labels


def _values(result) -> dict[str, float]:
    return {metric["id"]: round(metric["value"], 5) for metric in result["metrics"]}


def run_design(method: str, ng: int, rounds: int, max_genes: int) -> dict:
    import anndata as ad

    from app import design_methods

    scored = _scored_rounds(ng, rounds, max_genes)
    rounds = max(scored)
    truth = ad.read_h5ad(SPATIAL)
    reference = ad.read_h5ad(REFERENCE)
    target_genes = list(map(str, truth.var_names))

    def coords(adata):
        return adata.obs[["x", "y", "z"]].to_numpy(float)

    visible = design_methods.load_expression(
        reference, "X", list(map(str, reference.var_names)), target_genes, "celltype", coords(reference)
    )
    eligible = [g for g in target_genes if g not in set(PREVIOUS)]
    score, truth_values, labels = _scorer(
        ng, scored, reconstruct=lambda columns, values, model: design_methods.reconstruct(model, columns, values)
    )
    _, steps, facts = design_methods.run(
        method, visible, None, coords(truth), target_genes, PREVIOUS, eligible, ng, rounds, SEED,
        {g: g for g in target_genes}, score,
    )
    out = [{
        "round": step["round"],
        "n_measured": step["n_measured"],
        "genes": step["genes"],
        "expected_share": step["expected_share"],
        "metrics": _values(step["benchmark"]),
    } for step in steps]
    return {"rounds": out, "true_data": truth_values, "labels": labels, "facts": facts}


def run_dummy(method: str, ng: int, rounds: int, max_genes: int) -> dict:
    import anndata as ad

    from eval_team import dummy_method

    scored = _scored_rounds(ng, rounds, max_genes)
    rounds = max(scored)
    truth = ad.read_h5ad(SPATIAL)
    reference = ad.read_h5ad(REFERENCE)
    # One ordered gene list; round r takes the next `ng` of it. The dummy methods rank genes, so
    # taking prefixes equals running them round by round.
    sets = dummy_method.choose_gene_sets(
        reference, truth.var_names, rounds * ng // 4, DUMMY_HOW[method], SEED, exclude=PREVIOUS
    )
    ordered = [str(g) for gene_set in sets for g in gene_set]
    score, truth_values, labels = _scorer(ng, scored)
    measured = list(PREVIOUS)
    out = [{"round": 0, "n_measured": len(measured), "genes": [], "expected_share": None,
            "metrics": _values(score(measured))}]
    for round_number in range(1, rounds + 1):
        added = ordered[(round_number - 1) * ng: round_number * ng]
        if len(added) < ng:
            break
        measured = measured + added
        out.append({"round": round_number, "n_measured": len(measured), "genes": added,
                    "expected_share": None, "metrics": _values(score(measured))})
    return {"rounds": out, "true_data": truth_values, "labels": labels, "facts": {}}


def run_one(method: str, ng: int, rounds: int, max_genes: int, cache: Path) -> tuple[str, int, float]:
    started = time.time()
    runner = run_dummy if method in DUMMY_HOW else run_design
    result = runner(method, ng, rounds, max_genes)
    result.update(method=method, ng=ng, requested_rounds=rounds, scored_rounds=_scored_rounds(ng, rounds, max_genes))
    (cache / f"{method}_{ng}.json").write_text(json.dumps(result), encoding="utf-8")
    return method, ng, time.time() - started


def merge(cache: Path, out: Path, methods: list[str], sizes: list[int], rounds: int) -> None:
    runs, truth_values, labels = [], {}, {}
    for method in methods:
        for ng in sizes:
            path = cache / f"{method}_{ng}.json"
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            truth_values.update(data["true_data"])
            labels.update(data["labels"])
            runs.append({"method": method, "label": METHODS[method], "ng": ng, "rounds": data["rounds"]})
    document = {
        "dataset": "E7.5 mouse embryo (GSE278603): rep1 is the target, rep2 the reference",
        "previous_genes": PREVIOUS,
        "seed": SEED,
        "max_rounds": rounds,
        "metrics": [
            {"id": key, "label": labels[key], "true_data": truth_values.get(key),
             "higher_is_better": key != "celltype_tv"}
            for key in SCORE_KEYS if key in labels
        ],
        "notice": (
            "Retrospective simulation on the local E7.5 data with eval_team: round 0 is the four previously "
            "measured genes and each round adds the stated number of genes to every cell. Each round is "
            "reconstructed from all genes measured so far. Not a prospective performance guarantee."
        ),
        "runs": runs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(runs)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--ng", type=int, nargs="+", default=list(DEFAULT_NG))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--max-scored-genes", type=int, default=MAX_SCORED_GENES,
                        help="only score rounds whose measured-gene total is at most this (scoring is slow)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "prerun_cache")
    parser.add_argument("--out", type=Path, default=ROOT / "frontend" / "public" / "prerun" / "e75_prerun.json")
    parser.add_argument("--force", action="store_true", help="recompute cached results")
    args = parser.parse_args()

    for path in (SPATIAL, REFERENCE):
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Set MORPHO_DATA_DIR to the folder with the E7.5 .h5ad files.")
    args.cache.mkdir(parents=True, exist_ok=True)
    jobs = [(m, n) for n in sorted(args.ng, reverse=True) for m in args.methods
            if args.force or not (args.cache / f"{m}_{n}.json").is_file()]
    print(f"{len(jobs)} job(s) to run, {args.workers} worker(s)", flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_one, m, n, args.rounds, args.max_scored_genes, args.cache) for m, n in jobs]
            for future in as_completed(futures):
                method, ng, seconds = future.result()
                print(f"done {method} ng={ng} in {seconds / 60:.1f} min", flush=True)
    merge(args.cache, args.out, args.methods, args.ng, args.rounds)


if __name__ == "__main__":
    main()
