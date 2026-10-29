import argparse
from pathlib import Path
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dataset.load_dataset import convert_gene_names, load_dataset


REFERENCE_DATASETS = [
    {
        "dataset": "tome",
        "timepoint": "7.5",
        "layer": "logcounts",
    },
]


def _load_samples(dataset, timepoint=None):
    if isinstance(dataset, ad.AnnData):
        return {"input": convert_gene_names(dataset.copy())}
    if Path(dataset).is_file():
        return {Path(dataset).stem: convert_gene_names(ad.read_h5ad(dataset))}
    return load_dataset(str(dataset), timepoint=timepoint, output="anndata")


def _panel_expression(samples, genes, layer=None):
    frames = []
    for adata in samples.values():
        aliases = _reference_aliases(adata)
        identifiers = [aliases[_gene_key(gene)] for gene in genes]
        expression = adata[:, identifiers].to_df(layer=layer)
        expression.columns = genes
        frames.append(expression)
    return pd.concat(frames, ignore_index=True)


def _gene_key(name):
    name = str(name).upper()
    return re.sub(r"\.\d+$", "", name) if name.startswith("ENS") else name


def _reference_aliases(adata):
    aliases = {}

    def add_alias(symbol, identifier):
        aliases.setdefault(_gene_key(symbol), set()).add(identifier)

    for name in adata.var_names:
        add_alias(name, name)
    for column in ("gene_symbol", "gene_symbols", "symbol", "gene_id", "reference_gene_id"):
        if column in adata.var:
            for identifier, symbol in adata.var[column].dropna().items():
                add_alias(symbol, identifier)
    return {symbol: next(iter(ids)) for symbol, ids in aliases.items() if len(ids) == 1}


def select_dataset(
    adata,
    genes=None,
    reference_datasets=None,
    d=256,
    n_cells=1000,
    timepoint=None,
    input_layer=None,
    output_dir=None,
    reference_genes=None,
    verbose=False,
    top_n_var_genes=None,
    n_random_genes=None,
    seed=None,
):
    """Select the reference closest to the input panel's gene correlations.

    Accepts AnnData, an h5ad path, or a dataset name. `genes` defaults to all
    input genes; `reference_genes` restricts reference genes by mapped names.
    `genes="random"` samples `n_random_genes` input genes without replacement.
    `seed` controls gene and cell sampling; None uses fresh randomness.
    Samples up to `n_cells`, removes constant genes, and optionally retains
    `top_n_var_genes` mean-normalized HVGs (Seurat; requires log1p expression).

    Returns a dict with reference/score, panel correlation, all input gene names,
    and `common_gene_names` for retained reference genes aligned with the outputs:
    log counts (cells x genes), similarity (genes x genes), loadings (genes x d),
    and gene-filtering metadata. `verbose` prints progress; `output_dir` saves
    panel correlations and loadings as CSVs.
    """
    if verbose:
        print("Loading input dataset...", flush=True)
    samples = _load_samples(adata, timepoint)
    input_gene_names = list(adata.var_names) if isinstance(adata, ad.AnnData) else list(next(iter(samples.values())).var_names)
    if genes is None:
        genes = list(next(iter(samples.values())).var_names)
    elif isinstance(genes, str) and genes == "random":
        if n_random_genes is None:
            raise ValueError('Set n_random_genes when genes="random".')
        genes = np.random.default_rng(seed).choice(
            next(iter(samples.values())).var_names, size=n_random_genes, replace=False,
        )
    genes = list(genes)
    if len(genes) < 2 or len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("Provide at least two distinct panel genes.")
    if d < 1 or n_cells < 1:
        raise ValueError("d and n_cells must be positive.")
    if reference_genes is not None:
        reference_genes = list(reference_genes)
    if verbose:
        print(f"Computing input correlations for {len(genes)} panel genes...", flush=True)
    input_correlation = _panel_expression(samples, genes, input_layer).corr()
    pairs = np.triu_indices(len(genes), k=1)
    input_pairs = input_correlation.to_numpy()[pairs]
    if not np.isfinite(input_pairs).all():
        raise ValueError("Input panel correlations are undefined; check for constant genes.")

    references = REFERENCE_DATASETS if reference_datasets is None else reference_datasets
    best = None
    best_samples = None
    for reference in tqdm(references, desc="Reference datasets", disable=not verbose):
        if verbose:
            print(f"Loading reference {reference['dataset']} (timepoint={reference.get('timepoint')})...", flush=True)
        reference_samples = _load_samples(reference["dataset"], reference.get("timepoint"))
        excluded_reference_genes = []
        if reference_genes is not None:
            if verbose:
                print("Mapping reference genes...", flush=True)
            aliases = {
                sample: _reference_aliases(adata)
                for sample, adata in tqdm(reference_samples.items(), desc="Mapping samples", disable=not verbose, leave=False)
            }
            shared_genes = [gene for gene in tqdm(reference_genes, desc="Matching genes", disable=not verbose, leave=False) if all(
                _gene_key(gene) in mapping for mapping in aliases.values()
            )]
            excluded_reference_genes = [gene for gene in tqdm(reference_genes, desc="Checking excluded genes", disable=not verbose, leave=False) if gene not in shared_genes]
            if not shared_genes:
                raise ValueError(f"No shared mapped genes for {reference['dataset']}.")
            filtered_samples = {}
            for sample, adata in tqdm(reference_samples.items(), desc="Filtering samples", disable=not verbose, leave=False):
                selected_names = [aliases[sample][_gene_key(gene)] for gene in shared_genes]
                if len(set(selected_names)) != len(selected_names):
                    raise ValueError("reference_genes must resolve to distinct reference genes.")
                filtered = adata[:, selected_names].copy()
                filtered.var["reference_gene_id"] = filtered.var["gene_id"].to_numpy()
                filtered.var_names = shared_genes
                filtered_samples[sample] = filtered
            reference_samples = filtered_samples
            if verbose:
                print(f"Retained {len(shared_genes)} reference genes; excluded {len(excluded_reference_genes)}.", flush=True)
        if verbose:
            print("Computing reference panel correlations...", flush=True)
        expression = _panel_expression(
            reference_samples, genes, reference.get("layer"),
        )
        correlation = expression.corr()
        reference_pairs = correlation.to_numpy()[pairs]
        if not np.isfinite(reference_pairs).all():
            raise ValueError(f"Undefined panel correlations for reference {reference['dataset']}.")
        score = float(np.mean(np.abs(input_pairs - reference_pairs)))
        if verbose:
            print(f"Reference score: {score:.6g}", flush=True)
        if best is None or score < best["score"]:
            best = {
                "reference": reference, "score": score, "correlation": correlation,
                "excluded_reference_genes": excluded_reference_genes,
            }
            best_samples = reference_samples
    if best is None:
        raise ValueError("The reference dataset list is empty.")

    reference_samples = list(best_samples.values())
    selected = reference_samples[0] if len(reference_samples) == 1 else ad.concat(
        reference_samples, join="inner", merge="same",
    )
    sample_size = min(n_cells, selected.n_obs)
    indices = np.random.default_rng(seed).choice(selected.n_obs, size=sample_size, replace=False)
    expression = selected[indices, :].to_df(layer=best["reference"].get("layer"))
    variable_genes = expression.max(axis=0) > expression.min(axis=0)
    best["constant_genes"] = expression.columns[~variable_genes].tolist()
    expression = expression.loc[:, variable_genes]
    selected = selected[:, variable_genes.to_numpy()]
    if top_n_var_genes is not None:
        import scanpy as sc

        if verbose:
            print(f"Selecting top {top_n_var_genes} mean-normalized highly variable genes...", flush=True)
        hvg = sc.pp.highly_variable_genes(
            ad.AnnData(expression), n_top_genes=top_n_var_genes,
            flavor="seurat", inplace=False,
        )
        retained_genes = hvg["dispersions_norm"].nlargest(top_n_var_genes).index
        aliases = _reference_aliases(selected)
        panel_ids = {gene: aliases.get(_gene_key(gene)) for gene in genes}
        missing_genes = [gene for gene, identifier in panel_ids.items() if identifier not in retained_genes]
        if missing_genes:
            print(f"Warning: panel genes outside the top {top_n_var_genes} HVGs: {', '.join(missing_genes)}. Adding those available after constant-gene filtering.", flush=True)
        retained_genes = retained_genes.union(
            [identifier for identifier in panel_ids.values() if identifier is not None],
            sort=False,
        )
        expression = expression.loc[:, expression.columns.isin(retained_genes)]
        selected = selected[:, expression.columns]
    if selected.n_vars < 2 or d > min(sample_size, selected.n_vars):
        raise ValueError("Need at least two variable genes, and d cannot exceed the sampled cell or variable gene count.")
    best["gene_names"] = input_gene_names
    best["common_gene_names"] = list(selected.var_names)
    best["reference_gene_ids"] = (
        selected.var["reference_gene_id"].tolist()
        if "reference_gene_id" in selected.var else selected.var["gene_id"].tolist()
    )
    reference_name = best["reference"]["dataset"]
    if isinstance(reference_name, ad.AnnData):
        reference_name = "in_memory_reference"
    reference_timepoint = best["reference"].get("timepoint")
    best["gene_set_name"] = str(reference_name)
    if reference_timepoint is not None:
        best["gene_set_name"] += f":{reference_timepoint}"
    best["gene_log_counts"] = expression
    if verbose:
        print(f"Selected {best['gene_set_name']}: {sample_size} cells, {selected.n_vars} variable genes; computing similarity...", flush=True)
    best["gene_similarity"] = pd.DataFrame(
        np.corrcoef(expression.to_numpy(), rowvar=False, dtype=np.float32),
        index=pd.Index(selected.var_names, name="gene_id"),
        columns=selected.var_names,
    )
    model = FactorAnalysis(n_components=d, copy=True)
    if verbose:
        print(f"Fitting factor analysis with {d} components...", flush=True)
    model.fit(expression)
    best["loadings"] = pd.DataFrame(
        model.components_.T,
        index=pd.Index(selected.var_names, name="gene_id"),
        columns=[f"factor_{i + 1}" for i in range(d)],
    )
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        best["correlation"].to_csv(output_dir / "gene_correlation.csv", index_label="gene")
        best["loadings"].to_csv(output_dir / "gene_loadings.csv")
    if verbose:
        print("Dataset selection complete.", flush=True)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a reference by panel correlation differences and fit its retained gene set.")
    parser.add_argument("dataset", help="Input h5ad path or registered dataset name")
    parser.add_argument("--genes", nargs="+", help="Panel genes; defaults to all input genes")
    parser.add_argument(
        "--reference-genes", nargs="+",
        help="Restrict reference genes; must include the panel. Defaults to all reference genes.",
    )
    parser.add_argument("--timepoint", help="Input dataset timepoint")
    parser.add_argument("--input-layer", help="Input expression layer; defaults to X")
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--n-cells", type=int, default=1000)
    parser.add_argument(
        "--references", nargs="+", metavar="DATASET[:TIMEPOINT]",
        help="Reference loader names or h5ad paths; defaults to TOME 7.5. Configure layers and gene maps in REFERENCE_DATASETS or through the function.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "preprocessing" / "results" / "dataset_selection")
    args = parser.parse_args()
    references = None
    if args.references is not None:
        references = []
        for item in args.references:
            dataset, separator, stage = item.partition(":")
            reference = next(
                (dict(entry) for entry in REFERENCE_DATASETS if entry["dataset"] == dataset),
                {"dataset": dataset},
            )
            if separator:
                reference["timepoint"] = stage
            references.append(reference)
    result = select_dataset(
        args.dataset, genes=args.genes, reference_datasets=references,
        d=args.d, n_cells=args.n_cells, timepoint=args.timepoint,
        input_layer=args.input_layer, output_dir=args.output_dir,
        reference_genes=args.reference_genes,
    )
    print(f"Selected reference: {result['reference']['dataset']} (timepoint={result['reference'].get('timepoint')})")
    print(f"Mean absolute pairwise correlation difference: {result['score']:.6g}")
    print(f"Correlation matrix: {args.output_dir / 'gene_correlation.csv'}")
    print(f"Gene loadings: {args.output_dir / 'gene_loadings.csv'}")
