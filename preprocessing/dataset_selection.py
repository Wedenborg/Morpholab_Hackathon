import argparse
import gzip
from pathlib import Path
import re
import sys
from urllib.request import urlretrieve

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from dataset.load_dataset import load_dataset


REFERENCE_DATASETS = [
    {
        "dataset": "tome",
        "timepoint": "7.5",
        "layer": "logcounts",
        "gene_annotation_url": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz",
        "gene_map": {
            "SHH": "ENSMUSG00000002633",
            "SOX2": "ENSMUSG00000074637",
            "HOXA1": "ENSMUSG00000029844",
            "PMP22": "ENSMUSG00000018217",
        },
    },
]


def _load_samples(dataset, timepoint=None):
    if isinstance(dataset, ad.AnnData):
        return {"input": dataset}
    if Path(dataset).is_file():
        return {Path(dataset).stem: ad.read_h5ad(dataset)}
    return load_dataset(str(dataset), timepoint=timepoint, output="anndata")


def _panel_expression(samples, genes, layer=None, gene_map=None):
    gene_map = {key.upper(): value for key, value in (gene_map or {}).items()}
    frames = []
    for adata in samples.values():
        names = pd.Series(adata.var_names, index=adata.var_names.str.upper())
        if not names.index.is_unique:
            raise ValueError("Gene identifiers must be unique, ignoring case.")
        identifiers = [
            gene.upper() if gene.upper() in names.index
            else gene_map.get(gene.upper(), gene).upper()
            for gene in genes
        ]
        expression = adata[:, names.loc[identifiers].tolist()].to_df(layer=layer)
        expression.columns = genes
        frames.append(expression)
    return pd.concat(frames, ignore_index=True)


def _gene_key(name):
    name = str(name).upper()
    return re.sub(r"\.\d+$", "", name) if name.startswith("ENS") else name


def _reference_aliases(adata, reference, requested):
    aliases = {}

    def add_alias(symbol, identifier):
        aliases.setdefault(_gene_key(symbol), set()).add(identifier)

    identifiers = {_gene_key(name): name for name in adata.var_names}
    for name in adata.var_names:
        add_alias(name, name)
    for column in ("gene_symbol", "gene_symbols", "symbol"):
        if column in adata.var:
            for identifier, symbol in adata.var[column].dropna().items():
                add_alias(symbol, identifier)
    for symbol, identifier in reference.get("gene_map", {}).items():
        if _gene_key(identifier) in identifiers:
            add_alias(symbol, identifiers[_gene_key(identifier)])
    if reference.get("gene_annotation_url") and any(_gene_key(gene) not in aliases for gene in requested):
        url = reference["gene_annotation_url"]
        annotation = ROOT / "dataset" / "data" / "annotations" / url.rsplit("/", 1)[1]
        annotation.parent.mkdir(parents=True, exist_ok=True)
        if not annotation.exists():
            partial = annotation.with_name(annotation.name + ".part")
            urlretrieve(url, partial)
            partial.replace(annotation)
        with gzip.open(annotation, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) != 9 or fields[2] != "gene":
                    continue
                attributes = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
                identifier = identifiers.get(_gene_key(attributes["gene_id"]))
                if identifier is not None:
                    add_alias(attributes["gene_name"], identifier)
    return {symbol: next(iter(ids)) for symbol, ids in aliases.items() if len(ids) == 1}


def select_dataset(
    user_dataset,
    genes=None,
    reference_datasets=None,
    d=256,
    n_cells=1000,
    timepoint=None,
    input_layer=None,
    output_dir=None,
    reference_genes=None,
):
    """Rank references on the input panel; fit the selected reference gene set.

    user_dataset is an AnnData object, an h5ad path, or a loader dataset name.
    genes defaults to all genes in the first input sample. References are dicts
    with dataset, optional timepoint, layer, and symbol-to-ID gene_map fields;
    dataset accepts the same inputs as user_dataset. All references must contain
    the complete panel. The score is mean absolute Pearson-correlation difference
    over unique off-diagonal pairs, using all cells and the specified layers.
    reference_genes optionally restricts every reference sample before ranking
    and fitting, preserving the requested order. Names match var_names ignoring
    case or resolve through gene_map, var gene symbols, or gene_annotation_url.
    Keep the unambiguously mapped intersection across reference samples, labelled
    with the requested input names. The retained set must include the input panel.
    None starts with the full reference gene set. Genes constant in the sampled
    reference cells are removed before similarity and factor analysis.

    Return reference, score, correlation (panel genes x panel genes), and loadings
    (selected reference genes x d). Also return gene_set_name (dataset/timepoint),
    gene_names (ordered, aligned gene names), gene_log_counts (sampled cells x genes),
    and gene_similarity (retained-reference Pearson correlation, genes x genes).
    excluded_reference_genes reports unmapped/ambiguous genes; constant_genes
    reports genes removed for zero variance. reference_gene_ids preserves IDs.
    Factor analysis samples up to n_cells without
    replacement with seed 42, using the reference layer without variance scaling.
    If a reference pools samples, factor analysis uses their shared gene set.
    output_dir optionally saves gene_correlation.csv and gene_loadings.csv.
    """
    samples = _load_samples(user_dataset, timepoint)
    if genes is None:
        genes = list(next(iter(samples.values())).var_names)
    genes = list(genes)
    if len(genes) < 2 or len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("Provide at least two distinct panel genes.")
    if d < 1 or n_cells < 1:
        raise ValueError("d and n_cells must be positive.")
    if reference_genes is not None:
        reference_genes = list(reference_genes)
    input_correlation = _panel_expression(samples, genes, input_layer).corr()
    pairs = np.triu_indices(len(genes), k=1)
    input_pairs = input_correlation.to_numpy()[pairs]
    if not np.isfinite(input_pairs).all():
        raise ValueError("Input panel correlations are undefined; check for constant genes.")

    references = REFERENCE_DATASETS if reference_datasets is None else reference_datasets
    best = None
    best_samples = None
    for reference in references:
        reference_samples = _load_samples(reference["dataset"], reference.get("timepoint"))
        excluded_reference_genes = []
        if reference_genes is not None:
            aliases = {
                sample: _reference_aliases(adata, reference, reference_genes)
                for sample, adata in reference_samples.items()
            }
            shared_genes = [gene for gene in reference_genes if all(
                _gene_key(gene) in mapping for mapping in aliases.values()
            )]
            excluded_reference_genes = [gene for gene in reference_genes if gene not in shared_genes]
            if not shared_genes:
                raise ValueError(f"No shared mapped genes for {reference['dataset']}.")
            filtered_samples = {}
            for sample, adata in reference_samples.items():
                selected_names = [aliases[sample][_gene_key(gene)] for gene in shared_genes]
                if len(set(selected_names)) != len(selected_names):
                    raise ValueError("reference_genes must resolve to distinct reference genes.")
                filtered = adata[:, selected_names].copy()
                filtered.var["reference_gene_id"] = selected_names
                filtered.var_names = shared_genes
                filtered_samples[sample] = filtered
            reference_samples = filtered_samples
        expression = _panel_expression(
            reference_samples, genes, reference.get("layer"), reference.get("gene_map"),
        )
        correlation = expression.corr()
        reference_pairs = correlation.to_numpy()[pairs]
        if not np.isfinite(reference_pairs).all():
            raise ValueError(f"Undefined panel correlations for reference {reference['dataset']}.")
        score = float(np.mean(np.abs(input_pairs - reference_pairs)))
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
    indices = np.random.default_rng(42).choice(selected.n_obs, size=sample_size, replace=False)
    expression = selected[indices, :].to_df(layer=best["reference"].get("layer"))
    variable_genes = expression.max(axis=0) > expression.min(axis=0)
    best["constant_genes"] = expression.columns[~variable_genes].tolist()
    expression = expression.loc[:, variable_genes]
    selected = selected[:, variable_genes.to_numpy()]
    if selected.n_vars < 2 or d > min(sample_size, selected.n_vars):
        raise ValueError("Need at least two variable genes, and d cannot exceed the sampled cell or variable gene count.")
    best["gene_names"] = list(selected.var_names)
    best["reference_gene_ids"] = (
        selected.var["reference_gene_id"].tolist()
        if "reference_gene_id" in selected.var else list(selected.var_names)
    )
    reference_name = best["reference"]["dataset"]
    if isinstance(reference_name, ad.AnnData):
        reference_name = "in_memory_reference"
    reference_timepoint = best["reference"].get("timepoint")
    best["gene_set_name"] = str(reference_name)
    if reference_timepoint is not None:
        best["gene_set_name"] += f":{reference_timepoint}"
    best["gene_log_counts"] = expression
    best["gene_similarity"] = pd.DataFrame(
        np.corrcoef(expression.to_numpy(), rowvar=False, dtype=np.float32),
        index=pd.Index(selected.var_names, name="gene_id"),
        columns=selected.var_names,
    )
    model = FactorAnalysis(n_components=d, copy=True)
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
