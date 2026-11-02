import argparse
from pathlib import Path
import re
import sys

import anndata as ad
import numpy as np
import pandas as pd

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
    adata, genes=None, reference_datasets=None, d=256, n_cells=1000,
    timepoint=None, input_layer=None, output_dir=None, reference_genes=None,
    verbose=False, top_n_var_genes=None, n_random_genes=None, seed=None,
    *, data_dir="/insomnia001/depts/morpheus/projects/hackathon/Datasets",
    selection_score="ot_cluster_ari", scores=None, score_params=None,
    embedding_methods=("factor_analysis",), embedder_params=None,
    embedding_store=None, joint_top_k=None, return_gene_similarity=False,
    input_units=None, descriptor="", coords=None, min_panel_coverage=1.0,
):
    """Select a local reference with panelprep and return optional gene features.

    Omitted references use REFERENCE_DATASETS. Supply a dataset directory, file,
    registered local name, AnnData, or a list of reference dictionaries. Local
    names resolve under data_dir; this function never downloads source data.
    Directory samples are independent candidates, with converted H5AD preferred.

    selection_score determines the winner using its registered orientation;
    scores adds diagnostics. Default OT cluster ARI avoids gene correlations.
    Only finite, successful scores meeting min_panel_coverage can win.

    embedding_methods accepts registry names, "all", or [] (selection only).
    Separate embedders use the winner; joint methods use eligible references
    (optionally joint_top_k). External/prior features load from embedding_store
    when cache-only; unavailable/deferred sources are reported explicitly.
    d is requested dimensionality; methods may return a smaller effective rank.

    Inputs must have a known expression layer or explicitly declared units:
    counts, logcounts (natural log1p counts), or log1p_cp10k. Normalization uses
    the complete source gene axis before panel/output restriction. No QC runs.

    Returns reference, score, score_results, ranking, embeddings (GeneEmbedding
    objects), blocked_embeddings, load_failures, common_gene_names and aligned
    gene_log_counts. loadings is provided when factor_analysis succeeds.
    gene_similarity is computed only when return_gene_similarity=True.
    """
    from preprocessing.reference_selection import select_reference

    return select_reference(
        adata, genes=genes,
        references=REFERENCE_DATASETS if reference_datasets is None else reference_datasets,
        d=d, n_cells=n_cells, timepoint=timepoint, input_layer=input_layer,
        output_dir=output_dir, reference_genes=reference_genes, verbose=verbose,
        top_n_var_genes=top_n_var_genes, n_random_genes=n_random_genes, seed=seed,
        data_dir=data_dir, selection_score=selection_score, scores=scores,
        score_params=score_params, embedding_methods=embedding_methods,
        embedder_params=embedder_params, embedding_store=embedding_store,
        joint_top_k=joint_top_k, return_gene_similarity=return_gene_similarity,
        input_units=input_units, descriptor=descriptor, coords=coords,
        min_panel_coverage=min_panel_coverage,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select a local reference with panelprep and export gene embeddings.")
    parser.add_argument("dataset", help="Query H5AD path or local dataset name")
    parser.add_argument("--genes", nargs="+")
    parser.add_argument("--reference-genes", nargs="+")
    parser.add_argument("--timepoint")
    parser.add_argument("--input-layer")
    parser.add_argument("--input-units", choices=["counts", "logcounts", "log1p_cp10k"])
    parser.add_argument("--d", type=int, default=256)
    parser.add_argument("--n-cells", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--references", nargs="+", help="Directory, files, or DATASET[:TIMEPOINT]; omitted uses REFERENCE_DATASETS")
    parser.add_argument("--data-dir", default="/insomnia001/depts/morpheus/projects/hackathon/Datasets")
    parser.add_argument("--selection-score", default="ot_cluster_ari")
    parser.add_argument("--scores", nargs="*", default=[])
    parser.add_argument("--embedding-methods", nargs="*", default=["factor_analysis"], help="Names, all, or no values for selection only")
    parser.add_argument("--embedding-store", type=Path)
    parser.add_argument("--joint-top-k", type=int)
    parser.add_argument("--return-gene-similarity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "preprocessing/results/dataset_selection")
    args = parser.parse_args()
    references = None
    if args.references is not None:
        references = []
        for item in args.references:
            if Path(item).exists():
                references.append(item)
            else:
                dataset, separator, stage = item.partition(":")
                reference = next((dict(entry) for entry in REFERENCE_DATASETS if entry["dataset"] == dataset), {"dataset": dataset})
                if separator:
                    reference["timepoint"] = stage
                references.append(reference)
    result = select_dataset(
        args.dataset, genes=args.genes, reference_datasets=references,
        d=args.d, n_cells=args.n_cells, timepoint=args.timepoint,
        input_layer=args.input_layer, input_units=args.input_units,
        output_dir=args.output_dir, reference_genes=args.reference_genes,
        data_dir=args.data_dir, seed=args.seed, selection_score=args.selection_score,
        scores=args.scores, embedding_methods="all" if args.embedding_methods == ["all"] else args.embedding_methods,
        embedding_store=args.embedding_store, joint_top_k=args.joint_top_k,
        return_gene_similarity=args.return_gene_similarity, verbose=True,
    )
    print(f"Selected reference: {result['gene_set_name']}")
    print(f"{result['selection_score']}: {result['score']:.6g}")
    print(f"Embeddings: {list(result['embeddings'])}")
    print(f"Blocked embeddings: {result['blocked_embeddings']}")
    print(f"Outputs: {args.output_dir}")
