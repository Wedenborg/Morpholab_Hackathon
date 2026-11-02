"""Filter one source sample and produce three explicitly described count layers.

See qc_normalization.md for representation choices and examples. Inputs are read
directly, without dataset.load_dataset's transformations or gene-name conversion.
"""

import argparse
import os
from pathlib import Path
import re
import tempfile

import anndata as ad
import numpy as np
from scipy import sparse


DATA_ROOT = Path('/insomnia001/depts/morpheus/projects/hackathon/Datasets')
LAYER_NAMES = ('counts', 'logcounts', 'logcounts_normalized')
PAPERS = {
    'seu-3d': 'https://doi.org/10.1016/j.cell.2025.05.035',
    'pijuan-sala': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6522369/',
}


def _counts(adata, dataset, allow_corrected_counts):
    if dataset == 'tome':
        source = 'layers/counts' if 'counts' in adata.layers else 'X'
        matrix = adata.layers['counts'] if 'counts' in adata.layers else adata.X
        kind = 'original_rna_counts'
    elif dataset == 'seu-3d':
        if not allow_corrected_counts:
            raise ValueError('SEU-3D stores likely SCT-corrected counts, not original RNA '
                             'counts. Explicitly set allow_corrected_counts=True to use them.')
        if 'counts' in adata.layers:
            source, matrix = 'layers/counts', adata.layers['counts']
        elif adata.raw is not None and adata.raw.var_names.equals(adata.var_names):
            source, matrix = 'raw/X', adata.raw.X
        else:
            raise ValueError('SEU-3D needs a counts layer or gene-aligned raw.X.')
        kind = 'likely_sct_corrected_counts'
    else:
        raise ValueError('Supported datasets: tome, seu-3d')
    counts = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    counts.sort_indices()
    if counts.shape != adata.shape or not counts.nnz:
        raise ValueError('Counts must match the input shape and contain expression.')
    if not np.isfinite(counts.data).all() or (counts.data < 0).any():
        raise ValueError('Counts must be finite and nonnegative.')
    if not np.equal(counts.data, np.floor(counts.data)).all():
        raise ValueError('Expected integer-valued stored counts; provenance also needs metadata.')
    totals = np.asarray(counts.sum(axis=1)).ravel()
    metadata = 'nCount_RNA' if dataset == 'tome' else 'nCount_SCT'
    if metadata not in adata.obs or not np.array_equal(totals, adata.obs[metadata].to_numpy()):
        raise ValueError(f'Count totals must exactly match source obs[{metadata!r}].')
    if dataset == 'seu-3d':
        expected = counts.copy()
        expected.data = np.log1p(expected.data)
        difference = sparse.csr_matrix(adata.X) - expected
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-6:
            raise ValueError('SEU-3D X does not match log1p(stored counts).')
    return counts, source, kind, metadata


def _log1p(matrix):
    result = matrix.copy()
    result.data = np.log1p(result.data)
    return result


def qc_normalize(adata, *, dataset, min_genes=None, min_gene_variance=0.0,
                 allow_corrected_counts=False, source_study=None, min_gene_cells=None):
    """Return a filtered copy; variance is population variance of log1p counts.

    Keep cells with at least min_genes detected genes and positive count totals.
    Keep genes with variance strictly greater than min_gene_variance after cell
    filtering. Normalize retained counts to 10,000 per cell, then natural log1p.
    X and raw retain their source representations, subject to AnnData slicing.
    """
    if dataset == 'seu-3d':
        source_study = 'seu-3d'
    if min_genes is None:
        if source_study not in PAPERS:
            raise ValueError('No verified paper default for this source study. '
                             'Specify source_study or an explicit min_genes override.')
        min_genes = 400 if source_study == 'seu-3d' else 1000
        cell_threshold_origin = PAPERS[source_study]
    else:
        cell_threshold_origin = 'user override'
    if min_gene_cells is None:
        min_gene_cells = 4 if dataset == 'seu-3d' else 1
    if isinstance(min_gene_cells, bool) or not isinstance(min_gene_cells, (int, np.integer)) or min_gene_cells < 1:
        raise ValueError('min_gene_cells must be a positive integer.')
    if isinstance(min_genes, bool) or not isinstance(min_genes, (int, np.integer)) or min_genes < 0:
        raise ValueError('min_genes must be a nonnegative integer.')
    if not np.isfinite(min_gene_variance) or min_gene_variance < 0:
        raise ValueError('min_gene_variance must be finite and nonnegative.')
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError('Input must contain cells and genes.')
    if 'qc_normalization' in adata.uns:
        raise ValueError('Input already has qc_normalization provenance; use the source file.')
    counts, source, kind, metadata = _counts(adata, dataset, allow_corrected_counts)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    detected = np.diff(counts.indptr)
    cell_metric = detected
    cell_metric_name = 'detected genes in original counts'
    if dataset == 'seu-3d':
        cell_metric_name = 'source obs/nFeature_RNA'
        if 'nFeature_RNA' not in adata.obs:
            raise ValueError('SEU-3D cell filtering needs original nFeature_RNA metadata.')
        cell_metric = adata.obs['nFeature_RNA'].to_numpy(dtype=float)
        if not np.isfinite(cell_metric).all() or (cell_metric < 0).any():
            raise ValueError('Invalid nFeature_RNA metadata.')
    cells = (cell_metric >= min_genes) & (totals > 0)
    if cells.sum() < 2:
        raise ValueError('At least two cells must remain to filter genes by variance.')
    kept_counts = counts[cells]
    logged = _log1p(kept_counts)
    means = np.asarray(logged.mean(axis=0)).ravel()
    variances = np.maximum(0, np.asarray(logged.power(2).mean(axis=0)).ravel() - means**2)
    # Detect exact constants separately to avoid cancellation leaving tiny variance.
    varying = (kept_counts.max(axis=0).toarray().ravel()
               > kept_counts.min(axis=0).toarray().ravel())
    gene_detection = np.asarray((kept_counts > 0).sum(axis=0)).ravel()
    genes = varying & (variances > min_gene_variance) & (gene_detection >= min_gene_cells)
    if not genes.any():
        raise ValueError('No genes remain after variance filtering.')
    filtered = kept_counts[:, genes].tocsr()
    retained_totals = np.asarray(filtered.sum(axis=1)).ravel()
    if (retained_totals == 0).any():
        raise ValueError('Gene filtering left zero-count cells; lower min_gene_variance.')
    result = adata[cells, genes].copy()
    result.layers.clear()
    result.layers['counts'] = filtered
    result.layers['logcounts'] = _log1p(filtered)
    normalized = filtered.multiply((10000.0 / retained_totals)[:, None]).tocsr()
    result.layers['logcounts_normalized'] = _log1p(normalized)
    result.obs['qc_total_counts_before_gene_filter'] = totals[cells]
    result.obs['qc_n_genes_before_gene_filter'] = detected[cells]
    result.obs['qc_cell_filter_n_genes'] = cell_metric[cells]
    result.obs['qc_total_counts_retained'] = retained_totals
    result.obs['qc_normalization_factor'] = 10000.0 / retained_totals
    result.var['qc_n_cells_expressing_before_cell_filter'] = np.asarray((counts > 0).sum(axis=0)).ravel()[genes]
    result.var['qc_n_cells_expressing'] = np.asarray((kept_counts > 0).sum(axis=0)).ravel()[genes]
    result.var['qc_variance_log1p_counts'] = variances[genes]
    result.uns['qc_normalization'] = {
        'dataset': dataset, 'counts_source': source, 'counts_kind': kind,
        'validated_total_metadata': metadata,
        'min_genes': int(min_genes), 'min_gene_variance': float(min_gene_variance),
        'cell_filter_metric': cell_metric_name, 'cell_threshold_origin': cell_threshold_origin,
        'source_study': source_study or 'unspecified', 'min_gene_cells': int(min_gene_cells),
        'variance_threshold_origin': 'analysis choice, not a paper QC threshold',
        'variance_scale': 'natural log1p of unnormalized stored counts; population variance',
        'variance_rule': 'strictly greater than threshold; exact constant genes excluded',
        'normalization_target': 10000.0,
        'normalization_gene_universe': 'genes retained after variance filtering',
        'input_cells': adata.n_obs, 'input_genes': adata.n_vars,
        'removed_cells': int((~cells).sum()), 'removed_genes': int((~genes).sum()),
        'source_layers': list(adata.layers), 'X': 'source X, subset to retained cells and genes',
        'layers': {'counts': kind, 'logcounts': 'natural log1p(counts)',
                   'logcounts_normalized': 'natural log1p(counts * 10000 / retained cell total)'},
    }
    return result


def write_qc_h5ad(adata, destination):
    """Save completely before publishing; preserve nullable source string metadata."""
    destination = Path(destination).resolve()
    if destination.suffix != '.h5ad' or destination.exists():
        raise ValueError('Output must be a new .h5ad path.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.qc-') as directory:
        temporary = Path(directory) / destination.name
        with ad.settings.override(allow_write_nullable_strings=True):
            adata.write_h5ad(temporary, compression='gzip')
        os.link(temporary, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('tome', 'seu-3d'))
    parser.add_argument('input', type=Path, help='Absolute source path or path relative to --data-root')
    parser.add_argument('output', type=Path, help='New output .h5ad path')
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--min-genes', type=int, help='Override the source-paper cell threshold')
    parser.add_argument('--source-study', choices=('pijuan-sala',),
                        help='Confirm the underlying study for a TOME input')
    parser.add_argument('--min-gene-cells', type=int)
    parser.add_argument('--min-gene-variance', type=float, default=0.0)
    parser.add_argument('--allow-corrected-counts', action='store_true')
    args = parser.parse_args()
    source = args.input if args.input.is_absolute() else args.data_root / args.input
    source_study = args.source_study
    if args.dataset == 'tome' and source_study is None:
        match = re.fullmatch(r'seurat_object_(E[\d.]+[ab]?)\.h5ad', source.name)
        # These portal timepoints are exclusively attributed to Pijuan-Sala.
        if match and match[1] in {'E6.75', 'E7.0', 'E7.25', 'E7.5', 'E7.75',
                                  'E8', 'E8.0', 'E8.25', 'E8.5a'}:
            source_study = 'pijuan-sala'
    destination = args.output.resolve()
    if destination == source.resolve() or destination.exists():
        parser.error('Output must be a new path distinct from the source.')
    if destination.suffix != '.h5ad':
        parser.error('Output must end in .h5ad.')
    result = qc_normalize(ad.read_h5ad(source), dataset=args.dataset,
                          min_genes=args.min_genes, min_gene_variance=args.min_gene_variance,
                          allow_corrected_counts=args.allow_corrected_counts,
                          source_study=source_study, min_gene_cells=args.min_gene_cells)
    result.uns['qc_normalization']['source_file'] = str(source.resolve())
    write_qc_h5ad(result, destination)
    print(f'Saved {result.n_obs:,} cells x {result.n_vars:,} genes to {destination}')


if __name__ == '__main__':
    main()
