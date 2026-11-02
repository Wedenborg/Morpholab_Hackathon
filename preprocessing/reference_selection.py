"""Local reference discovery, panelprep selection, and aligned gene features.

Shared inputs are read-only. R conversions are staged in a writable cache;
unknown raw tables require a dataset.json reader specification. No downloads
or QC run here. Normalization always precedes panel/gene restriction.
"""
from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread

from dataset.load_dataset import _discover, _read_sample, _preferred_path, _select_timepoint, _parse_timepoint
from .panelprep.embed import EMBEDDERS, run_embedders
from .panelprep.embed.store import EmbeddingStore
from .panelprep.integrations.morpholab import _coords_from, _guess_category  # registers factor_analysis
from .panelprep.normalize import gene_key, resolve_genes
from .panelprep.priors import PRIORS, build_priors, to_gene_embedding
from .panelprep.run import CACHED_ONLY
from .panelprep.schemas import CandidateDataset, PanelQuery
from .panelprep.scores import SCORES, run_scores

DEFAULT_DATA_DIR = Path('/insomnia001/depts/morpheus/projects/hackathon/Datasets')


def available_embedding_methods():
    """All expression, external, and prior registry names (including deferred sources)."""
    return sorted(set(EMBEDDERS.names()) | set(PRIORS.names()))


def discover_references(references, *, data_dir=DEFAULT_DATA_DIR):
    """Return (sample specifications, discovery issues), without reading matrices.

    A directory containing dataset.json is one collection; otherwise scan each
    child directory separately. Reader specifications use dataset/load_dataset.py's
    format, orientation, obs/var and spatial_columns fields. Each sample is a
    separate scoring candidate. Duplicate converted exports prefer H5AD.
    """
    root = Path(data_dir)
    entries, issues = [], {}
    if isinstance(references, (str, Path, dict, ad.AnnData)):
        references = [references]

    def directory(folder, options):
        manifest = folder / 'dataset.json'
        try:
            settings = json.loads(manifest.read_text()) if manifest.exists() else {}
            specs = settings.get('samples') or _discover(folder)
            # Public Pijuan-Sala export contains a PCA RDS alongside the actual
            # sparse expression matrix. The PCA file is not a reference matrix.
            atlas = next((p.parent for p in folder.rglob('raw_counts.mtx')
                          if (p.parent / 'genes.tsv').exists() and (p.parent / 'meta.tab').exists()), None)
            if not manifest.exists() and atlas is not None:
                specs = {'atlas': {'path': str(atlas), 'adapter': 'pijuan_sala', 'units': 'counts'}}
            metadata = sorted(folder.glob('GSE169210_*_single_cell_metadata.txt.gz'))
            if not manifest.exists() and metadata:
                specs = {p.name.removeprefix('GSE169210_').removesuffix('_single_cell_metadata.txt.gz'):
                         {'path': str(p), 'adapter': 'marsseq', 'units': 'counts',
                          'category': 'causal' if 'foxc12_dko' in p.name else 'single_cell'} for p in metadata}
            if not specs and folder.name == 'mdcla':
                issues[str(folder)] = 'OMIX014646-01.csv contains cell metadata only; expression matrix is not downloaded.'
                return
            requested = options.get('timepoint')
            if requested is not None and str(requested).lower() != 'all':
                specs = _select_timepoint(specs, _parse_timepoint(requested))
            if not specs:
                issues[str(folder)] = 'No readable samples; tables need dataset.json with format, orientation and metadata.'
            for sample, spec in specs.items():
                path = _preferred_path(folder / spec['path'])
                entry = {**settings.get('reference_options', {}), **options, **spec}
                entry.update(dataset=str(path), folder=str(folder), reader={**spec, 'path': str(path)},
                             name=options.get('name', f'{folder.name}/{sample}'))
                entries.append(entry)
            # Keep otherwise undiscovered tables/archives visible to the caller.
            if specs and not manifest.exists():
                extra = [p.name for p in folder.iterdir() if p.is_file()
                         and p.name.endswith(('.csv', '.txt.gz', '.mtx.gz', '.tar.gz'))]
                if extra:
                    issues[f'{folder}:unclassified'] = f'Unclassified tables/archives (not additional candidates): {extra}'
        except Exception as exc:
            issues[str(folder)] = f'{type(exc).__name__}: {exc}'

    for item in references:
        options = dict(item) if isinstance(item, dict) else {'dataset': item}
        value = options['dataset']
        if isinstance(value, ad.AnnData):
            entries.append({**options, 'name': options.get('name', f'in_memory_{len(entries)}')})
            continue
        path = Path(value).expanduser()
        if not path.exists():
            local = root / ('GSE278603_RAW' if str(value) == 'seu-3d' else str(value))
            if local.is_dir():
                path = local
            else:
                issues[str(value)] = f'No local file or dataset directory under {root}'
                continue
        if path.is_file():
            path = _preferred_path(path)
            entries.append({**options, 'dataset': str(path), 'name': options.get('name', str(path)),
                            'reader': {'path': str(path)}, 'folder': str(path.parent)})
        elif (path / 'dataset.json').exists():
            directory(path, options)
        else:
            # A study can contain nested sample folders, while a dataset root
            # contains studies. _discover handles nesting within each collection.
            if any(p.is_file() and p.suffix.lower() in ('.h5ad', '.loom', '.rds', '.h5td') for p in path.iterdir()):
                directory(path, options)
            else:
                children = [p for p in sorted(path.iterdir()) if p.is_dir() and p.name != 'annotations']
                if not children:
                    directory(path, options)
                for folder in children:
                    directory(folder, options)
    seen, unique = set(), []
    for entry in entries:
        value = entry['dataset']
        identity = id(value) if isinstance(value, ad.AnnData) else str(Path(value).resolve())
        # Explicit configurations of different layers remain distinct.
        key = (identity, entry.get('layer'), entry.get('units'))
        if key not in seen:
            unique.append(entry)
            seen.add(key)
    names = [e['name'] for e in unique]
    if len(set(names)) != len(names):
        raise ValueError('Reference names must be unique; provide distinct name fields.')
    return unique, issues


def _raw_sample(entry, max_obs, seed):
    """Read known raw atlas exports using their declared orientation/metadata."""
    path = Path(entry['dataset'])
    if entry['adapter'] == 'pijuan_sala':
        matrix = sparse.csr_matrix(mmread(path / 'raw_counts.mtx')).T.tocsr()
        var = pd.read_csv(path / 'genes.tsv', sep='\t', header=None, names=['gene_id', 'gene_symbol'])
        var.index = var.gene_id.astype(str)
        obs = pd.read_csv(path / 'meta.tab', sep='\t', index_col=0)
        # Metadata row order is documented as identical to matrix columns.
        return ad.AnnData(matrix, obs=obs, var=var)
    from .panelprep.io.loaders import _read_marsseq_plate
    metadata = pd.read_csv(path, sep='\t', index_col=0)
    plates = {p.name.split('_', 1)[1].removesuffix('.txt.gz'): p
              for p in path.parent.glob('GSM*.txt.gz')}
    metadata = metadata[metadata['plate'].astype(str).isin(plates)]
    if max_obs is not None and len(metadata) > max_obs:
        metadata = metadata.iloc[np.sort(np.random.default_rng(seed).choice(len(metadata), max_obs, replace=False))]
    blocks = []
    for plate, rows in metadata.groupby('plate', sort=True):
        counts = _read_marsseq_plate(str(plates[str(plate)]))
        wells = [str(cell).split('.', 1)[-1] for cell in rows.index]
        present = np.array([well in counts.index for well in wells])
        block = counts.loc[[well for well, ok in zip(wells, present) if ok]].copy()
        block.index = rows.index[present]
        blocks.append(block)
    if not blocks:
        raise ValueError('No metadata cells matched downloaded plate expression.')
    counts = pd.concat(blocks).fillna(0)
    return ad.AnnData(sparse.csr_matrix(counts.to_numpy()), obs=metadata.loc[counts.index],
                      var=pd.DataFrame(index=counts.columns))


def _read(entry, max_obs, seed):
    value = entry['dataset']
    if entry.get('adapter'):
        source = _raw_sample(entry, max_obs, seed)
    elif isinstance(value, ad.AnnData):
        source = value
    else:
        path = Path(value)
        if path.suffix.lower() in ('.rds', '.rda', '.rdata'):
            # Existing RDS readers persist their conversion beside the R file.
            # Stage it in our writable cache so the shared source stays intact.
            identity = f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
            digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
            cache = Path(entry.get('reader_cache_dir') or Path(__file__).parent / 'results/reference_readers') / digest
            cache.mkdir(parents=True, exist_ok=True)
            staged = cache / path.name
            if not staged.with_suffix('.h5ad').exists() and not staged.exists():
                with path.open('rb') as src, staged.with_suffix(staged.suffix + '.part').open('wb') as dst:
                    shutil.copyfileobj(src, dst)
                staged.with_suffix(staged.suffix + '.part').replace(staged)
            path = _preferred_path(staged)
            reader = {**entry.get('reader', {}), 'path': str(path)}
            if path.suffix.lower() == '.h5ad':
                reader['format'] = 'h5ad'
            source = _read_sample(cache, reader, verbose=False)
        else:
            source = ad.read_h5ad(path, backed='r') if path.suffix.lower() == '.h5ad' else _read_sample(
                Path(entry.get('folder', path.parent)), entry.get('reader', {'path': str(path)}), verbose=False)
    try:
        if source.n_obs == 0 or source.n_vars == 0:
            raise ValueError('Empty expression matrix.')
        idx = np.arange(source.n_obs)
        if max_obs is not None and len(idx) > max_obs:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_obs, replace=False))
        view = source[idx]
        data = view.to_memory() if source.isbacked else view.copy()
        if entry.get('coords') is not None:
            coordinates = np.asarray(entry['coords'], dtype=float)
            if coordinates.ndim != 2 or coordinates.shape[0] != source.n_obs or coordinates.shape[1] < 2:
                raise ValueError('Explicit coords must have one row per input cell and at least two columns.')
            data.obsm['spatial'] = coordinates[idx, :2]
        return data
    finally:
        if source.isbacked:
            source.file.close()


def _prepare(entry, max_obs, seed):
    """Read bounded cells, normalize on the full gene axis, resolve symbols."""
    data = _read(entry, max_obs, seed)
    layer = entry.get('layer')
    units = entry.get('units')
    if layer is None and units is None:
        if 'logcounts_normalized' in data.layers:
            layer, units = 'logcounts_normalized', units or 'log1p_cp10k'
        elif 'counts' in data.layers:
            layer, units = 'counts', units or 'counts'
        elif 'logcounts' in data.layers:
            layer, units = 'logcounts', units or 'logcounts'
        elif 'log1p' in data.uns or 'GSE278603' in str(entry['dataset']):
            units = units or 'logcounts'
    units = units or {'counts': 'counts', 'logcounts': 'logcounts',
                      'logcounts_normalized': 'log1p_cp10k'}.get(layer)
    if units not in ('counts', 'logcounts', 'log1p_cp10k'):
        # Count totals can establish the meaning of X without claiming that
        # stored SCT-corrected counts are original RNA counts.
        candidate = sparse.csr_matrix(data.X, dtype=np.float64)
        totals = np.asarray(candidate.sum(axis=1)).ravel()
        matched = [column for column in ('nCount_RNA', 'nCount_SCT', 'n_counts')
                   if column in data.obs and np.allclose(totals, data.obs[column].to_numpy(dtype=float), rtol=1e-5)]
        if matched:
            units = 'counts'
        else:
            raise ValueError('Declare units=counts, logcounts, or log1p_cp10k for this source; X units cannot be guessed.')
    matrix = data.X if layer is None else data.layers[layer]
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('Expression must contain finite, nonnegative values.')
    if units != 'log1p_cp10k':
        if units == 'logcounts':
            matrix.data = np.expm1(matrix.data)
        totals = np.asarray(matrix.sum(axis=1)).ravel()
        if not np.isfinite(totals).all() or (totals <= 0).any():
            raise ValueError('Nonfinite or zero-count cells; apply QC before selection.')
        matrix = matrix.multiply((10000 / totals)[:, None]).tocsr()
        matrix.data = np.log1p(matrix.data)
    symbols = pd.Series(data.var_names.astype(str), index=data.var_names)
    for column in ('gene_symbol', 'gene_symbols', 'symbol'):
        if column in data.var:
            supplied = data.var[column]
            symbols = supplied.fillna(symbols).astype(str)
            break
    keys = symbols.map(gene_key)
    keep = ~keys.duplicated(keep=False)
    if not keep.any():
        raise ValueError('No unambiguous gene identifiers remain.')
    names = keys[keep].tolist()
    frame = pd.DataFrame(matrix[:, keep.to_numpy()].toarray().astype('float32'),
                         index=data.obs_names, columns=names)
    coords = _coords_from(data)
    category = entry.get('category') or _guess_category(str(entry['dataset']), data)
    ids = data.var['gene_id'] if 'gene_id' in data.var else pd.Series(data.var_names, index=data.var_names)
    aliases = {}
    for index in np.flatnonzero(keep.to_numpy()):
        identifiers = [data.var_names[index], keys.iloc[index]]
        identifiers.extend(data.var.iloc[index][column] for column in (
            'gene_id', 'reference_gene_id', 'gene_symbol', 'gene_symbols', 'symbol') if column in data.var)
        for identifier in identifiers:
            if pd.notna(identifier):
                aliases.setdefault(gene_key(identifier), set()).add(keys.iloc[index])
    meta = {'layer': layer, 'source_units': units, 'normalization': 'log1p_cp10k',
            'source': str(entry['dataset']) if not isinstance(entry['dataset'], ad.AnnData) else 'in_memory',
            'reference_gene_ids': dict(zip(names, ids[keep].astype(str))),
            'gene_aliases': {key: next(iter(values)) for key, values in aliases.items() if len(values) == 1},
            'qc_normalization': dict(data.uns.get('qc_normalization', {})),
            'expression_kind': entry.get('expression_kind', 'imputed' if 'imputed' in str(entry['name']).lower() else 'measured_or_unspecified')}
    return CandidateDataset(name=entry['name'], category=category, expression=frame,
                            coords=coords, obs=data.obs.copy(), units='log1p_cp10k',
                            obs_unit=entry.get('obs_unit', 'cell'), meta=meta)


def _resolve_columns(genes, dataset):
    aliases = dataset.meta.get('gene_aliases', {})
    names = [aliases.get(gene_key(g), gene_key(g)) for g in genes]
    available = set(dataset.expression.columns)
    resolved = [name if name in available else None for name in names]
    return resolved, [g for g, name in zip(genes, resolved) if name is None]


def _aligned_embedding(embedding, genes, aliases=None):
    embedding.matrix.index = embedding.matrix.index.map(gene_key)
    aliases = aliases or {}
    index = set(embedding.matrix.index)
    embedding.matrix = embedding.matrix.loc[~embedding.matrix.index.duplicated(keep=False)].reindex(
        [gene_key(g) if gene_key(g) in index else aliases.get(gene_key(g), gene_key(g)) for g in genes])
    embedding.matrix.index = pd.Index(genes, name='gene')
    return embedding


def select_reference(adata, *, genes, references, d, n_cells, timepoint,
                     input_layer, output_dir, reference_genes, verbose,
                     top_n_var_genes, n_random_genes, seed, data_dir,
                     selection_score, scores, score_params, embedding_methods,
                     embedder_params, embedding_store, joint_top_k,
                     return_gene_similarity, input_units, descriptor, coords,
                     min_panel_coverage):
    seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
    if d < 1 or n_cells < 1 or not 0 < min_panel_coverage <= 1:
        raise ValueError('d/n_cells must be positive and min_panel_coverage must be in (0, 1].')
    if joint_top_k is not None and joint_top_k < 1:
        raise ValueError('joint_top_k must be positive or None.')
    requested = available_embedding_methods() if embedding_methods == 'all' else (
        [embedding_methods] if isinstance(embedding_methods, str) else list(embedding_methods))
    unknown = set(requested) - set(available_embedding_methods())
    if unknown:
        raise ValueError(f'Unknown embedding methods: {sorted(unknown)}')
    score_names = list(dict.fromkeys([selection_score] + list(scores or [])))
    for name in score_names:
        SCORES[name]  # fail early for configuration errors
    entries, issues = discover_references(references, data_dir=data_dir)
    query_entries, query_issues = discover_references(
        {'dataset': adata, 'timepoint': timepoint, 'layer': input_layer, 'units': input_units}, data_dir=data_dir)
    if not query_entries:
        raise ValueError(f'Cannot load query: {query_issues}')
    if len(query_entries) != 1:
        raise ValueError('Query must resolve to one sample; pass an individual file or AnnData.')
    query_source = query_entries[0]['dataset']
    if not isinstance(query_source, ad.AnnData):
        query_path = Path(query_source).resolve()
        entries = [entry for entry in entries if isinstance(entry['dataset'], ad.AnnData)
                   or Path(entry['dataset']).resolve() != query_path]
    query_entries[0]['coords'] = coords
    query_data = _prepare(query_entries[0], n_cells, seed)
    original_genes = list(adata.var_names) if isinstance(adata, ad.AnnData) else query_data.genes
    if isinstance(genes, str) and genes == 'random':
        if n_random_genes is None:
            raise ValueError('Set n_random_genes when genes="random".')
        genes = np.random.default_rng(seed).choice(query_data.genes, size=n_random_genes, replace=False).tolist()
    elif genes is None:
        genes = query_data.genes
    else:
        genes = list(genes)
    if len(genes) < 2 or len({gene_key(g) for g in genes}) != len(genes):
        raise ValueError('Provide at least two distinct panel genes.')
    resolved, missing = _resolve_columns(genes, query_data)
    if missing:
        raise ValueError(f'Panel genes absent from query: {missing}')
    panel = query_data.expression.loc[:, resolved].copy()
    panel.columns = genes
    if not np.isfinite(panel.to_numpy()).all():
        raise ValueError('Query panel contains nonfinite expression.')
    query_coords = query_data.coords
    # Coordinates are unused by nonspatial scores; NaNs explicitly mark absence.
    query = PanelQuery(panel, query_coords if query_coords is not None else np.full((len(panel), 2), np.nan),
                       descriptor, units=query_data.units, obs=query_data.obs)
    if query_coords is None and any(SCORES[name].meta.get('needs_coords') for name in score_names):
        raise ValueError('Spatial scores require query coordinates in AnnData.obsm or coords.')
    # Only the panel is retained for scoring; full expression is reloaded for
    # the winner and (when requested) the ranked views for joint methods.
    del query_data
    tables = []
    for entry in entries:
        try:
            candidate = _prepare(entry, n_cells, seed)
            resolved, _ = _resolve_columns(genes, candidate)
            candidate.expression = candidate.expression.loc[:, [name for name in resolved if name is not None]].copy()
            candidate.expression.columns = [g for g, name in zip(genes, resolved) if name is not None]
            table = run_scores(query, [candidate], score_names=score_names, genes=genes,
                               params=score_params, seed=seed, max_obs=n_cells)
            tables.append(table)
            if verbose:
                print(f"Scored {entry['name']}", flush=True)
        except Exception as exc:
            issues[entry['name']] = f'{type(exc).__name__}: {exc}'
    results = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    if results.empty:
        raise ValueError(f'No references could be scored: {issues}')
    ranking = results.loc[results.score == selection_score].copy()
    ranking['panel_coverage'] = ranking.n_genes_used / ranking.n_genes_requested
    ranking['eligible'] = (ranking.status == 'ok') & np.isfinite(ranking.value) & (ranking.panel_coverage >= min_panel_coverage)
    ascending = SCORES[selection_score].meta['orientation'] == 'lower_is_better'
    ranking = ranking.sort_values(['eligible', 'value', 'dataset'], ascending=[False, ascending, True], kind='stable').reset_index(drop=True)
    eligible = ranking.loc[ranking.eligible, 'dataset'].tolist()
    if not eligible:
        raise ValueError(f'No eligible reference for {selection_score}; scores: {ranking.to_dict("records")}; load issues: {issues}')
    lookup = {e['name']: e for e in entries}
    winner = eligible[0]
    selected = _prepare(lookup[winner], n_cells, seed)
    expression = selected.expression
    excluded = []
    if reference_genes is not None:
        output_genes = list(reference_genes)
        resolved, excluded = _resolve_columns(output_genes, selected)
        output_genes = [g for g, hit in zip(output_genes, resolved) if hit is not None]
        expression = expression.loc[:, [hit for hit in resolved if hit is not None]].copy()
        expression.columns = output_genes
    constant = expression.columns[expression.max() == expression.min()].tolist()
    expression = expression.drop(columns=constant)
    if top_n_var_genes is not None:
        import scanpy as sc
        hvg = sc.pp.highly_variable_genes(ad.AnnData(expression), n_top_genes=top_n_var_genes,
                                         flavor='seurat', inplace=False)
        retained = set(hvg.dispersions_norm.nlargest(top_n_var_genes).index)
        retained.update(g for g in expression if gene_key(g) in {gene_key(p) for p in genes})
        expression = expression.loc[:, expression.columns.isin(retained)]
    if expression.shape[1] < 2:
        raise ValueError('The selected reference has fewer than two retained variable genes.')
    output_genes = expression.columns.tolist()
    if len({gene_key(g) for g in output_genes}) != len(output_genes):
        raise ValueError('reference_genes must resolve to distinct genes.')
    resolved_ids, _ = _resolve_columns(output_genes, selected)
    if len(set(resolved_ids)) != len(resolved_ids):
        raise ValueError('reference_genes must resolve to distinct genes.')
    selected.expression = expression
    aliases = selected.meta['gene_aliases']
    embeddings, blocked = {}, {}
    store = EmbeddingStore(embedding_store) if isinstance(embedding_store, (str, Path)) else embedding_store
    params = embedder_params or {}
    joint = None
    for method in requested:
        try:
            if method in CACHED_ONLY:
                hits = [] if store is None else [n for n in store.names() if n == method or n.startswith(method + '::')]
                if not hits:
                    blocked[method] = 'Not in embedding_store. Build with preprocessing/scripts/build_embedding_store.py first.'
                    continue
                embeddings.update({hit: _aligned_embedding(store.get(hit), output_genes, aliases) for hit in hits})
                continue
            if method in EMBEDDERS:
                mode = EMBEDDERS[method].meta.get('mode', 'joint')
                views = {winner: selected}
                if mode == 'joint':
                    if joint is None:
                        joint = {winner: selected}
                        for name in eligible[1:joint_top_k]:
                            try:
                                view = _prepare(lookup[name], n_cells, seed)
                                columns, _ = _resolve_columns(output_genes, view)
                                view.expression = view.expression.loc[:, [c for c in columns if c is not None]].copy()
                                view.expression.columns = [g for g, c in zip(output_genes, columns) if c is not None]
                                joint[name] = view
                            except Exception as exc:
                                issues[f'joint:{name}'] = str(exc)
                    views = joint
                    if len(views) < 2:
                        blocked[method] = 'Requires at least two eligible references.'
                        continue
                got = run_embedders(views, embedder_names=[method], genes=output_genes, dim=d,
                                    params=params, seed=seed, max_obs=n_cells)
                embeddings.update(got)
            else:
                hits = [] if store is None else [n for n in store.names() if n == method or n.startswith(method + '::')]
                if hits:
                    embeddings.update({hit: _aligned_embedding(store.get(hit), output_genes, aliases) for hit in hits})
                    continue
                pf = build_priors(output_genes, sources=[method], dim=d, offline=True,
                                  datasets={winner: selected}, rng=np.random.default_rng(seed),
                                  params={method: params.get(method, {})})[method]
                embeddings[method] = to_gene_embedding(pf)
        except Exception as exc:
            blocked[method] = f'{type(exc).__name__}: {exc}'
    for name, emb in embeddings.items():
        _aligned_embedding(emb, output_genes, aliases)
        status = str(emb.provenance.get('status', 'ok'))
        if status.startswith(('blocked', 'deferred')) or not np.isfinite(emb.matrix.to_numpy()).any():
            blocked[name] = emb.provenance.get('reason', status)
    result = dict(reference=lookup[winner], score=float(ranking.iloc[0].value), selection_score=selection_score,
                  score_results=results, ranking=ranking, load_failures=issues, embeddings=embeddings,
                  blocked_embeddings=blocked, gene_names=original_genes, common_gene_names=output_genes,
                  reference_gene_ids=[selected.meta['reference_gene_ids'][key] for key in resolved_ids],
                  gene_set_name=winner, gene_log_counts=expression, constant_genes=constant,
                  excluded_reference_genes=excluded, panel_genes=genes, seed=seed,
                  reference_metadata=selected.meta)
    factor_key = f'factor_analysis::{winner}'
    if factor_key in embeddings:
        result['loadings'] = embeddings[factor_key].matrix
    if return_gene_similarity:
        result['gene_similarity'] = expression.corr().astype('float32')
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results.to_csv(out / 'score_results.csv', index=False)
        ranking.to_csv(out / 'reference_ranking.csv', index=False)
        expression.to_csv(out / 'gene_log_counts.csv')
        (out / 'selection_status.json').write_text(json.dumps({
            'selected': winner, 'selection_score': selection_score, 'seed': seed,
            'load_failures': issues, 'blocked_embeddings': blocked,
            'reference_metadata': selected.meta}, indent=2, default=str))
        for i, (name, embedding) in enumerate(embeddings.items()):
            embedding.to_parquet(out / f'embedding_{i}_{name.split("::")[0]}.parquet')
        (out / 'embedding_inventory.json').write_text(json.dumps({name: {
            'path': f'embedding_{i}_{name.split("::")[0]}.parquet',
            'shape': list(emb.matrix.shape), 'provenance': emb.provenance}
            for i, (name, emb) in enumerate(embeddings.items())}, indent=2, default=str))
        if 'loadings' in result:
            result['loadings'].to_csv(out / 'gene_loadings.csv')
        if return_gene_similarity:
            result['gene_similarity'].to_csv(out / 'gene_similarity.csv')
    return result
