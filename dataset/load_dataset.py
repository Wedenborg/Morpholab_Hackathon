import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import warnings
from urllib.parse import unquote, urlparse
from urllib.request import urlretrieve
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATASET_NAMES = (
    "seu-3d", "mosta", "harland", "lohoff", "egastrulation",
    "wang-gastrulation", "peng-epiblast", "embryo-slide-seq", "spateo",
    "most", "embryo-visium", "dbit-seq", "mpsta", "gut-development",
    "endothelial", "embryo-spatial-atac-visium", "visium-hd-embryo",
    "xenium-neonatal", "tome", "pijuan-sala", "extended-mouse-atlas",
    "somite-atlas", "mittnenzweig", "omg", "moca", "mela", "nextcell",
    "gut-endoderm", "scialdone", "tyser-heart", "hand2", "mesp1",
    "embryo-multiome", "scnmt-gastrulation", "st-magic",
    "mohammed", "cheng", "xue", "mdcla", "foregut", "developing-brain",
    "cortical-diversification", "spinal-cord", "pancreatic-endocrinogenesis",
    "kidney-development", "tabula-muris", "tabula-muris-senis",
    "mouse-cell-atlas", "allen-brain", "starmap-plus", "osmfish",
    "slide-seq-v2", "liver-cell-atlas",
)
DOWNLOADS = {
    "tome": [
        "https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/"
        f"tome_summary_data/mm/seurat_object_{stage}.rds"
        for stage in (
            "E3.5", "E4.5", "E5.25", "E5.5", "E6.25", "E6.5", "E6.75",
            "E7.0", "E7.25", "E7.5", "E7.75", "E8", "E8.25", "E8.5a",
            "E8.5b", "E9.5", "E10.5", "E11.5", "E12.5", "E13.5",
        )
    ],
    "seu-3d": ["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE278nnn/GSE278603/suppl/GSE278603_RAW.tar"],
    "extended-mouse-atlas": ["https://bioinformatics.stemcells.cam.ac.uk/rlh60/Supplemental/ExtendedMouseAtlas/ExtendedMouseAtlas.tar.gz"],
    "spateo": [
        f"https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/mouse_{stage}_{tissue}.h5ad"
        for stage in ("E9.5", "E11.5") for tissue in ("embryo", "heart")
    ],
}


def _log(message, verbose):
    if verbose:
        print(message, flush=True)


def _download(url, destination, verbose):
    _log(f"Downloading {url}\n  -> {destination}", verbose)
    last_report = 0.0

    def progress(blocks, block_size, total):
        nonlocal last_report
        now = time.monotonic()
        if now - last_report < 5:
            return
        last_report = now
        received = blocks * block_size
        if total > 0:
            received = min(received, total)
            status = f"{received / 2**20:,.1f} / {total / 2**20:,.1f} MiB ({received / total:.0%})"
        else:
            status = f"{received / 2**20:,.1f} MiB"
        _log(f"  {status}", verbose)

    urlretrieve(url, destination, reporthook=progress if verbose else None)
    _log(f"Download complete: {destination.stat().st_size / 2**20:,.1f} MiB", verbose)


def convert_gene_names(adata, data_dir=None, verbose=True):
    """Convert gene IDs to unique symbols in place, preserving original IDs."""
    if "gene_id" not in adata.var:
        adata.var["gene_id"] = list(adata.var_names)
    symbols = pd.Series(list(adata.var_names), index=adata.var_names)
    for column in ("gene_symbol", "gene_symbols", "symbol"):
        if column in adata.var:
            symbols = symbols.where(~symbols.str.startswith("ENSMUSG"), adata.var[column].fillna(symbols))
    unresolved = symbols.str.startswith("ENSMUSG")
    if unresolved.any():
        url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
        annotation = Path(data_dir if data_dir is not None else DATA_DIR) / "annotations" / url.rsplit("/", 1)[1]
        annotation.parent.mkdir(parents=True, exist_ok=True)
        if not annotation.exists():
            partial = annotation.with_name(annotation.name + ".part")
            _download(url, partial, verbose)
            partial.replace(annotation)
        mapping = {}
        with gzip.open(annotation, "rt") as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) == 9 and fields[2] == "gene":
                    attributes = dict(re.findall(r'(\w+) "([^"]*)"', fields[8]))
                    mapping[attributes["gene_id"].split(".")[0]] = attributes["gene_name"]
        symbols.loc[unresolved] = symbols.loc[unresolved].map(
            lambda name: mapping.get(name.split(".")[0], name)
        )
    duplicates = symbols.str.upper().duplicated(keep=False)
    symbols.loc[duplicates] = adata.var_names[duplicates]
    adata.var_names = pd.Index(symbols)
    return adata


def _normalize(name):
    return re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")


def _catalog():
    entries = {}
    current = None
    for line in (ROOT / "Datasets.md").read_text().splitlines():
        match = re.match(r"### (\d+)\. (.+)", line) or re.match(r"\| (\d+) \| (.+?) \|", line)
        if match:
            current = int(match[1])
            entries[current] = {"title": match[2].replace("**", ""), "text": line}
        elif line.startswith("## "):
            current = None
        elif current is not None:
            entries[current]["text"] += "\n" + line
    return entries


def _resolve_name(dataset_name):
    query = _normalize(dataset_name)
    matches = []
    catalog = _catalog()
    for number, name in enumerate(DATASET_NAMES, 1):
        entry = catalog[number]
        aliases = [name, str(number), entry["title"]]
        aliases += re.findall(r"\b(?:GSE\d+|E-MTAB-\d+|OMIX\d+|STDS\d+)\b", entry["text"])
        if query in {_normalize(alias) for alias in aliases}:
            matches.append(name)
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous dataset {dataset_name!r}. Use --list for dataset names.")
    return matches[0]


def _extract(archive, member, name, folder, zip_file=False):
    destination = (folder / name).resolve()
    if not destination.is_relative_to(folder.resolve()):
        raise ValueError(f"Archive member escapes dataset directory: {name}")
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = archive.open(member) if zip_file else archive.extractfile(member)
        with source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


def _unpack(folder):
    seen = set()
    while True:
        archives = [
            path for path in folder.rglob("*")
            if path.is_file() and path not in seen
            and path.name.lower().endswith((".tar", ".tar.gz", ".tgz", ".zip", ".tar.bz2", ".tar.xz"))
        ]
        if not archives:
            return
        for path in archives:
            seen.add(path)
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if not member.is_dir():
                            _extract(archive, member, member.filename, path.parent, zip_file=True)
            else:
                with tarfile.open(path) as archive:
                    for member in archive.getmembers():
                        if member.isfile():
                            _extract(archive, member, member.name, path.parent)


def _table(folder, spec):
    if isinstance(spec, str):
        spec = {"path": spec}
    path = folder / spec["path"]
    separator = spec.get("sep", "," if ".csv" in path.suffixes else "\t")
    index = spec.get("index_col", 0)
    frame = pd.read_csv(
        path, sep=separator, index_col=index, header=spec.get("header", "infer"),
        converters={index: str} if index is not None else None,
    )
    frame.index = frame.index.astype(str)
    return frame


def _preferred_path(path):
    """Prefer a saved AnnData conversion, even if the RDS source was removed."""
    if path.suffix.lower() == ".rds" and path.with_suffix(".h5ad").is_file():
        return path.with_suffix(".h5ad")
    return path


def _read_legacy_seurat(path):
    """Adapt older Seurat assays using matrix dimnames, without loading R.

    readseurat 0.1.0 expects a features slot that older Assay objects do not
    contain. Its default sparse constructor also discards matrix dimnames.
    Keep those names during deserialization and supply the missing features.
    Internal helpers are from the pinned readseurat==0.1.0 dependency.
    """
    from readseurat.rdata import read_rds
    from readseurat.rdata.conversion import DEFAULT_CLASS_MAP
    from readseurat.rdata.conversion._conversion import dgcmatrix_constructor
    from readseurat.convert.seurat import parse
    from readseurat.convert.seurat.convert import convert_to_anndata
    from readseurat.convert.seurat.model import SeuratObject

    def named_matrix(obj, attrs):
        matrix = dgcmatrix_constructor(obj, attrs)
        matrix._r_dimnames = attrs.get("Dimnames")
        return matrix

    def legacy_assay(obj, attrs):
        if getattr(obj, "features", None) is None:
            for key in ("counts", "data"):
                dimnames = getattr(getattr(obj, key, None), "_r_dimnames", None)
                if dimnames is not None and dimnames[0] is not None:
                    obj.features = np.asarray(dimnames[0], dtype=str)
                    break
            else:
                raise ValueError("Seurat assay has no gene names in its matrix Dimnames")
        return obj

    constructors = {**DEFAULT_CLASS_MAP, "dgCMatrix": named_matrix, "Assay": legacy_assay}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Missing constructor for R class")
        obj = read_rds(str(path), constructor_dict=constructors)
    slots = vars(obj)
    version = parse._extract_version(slots)
    cells = parse._extract_cell_names(obj, slots)
    seurat = SeuratObject(
        project=parse._extract_project_name(slots),
        active_assay=parse._extract_active_assay(slots),
        version=version,
        cell_names=cells,
        meta_data=parse._extract_meta_data(slots, cells),
        assays=parse._extract_assays(obj, slots, int(version.split(".")[0])),
        reductions=parse._extract_reductions(slots),
        graphs=parse._extract_graphs(slots),
        images=parse._extract_images(slots),
        misc=parse._extract_misc(slots),
        tools=parse._extract_tools(slots),
        commands=parse._extract_commands(slots),
    )
    return convert_to_anndata(seurat)


def _read_r(path, spec, verbose=True):
    preferred = _preferred_path(path)
    if preferred != path:
        _log(f"Reusing converted AnnData: {preferred}", verbose)
        return ad.read_h5ad(preferred)
    if path.suffix.lower() == ".rds":
        try:
            from readseurat import read_seurat
        except ImportError as error:
            raise ImportError(
                "Cannot import readseurat in this Python environment. Run "
                "'python -m pip install -r requirements.txt' from the repository root. "
                "RDS conversion does not fall back to R."
            ) from error
        try:
            _log(f"Converting {path.name} to AnnData with readseurat (Python)", verbose)
            legacy = False
            try:
                result = read_seurat(str(path))
            except Exception as error:
                if "No features found in assay" not in str(error):
                    raise
                legacy = True
            # Leave the exception handler first so its traceback releases the
            # potentially large original object before deserializing it again.
            if legacy:
                _log("Reading legacy Seurat gene names from matrix Dimnames (Python)", verbose)
                result = _read_legacy_seurat(path)
        except Exception as error:
            raise RuntimeError(
                f"Python Seurat RDS conversion failed for {path}: {error}. "
                "No Rscript was invoked."
            ) from error
        if "counts" in result.layers:
            if "logcounts" not in result.layers:
                result.layers["logcounts"] = result.X.copy()
            result.X = result.layers["counts"].copy()
        _save_rds_conversion(result, path, verbose)
        return result
    if shutil.which("Rscript") is None:
        raise RuntimeError(
            f"Reading {path} requires Rscript, SingleCellExperiment, and zellkonverter "
            "(plus Seurat for Seurat objects). Seurat RDS files can use readseurat without R."
        )
    script = '''
args <- commandArgs(trailingOnly=TRUE)
if (tolower(tools::file_ext(args[1])) %in% c("rdata", "rda")) {
    env <- new.env()
    objects <- load(args[1], envir=env)
    selected <- args[3]
    if (selected == "") {
        if (length(objects) != 1) stop("Set object in dataset.json for multi-object RData files")
        selected <- objects[1]
    }
    x <- env[[selected]]
} else {
    x <- readRDS(args[1])
}
if (inherits(x, "Seurat")) x <- Seurat::as.SingleCellExperiment(x)
if (is.data.frame(x)) x <- as.matrix(x)
if (is.matrix(x) || inherits(x, "Matrix")) {
    x <- SingleCellExperiment::SingleCellExperiment(assays=list(counts=x))
}
if (!inherits(x, "SingleCellExperiment")) stop("Expected Seurat, SingleCellExperiment, or expression matrix")
zellkonverter::writeH5AD(x, args[2])
'''
    _log(f"Converting {path.name} to AnnData with Rscript", verbose)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "converted.h5ad"
        subprocess.run(
            ["Rscript", "-e", script, str(path), str(output), spec.get("object", "")],
            check=True,
        )
        result = ad.read_h5ad(output)
    if path.suffix.lower() == ".rds":
        _save_rds_conversion(result, path, verbose)
    return result


def _save_rds_conversion(adata, path, verbose=True):
    """Publish only complete conversions so interrupted writes cannot be reused."""
    destination = path.with_suffix(".h5ad")
    _log(f"Saving converted AnnData: {destination}", verbose)
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".rds-conversion-") as directory:
        temporary = Path(directory) / destination.name
        adata.write_h5ad(temporary)
        temporary.replace(destination)


def _read_sample(folder, spec, verbose=True):
    path = folder / spec["path"]
    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = path.with_suffix("").suffix.lower()
    kind = spec.get("format", suffix.lstrip("."))
    if kind in ("mtx", "table", "csv", "tsv", "txt", "tab", "h5", "hdf5", "hdf"):
        if spec.get("orientation") not in ("genes_by_cells", "cells_by_genes"):
            raise ValueError(f"Set orientation to genes_by_cells or cells_by_genes for {path}")
    if kind == "h5ad":
        _log(f"Reading existing AnnData: {path}", verbose)
        result = ad.read_h5ad(path)
    elif kind == "h5td":
        import treedata
        result = treedata.read_h5td(path)
    elif kind == "loom":
        result = ad.io.read_loom(path, **spec.get("reader_options", {}))
    elif kind in ("rds", "rdata", "rda"):
        result = _read_r(path, spec, verbose=verbose)
    elif kind in ("10x_h5", "10x_mtx", "visium"):
        import scanpy as sc
        readers = {"10x_h5": sc.read_10x_h5, "10x_mtx": sc.read_10x_mtx, "visium": sc.read_visium}
        result = readers[kind](path, **spec.get("reader_options", {}))
    elif kind in ("h5", "hdf5", "hdf"):
        result = ad.io.read_hdf(path, spec["key"])
        if spec["orientation"] == "genes_by_cells":
            result = result.T.copy()
    elif kind == "mtx":
        matrix = io.mmread(path)
        if spec["orientation"] == "genes_by_cells":
            matrix = matrix.T
        result = ad.AnnData(sparse.csr_matrix(matrix))
        result.obs = _table(folder, spec["obs"])
        result.var = _table(folder, spec["var"])
    elif kind in ("table", "csv", "tsv", "txt", "tab"):
        frame = _table(folder, spec)
        if spec["orientation"] == "genes_by_cells":
            frame = frame.T
        result = ad.AnnData(frame.apply(pd.to_numeric))
    else:
        raise ValueError(f"Unsupported expression format: {kind}")
    for axis in ("obs", "var"):
        if axis in spec and kind != "mtx":
            metadata = _table(folder, spec[axis])
            existing = getattr(result, axis)
            aligned = metadata.loc[existing.index]
            setattr(result, axis, existing.drop(columns=existing.columns.intersection(aligned.columns)).join(aligned))
    if "spatial_columns" in spec:
        result.obsm["spatial"] = result.obs[spec["spatial_columns"]].to_numpy(dtype=float)
    if "newick" in spec:
        result.uns["lineage_newick"] = (folder / spec["newick"]).read_text()
    return result


def _discover(folder):
    samples = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".h5ad", ".h5td", ".loom", ".rds", ".rdata", ".rda"):
            relative = path.relative_to(folder)
            name = str(relative.with_suffix(""))
            if name not in samples or path.suffix.lower() == ".h5ad":
                samples[name] = {"path": str(relative)}
    if samples:
        return samples
    for path in sorted(folder.rglob("*matrix.mtx*")):
        suffix = ".gz" if path.suffix == ".gz" else ""
        prefix = path.name.split("matrix.mtx")[0]
        if (path.parent / f"{prefix}barcodes.tsv{suffix}").exists():
            name = str(path.relative_to(folder)).removesuffix(suffix).removesuffix(".mtx")
            samples[name] = {
                "path": str(path.parent.relative_to(folder)), "format": "10x_mtx",
                "reader_options": {"prefix": prefix, "var_names": "gene_ids", "make_unique": False,
                                   "gex_only": False, "compressed": bool(suffix)},
            }
    for path in sorted(folder.rglob("*.h5")):
        import h5py
        with h5py.File(path) as handle:
            is_10x = "matrix" in handle and isinstance(handle["matrix"], h5py.Group) and "barcodes" in handle["matrix"]
        if is_10x:
            relative = path.relative_to(folder)
            if (path.parent / "spatial").is_dir():
                samples[str(relative.with_suffix(""))] = {
                    "path": str(relative.parent), "format": "visium",
                    "reader_options": {"count_file": path.name},
                }
            else:
                samples[str(relative.with_suffix(""))] = {
                    "path": str(relative), "format": "10x_h5", "reader_options": {"gex_only": False},
                }
    return samples


def _parse_timepoint(value):
    match = re.fullmatch(r"[eE]?(\d+(?:\.\d+)?)([aAbB]?)", str(value).strip())
    if not match:
        raise ValueError("Use an embryonic timepoint such as 7.5, E8.0, or E8.5a")
    return float(match[1]), match[2].lower()


def _file_timepoint(value):
    match = re.search(r"(?:^|[^a-z0-9])E(\d+(?:\.\d+)?)([ab]?)(?=[^a-z0-9]|$)", str(value), re.I)
    return (float(match[1]), match[2].lower()) if match else None


def _matches_timepoint(stage, requested):
    return stage is not None and stage[0] == requested[0] and (not requested[1] or stage[1] == requested[1])


def _select_timepoint(samples, requested):
    if requested is None:
        return samples
    return {
        sample: spec for sample, spec in samples.items()
        if _matches_timepoint(
            _parse_timepoint(spec["timepoint"]) if "timepoint" in spec
            else _file_timepoint(sample) or _file_timepoint(spec["path"]), requested,
        )
    }


def load_dataset(dataset_name="seu-3d", output="anndata", timepoint="all", data_dir=None, verbose=True):
    """Load samples, downloading into data_dir/<dataset> (default: dataset/data).

    Seurat RDS inputs use readseurat without R when installed. Conversions are saved
    as same-stem .h5ad files and preferred on subsequent calls, including manifests.
    Gene annotation downloads also stay under the selected data directory.
    All timepoints are selected by default; None is also accepted for compatibility.
    Progress is printed by default; set verbose=False to silence it.
    """
    return dict(iter_dataset(dataset_name, output, timepoint, data_dir, verbose))


def iter_dataset(dataset_name="seu-3d", output="anndata", timepoint="all", data_dir=None, verbose=True):
    """Yield (sample_name, data) pairs without retaining previous samples in memory."""
    if output not in ("anndata", "arrays"):
        raise ValueError("output must be 'anndata' or 'arrays'")
    name = _resolve_name(dataset_name)
    requested = None if timepoint is None or str(timepoint).strip().lower() == "all" else _parse_timepoint(timepoint)
    download_root = Path(data_dir).expanduser() if data_dir is not None else DATA_DIR
    folder = download_root / ("GSE278603_RAW" if name == "seu-3d" else name)
    if data_dir is None and name == "seu-3d" and not folder.exists() and (ROOT / "GSE278603_RAW").exists():
        folder = ROOT / "GSE278603_RAW"
    folder.mkdir(parents=True, exist_ok=True)
    _log(f"Dataset: {name} | Timepoints: {'all' if requested is None else timepoint}\nDirectory: {folder}", verbose)
    manifest_path = folder / "dataset.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    samples = _select_timepoint(manifest.get("samples") or _discover(folder), requested)
    urls = manifest.get("downloads", DOWNLOADS.get(name, []))
    if requested is not None:
        if any(_file_timepoint(Path(unquote(urlparse(url).path)).name) for url in urls):
            urls = [url for url in urls if _matches_timepoint(
                _file_timepoint(Path(unquote(urlparse(url).path)).name), requested,
            )]
        elif name == "seu-3d":
            if requested not in ((7.5, ""), (7.75, ""), (8.0, "")):
                raise ValueError("SEU-3D timepoints are 7.5, 7.75, and 8.0")
        elif not samples:
            raise ValueError(
                f"No timepoint-labelled samples for {name} at {timepoint}. "
                "Use filenames containing E<stage> or a timepoint field in dataset.json samples."
            )
        if not urls and not samples:
            raise ValueError(f"No available timepoint {timepoint} for {name}")
    missing_stage_files = name in ("tome", "spateo") and not manifest.get("samples") and any(
        not _preferred_path(folder / Path(unquote(urlparse(url).path)).name).exists() for url in urls
    )
    _log(f"Selected {len(urls)} download source(s); found {len(samples)} local sample(s).", verbose)
    if missing_stage_files or not samples or any(not _preferred_path(folder / spec["path"]).exists() for spec in samples.values()):
        for index, url in enumerate(urls, 1):
            filename = Path(unquote(urlparse(url).path)).name
            archive_path = folder / filename
            _log(f"[File {index}/{len(urls)}] {filename}", verbose)
            if _preferred_path(archive_path) != archive_path:
                _log(f"Reusing converted AnnData: {_preferred_path(archive_path)}", verbose)
                continue
            if name == "seu-3d" and (folder.parent / filename).exists():
                archive_path = folder.parent / filename
                with tarfile.open(archive_path) as archive:
                    for member in archive.getmembers():
                        if member.isfile() and member.name.endswith(".h5ad"):
                            _extract(archive, member, member.name, folder)
            elif not archive_path.exists():
                partial_path = archive_path.with_name(archive_path.name + ".part")
                _download(url, partial_path, verbose)
                partial_path.replace(archive_path)
            else:
                _log(f"Reusing downloaded file: {archive_path}", verbose)
        _log("Checking and extracting archives...", verbose)
        _unpack(folder)
        samples = _select_timepoint(manifest.get("samples") or _discover(folder), requested)
    if not samples:
        raise FileNotFoundError(
            f"No self-describing expression files in {folder}. Download processed files from "
            f"Datasets.md entry {DATASET_NAMES.index(name) + 1}. For tables, generic HDF5, or "
            "non-10x MTX, add dataset.json with samples mapping sample names to path, format, "
            "orientation (cells_by_genes or genes_by_cells), and obs/var metadata paths. "
            "Raw reads and images require upstream processing."
        )
    _log(f"Loading {len(samples)} sample(s). RDS conversions will be saved as .h5ad.", verbose)
    for index, (sample, spec) in enumerate(samples.items(), 1):
        path = folder / spec["path"]
        _log(f"[Sample {index}/{len(samples)}] {sample}: {_preferred_path(path)}", verbose)
        adata = convert_gene_names(_read_sample(folder, spec, verbose=verbose), data_dir=download_root, verbose=verbose)
        if name == "tome" and "logcounts" in adata.layers:
            counts = sparse.csr_matrix(adata.X)
            logcounts = sparse.csr_matrix(adata.layers["logcounts"])
            if (counts != logcounts).nnz == 0 and np.all(counts.data >= 0) and np.all(counts.data == np.floor(counts.data)):
                logcounts = counts.astype(float, copy=True)
                logcounts.data = np.log1p(logcounts.data)
                adata.layers["logcounts"] = logcounts
            del counts, logcounts
        _log(f"Ready: {sample} | {adata.n_obs:,} cells x {adata.n_vars:,} genes", verbose)
        if output == "arrays":
            yield sample, adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        else:
            yield sample, adata
        del adata
    _log(f"Completed {name}: {len(samples)} sample(s). Files: {folder}", verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and load a named Datasets.md entry one sample at a time (all timepoints by default).",
        epilog='Files live in dataset/data/<name> (SEU-3D uses GSE278603_RAW). '
        'Automatic downloads: seu-3d, spateo, extended-mouse-atlas, tome. Other entries need local '
        'processed files or dataset.json downloads. For custom matrices, dataset.json contains '
        '{"samples": {"sample": {"path": "counts.csv", "orientation": "genes_by_cells"}}}. '
        'MTX also needs obs and var metadata paths; generic HDF5 needs key. '
        'Supported formats: h5ad, h5td, loom, rds/rdata, table/csv/tsv/txt, mtx, hdf, '
        '10x_h5, 10x_mtx, visium. Seurat RDS uses readseurat without R and caches .h5ad. '
        'RDS conversion never falls back to R. RData/RDA objects require Rscript, '
        'SingleCellExperiment, zellkonverter, and Seurat as needed.',
    )
    parser.add_argument("dataset_name", nargs="?", default="seu-3d")
    parser.add_argument(
        "--data-dir", type=Path,
        help="Download/cache root; datasets use subfolders here (default: dataset/data)",
    )
    parser.add_argument(
        "--timepoint", default="all",
        help="Timepoint or 'all' (default); e.g. E7.5; E8.5 includes both TOME E8.5 files",
    )
    parser.add_argument(
        "--output", choices=("anndata", "arrays"), default="anndata",
        help="Output type; arrays returns dense cells-by-genes X values without metadata",
    )
    parser.add_argument("--list", action="store_true", help="List catalog names and entry numbers")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    args = parser.parse_args()
    if args.list:
        catalog = _catalog()
        for number, name in enumerate(DATASET_NAMES, 1):
            print(f"{number:2}  {name:30} {catalog[number]['title']}")
    else:
        for name, data in iter_dataset(
            args.dataset_name, output=args.output, timepoint=args.timepoint,
            data_dir=args.data_dir, verbose=not args.quiet,
        ):
            if not args.quiet:
                print(name, data, sep="\n", flush=True)
            del data
