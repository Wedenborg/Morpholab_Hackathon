import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile
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


def convert_gene_names(adata):
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
        annotation = DATA_DIR / "annotations" / url.rsplit("/", 1)[1]
        annotation.parent.mkdir(parents=True, exist_ok=True)
        if not annotation.exists():
            partial = annotation.with_name(annotation.name + ".part")
            urlretrieve(url, partial)
            partial.replace(annotation)
        mapping = {}
        with gzip.open(annotation, "rt", encoding="utf-8") as handle:
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
    for line in (ROOT / "Datasets.md").read_text(encoding="utf-8").splitlines():
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


def _read_r(path, spec):
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
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "converted.h5ad"
        subprocess.run(
            ["Rscript", "-e", script, str(path), str(output), spec.get("object", "")],
            check=True,
        )
        return ad.read_h5ad(output)


def _read_sample(folder, spec):
    path = folder / spec["path"]
    suffix = path.suffix.lower()
    if suffix == ".gz":
        suffix = path.with_suffix("").suffix.lower()
    kind = spec.get("format", suffix.lstrip("."))
    if kind in ("mtx", "table", "csv", "tsv", "txt", "tab", "h5", "hdf5", "hdf"):
        if spec.get("orientation") not in ("genes_by_cells", "cells_by_genes"):
            raise ValueError(f"Set orientation to genes_by_cells or cells_by_genes for {path}")
    if kind == "h5ad":
        result = ad.read_h5ad(path)
    elif kind == "h5td":
        import treedata
        result = treedata.read_h5td(path)
    elif kind == "loom":
        result = ad.io.read_loom(path, **spec.get("reader_options", {}))
    elif kind in ("rds", "rdata", "rda"):
        result = _read_r(path, spec)
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
        result.uns["lineage_newick"] = (folder / spec["newick"]).read_text(encoding="utf-8")
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


def load_dataset(dataset_name="seu-3d", output="anndata", timepoint=None):
    if output not in ("anndata", "arrays"):
        raise ValueError("output must be 'anndata' or 'arrays'")
    name = _resolve_name(dataset_name)
    requested = _parse_timepoint(timepoint) if timepoint is not None else None
    folder = DATA_DIR / ("GSE278603_RAW" if name == "seu-3d" else name)
    if name == "seu-3d" and not folder.exists() and (ROOT / "GSE278603_RAW").exists():
        folder = ROOT / "GSE278603_RAW"
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "dataset.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
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
        not (folder / Path(unquote(urlparse(url).path)).name).exists() for url in urls
    )
    if missing_stage_files or not samples or any(not (folder / spec["path"]).exists() for spec in samples.values()):
        for url in urls:
            filename = Path(unquote(urlparse(url).path)).name
            archive_path = folder / filename
            if name == "seu-3d" and (folder.parent / filename).exists():
                archive_path = folder.parent / filename
                with tarfile.open(archive_path) as archive:
                    for member in archive.getmembers():
                        if member.isfile() and member.name.endswith(".h5ad"):
                            _extract(archive, member, member.name, folder)
            elif not archive_path.exists():
                partial_path = archive_path.with_name(archive_path.name + ".part")
                urlretrieve(url, partial_path)
                partial_path.replace(archive_path)
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
    datasets = {}
    for sample, spec in samples.items():
        adata = convert_gene_names(_read_sample(folder, spec))
        if name == "tome" and "logcounts" in adata.layers:
            counts = sparse.csr_matrix(adata.X)
            logcounts = sparse.csr_matrix(adata.layers["logcounts"])
            if (counts != logcounts).nnz == 0 and np.all(counts.data >= 0) and np.all(counts.data == np.floor(counts.data)):
                logcounts = counts.astype(float, copy=True)
                logcounts.data = np.log1p(logcounts.data)
                adata.layers["logcounts"] = logcounts
        if output == "arrays":
            datasets[sample] = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
        else:
            datasets[sample] = adata
    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a named Datasets.md entry as a dictionary of sample AnnData objects or NumPy arrays.",
        epilog='Files live in dataset/data/<name> (SEU-3D uses GSE278603_RAW). '
        'Automatic downloads: seu-3d, spateo, extended-mouse-atlas, tome. Other entries need local '
        'processed files or dataset.json downloads. For custom matrices, dataset.json contains '
        '{"samples": {"sample": {"path": "counts.csv", "orientation": "genes_by_cells"}}}. '
        'MTX also needs obs and var metadata paths; generic HDF5 needs key. '
        'Supported formats: h5ad, h5td, loom, rds/rdata, table/csv/tsv/txt, mtx, hdf, '
        '10x_h5, 10x_mtx, visium. R objects require Rscript, SingleCellExperiment, '
        'zellkonverter, and Seurat for Seurat objects.',
    )
    parser.add_argument("dataset_name", nargs="?", default="seu-3d")
    parser.add_argument(
        "--timepoint", help="Select stage-labelled samples, e.g. 7.5 or E7.5; 8.5 includes both TOME E8.5 files",
    )
    parser.add_argument(
        "--output", choices=("anndata", "arrays"), default="anndata",
        help="Output type; arrays returns dense cells-by-genes X values without metadata",
    )
    parser.add_argument("--list", action="store_true", help="List catalog names and entry numbers")
    args = parser.parse_args()
    if args.list:
        catalog = _catalog()
        for number, name in enumerate(DATASET_NAMES, 1):
            print(f"{number:2}  {name:30} {catalog[number]['title']}")
    else:
        datasets = load_dataset(args.dataset_name, output=args.output, timepoint=args.timepoint)
        for name, adata in datasets.items():
            print(name)
            print(adata)
