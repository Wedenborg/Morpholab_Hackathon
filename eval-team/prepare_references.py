"""Download and summarise the two external references used by the region benchmark.

1. Peng et al. 2019 (GEO GSE120963): Geo-seq of small tissue pieces cut from known
   positions of an E7.5 embryo. We keep the combined E7.5 reference table
   (log2(FPKM + 1), genes x 83 samples).
2. Pijuan-Sala et al. 2019 atlas (scRNA-seq, E6.5-E8.5). We stream the 6.7 GB count
   matrix once and keep the mean log2(normalised count + 1) per cell type for a few
   stages. The full matrix is never loaded into memory.

Usage:
    python prepare_references.py            # both references
    python prepare_references.py --skip-atlas

Files are written to $MORPHO_REF_DIR (default: ../../reference_data relative to this file,
i.e. next to the repo, outside git).
"""
import argparse
import os
import pathlib
import subprocess
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
REF_DIR = pathlib.Path(os.environ.get("MORPHO_REF_DIR", HERE.parent.parent / "reference_data"))

PENG_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE120nnn/GSE120963/suppl/"
            "GSE120963_E7.5_Ref_all.combine.gene.fpkm.txt.gz")
ATLAS_URL = "https://content.cruk.cam.ac.uk/jmlab/atlas_data.tar.gz"
ATLAS_STAGES = ["E7.5", "E7.75", "E8.0"]


def _download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"downloading {url}", file=sys.stderr)
    # curl resumes interrupted transfers; the atlas server drops long connections.
    for _ in range(10):
        r = subprocess.run(["curl", "-sSL", "-C", "-", "-o", str(dest), url])
        if r.returncode == 0:
            return dest
    raise RuntimeError(f"download failed: {url}")


def prepare_peng():
    src = _download(PENG_URL, REF_DIR / "peng_2019" / pathlib.Path(PENG_URL).name)
    df = pd.read_csv(src, sep="\t", index_col=0)
    df = df[~df.index.duplicated()]
    out = np.log2(df + 1)
    out.columns = [c.replace(".E7.5", "") for c in out.columns]
    dest = REF_DIR / "peng_2019" / "peng_E7.5_log2fpkm.csv.gz"
    out.to_csv(dest)
    print(f"wrote {dest} {out.shape}", file=sys.stderr)


def prepare_atlas(stages=ATLAS_STAGES, chunksize=20_000_000):
    d = REF_DIR / "pijuan_sala_2019"
    tar = _download(ATLAS_URL, d / "atlas_data.tar.gz")
    meta_files = ["atlas/meta.tab", "atlas/genes.tsv", "atlas/sizefactors.tab"]
    if not all((d / f).exists() for f in meta_files):
        with tarfile.open(tar) as t:
            for f in meta_files:
                t.extract(f, d)
    meta = pd.read_csv(d / "atlas/meta.tab", sep="\t", low_memory=False)
    genes = pd.read_csv(d / "atlas/genes.tsv", sep="\t", header=None)[1].to_numpy()
    sf = pd.read_csv(d / "atlas/sizefactors.tab", header=None)[0].to_numpy()

    keep = meta["stage"].isin(stages) & meta["celltype"].notna() & ~meta["doublet"].astype(bool) \
        & ~meta["stripped"].astype(bool)
    groups = (meta["stage"] + "|" + meta["celltype"].astype(str)).where(keep)
    codes, uniq = pd.factorize(groups)  # -1 for cells we drop
    n_groups, n_genes = len(uniq), len(genes)
    sums = np.zeros(n_genes * n_groups)
    detected = np.zeros(n_genes * n_groups)

    # Stream the MatrixMarket file (genes x cells, 1-indexed) straight out of the tarball.
    proc = subprocess.Popen(["tar", "-xzOf", str(tar), "atlas/raw_counts.mtx"], stdout=subprocess.PIPE)
    f = proc.stdout
    line = f.readline()
    while line.startswith(b"%"):
        line = f.readline()
    n_rows, n_cols, nnz = map(int, line.split())
    assert n_rows == n_genes and n_cols == len(meta), (n_rows, n_cols)
    seen = 0
    reader = pd.read_csv(f, sep=" ", header=None, chunksize=chunksize,
                         dtype={0: np.int32, 1: np.int32, 2: np.float64})
    for chunk in reader:
        g = chunk[0].to_numpy() - 1
        c = chunk[1].to_numpy() - 1
        v = chunk[2].to_numpy()
        grp = codes[c]
        m = grp >= 0
        idx = g[m].astype(np.int64) * n_groups + grp[m]
        sums += np.bincount(idx, weights=np.log2(v[m] / sf[c[m]] + 1), minlength=sums.size)
        detected += np.bincount(idx, minlength=sums.size)
        seen += len(chunk)
        print(f"  {seen / nnz:.0%} of matrix entries", file=sys.stderr)
    proc.wait()

    n_cells = pd.Series(codes[codes >= 0]).value_counts().reindex(range(n_groups)).to_numpy()
    mean = sums.reshape(n_genes, n_groups) / n_cells
    frac = detected.reshape(n_genes, n_groups) / n_cells
    for stage in stages:
        cols = [i for i, u in enumerate(uniq) if u.split("|")[0] == stage]
        names = [uniq[i].split("|")[1] for i in cols]
        for arr, tag in [(mean, "mean_log2norm"), (frac, "frac_detected")]:
            df = pd.DataFrame(arr[:, cols], index=genes, columns=names)
            df = df[~df.index.duplicated()]
            df.to_csv(d / f"atlas_{stage}_{tag}.csv.gz")
        pd.Series(n_cells[cols], index=names, name="n_cells").to_csv(d / f"atlas_{stage}_ncells.csv")
        print(f"wrote atlas tables for {stage}: {len(names)} cell types", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-atlas", action="store_true")
    args = ap.parse_args()
    prepare_peng()
    if not args.skip_atlas:
        prepare_atlas()
