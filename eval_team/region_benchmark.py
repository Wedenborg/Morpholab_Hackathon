"""Region benchmark for a reconstructed E7.5 embryo.

A method returns a reconstructed expression matrix for a hidden Stereo-seq embryo:
one row per cell of that embryo and one column per gene. The benchmark

1. splits the embryo's cells into anatomical regions: germ layer x position around the
   embryo (ectoderm: anterior / anterior-lateral / posterior-lateral / posterior;
   mesoderm and endoderm: anterior / posterior), optionally also x level along the
   proximal-distal axis. These are the pieces Peng et al. 2019 cut and sequenced at E7.5.
2. averages the reconstruction within each region, and within each cell type,
3. compares, gene by gene, the pattern across regions with Peng's Geo-seq data and the
   pattern across cell types with the Pijuan-Sala 2019 scRNA-seq atlas (Spearman
   correlation, so platform scale differences do not matter),
4. reports the same scores for the true measured data of the hidden embryo, as a reference
   point. It is not an upper bound: the true data are noisy, and a reconstruction that
   denoises them (or copies another embryo) can score higher against Peng and the atlas,
5. scores agreement with the hidden embryo's own measured data, region by region
   ("truth_region_spearman"). This is the only score that tells a faithful reconstruction
   of this embryo from a plausible-looking copy of another one.

Region and cell-type assignment use the hidden embryo's own labels and coordinates, which
only the benchmark sees. Nothing about the reconstruction is used to define them.

Input scale: the same as the h5ad files' X, i.e. log1p of counts. Any per-gene monotone
rescaling of the averages leaves the scores unchanged.

Usage:
    python -m eval_team.region_benchmark --truth <hidden embryo .h5ad> --recon <reconstruction .h5ad>
"""
import argparse
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from .paths import REF_DIR

# Cell types that belong to the embryo proper at E7.5. Everything else in the Stereo-seq
# labels is extra-embryonic or lines the outside of the conceptus (ExE tissues, yolk-sac
# blood, parietal endoderm, proximal "Surface ectoderm"/"Mesenchyme"), which Peng did not
# sample. They sit at high z in both E7.5 embryos.
EMBRYONIC_TYPES = {
    "Epiblast", "Rostral neurectoderm", "Caudal neurectoderm", "Caudal epiblast",
    "Primitive Streak", "Anterior Primitive Streak",
    "Nascent mesoderm", "Mixed mesoderm", "Paraxial mesoderm", "Pre-somitic",
    "Caudal Mesoderm", "Pharyngeal mesoderm", "Intermediate mesoderm",
    "Def. endoderm", "Gut", "Visceral endoderm", "Notochord",
}
# Used only to find the anterior-posterior direction of each embryo.
POSTERIOR_TYPES = ["Primitive Streak", "Caudal epiblast", "Nascent mesoderm", "Caudal Mesoderm"]
ANTERIOR_TYPES = ["Rostral neurectoderm", "Def. endoderm"]
LAYERS = {"ectoderm": "ecto", "mesoderm": "meso", "endoderm": "endo"}
LEVELS = ("distal", "mid", "proximal")


def canonical_celltype(s):
    return pd.Series(s, dtype=str).replace({"pre-somitic": "Pre-somitic"}).to_numpy()


# --------------------------------------------------------------------------- regions
def _ecto_sector(angle):
    """angle: degrees from the anterior direction, 0..180 (left and right merged)."""
    return np.select([angle < 30, angle < 90, angle < 150], ["A", "AL", "PL"], "P")


def assign_regions(obs):
    """Assign each embryonic cell to a region. Returns a DataFrame indexed like obs with
    columns layer, sector, level, region (NaN region for cells outside the embryo proper)."""
    ct = canonical_celltype(obs["celltype"])
    layer = pd.Series(obs["germ_layer"].astype(str).to_numpy()).map(LAYERS).to_numpy()
    emb = np.isin(ct, list(EMBRYONIC_TYPES)) & pd.notna(layer)
    xy = obs[["x", "y"]].to_numpy(float)
    z = obs["z"].to_numpy(float)

    post = xy[emb & np.isin(ct, POSTERIOR_TYPES)].mean(0)
    ant = xy[emb & np.isin(ct, ANTERIOR_TYPES)].mean(0)
    a_dir = (ant - post) / np.linalg.norm(ant - post)

    # Angle around the centre of the embryo proper in each section.
    centre = np.full_like(xy, np.nan)
    for zz in np.unique(z[emb]):
        m = emb & (z == zz)
        centre[z == zz] = xy[m].mean(0)
    v = xy - centre
    cosang = (v @ a_dir) / np.maximum(np.linalg.norm(v, axis=1), 1e-9)
    angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # Level: thirds of the z range spanned by the embryo proper (2nd-98th percentile).
    lo, hi = np.percentile(z[emb], [2, 98])
    rel = np.clip((z - lo) / max(hi - lo, 1e-9), 0, 1)
    level = np.array(LEVELS)[np.minimum((rel * 3).astype(int), 2)]

    sector = np.where(layer == "ecto", _ecto_sector(angle), np.where(angle < 90, "A", "P"))
    region = np.where(emb, pd.Series(layer).astype(str) + "_" + sector + "_" + level, None)
    return pd.DataFrame({"celltype": ct, "layer": np.where(emb, layer, None),
                         "sector": np.where(emb, sector, None), "level": np.where(emb, level, None),
                         "angle": angle, "rel_z": rel, "region": region}, index=obs.index)


def peng_sample_regions(sample):
    """Map a Peng E7.5 sample name (e.g. '3L1', '10MP', '1EAP') to benchmark region ids.

    Section numbers run distal (1) to proximal (9 for ectoderm, 10 for endo/mesoderm).
    Ectoderm pieces: A, P, L1/R1 (anterior-lateral), L2/R2 (posterior-lateral). Plain L/R
    (sections 2 and 9) cannot be placed anterior or posterior and are dropped."""
    import re
    m = re.fullmatch(r"(\d+)(EAP|EA|EP|MA|MP|L1|L2|R1|R2|A|P|L|R)", sample)
    if m is None:
        raise ValueError(sample)
    sec, piece = int(m.group(1)), m.group(2)
    level = LEVELS[0] if sec <= 3 else LEVELS[1] if sec <= 6 else LEVELS[2]
    if piece[0] in "EM":
        layer = "endo" if piece[0] == "E" else "meso"
        sectors = ["A", "P"] if piece[1:] == "AP" else [piece[1]]
    else:
        layer = "ecto"
        sectors = {"A": ["A"], "P": ["P"], "L1": ["AL"], "R1": ["AL"], "L2": ["PL"], "R2": ["PL"]}.get(piece, [])
    return [f"{layer}_{s}_{level}" for s in sectors]


def peng_region_table(peng, levels=False):
    """Average Peng samples (genes x samples, log2 FPKM) into benchmark regions.
    levels=False merges the three proximal-distal levels (layer x sector, 8 regions)."""
    cols = {}
    for s in peng.columns:
        for r in peng_sample_regions(s):
            key = r if levels else r.rsplit("_", 1)[0]
            cols.setdefault(key, []).append(s)
    return pd.DataFrame({r: peng[ss].mean(1) for r, ss in cols.items()})


# --------------------------------------------------------------------------- scoring
def group_means(X, labels, groups):
    """Mean of rows of X (cells x genes) for each group label. Returns groups x genes."""
    idx = pd.Index(groups).get_indexer(labels)
    keep = idx >= 0
    M = sp.csr_matrix((np.ones(keep.sum()), (idx[keep], np.where(keep)[0])), shape=(len(groups), X.shape[0]))
    M = sp.diags(1 / np.maximum(np.asarray(M.sum(1)).ravel(), 1)) @ M
    out = M @ X
    return np.asarray(out.todense()) if sp.issparse(out) else np.asarray(out)


def rowwise_spearman(A, B):
    """Spearman correlation between matching rows of A and B. A constant row scores 0."""
    ra = rankdata(A, axis=1)
    rb = rankdata(B, axis=1)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ra * rb).sum(1) / den
    return np.where(den > 0, r, 0.0)


def _matrix(recon, cells, genes):
    """Pull a cells x genes float matrix out of an AnnData or DataFrame reconstruction.
    Genes the reconstruction does not provide come back as all-zero columns."""
    if hasattr(recon, "obs_names"):
        rows = pd.Index(recon.obs_names).get_indexer(cells)
        cols = pd.Index(recon.var_names).get_indexer(genes)
        X = recon.X
    else:
        rows = recon.index.get_indexer(cells)
        cols = recon.columns.get_indexer(genes)
        X = recon.to_numpy()
    if (rows < 0).any():
        raise ValueError(f"reconstruction is missing {(rows < 0).sum()} cells of the hidden embryo")
    present = cols >= 0
    sub = X[rows][:, cols[present]]
    sub = sub.toarray() if sp.issparse(sub) else np.asarray(sub, dtype=float)
    out = np.zeros((len(cells), len(genes)))
    out[:, present] = sub
    return out, int((~present).sum())


class RegionBenchmark:
    def __init__(self, truth, ref_dir=REF_DIR, stage="E7.5", levels=False, min_cells=20,
                 min_detect=0.02, min_peng_range=3.0, min_atlas_range=1.0):
        """truth: AnnData of the hidden embryo (the h5ad file as distributed).

        levels: also split regions into distal / mid / proximal thirds (24 regions instead
            of 8). The z thirds line up only roughly with Peng's sections, so this is noisier.
        min_cells: regions and cell types with fewer cells (in the hidden embryo, and in the
            atlas) are left out.
        min_detect: genes detected in fewer than this fraction of the embryo's cells are
            left out, because their regional averages are mostly noise even in the true data.
        min_peng_range, min_atlas_range: keep genes whose reference values differ between
            regions (cell types) by at least this much, in log2 units. 3.0 = 8-fold in Peng;
            1.0 in the atlas, whose per-type means of log values are compressed by zeros.
            Genes without a clear pattern in the reference cannot be scored meaningfully."""
        self.truth = truth
        reg = assign_regions(truth.obs)
        self.regions_df = reg
        emb = reg["region"].notna().to_numpy()
        self.cells = truth.obs_names[emb]
        full = reg["region"].to_numpy()[emb]
        self.region = full if levels else np.array([r.rsplit("_", 1)[0] for r in full])
        self.celltype = reg["celltype"].to_numpy()[emb]
        X = sp.csr_matrix(truth.X)[np.where(emb)[0]]
        detect = np.asarray((X > 0).mean(0)).ravel()
        detected = pd.Index(truth.var_names[detect >= min_detect])
        self.min_cells = min_cells

        counts = pd.Series(self.region).value_counts()
        ref_dir = pathlib.Path(ref_dir)
        self.refs = {}

        peng_path = ref_dir / "peng_2019" / "peng_E7.5_log2fpkm.csv.gz"
        if stage == "E7.5" and peng_path.exists():
            peng = peng_region_table(pd.read_csv(peng_path, index_col=0), levels=levels)
            groups = [r for r in peng.columns if counts.get(r, 0) >= min_cells]
            ref = peng[groups]
            rng = ref.max(1) - ref.min(1)
            genes = detected.intersection(ref.index[rng >= min_peng_range])
            self.refs["peng"] = dict(labels=self.region, groups=groups, genes=genes,
                                     table=ref.loc[genes, groups].to_numpy())

        atlas_path = ref_dir / "pijuan_sala_2019" / f"atlas_{stage}_mean_log2norm.csv.gz"
        if atlas_path.exists():
            atlas = pd.read_csv(atlas_path, index_col=0)
            n_atlas = pd.read_csv(ref_dir / "pijuan_sala_2019" / f"atlas_{stage}_ncells.csv", index_col=0)["n_cells"]
            ct_counts = pd.Series(self.celltype).value_counts()
            groups = [c for c in atlas.columns
                      if ct_counts.get(c, 0) >= min_cells and n_atlas.get(c, 0) >= min_cells]
            ref = atlas[groups]
            rng = ref.max(1) - ref.min(1)
            genes = detected.intersection(ref.index[rng >= min_atlas_range])
            self.refs["atlas"] = dict(labels=self.celltype, groups=groups, genes=genes,
                                      table=ref.loc[genes, groups].to_numpy())

        # Agreement with the hidden embryo's own measured data, region by region.
        self.truth_genes = detected
        groups = [r for r in counts.index if counts[r] >= min_cells]
        self.truth_groups = groups
        Xt, _ = _matrix(truth, self.cells, list(detected))
        self.truth_table = group_means(Xt, self.region, groups).T

        self.true_data = self._score(truth, external_only=True)[0] if self.refs else {}

    def _score(self, recon, external_only=False):
        summary, per_gene = {}, []
        for name, ref in self.refs.items():
            Xr, missing = _matrix(recon, self.cells, list(ref["genes"]))
            means = group_means(Xr, ref["labels"], ref["groups"]).T
            r = rowwise_spearman(means, ref["table"])
            summary[f"{name}_spearman"] = float(r.mean())
            summary[f"{name}_n_genes"] = len(ref["genes"])
            summary[f"{name}_n_groups"] = len(ref["groups"])
            summary[f"{name}_missing_genes"] = missing
            per_gene.append(pd.Series(r, index=ref["genes"], name=f"{name}_spearman"))
        if not external_only:
            Xr, missing = _matrix(recon, self.cells, list(self.truth_genes))
            means = group_means(Xr, self.region, self.truth_groups).T
            r = rowwise_spearman(means, self.truth_table)
            summary["truth_region_spearman"] = float(r.mean())
            summary["truth_n_genes"] = len(self.truth_genes)
            summary["truth_missing_genes"] = missing
            per_gene.append(pd.Series(r, index=self.truth_genes, name="truth_region_spearman"))
        return summary, pd.concat(per_gene, axis=1)

    def score(self, recon):
        """Score a reconstruction (AnnData or cells x genes DataFrame indexed by cell name).
        Returns (summary dict, per-gene DataFrame). For each external reference the summary
        also gives the score of the true data ("<ref>_spearman_true_data") and the reconstruction's
        score divided by it ("<ref>_spearman_relative"), which can exceed 1."""
        summary, per_gene = self._score(recon)
        for name in self.refs:
            c = self.true_data[f"{name}_spearman"]
            summary[f"{name}_spearman_true_data"] = c
            summary[f"{name}_spearman_relative"] = summary[f"{name}_spearman"] / c if c > 0 else np.nan
        return summary, per_gene


def main():
    import anndata as ad
    warnings.filterwarnings("ignore", category=FutureWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", required=True, help="hidden embryo .h5ad")
    ap.add_argument("--recon", required=True, help="reconstruction .h5ad (cells x genes, same cell names)")
    ap.add_argument("--per-gene", help="optional CSV path for per-gene scores")
    ap.add_argument("--levels", action="store_true", help="split regions by proximal-distal level too")
    args = ap.parse_args()
    bench = RegionBenchmark(ad.read_h5ad(args.truth), levels=args.levels)
    summary, per_gene = bench.score(ad.read_h5ad(args.recon))
    print(json.dumps(summary, indent=2))
    if args.per_gene:
        per_gene.to_csv(args.per_gene)


if __name__ == "__main__":
    main()
