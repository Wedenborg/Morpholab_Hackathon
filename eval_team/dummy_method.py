"""A dummy method that produces a submission the benchmark can score.

It is a stand-in for the method team's output, not a proposal. It only touches the hidden
embryo through the MeasurementOracle (positions + up to 4 genes per experiment), and it
uses the visible embryo (rep1 by default) freely.

Design: one experiment per section. Sections take turns through `n_sets` sets of 4 genes,
so the embryo sees 4 * n_sets distinct genes in total. Genes are cell-type markers of the
visible embryo (default), its most variable genes, or random expressed genes.

Reconstruction: each hidden cell is matched to the visible cells closest to it on the 4
genes measured in its section (all cells tied with the k-th nearest are included, which
matters because many cells read zero on all 4 genes). The cell gets their average
profile over the whole gene library. Measured values replace predictions where they exist.

Usage:
    python -m eval_team.dummy_method --hidden-rep 2 --visible-rep 1 --sets 8 --out-dir dummy_rep2_8sets
writes reconstruction.h5ad and experiments.csv.
"""
import argparse
import pathlib
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist

from .harness import load_embryo as load
from .submission import K_MAX, MeasurementOracle

def choose_gene_sets(visible, library, n_sets, how="markers", seed=0, k=K_MAX):
    genes = pd.Index(visible.var_names).intersection(library)
    X = sp.csr_matrix(visible[:, genes].X)
    if how == "variance":
        m = np.asarray(X.mean(0)).ravel()
        var = np.asarray(X.multiply(X).mean(0)).ravel() - m ** 2
        picked = genes[np.argsort(-var)[: k * n_sets]]
    elif how == "markers":
        # Round-robin over the visible embryo's cell types: each type's next-best marker
        # (largest mean difference to all other cells), as Tangram picks training genes.
        ct = visible.obs["celltype"].astype(str).replace({"pre-somitic": "Pre-somitic"}).to_numpy()
        types = [t for t in pd.unique(ct) if (ct == t).sum() >= 20]
        means = np.vstack([np.asarray(X[ct == t].mean(0)).ravel() for t in types])
        rest = np.vstack([np.asarray(X[ct != t].mean(0)).ravel() for t in types])
        ranked = [list(genes[np.argsort(-(means[i] - rest[i]))]) for i in range(len(types))]
        picked, depth = [], 0
        while len(picked) < k * n_sets:
            for r in ranked:
                if r[depth] not in picked and len(picked) < k * n_sets:
                    picked.append(r[depth])
            depth += 1
    elif how == "random":
        expressed = genes[np.asarray((X > 0).mean(0)).ravel() >= 0.02]
        picked = np.random.default_rng(seed).choice(expressed, k * n_sets, replace=False)
    else:
        raise ValueError(how)
    return [list(picked[i * k:(i + 1) * k]) for i in range(n_sets)]


def run(oracle, visible, n_sets=8, n_neighbours=20, how="markers", seed=0):
    library = oracle.genes
    sets = choose_gene_sets(visible, library, n_sets, how, seed)
    # Visible expression in library gene order (zeros for genes the visible embryo lacks).
    vcols = pd.Index(visible.var_names).get_indexer(library)
    S = sp.csr_matrix((np.ones((vcols >= 0).sum()), (vcols[vcols >= 0], np.where(vcols >= 0)[0])),
                      shape=(visible.n_vars, len(library)))
    Xv = sp.csr_matrix(visible.X) @ S
    lib_pos = {g: i for i, g in enumerate(library)}

    cells = oracle.cells.index
    row_of = {c: i for i, c in enumerate(cells)}
    W_rows, W_cols, W_vals = [], [], []
    meas_rows, meas_cols, meas_vals = [], [], []
    for i, z in enumerate(oracle.sections):
        genes = sets[i % n_sets]
        m = oracle.measure(genes, section=z)
        rows = np.array([row_of[c] for c in m.index])
        gcols = np.array([lib_pos[g] for g in genes])
        Q = m.to_numpy()
        R = Xv[:, gcols].toarray()
        uq, inv = np.unique(Q, axis=0, return_inverse=True)
        D = cdist(uq, R)
        k = min(n_neighbours, R.shape[0])
        thr = np.partition(D, k - 1, axis=1)[:, k - 1]
        sel = D <= thr[:, None] + 1e-9
        for qi, r in enumerate(rows):
            nb = np.where(sel[inv.ravel()[qi]])[0]
            W_rows.append(np.full(len(nb), r))
            W_cols.append(nb)
            W_vals.append(np.full(len(nb), 1 / len(nb)))
        meas_rows.append(np.repeat(rows, len(genes)))
        meas_cols.append(np.tile(gcols, len(rows)))
        meas_vals.append(Q.ravel())
    W = sp.csr_matrix((np.concatenate(W_vals), (np.concatenate(W_rows), np.concatenate(W_cols))),
                      shape=(len(cells), visible.n_obs))
    recon = (W @ Xv).tocsr()
    mr, mc, mv = map(np.concatenate, (meas_rows, meas_cols, meas_vals))
    measured = sp.csr_matrix((np.ones(len(mr)), (mr, mc)), shape=recon.shape)
    values = sp.csr_matrix((mv, (mr, mc)), shape=recon.shape)
    recon = recon - recon.multiply(measured) + values
    recon.eliminate_zeros()
    out = ad.AnnData(X=recon.astype(np.float32), obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=library))
    return out, oracle.log()


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden-rep", type=int, default=2)
    ap.add_argument("--visible-rep", type=int, default=1)
    ap.add_argument("--sets", type=int, default=8, help="number of 4-gene sets rotated across sections")
    ap.add_argument("--genes", choices=["markers", "variance", "random"], default="markers")
    ap.add_argument("--neighbours", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    oracle = MeasurementOracle(load(args.hidden_rep))
    recon, log = run(oracle, load(args.visible_rep), args.sets, args.neighbours, args.genes)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recon.write_h5ad(out / "reconstruction.h5ad")
    log.to_csv(out / "experiments.csv", index=False)
    print(f"{len(log)} experiments, {log['genes'].str.split(';').explode().nunique()} distinct genes; "
          f"wrote {out}/reconstruction.h5ad and experiments.csv")


if __name__ == "__main__":
    main()
