"""Loaders that turn catalogue entries into ``CandidateDataset`` objects.

Every loader is responsible for three things, and the rest of the package
trusts it on all three:

1. **Orientation** -- return observations x genes, never transposed blindly.
2. **Units** -- convert to ``log1p_cp10k`` and say so, or label the real units.
3. **Observation unit** -- record cell / spot / bin truthfully.

MODIFY HERE
-----------
Add a function and register it in ``LOADERS`` at the bottom. Signature:
``fn(entry, data_dir, *, max_obs=None, rng=None, **loader_kwargs) -> CandidateDataset``.
"""
from __future__ import annotations

import glob
import gzip
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from ..normalize import to_log1p_cp10k
from ..schemas import CandidateDataset, PanelQuery

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _subsample_rows(n: int, max_obs: Optional[int], rng) -> Optional[np.ndarray]:
    if max_obs is None or n <= max_obs:
        return None
    rng = np.random.default_rng(0) if rng is None else rng
    return np.sort(rng.choice(n, size=max_obs, replace=False))


def _dense(X) -> np.ndarray:
    from scipy import sparse

    return np.asarray(X.todense()) if sparse.issparse(X) else np.asarray(X)


def renormalise_lognorm(X, target_sum: float = 1e4) -> np.ndarray:
    """``log1p(normalised)`` -> ``log1p(CP10K)``.

    SEU-3D ``X`` is ``log1p`` of *SCT-corrected* counts, whose per-cell totals
    are **not** constant (they equal ``nCount_SCT``). Comparing those values
    with a library-size-normalised atlas would compare two different scales, so
    we invert the log, renormalise each observation to ``target_sum``, and
    re-apply ``log1p``. The SCT variance-stabilisation upstream of this step is
    not undone; it is recorded in ``meta["normalization_note"]``.
    """
    arr = np.expm1(np.asarray(X, dtype=np.float64))
    arr[arr < 0] = 0.0
    totals = arr.sum(axis=1, keepdims=True)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(arr * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_seu3d(entry: Dict[str, Any], data_dir: str, *, filename: str,
               coord_key: str = "X_sagittal", max_obs: Optional[int] = None,
               rng=None, keep_obs: Sequence[str] = ("celltype", "germ_layer", "z", "side"),
               **_) -> CandidateDataset:
    """Xie et al. 2025 Stereo-seq embryo (GSE278603).

    ``coord_key`` -- ``X_sagittal`` is the whole-embryo 2D sagittal projection
    (x, z) and is the default "2D location" for this dataset; ``X_spatial`` is
    the within-section (x, y) coordinate, and ``X_image`` is pixel space.
    """
    import anndata as ad

    path = Path(data_dir) / "raw" / "GSE278603" / filename
    adata = ad.read_h5ad(path)
    idx = _subsample_rows(adata.n_obs, max_obs, rng)
    if idx is not None:
        adata = adata[idx].copy()
    expr = pd.DataFrame(renormalise_lognorm(_dense(adata.X)),
                        index=adata.obs_names.astype(str), columns=adata.var_names.astype(str))
    coords = np.asarray(adata.obsm[coord_key], dtype=float)[:, :2]
    obs = adata.obs[[c for c in keep_obs if c in adata.obs]].copy()
    meta = {k: v for k, v in entry.items() if k not in ("loader", "loader_kwargs")}
    meta["normalization_note"] = (
        "source X is log1p of SCT-corrected counts; renormalised to log1p(CP10K) per observation")
    meta["coord_key"] = coord_key
    meta["source_file"] = str(path)
    # The default 2D view (X_sagittal) is a projection that superimposes cells
    # from every z-section. Spatial autocorrelation on a projection is not
    # comparable across embryos reconstructed from different section counts, so
    # the within-section coordinates and the section label are carried too and
    # used by panelprep.scores.spatial when present.
    if "X_spatial" in adata.obsm:
        meta["coords_section"] = np.asarray(adata.obsm["X_spatial"], dtype=float)[:, :2]
    if "z" in adata.obs:
        meta["section_label"] = adata.obs["z"].to_numpy()
    return CandidateDataset(name=entry["name"], category=entry["category"], expression=expr,
                            coords=coords, obs=obs, units="log1p_cp10k",
                            obs_unit=entry.get("obs_unit", "cell"), meta=meta)


def _clean_symbol(name: str) -> str:
    """MARS-seq tables list synonyms as ``Sox2;Srg`` -- keep the primary symbol."""
    return str(name).split(";")[0].strip()


def _read_marsseq_plate(path: str) -> pd.DataFrame:
    """One MARS-seq plate table: genes x wells on disk -> wells x genes counts.

    The file is genes-by-wells, which is the opposite of the package contract,
    so it is transposed here and nowhere else.
    """
    df = pd.read_csv(path, sep="\t", index_col=0, compression="gzip")
    df.index = [_clean_symbol(i) for i in df.index]
    df = df.groupby(level=0).sum()        # collapse duplicate primary symbols
    return df.T


def load_mittnenzweig(entry: Dict[str, Any], data_dir: str, *,
                      groups: Sequence[str] = ("wt_bl6",),
                      control_groups: Sequence[str] = (),
                      time_range: Optional[tuple] = None,
                      gfp_threshold: float = 500.0,
                      max_obs: Optional[int] = None, rng=None,
                      min_umis: int = 500, **_) -> CandidateDataset:
    """Mittnenzweig et al. 2021 MARS-seq gastrulation atlas (GSE169210).

    Observation unit: one cell (one MARS-seq well). Counts are raw UMIs and are
    normalised here to ``log1p(CP10K)``.

    Cells come from the per-experiment metadata files, which carry plate,
    embryo identity and -- for the wild-type arms -- ``developmental_time`` (the
    study's model-derived developmental age, **not** the dissection label) and a
    metacell id. Only wells listed in the metadata are read; the plate tables
    also contain empty and QC-failed wells.

    Parameters
    ----------
    groups, control_groups
        Metadata groups to include. ``control_groups`` are flagged
        ``obs["condition"] == "control"`` so a causal contrast is available;
        ``groups`` are flagged ``perturbed`` when controls are supplied,
        ``wildtype`` otherwise.
    time_range
        ``(lo, hi)`` filter on ``developmental_time``. Use this to build a
        stage-matched reference. Groups without the column are dropped when a
        range is requested, and that is recorded in ``meta["excluded_groups"]``.
    gfp_threshold
        FACS ``gfp_a`` gate used to label chimera cells as GFP-positive
        (knockout-derived) in ``obs["gfp_positive"]``. This re-derives a gate
        from the deposited FACS values; it is a threshold choice made here, not
        a label taken from the paper, and is recorded in ``meta``.

    MODIFY HERE: ``min_umis`` (per-cell depth filter), ``time_range``,
    ``gfp_threshold``.
    """
    root = Path(data_dir) / "raw" / "GSE169210"
    meta_frames, dropped = [], []
    spec = [(g, "perturbed" if control_groups else "wildtype") for g in groups] + \
           [(g, "control") for g in control_groups]
    for grp, cond in spec:
        mp = root / f"{grp}_meta.txt.gz"
        if not mp.exists():
            dropped.append(f"{grp}: metadata file absent")
            continue
        m = pd.read_csv(mp, sep="\t", index_col=0, compression="gzip")
        m.index = m.index.astype(str)
        m["group"] = grp
        m["condition"] = cond
        if time_range is not None:
            if "developmental_time" not in m.columns:
                dropped.append(f"{grp}: no developmental_time column, excluded by time_range")
                continue
            lo, hi = time_range
            m = m[(m["developmental_time"] >= lo) & (m["developmental_time"] <= hi)]
        if "gfp_a" in m.columns:
            m["gfp_positive"] = m["gfp_a"] >= gfp_threshold
        meta_frames.append(m)
    if not meta_frames:
        raise FileNotFoundError(f"no usable metadata for groups {groups}{control_groups} under {root}")
    meta_df = pd.concat(meta_frames)
    if meta_df.empty:
        raise ValueError(f"no cells left after time_range={time_range}")

    plate_files = sorted(glob.glob(str(root / "plates" / "*.txt.gz")))
    if not plate_files:
        raise FileNotFoundError(f"no plate tables under {root / 'plates'}")
    # Filenames are GSM<id>_<plate>.txt.gz; metadata cell ids are "<plate>.<well>".
    plate_of = {os.path.basename(f).split("_", 1)[1][: -len(".txt.gz")]: f for f in plate_files}

    wanted = meta_df.copy()
    wanted["_well"] = [str(i).split(".", 1)[1] if "." in str(i) else "" for i in wanted.index]
    wanted = wanted[wanted["plate"].astype(str).isin(plate_of) & (wanted["_well"] != "")]
    if wanted.empty:
        raise ValueError("no metadata cells matched a plate table; check the cell_id format")

    rng = np.random.default_rng(0) if rng is None else rng
    plates = list(wanted["plate"].astype(str).unique())
    rng.shuffle(plates)
    blocks, obs_blocks, total = [], [], 0
    for p in plates:
        rows = wanted[wanted["plate"].astype(str) == p]
        counts = _read_marsseq_plate(plate_of[p])
        present = [w in counts.index for w in rows["_well"]]
        if not any(present):
            continue
        sub = counts.loc[[w for w, ok in zip(rows["_well"], present) if ok]]
        sub.index = rows.index[present]
        sub = sub[sub.sum(axis=1) >= min_umis]
        if sub.empty:
            continue
        blocks.append(sub)
        obs_blocks.append(wanted.loc[sub.index].drop(columns=["_well"]))
        total += sub.shape[0]
        if max_obs is not None and total >= max_obs:
            break
    if not blocks:
        raise ValueError(f"no cells passed the min_umis={min_umis} filter")
    counts = pd.concat(blocks).fillna(0.0)
    obs = pd.concat(obs_blocks).loc[counts.index]
    if max_obs is not None and counts.shape[0] > max_obs:
        idx = np.sort(rng.choice(counts.shape[0], max_obs, replace=False))
        counts, obs = counts.iloc[idx], obs.iloc[idx]
    expr = to_log1p_cp10k(counts)
    meta = {k: v for k, v in entry.items() if k not in ("loader", "loader_kwargs")}
    meta["normalization_note"] = "raw MARS-seq UMI counts normalised to log1p(CP10K)"
    meta["groups"] = list(groups) + list(control_groups)
    meta["n_plates_read"] = len(blocks)
    meta["min_umis"] = min_umis
    meta["time_range"] = time_range
    meta["gfp_threshold"] = gfp_threshold if "gfp_a" in obs.columns else None
    if dropped:
        meta["excluded_groups"] = dropped
    return CandidateDataset(name=entry["name"], category=entry["category"], expression=expr,
                            coords=None, obs=obs, units="log1p_cp10k",
                            obs_unit=entry.get("obs_unit", "cell"), meta=meta)


def load_none(entry: Dict[str, Any], data_dir: str, **_) -> CandidateDataset:
    """Embedding-only catalogue entry: no cell x gene matrix exists."""
    meta = {k: v for k, v in entry.items() if k not in ("loader", "loader_kwargs")}
    return CandidateDataset(name=entry["name"], category=entry["category"], expression=None,
                            coords=None, obs=None, units="n/a",
                            obs_unit=entry.get("obs_unit", "cell"), meta=meta)


LOADERS = {"seu3d": load_seu3d, "mittnenzweig": load_mittnenzweig, "none": load_none}


def load_candidate(entry: Dict[str, Any], data_dir: str = "data", *,
                   max_obs: Optional[int] = None, rng=None) -> CandidateDataset:
    fn = LOADERS[entry["loader"]]
    return fn(entry, data_dir, max_obs=max_obs, rng=rng, **entry.get("loader_kwargs", {}))


# ---------------------------------------------------------------------------
# building the query
# ---------------------------------------------------------------------------

def query_from_seu3d(
    filename: str,
    panel: Sequence[str],
    descriptor: str,
    *,
    data_dir: str = "data",
    coord_key: str = "X_sagittal",
    max_obs: Optional[int] = None,
    rng=None,
    name: str = "query",
) -> PanelQuery:
    """Build the ``PanelQuery`` the team is handed: N x panel + 2D coords + text.

    Only the ``panel`` columns are retained -- this is the honest simulation of
    "we could only measure k genes". The full matrix is not carried forward.
    """
    ds = load_seu3d({"name": name, "category": "spatial"}, data_dir, filename=filename,
                    coord_key=coord_key, max_obs=max_obs, rng=rng)
    missing = [g for g in panel if g not in ds.expression.columns]
    if missing:
        raise KeyError(f"panel genes absent from {filename}: {missing}")
    return PanelQuery(
        expression=ds.expression.loc[:, list(panel)], coords=ds.coords, descriptor=descriptor,
        organism="mus_musculus", units="log1p_cp10k", obs_unit="cell", obs=ds.obs, name=name,
        meta={"source_file": filename, "coord_key": coord_key,
              "normalization_note": ds.meta.get("normalization_note"),
              "coords_section": ds.meta.get("coords_section"),
              "section_label": ds.meta.get("section_label")},
    )
