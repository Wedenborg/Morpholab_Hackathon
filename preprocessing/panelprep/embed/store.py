"""On-disk store for gene embeddings: compute once, load forever.

Why this module exists
---------------------
Three of the embedding sources in this package are expensive and *do not depend
on the single-cell values at all*: ``esm2_sequence`` (a protein language model
over UniProt sequences), ``literature_text`` (BioGPT) and ``ppi_network``
(node2vec over the STRING graph). Re-running them inside a notebook costs
tens of minutes to hours and returns a bit-identical answer every time, because
their only input is the gene symbol list.

So they are treated as **artefacts, not computations**: built once by
``scripts/build_embedding_store.py`` (or by whatever produced them), written
here, and afterwards *loaded*. The expression embedders (PCA/AE/VAE/joint) are
cheap-ish and dataset-dependent, so they are cached under the same interface
but keyed by dataset.

Layout
------
::

    <root>/
      manifest.csv                     # the inventory: one row per embedding
      expression/<method>__<dataset>.parquet   + .json
      external/<source>.parquet                + .json
      prior/<source>.parquet                   + .json
      custom/<name>.parquet                    + .json

Each ``.parquet`` is the ``(G, d)`` matrix -- **one row per gene, indexed by
gene symbol** (index name ``gene``), ``float32``, columns ``d00, d01, ...``
(or ``f000, f001, ...`` for priors, whose widths are the source's own).
Each ``.json`` sidecar carries ``method``, ``category`` and the full
``provenance`` dict, so a loaded :class:`~panelprep.schemas.GeneEmbedding` is
indistinguishable from a freshly computed one.

Uncovered genes stay as **all-NaN rows**; they are never filled with zeros,
because a zero vector is a position in the embedding space and would silently
cluster every uncovered gene together.

Typical use
-----------
>>> store = EmbeddingStore("outputs/real_pilot/store")
>>> emb = store.ensure("esm2_sequence",                       # loads if present
...                    lambda: build_esm2(genes),             # only called on a miss
...                    group="external")
>>> store.inventory()[["name", "shape", "path"]]
>>> embs = store.load_all(group="expression", genes=universe) # aligned to one gene list

MODIFY HERE
-----------
* ``GROUPS`` -- add a new sub-directory if you add a new kind of embedding.
* ``_COLUMN_PREFIX`` -- rename the generated column labels.
* :meth:`EmbeddingStore.ensure` -- change the staleness policy (currently:
  a file that exists is trusted; pass ``recompute=True`` to overwrite).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ..schemas import CATEGORIES, GeneEmbedding

__all__ = ["EmbeddingStore", "GROUPS"]

#: Sub-directories. MODIFY HERE to add a new kind of embedding.
GROUPS: tuple[str, ...] = ("expression", "external", "prior", "custom")

_COLUMN_PREFIX = {"prior": "f", "default": "d"}


def _fname(name: str) -> str:
    """``pca_baseline::seu3d_E7.5_rep2`` -> ``pca_baseline__seu3d_E7.5_rep2``."""
    return name.replace("::", "__").replace("/", "__")


def _columns(width: int, group: str) -> list[str]:
    prefix = _COLUMN_PREFIX.get(group, _COLUMN_PREFIX["default"])
    pad = max(2, len(str(max(width - 1, 0))))
    return [f"{prefix}{i:0{pad}d}" for i in range(width)]


class EmbeddingStore:
    """A directory of gene embeddings with a manifest.

    Parameters
    ----------
    root
        Directory to read/write. Created on demand.
    strict
        If True (default) :meth:`get` raises on a missing embedding; if False it
        returns ``None``, which is convenient in notebooks.
    """

    def __init__(self, root: str | Path, *, strict: bool = True) -> None:
        self.root = Path(root)
        self.strict = bool(strict)
        for group in GROUPS:
            (self.root / group).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def path(self, name: str, group: str = "custom") -> Path:
        """Parquet path for ``name``; does not check existence."""
        if group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}; got {group!r}")
        return self.root / group / f"{_fname(name)}.parquet"

    def _sidecar(self, path: Path) -> Path:
        return path.with_suffix(".json")

    def _find(self, name: str) -> Optional[tuple[Path, str]]:
        """Locate ``name`` in any group. First match in ``GROUPS`` order wins."""
        for group in GROUPS:
            p = self.path(name, group)
            if p.exists():
                return p, group
        return None

    # ------------------------------------------------------------------- read
    def has(self, name: str) -> bool:
        """True if ``name`` is already on disk (in any group)."""
        return self._find(name) is not None

    def get(self, name: str, *, genes: Optional[Sequence[str]] = None
            ) -> Optional[GeneEmbedding]:
        """Load one embedding.

        ``genes`` reindexes the result to exactly that gene list (missing genes
        become all-NaN rows), which is how you align several embeddings that
        were built over different universes.
        """
        found = self._find(name)
        if found is None:
            if self.strict:
                raise KeyError(
                    f"{name!r} is not in the store at {self.root}. "
                    f"Available: {', '.join(self.names()) or '(empty)'}"
                )
            return None
        path, group = found
        matrix = pd.read_parquet(path)
        meta: Dict[str, Any] = {}
        side = self._sidecar(path)
        if side.exists():
            meta = json.loads(side.read_text())
        if genes is not None:
            matrix = matrix.reindex(list(genes))
        emb = GeneEmbedding(
            matrix=matrix.astype("float32"),
            method=meta.get("method", name.split("::")[0]),
            category=meta.get("category", "single_cell"),
            provenance=dict(meta.get("provenance", {})),
        )
        emb.provenance.setdefault("loaded_from", str(path))
        emb.provenance["store_group"] = group
        return emb

    def load_all(
        self,
        names: Optional[Sequence[str]] = None,
        *,
        group: Optional[str] = None,
        category: Optional[str] = None,
        genes: Optional[Sequence[str]] = None,
    ) -> Dict[str, GeneEmbedding]:
        """Load many embeddings as ``{name: GeneEmbedding}``.

        With ``genes`` every returned matrix carries the *same* index, so the
        agreement and pair-plot functions can be applied directly.
        """
        wanted = list(names) if names is not None else self.names(
            group=group, category=category)
        return {n: self.get(n, genes=genes) for n in wanted}

    def names(self, *, group: Optional[str] = None,
              category: Optional[str] = None) -> list[str]:
        """Sorted embedding names currently on disk."""
        inv = self.inventory()
        if inv.empty:
            return []
        if group is not None:
            inv = inv[inv["group"] == group]
        if category is not None:
            inv = inv[inv["category"] == category]
        return sorted(inv["name"].tolist())

    # ------------------------------------------------------------------ write
    def put(
        self,
        embedding: GeneEmbedding,
        name: Optional[str] = None,
        *,
        group: str = "custom",
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Write one embedding (matrix + provenance sidecar) and return its path."""
        name = name or embedding.method
        path = self.path(name, group)
        matrix = pd.DataFrame(embedding.matrix).astype("float32")
        matrix.index = pd.Index([str(g) for g in matrix.index], name="gene")
        if not all(isinstance(c, str) for c in matrix.columns):
            matrix.columns = _columns(matrix.shape[1], group)
        matrix.to_parquet(path)
        side = {
            "name": name,
            "method": embedding.method,
            "category": embedding.category,
            "group": group,
            "n_genes": int(matrix.shape[0]),
            "dim": int(matrix.shape[1]),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "provenance": _jsonable(embedding.provenance),
        }
        if extra:
            side.update(_jsonable(dict(extra)))
        self._sidecar(path).write_text(json.dumps(side, indent=2, sort_keys=True))
        return path

    def ensure(
        self,
        name: str,
        build: Callable[[], GeneEmbedding],
        *,
        group: str = "custom",
        recompute: bool = False,
        verbose: bool = True,
    ) -> GeneEmbedding:
        """Load ``name`` if it exists, otherwise build it once and save it.

        This is the function that keeps ESM-2 / BioGPT / node2vec out of the
        notebook's critical path. ``build`` is a zero-argument callable (use a
        ``lambda`` or ``functools.partial``) and is invoked **only** on a cache
        miss, or when ``recompute=True``.
        """
        if self.has(name) and not recompute:
            if verbose:
                print(f"  [store] load  {name}  <- {self._find(name)[0]}")
            return self.get(name)
        if verbose:
            print(f"  [store] build {name}  (cache {'bypassed' if recompute else 'miss'})")
        t0 = time.time()
        emb = build()
        emb.provenance.setdefault("build_seconds", round(time.time() - t0, 2))
        path = self.put(emb, name, group=group)
        if verbose:
            print(f"  [store] saved {name}  -> {path}  "
                  f"({emb.matrix.shape[0]} x {emb.matrix.shape[1]}, "
                  f"{emb.provenance['build_seconds']:.1f}s)")
        return emb

    def put_priors(self, priors: Mapping[str, Any], *, verbose: bool = True
                   ) -> Dict[str, Path]:
        """Ingest ``{source: PriorFeatures}`` from :mod:`panelprep.priors`.

        ``PriorFeatures`` and ``GeneEmbedding`` have the same ``(G, f)`` matrix
        contract, so this is a straight field copy into the ``prior`` group.
        """
        out: Dict[str, Path] = {}
        for source, pf in priors.items():
            prov = dict(getattr(pf, "provenance", {}) or {})
            prov.setdefault("modality", getattr(pf, "modality", ""))
            emb = GeneEmbedding(
                matrix=pf.matrix, method=source,
                category=getattr(pf, "category", "non_transcriptomic"),
                provenance=prov,
            )
            out[source] = self.put(emb, source, group="prior")
            if verbose:
                print(f"  [store] prior {source}: {pf.matrix.shape} -> {out[source]}")
        return out

    def import_parquet(
        self,
        path: str | Path,
        name: str,
        *,
        group: str = "prior",
        category: str = "non_transcriptomic",
        method: Optional[str] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        """Adopt a parquet written elsewhere (e.g. by a worker) into the store.

        The file must be gene-indexed; everything else is metadata you supply.
        """
        matrix = pd.read_parquet(path)
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}; got {category!r}")
        emb = GeneEmbedding(matrix=matrix, method=method or name, category=category,
                            provenance=dict(provenance or {"imported_from": str(path)}))
        return self.put(emb, name, group=group)

    # -------------------------------------------------------------- inventory
    def inventory(self, *, write: bool = False) -> pd.DataFrame:
        """The table "what embeddings do I have, what shape, and where".

        One row per embedding with ``name, group, method, category, n_genes,
        dim, shape, n_nan_genes, path, size_kb`` -- i.e. the answer to both
        "what is the output shape" and "where is it saved".
        """
        rows = []
        for group in GROUPS:
            for path in sorted((self.root / group).glob("*.parquet")):
                side = self._sidecar(path)
                meta = json.loads(side.read_text()) if side.exists() else {}
                try:
                    matrix = pd.read_parquet(path)
                    n_genes, dim = matrix.shape
                    n_nan = int(matrix.isna().all(axis=1).sum())
                except Exception as exc:                      # unreadable file
                    n_genes, dim, n_nan = -1, -1, -1
                    meta.setdefault("error", str(exc))
                rows.append({
                    "name": meta.get("name", path.stem),
                    "group": group,
                    "method": meta.get("method", path.stem.split("__")[0]),
                    "category": meta.get("category", ""),
                    "n_genes": n_genes,
                    "dim": dim,
                    "shape": f"{n_genes} x {dim}",
                    "n_nan_genes": n_nan,
                    "path": str(path),
                    "size_kb": round(path.stat().st_size / 1024, 1),
                    "written_at": meta.get("written_at", ""),
                })
        inv = pd.DataFrame(rows)
        if not inv.empty:
            inv = inv.sort_values(["group", "name"]).reset_index(drop=True)
        if write:
            (self.root / "manifest.csv").parent.mkdir(parents=True, exist_ok=True)
            inv.to_csv(self.root / "manifest.csv", index=False)
        return inv

    def common_genes(self, names: Optional[Sequence[str]] = None,
                     *, how: str = "intersection") -> list[str]:
        """Gene index shared by several embeddings (``intersection``/``union``)."""
        names = list(names) if names is not None else self.names()
        idx: Optional[pd.Index] = None
        for n in names:
            found = self._find(n)
            if found is None:
                continue
            gi = pd.read_parquet(found[0], columns=[]).index
            idx = gi if idx is None else (
                idx.intersection(gi) if how == "intersection" else idx.union(gi))
        return sorted(idx) if idx is not None else []

    def __repr__(self) -> str:
        inv = self.inventory()
        by = "" if inv.empty else ", ".join(
            f"{g}={int((inv['group'] == g).sum())}" for g in GROUPS
            if int((inv["group"] == g).sum()))
        return f"EmbeddingStore({str(self.root)!r}, {len(inv)} embeddings: {by})"


def _jsonable(obj: Any) -> Any:
    """Make provenance JSON-safe without losing information silently."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return {"_dataframe_shape": list(obj.shape)}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)
