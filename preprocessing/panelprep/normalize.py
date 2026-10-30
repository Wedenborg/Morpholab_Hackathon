"""Gene-identifier and expression-unit handling.

These two things are where cross-dataset comparisons silently go wrong, so
they live in one small module that every other module imports.

MODIFY HERE
-----------
* ``gene_key`` -- change if you want alias resolution (MGI synonyms) instead of
  plain case folding. Nothing else in the package assumes a particular scheme.
* ``to_log1p_cp10k`` -- change the target normalisation; update
  ``schemas.CANONICAL_UNITS`` to match.
"""
from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

_ENS_VERSION = re.compile(r"\.\d+$")


def gene_key(name: object) -> str:
    """Canonical join key for a gene identifier.

    Upper-cases, and strips the version suffix from Ensembl IDs
    (``ENSMUSG00000051951.6`` -> ``ENSMUSG00000051951``). Deliberately does
    *not* resolve aliases: an unmatched gene is reported, not guessed.
    """
    s = str(name).strip().upper()
    return _ENS_VERSION.sub("", s) if s.startswith("ENS") else s


def key_map(names: Iterable[object]) -> dict[str, str]:
    """``{canonical_key: original_name}``, dropping ambiguous keys."""
    seen: dict[str, set] = {}
    for n in names:
        seen.setdefault(gene_key(n), set()).add(str(n))
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}


def resolve_genes(genes: Sequence[str], available: Iterable[object]) -> tuple[list[str], list[str]]:
    """Map ``genes`` onto the names actually present in ``available``.

    Returns ``(resolved_names, missing_genes)``.

    ``resolved_names`` is **the same length as** ``genes`` and positionally
    aligned with it, carrying ``None`` where a gene has no unambiguous match --
    callers zip it against ``genes`` and build presence masks from it, so a
    compacted list would silently misalign them. Names are in the *available*
    namespace, so the non-``None`` entries can index a DataFrame directly.
    ``missing_genes`` lists the requested symbols with no unambiguous match.
    """
    lookup = key_map(available)
    resolved, missing = [], []
    for g in genes:
        hit = lookup.get(gene_key(g))
        resolved.append(hit)
        if hit is None:
            missing.append(str(g))
    return resolved, missing


def to_log1p_cp10k(counts: pd.DataFrame | np.ndarray, target_sum: float = 1e4) -> pd.DataFrame:
    """Counts -> log1p(CP10K). Rows are observations.

    Rows with zero total are left as zeros rather than producing NaN; they are
    reported by ``describe_units`` as empty observations.
    """
    df = pd.DataFrame(counts)
    totals = df.to_numpy(dtype=float).sum(axis=1, keepdims=True)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
    return pd.DataFrame(np.log1p(df.to_numpy(dtype=float) * scale),
                        index=df.index, columns=df.columns)


def looks_like_counts(x: pd.DataFrame | np.ndarray, sample: int = 2000, seed: int = 0) -> bool:
    """True when the matrix looks like raw integer counts (so it needs normalising)."""
    arr = np.asarray(pd.DataFrame(x).to_numpy(), dtype=float)
    if arr.size == 0:
        return False
    rng = np.random.default_rng(seed)
    if arr.shape[0] > sample:
        arr = arr[rng.choice(arr.shape[0], sample, replace=False)]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return False
    return bool(np.all(finite >= 0) and np.allclose(finite, np.rint(finite)) and finite.max() > 30)


def describe_units(x: pd.DataFrame | np.ndarray) -> dict:
    """Cheap numeric fingerprint used in score diagnostics and the data report."""
    arr = np.asarray(pd.DataFrame(x).to_numpy(), dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "n_obs": int(arr.shape[0]),
        "n_genes": int(arr.shape[1]) if arr.ndim == 2 else 0,
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "frac_zero": float((finite == 0).mean()) if finite.size else float("nan"),
        "frac_nonfinite": float(1 - finite.size / arr.size) if arr.size else 0.0,
        "integer_like": looks_like_counts(arr),
    }


def units_compatible(a: str, b: str) -> bool:
    """Whether two unit strings may be compared by a distributional score."""
    return str(a) == str(b)
