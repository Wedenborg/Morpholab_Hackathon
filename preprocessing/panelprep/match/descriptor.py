"""Rank catalogue entries by how well they match the experiment descriptor.

This is a *text* match over catalogue metadata (organism, stage, modality,
tissue, description) -- it says which datasets plausibly describe the same
biology. It is deliberately separate from the agreement scores, which measure
whether the measured panel genes actually behave the same way. The report
shows both: a dataset can match the story and still disagree with the data.

MODIFY HERE
-----------
Register another matcher with ``@MATCHERS.register("name")``; signature
``fn(descriptor, entries, **params) -> dict[name, float]`` with higher = better
match. ``Config.match_method`` selects it.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..registry import Registry

MATCHERS = Registry("matcher")

_TOKEN = re.compile(r"[a-z0-9.]+")

#: Stage tokens are weighted up: "E7.5" matching "E7.5" is far more
#: informative than a shared word like "mouse".
STAGE_RE = re.compile(r"\b[eE](\d+(?:\.\d+)?)\b")


def _tokens(text: str) -> List[str]:
    return _TOKEN.findall(str(text).lower())


def _entry_text(e: Dict[str, Any]) -> str:
    keys = ("name", "category", "organism", "stage", "modality", "tissue", "description", "accession")
    return " ".join(str(e.get(k, "")) for k in keys)


@MATCHERS.register("tfidf", description="cosine similarity of TF-IDF vectors over catalogue text")
def tfidf_match(descriptor: str, entries: Sequence[Dict[str, Any]], *,
                stage_bonus: float = 0.25, **_) -> Dict[str, float]:
    """TF-IDF cosine similarity plus an explicit embryonic-stage bonus."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [_entry_text(e) for e in entries]
    vec = TfidfVectorizer(token_pattern=r"[a-zA-Z0-9.]+", lowercase=True, sublinear_tf=True)
    M = vec.fit_transform(docs + [descriptor])
    D, q = M[:-1], M[-1]
    num = np.asarray((D @ q.T).todense()).ravel()
    den = np.sqrt(np.asarray(D.multiply(D).sum(1)).ravel()) * np.sqrt((q.multiply(q)).sum())
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    q_stages = set(STAGE_RE.findall(descriptor))
    out = {}
    for e, c in zip(entries, cos):
        s = set(STAGE_RE.findall(_entry_text(e)))
        out[e["name"]] = float(c + (stage_bonus if q_stages & s else 0.0))
    return out


@MATCHERS.register("keyword", description="fraction of descriptor tokens present in the entry text")
def keyword_match(descriptor: str, entries: Sequence[Dict[str, Any]], **_) -> Dict[str, float]:
    q = set(_tokens(descriptor))
    return {e["name"]: (len(q & set(_tokens(_entry_text(e)))) / max(len(q), 1)) for e in entries}


def match_descriptor(descriptor: str, entries: Sequence[Dict[str, Any]], *,
                     method: str = "tfidf", top_k: int = 0, **params) -> pd.DataFrame:
    """Ranked match table. ``top_k=0`` keeps every candidate (ranked)."""
    scores = MATCHERS[method].fn(descriptor, list(entries), **params)
    df = pd.DataFrame({
        "dataset": [e["name"] for e in entries],
        "category": [e["category"] for e in entries],
        "stage": [e.get("stage") for e in entries],
        "match_score": [scores[e["name"]] for e in entries],
    }).sort_values("match_score", ascending=False).reset_index(drop=True)
    df["match_rank"] = np.arange(1, len(df) + 1)
    return df.head(top_k) if top_k else df
