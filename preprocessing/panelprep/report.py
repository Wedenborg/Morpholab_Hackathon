"""The comparison report: markdown generated FROM THE REGISTRIES.

Why generated rather than written
---------------------------------
The brief's hard requirement is "make it easy that we add a new score and
visualize its effect" / "adding a metric must automatically update the
comparison report". A hand-written methods list breaks that the first time
someone registers a score and forgets to edit the prose. So every list in this
report is derived:

* the **methods table** iterates ``panelprep.scores.SCORES`` and
  ``panelprep.embed.EMBEDDERS`` and reads ``entry.meta`` -- there is no literal
  method name anywhere in this module;
* the **agreement table** is built by ``panelprep.viz.score_table`` from the
  long-format frame, so its columns are whatever actually ran;
* the **status section** takes *implemented* from the registries and
  *evaluated* from what produced a non-null result in this run, so a method
  cannot be claimed to work because someone wrote that it does.

``tests/test_viz.py`` registers a dummy score inside the test and asserts it
appears here with no edit to this file.

Four statuses, with fixed meanings
----------------------------------
``implemented``
    registered in a registry; it can be called.
``evaluated``
    it ran in *this* report's inputs and produced at least one usable value
    (a score: a row with ``status == "ok"`` and a finite value; an embedder: an
    embedding with at least one non-``NaN`` gene row).
``blocked``
    requested in the configuration but unavailable -- not registered, a missing
    dependency, or an error. Taken from the config/registry mismatch and from
    ``manual_status``.
``deferred``
    deliberately out of scope for this run; must come from ``manual_status``
    with a reason, because only a human can defer something.

MODIFY HERE
-----------
* ``manual_status`` -- ``{"name": {"status": "deferred", "reason": "..."}}``
  (a bare string ``"blocked: no network"`` is also accepted) for methods whose
  state cannot be read off the run.
* ``SECTIONS`` -- the section order.
* ``build_report(..., compute_quality=False)`` to skip the permutation nulls
  for a fast draft.
"""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .schemas import GeneAnnotation, GeneEmbedding

#: Section order of the generated markdown. MODIFY HERE to reorder/remove.
SECTIONS: tuple[str, ...] = (
    "configuration", "methods", "match", "agreement", "status", "embeddings",
    "annotations", "figures", "notes",
)

STATUS_ORDER = ("evaluated", "implemented", "blocked", "deferred")


# ---------------------------------------------------------------------------
# registry-derived pieces
# ---------------------------------------------------------------------------

def _registries() -> Dict[str, Any]:
    """``{"score": SCORES, "embedder": EMBEDDERS}``; a failed import is reported."""
    out: Dict[str, Any] = {}
    problems: Dict[str, str] = {}
    try:
        from .scores import SCORES
        out["score"] = SCORES
    except Exception as exc:                      # pragma: no cover - defensive
        problems["score"] = f"{type(exc).__name__}: {exc}"
    try:
        from .embed import EMBEDDERS
        out["embedder"] = EMBEDDERS
    except Exception as exc:                      # pragma: no cover - defensive
        problems["embedder"] = f"{type(exc).__name__}: {exc}"
    out["_problems"] = problems
    return out


def methods_table() -> pd.DataFrame:
    """Every registered method and its declared metadata, as a DataFrame.

    Observation unit: one registered callable. Columns are ``kind``, ``name``
    and the union of the metadata keys the registries carry (``orientation``,
    ``applies_to``, ``description``, ``mode``, ``category``, ...). Nothing here
    is hand-maintained: register a method and it appears. Missing-data
    behaviour: a metadata key a method does not declare is empty, not guessed.

    MODIFY HERE: nothing -- add metadata to the ``@SCORES.register`` /
    ``@EMBEDDERS.register`` call and the column appears automatically.
    """
    regs = _registries()
    rows = []
    for kind in ("score", "embedder"):
        reg = regs.get(kind)
        if reg is None:
            continue
        for e in sorted(reg, key=lambda e: e.name):
            meta = {k: v for k, v in dict(e.meta).items() if k != "defaults"}
            defaults = dict(e.meta).get("defaults") or {}
            rows.append({"kind": kind, "name": e.name,
                         **{k: _scalar(v) for k, v in meta.items()},
                         "defaults": ", ".join(f"{k}={v}" for k, v in defaults.items())})
    if not rows:
        return pd.DataFrame(columns=["kind", "name", "description"])
    df = pd.DataFrame(rows)
    front = [c for c in ("kind", "name", "orientation", "category", "mode", "applies_to",
                         "needs_coords", "units_sensitive") if c in df.columns]
    rest = [c for c in df.columns if c not in front and c != "description"]
    cols = front + rest + (["description"] if "description" in df.columns else [])
    return df.loc[:, cols].fillna("")


def _scalar(v: Any) -> Any:
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x) for x in v)
    return v


# ---------------------------------------------------------------------------
# run-derived pieces
# ---------------------------------------------------------------------------

def _evaluated_scores(score_results: Optional[pd.DataFrame]) -> Dict[str, dict]:
    """Per score: did it produce a usable number in this run, and on how many datasets."""
    out: Dict[str, dict] = {}
    if score_results is None or len(score_results) == 0:
        return out
    df = pd.DataFrame(score_results)
    for name, grp in df.groupby("score", sort=True):
        vals = pd.to_numeric(grp["value"], errors="coerce")
        ok = grp["status"].eq("ok") & np.isfinite(vals)
        out[str(name)] = {
            "n_ok": int(ok.sum()), "n_cells": int(len(grp)),
            "statuses": ", ".join(f"{k}:{v}" for k, v in
                                  grp["status"].value_counts().sort_index().items()),
        }
    return out


def _evaluated_embedders(embeddings: Optional[Mapping[str, GeneEmbedding]]) -> Dict[str, dict]:
    """Per embedder name: how many embeddings it produced with >=1 usable gene."""
    out: Dict[str, dict] = {}
    for key, emb in (embeddings or {}).items():
        base = key.split("::")[0]
        M = pd.DataFrame(emb.matrix).astype(float)
        usable = int(np.isfinite(M.to_numpy()).all(axis=1).sum())
        rec = out.setdefault(base, {"n_embeddings": 0, "n_usable_genes": 0, "keys": []})
        rec["n_embeddings"] += 1
        rec["n_usable_genes"] += usable
        rec["keys"].append(key)
    return out


def _parse_manual(entry: Any) -> tuple[str, str]:
    if isinstance(entry, Mapping):
        return str(entry.get("status", "blocked")), str(entry.get("reason", ""))
    s = str(entry)
    if ":" in s:
        head, _, tail = s.partition(":")
        if head.strip() in STATUS_ORDER:
            return head.strip(), tail.strip()
    return (s.strip(), "") if s.strip() in STATUS_ORDER else ("blocked", s.strip())


def method_status_table(
    config,
    *,
    score_results: Optional[pd.DataFrame] = None,
    embeddings: Optional[Mapping[str, GeneEmbedding]] = None,
    manual_status: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """One row per method: implemented / evaluated / blocked / deferred, with evidence.

    Observation unit: one method (a registry entry, a configuration request, or
    a ``manual_status`` key). ``implemented`` comes from the registries;
    ``evaluated`` from what produced a non-null result in the inputs passed to
    :func:`build_report`; ``blocked`` from a configuration request with no
    registry entry or from ``manual_status``; ``deferred`` only from
    ``manual_status`` -- a run cannot infer intent. Missing-data behaviour: a
    registered method that was not requested in the configuration is reported
    ``implemented`` with the evidence "not selected in this run", not silently
    dropped.

    MODIFY HERE: ``manual_status`` for anything whose state is not visible in
    the run (an embedder blocked by a missing weight file, a metric deferred).
    """
    regs = _registries()
    manual = {str(k): _parse_manual(v) for k, v in (manual_status or {}).items()}
    ev_s, ev_e = _evaluated_scores(score_results), _evaluated_embedders(embeddings)
    requested = {"score": list(getattr(config, "scores", []) or []),
                 "embedder": list(getattr(config, "embedders", []) or [])}

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for kind in ("score", "embedder"):
        reg = regs.get(kind)
        names = sorted(set((reg.names() if reg is not None else [])) | set(requested[kind]))
        for name in names:
            registered = reg is not None and name in reg
            req = name in requested[kind]
            if kind == "score":
                ev = ev_s.get(name)
                evaluated = bool(ev and ev["n_ok"] > 0)
                evidence = (f"{ev['n_ok']}/{ev['n_cells']} dataset cells ok ({ev['statuses']})"
                            if ev else "no result rows in this run")
            else:
                ev = ev_e.get(name)
                evaluated = bool(ev and ev["n_usable_genes"] > 0)
                evidence = (f"{ev['n_embeddings']} embedding(s), "
                            f"{ev['n_usable_genes']} usable gene rows"
                            if ev else "no embedding produced in this run")
            if not registered:
                status, reason = "blocked", "requested in config but not registered"
            elif evaluated:
                status, reason = "evaluated", ""
            else:
                status = "implemented"
                reason = "" if req else "not selected in this run"
            if name in manual:                     # explicit human call wins
                status, reason = manual[name]
                evidence = evidence + (f"; manual: {reason}" if reason else "; manual")
            rows.append({"kind": kind, "method": name, "registered": registered,
                         "requested": req, "status": status,
                         "evidence": evidence if not reason or status in ("evaluated",
                                                                          "implemented")
                                     else f"{reason}; {evidence}",
                         "reason": reason})
            seen.add((kind, name))

    for name, (status, reason) in manual.items():   # manual-only entries (deferred ideas)
        if any(name == r["method"] for r in rows):
            continue
        rows.append({"kind": "other", "method": name, "registered": False, "requested": False,
                     "status": status, "reason": reason,
                     "evidence": f"declared by hand: {reason}" if reason else "declared by hand"})

    df = pd.DataFrame(rows)
    order = {s: i for i, s in enumerate(STATUS_ORDER)}
    return df.sort_values(["kind", "status", "method"],
                          key=lambda c: c.map(order) if c.name == "status" else c) \
             .reset_index(drop=True)


def embedding_inventory(
    embeddings: Optional[Mapping[str, GeneEmbedding]],
    annotation: Optional[GeneAnnotation] = None,
    *,
    label_col: str = "top_pathway",
    compute_quality: bool = True,
    quality_kwargs: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Per-embedding shape, NaN-gene count and label-organisation numbers.

    Observation unit: one embedding (one ``GeneEmbedding``; genes are its
    rows). Missing-data behaviour: ``n_genes_nan`` counts genes with no usable
    embedding row -- they are excluded from the quality statistics, which are
    reported together with their label-permutation null so a silhouette of
    0.05 cannot be read as success.

    MODIFY HERE: ``compute_quality=False`` for a fast draft;
    ``quality_kwargs`` (e.g. ``n_permutations``) for the null resolution.
    """
    from .viz.embeddings import clean_matrix, embedding_quality

    rows = []
    for key, emb in (embeddings or {}).items():
        M, counts = clean_matrix(emb)
        rec = {
            "embedding": key, "method": emb.method, "category": emb.category,
            "n_genes": counts["n_total"], "dim": counts["dim"],
            "shape": f"{counts['n_total']} x {counts['dim']}",
            "n_genes_nan": counts["n_total"] - counts["n_kept"],
            "n_genes_usable": counts["n_kept"],
            "provenance": ", ".join(f"{k}={v}" for k, v in
                                    list(dict(emb.provenance or {}).items())[:4]),
        }
        if compute_quality and annotation is not None and counts["n_kept"] >= 3:
            q = embedding_quality(emb, annotation, label_col,
                                  **dict(quality_kwargs or {"n_permutations": 20}))
            rec.update({k: q.get(k) for k in ("status", "silhouette", "silhouette_null_mean",
                                              "silhouette_p", "knn_agreement", "knn_null_mean",
                                              "knn_p", "n_genes_scored", "n_classes")})
        else:
            rec["status"] = "quality_not_computed" if annotation is not None else "no_annotation"
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# markdown helpers
# ---------------------------------------------------------------------------

def _md_table(df: Optional[pd.DataFrame], *, index: bool = False, floatfmt: str = "{:.4g}",
              empty: str = "_(none)_") -> str:
    if df is None or len(df) == 0:
        return empty
    d = pd.DataFrame(df).copy()
    if index:
        d = d.reset_index()
    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "| " + " | ".join("---" for _ in d.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _md_matrix(raw: pd.DataFrame, status: pd.DataFrame, *, labels: Mapping[str, str],
               floatfmt: str = "{:.4g}") -> str:
    """Dataset x score matrix where value-less cells show their status label."""
    if raw.empty:
        return "_(none)_"
    head = "| dataset | " + " | ".join(str(c) for c in raw.columns) + " |"
    rule = "| --- | " + " | ".join("---" for _ in raw.columns) + " |"
    body = []
    for i, ds in enumerate(raw.index):
        cells = []
        for j in range(raw.shape[1]):
            st = str(status.iat[i, j])
            v = raw.iat[i, j]
            cells.append(floatfmt.format(v) if st == "ok" and np.isfinite(v)
                         else f"_{labels.get(st, st)}_")
        body.append(f"| {ds} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


def _rel(path: str, base: Optional[Path]) -> str:
    p = Path(str(path))
    if base is None:
        return str(p)
    try:
        return str(p.relative_to(base))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def build_report(
    config,
    *,
    match_table: Optional[pd.DataFrame] = None,
    score_results: Optional[pd.DataFrame] = None,
    embeddings: Optional[Mapping[str, GeneEmbedding]] = None,
    annotation: Optional[GeneAnnotation] = None,
    figures: Optional[Mapping[str, str]] = None,
    out_path: Optional[str] = None,
    manual_status: Optional[Mapping[str, Any]] = None,
    normalise: str = "rank",
    label_col: str = "top_pathway",
    compute_quality: bool = True,
    quality_kwargs: Optional[Mapping[str, Any]] = None,
    extra_notes: Optional[Sequence[str]] = None,
) -> str:
    """Write (and return) the markdown comparison report for one run.

    Observation unit: one run of the pipeline. Everything is derived from the
    arguments and the registries; nothing is hand-listed, so registering a new
    score or embedder updates the methods table, the agreement table and the
    status section with no edit here. Normalisation assumption: the agreement
    ranking uses the same column-wise, orientation-corrected normalisation as
    the heatmap (``panelprep.viz.score_table.agreement_frame``), and the raw
    values are printed next to it so nothing is hidden. Missing-data
    behaviour: every argument is optional -- an absent input produces a section
    that says it is absent; value-less score cells are shown with their status
    label and excluded from the ranking; genes with no embedding are counted in
    the inventory.

    Parameters
    ----------
    config
        The resolved ``panelprep.config.Config``.
    figures
        ``{caption: path}``; paths are rewritten relative to ``out_path``.
    manual_status
        ``{"method": {"status": "deferred", "reason": "..."}}`` for blocked or
        deferred methods -- the only place a human statement enters the report.

    MODIFY HERE: ``SECTIONS`` (order), ``manual_status``, ``compute_quality``.
    """
    from .viz.score_table import STATUS_LABELS, agreement_frame

    out_dir = Path(out_path).parent if out_path else None
    notes: list[str] = list(extra_notes or [])
    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    tab = None
    if score_results is not None and len(score_results):
        tab = agreement_frame(score_results, normalise=normalise)
        notes += [f"score table: {n}" for n in tab.notes]
    regs = _registries()
    for kind, msg in (regs.get("_problems") or {}).items():
        notes.append(f"{kind} registry failed to import: {msg}")

    mt = methods_table()
    st = method_status_table(config, score_results=score_results, embeddings=embeddings,
                             manual_status=manual_status)
    inv = embedding_inventory(embeddings, annotation, label_col=label_col,
                              compute_quality=compute_quality, quality_kwargs=quality_kwargs)

    parts: Dict[str, str] = {}

    parts["configuration"] = "\n".join([
        "## Resolved configuration",
        "",
        "Single entry point: `panelprep.config.Config`. Everything below drove this run.",
        "",
        _md_table(pd.DataFrame({"setting": list(cfg), "value": [_scalar(v) for v in cfg.values()]})),
    ])

    n_scores = int((mt["kind"] == "score").sum()) if len(mt) else 0
    n_emb = int((mt["kind"] == "embedder").sum()) if len(mt) else 0
    parts["methods"] = "\n".join([
        "## Methods available (generated from the registries)",
        "",
        f"{n_scores} score(s) and {n_emb} embedder(s) are registered. This table is produced by "
        "iterating `panelprep.scores.SCORES` and `panelprep.embed.EMBEDDERS` and reading "
        "`entry.meta`: registering a method with `@SCORES.register(...)` / "
        "`@EMBEDDERS.register(...)` is sufficient for it to appear here, in the agreement table "
        "and in the heatmap.",
        "",
        _md_table(mt),
    ])

    parts["match"] = "\n".join([
        "## Descriptor match ranking (stage 1a)",
        "",
        "Text match of the experiment descriptor against catalogue metadata. This says which "
        "datasets plausibly describe the same biology; the agreement scores below say whether "
        "the measured panel genes actually behave the same way. A dataset can match the story "
        "and still disagree with the data.",
        "",
        _md_table(match_table, empty="_(no match table supplied)_"),
    ])

    if tab is None or tab.raw.empty:
        parts["agreement"] = "## Panel-gene agreement (stage 1b)\n\n_(no score results supplied)_"
    else:
        legend = ", ".join(f"`{v}` = {k.replace('_', ' ')}"
                           for k, v in STATUS_LABELS.items()
                           if (tab.status == k).to_numpy().any())
        n_grey = int((tab.status != "ok").to_numpy().sum())
        cov = tab.coverage.copy()
        cov.insert(0, "dataset", cov.index)
        parts["agreement"] = "\n".join([
            "## Panel-gene agreement (stage 1b)",
            "",
            "### Raw values",
            "",
            "Orientation differs per column, so these numbers must not be compared across "
            "columns: " + "; ".join(f"`{s}` {tab.orientation[s].replace('_', ' ')}"
                                    for s in tab.raw.columns) + ".",
            "",
            _md_matrix(tab.raw, tab.status, labels=STATUS_LABELS),
            "",
            f"Value-less cells: {n_grey} of {tab.raw.size}." + (f" Legend: {legend}." if legend else ""),
            "",
            f"### Agreement-normalised ({tab.normalise} within each column, "
            "direction-corrected)",
            "",
            "Every column is mapped onto a common agreement direction using the `orientation` "
            "carried by each `ScoreResult` (`lower_is_better` columns are negated), then "
            "normalised across datasets. High = agrees with the query panel, in every column. "
            "This is the quantity the heatmap colours red.",
            "",
            _md_matrix(tab.agreement, tab.status, labels=STATUS_LABELS),
            "",
            "### Ranking by mean agreement",
            "",
            _md_table(tab.ranking),
            "",
            "### Panel-gene coverage (used/requested per cell)",
            "",
            "A score computed on 2 of 4 panel genes is not the same measurement as one computed "
            "on all 4; the counts are kept per cell rather than averaged away.",
            "",
            _md_table(cov),
        ])

    counts = st["status"].value_counts().reindex(STATUS_ORDER).fillna(0).astype(int)
    parts["status"] = "\n".join([
        "## Status of every requested method",
        "",
        "`implemented` = registered and callable; `evaluated` = produced at least one usable "
        "value in this run (a score cell with `status == ok` and a finite value, or an "
        "embedding with at least one non-NaN gene row); `blocked` = requested but unavailable; "
        "`deferred` = deliberately out of scope (only ever stated by hand, with a reason).",
        "",
        "  ".join(f"**{k}**: {v}" for k, v in counts.items()),
        "",
        _md_table(st),
    ])

    if len(inv) == 0:
        parts["embeddings"] = "## Gene-embedding inventory (items 0, 2, 3)\n\n" \
                              "_(no embeddings supplied)_"
    else:
        by_cat = inv.groupby("category")["embedding"].count().to_dict() if "category" in inv else {}
        parts["embeddings"] = "\n".join([
            "## Gene-embedding inventory (items 0, 2, 3)",
            "",
            f"{len(inv)} embedding(s) by category: "
            + (", ".join(f"{k}: {v}" for k, v in sorted(by_cat.items())) or "-") + ".",
            "",
            "`silhouette` and `knn_agreement` measure how well the annotation labels "
            f"(`{label_col}`) are organised in the **embedding** space (not the 2D picture), "
            "each with a label-permutation null and a one-sided p-value; `n_genes_nan` counts "
            "genes the embedder could not place, which are excluded from both statistics.",
            "",
            _md_table(inv),
        ])

    if annotation is None:
        parts["annotations"] = "## Gene annotations (item 4)\n\n_(no annotation supplied)_"
    else:
        at = pd.DataFrame(annotation.table)
        flags = {c: int(at[c].fillna(False).astype(bool).sum())
                 for c in ("is_ligand", "is_receptor", "is_tf") if c in at.columns}
        top = at[label_col].value_counts().head(10) if label_col in at.columns else pd.Series(dtype=int)
        parts["annotations"] = "\n".join([
            "## Gene annotations (item 4)",
            "",
            f"{len(at)} genes annotated; roles: "
            + (", ".join(f"{k.replace('is_', '')} {v}" for k, v in flags.items()) or "-")
            + f". Sources: {_scalar(annotation.sources) or '-'}.",
            "",
            _md_table(pd.DataFrame({label_col: top.index, "n_genes": top.to_numpy()}),
                      empty="_(no pathway labels)_"),
        ])

    if figures:
        parts["figures"] = "\n".join(
            ["## Figures", ""]
            + [f"**{cap}**\n\n![{cap}]({_rel(p, out_dir)})\n" for cap, p in figures.items()])
    else:
        parts["figures"] = "## Figures\n\n_(none generated)_"

    parts["notes"] = "\n".join(["## Notes and caveats", ""] +
                               ([f"- {n}" for n in notes] or ["_(none)_"]))

    try:
        from . import __version__ as _ver
    except Exception:                               # pragma: no cover
        _ver = "unknown"
    header = "\n".join([
        f"# panelprep comparison report - {cfg.get('run_name', 'run')}",
        "",
        f"generated {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by panelprep {_ver}; "
        "every table below is generated from the registries and from this run's results.",
    ])
    md = "\n\n".join([header] + [parts[s] for s in SECTIONS if s in parts]) + "\n"

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(md)
    return md
