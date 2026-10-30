"""The agreement heatmap: candidate datasets x scores (the brief's item 1 table).

Requested formulation
---------------------
    "these scores should be visualized nicely in a table where red is a high
    score and a nice blue the lowest so for each dataset in the target set of
    datasets D by how many scores they have; make it easy that we add a new
    score and visualize its effect"

Rows are datasets, columns are scores, exactly as asked.

THE CORRECTION APPLIED HERE (and why)
-------------------------------------
Colouring the *raw* values would make red mean opposite things in neighbouring
columns, because the registered scores do not share a direction or a scale:

=========================  ===============  =====================================
score                      orientation      raw range
=========================  ===============  =====================================
``avg_kl``                 lower_is_better  ``[0, inf)`` nats, unbounded
``avg_js``                 lower_is_better  ``[0, ln 2]`` nats
``corr_frobenius``         lower_is_better  ``[0, 2]`` RMS correlation difference
``corr_cosine``            higher_is_better ``[-1, 1]``
``ot_cluster_ari``         higher_is_better ``(-0.5, 1]``
=========================  ===============  =====================================

A single colour map over raw values would paint a *good* ``avg_kl`` (small) and
a *good* ``ot_cluster_ari`` (large) at opposite ends of the bar. So the renderer:

1. **maps every column onto a common agreement direction** using the
   ``orientation`` column that ``ScoreResult`` carries (``lower_is_better``
   columns are negated; nothing is assumed from the score's name);
2. **normalises within each column** across datasets, because the scales are
   not commensurable -- rank by default (robust to the unbounded tail of
   ``avg_kl`` and to a single outlying dataset), with ``minmax`` and ``zscore``
   available;
3. **colours the normalised agreement blue -> red**, so red = strong agreement
   in *every* column and the colourbar says so;
4. **prints the raw value in every cell**, so the normalisation hides nothing
   and a reader can still see that ``avg_js = 0.31`` nats.

The normalisation is *within a column across datasets*, which means the colour
answers "which candidate agrees most, by this score", not "is 0.31 nats a lot".
With one dataset a rank normalisation is undefined, so a single-row table is
painted at the neutral midpoint and the note is recorded on the returned table.

Statuses are not numbers
------------------------
A cell whose ``status`` is not ``ok`` (``not_applicable`` for a PPI graph that
has no expression matrix, ``units_mismatch``, ``insufficient_genes``,
``error``) has no value. It is drawn grey and annotated with the status label,
never as 0 and never in a colour that reads like a number; such cells are
excluded from the column normalisation and from the mean-agreement ranking.

MODIFY HERE
-----------
* ``normalise`` -- ``"rank"`` (default) / ``"minmax"`` / ``"zscore"``.
* ``direction`` -- ``"agreement"`` (default) or ``"raw"`` to colour the raw
  numbers and see why that is a bad idea.
* ``cmap`` -- any diverging matplotlib map; the default ``RdBu_r`` is the
  brief's "red high, nice blue low".
* ``STATUS_LABELS`` / ``STATUS_COLOUR`` -- how non-``ok`` cells are shown.
* ``sort_datasets`` -- row order: ``"agreement"`` / ``"name"`` / ``"none"``.

Adding a new score needs NO edit to this file. Everything the renderer needs
(the column set, each column's direction, its description) is read from the
long-format frame produced by ``panelprep.scores.run_scores`` and from
``panelprep.scores.SCORES[name].meta``. Registering a score with
``@SCORES.register(...)`` and adding its name to ``Config.scores`` is
sufficient; ``tests/test_viz.py`` asserts exactly that by registering a dummy
score inside the test and checking it appears here and in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Short labels drawn inside a cell that has no value. Anything not listed is
#: shown as the first 6 characters of the status string.
STATUS_LABELS: Dict[str, str] = {
    "not_applicable": "n/a",
    "units_mismatch": "units",
    "insufficient_genes": "genes",
    "error": "err",
    "missing": "-",
    "nan": "nan",
}

#: Grey used for every value-less cell. Deliberately outside the diverging map.
STATUS_COLOUR = "#b8b8b8"

#: Columns the long-format score frame is expected to carry.
LONG_COLUMNS = ("dataset", "score", "value", "orientation", "status")

_NORMALISERS = ("rank", "minmax", "zscore")


# ---------------------------------------------------------------------------
# the frame -> (raw, agreement, colour, status) transformation
# ---------------------------------------------------------------------------

@dataclass
class AgreementTable:
    """Everything the heatmap, the Styler and the markdown report draw from.

    Observation unit: one cell is one ``(dataset, score)`` pair -- one
    ``ScoreResult``. Rows are candidate datasets, columns are scores.

    Attributes
    ----------
    raw
        Raw score values as returned by the scores, ``NaN`` where there is none.
    agreement
        Column-normalised agreement, high = agrees with the query. ``rank`` and
        ``minmax`` are in ``[0, 1]``; ``zscore`` is in standard deviations.
    colour
        Position in ``[0, 1]`` passed to the colour map; ``NaN`` = draw grey.
    status
        Per-cell status string (``ok`` or the reason there is no number).
    coverage
        ``"used/requested"`` panel genes per cell, as a string.
    ranking
        Per-dataset mean agreement over the ``ok`` cells, with the number of
        scores that produced a value (the brief's "by how many scores they
        have").
    """

    raw: pd.DataFrame
    agreement: pd.DataFrame
    colour: pd.DataFrame
    status: pd.DataFrame
    coverage: pd.DataFrame
    ranking: pd.DataFrame
    orientation: Dict[str, str]
    descriptions: Dict[str, str]
    normalise: str
    direction: str
    notes: List[str] = field(default_factory=list)
    fig: Any = None
    out_path: Optional[str] = None

    @property
    def datasets(self) -> List[str]:
        return list(self.raw.index)

    @property
    def scores(self) -> List[str]:
        return list(self.raw.columns)


def _registry_meta() -> Dict[str, dict]:
    """``{score_name: meta}`` from the live registry; empty if import fails.

    Imported lazily so this module can render a frame that was loaded from CSV
    in a session where the score package was never imported.
    """
    try:
        from ..scores import SCORES
    except Exception:  # pragma: no cover - defensive
        return {}
    return {e.name: dict(e.meta) for e in SCORES}


def _orientation_map(results: pd.DataFrame, notes: List[str]) -> Dict[str, str]:
    """Direction per score: the frame's own column first, registry as fallback."""
    meta = _registry_meta()
    out: Dict[str, str] = {}
    for name, grp in results.groupby("score", sort=False):
        vals = {str(v) for v in grp.get("orientation", pd.Series(dtype=object)).dropna().unique()}
        vals = {v for v in vals if v in ("higher_is_better", "lower_is_better")}
        if len(vals) == 1:
            out[name] = vals.pop()
            continue
        if len(vals) > 1:
            notes.append(f"score {name!r} reports conflicting orientations {sorted(vals)}; "
                         f"used the registry value")
        reg = meta.get(name, {}).get("orientation")
        if reg in ("higher_is_better", "lower_is_better"):
            out[name] = reg
        else:
            out[name] = "lower_is_better"
            notes.append(f"score {name!r} has no orientation in the frame or the registry; "
                         f"assumed lower_is_better")
    return out


def _normalise_column(signed: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Normalise one column of *signed* (already direction-corrected) values.

    Returns ``(agreement, colour_position)``. ``signed`` may contain ``NaN``
    for cells with no value; those stay ``NaN`` in both outputs and take no
    part in the normalisation.
    """
    out = np.full(signed.shape, np.nan, dtype=float)
    pos = np.full(signed.shape, np.nan, dtype=float)
    ok = np.isfinite(signed)
    n = int(ok.sum())
    if n == 0:
        return out, pos
    v = signed[ok]
    if n == 1:
        out[ok], pos[ok] = 0.5 if method != "zscore" else 0.0, 0.5
        return out, pos
    if method == "rank":
        from scipy.stats import rankdata
        r = rankdata(v, method="average")
        out[ok] = (r - 1.0) / (n - 1.0)
        pos[ok] = out[ok]
    elif method == "minmax":
        lo, hi = float(v.min()), float(v.max())
        out[ok] = 0.5 if hi <= lo else (v - lo) / (hi - lo)
        pos[ok] = out[ok]
    elif method == "zscore":
        sd = float(v.std(ddof=0))
        z = np.zeros_like(v) if sd == 0 else (v - float(v.mean())) / sd
        out[ok] = z
        m = float(np.max(np.abs(z))) or 1.0
        pos[ok] = np.clip(0.5 + z / (2 * m), 0.0, 1.0)
    else:
        raise ValueError(f"normalise must be one of {_NORMALISERS}; got {method!r}")
    return out, pos


def agreement_frame(
    results: pd.DataFrame,
    *,
    normalise: str = "rank",
    direction: str = "agreement",
    sort_datasets: str = "agreement",
    score_order: Optional[Sequence[str]] = None,
    dataset_order: Optional[Sequence[str]] = None,
) -> AgreementTable:
    """Long-format score rows -> the matrices the renderers colour.

    Observation unit: one ``(dataset, score)`` cell. Normalisation assumption:
    raw values are *not* comparable across columns, so every column is
    direction-corrected via ``orientation`` and then normalised across datasets
    (``normalise``); no normalisation is applied across rows. Missing-data
    behaviour: cells with ``status != "ok"``, and ``ok`` cells whose value is
    not finite, are excluded from the normalisation and the ranking and are
    marked for grey rendering; absent ``(dataset, score)`` combinations get
    status ``"missing"``.

    MODIFY HERE: ``normalise`` and ``direction``; ``score_order`` /
    ``dataset_order`` to force a layout instead of the default
    (registry order for columns, mean agreement for rows).
    """
    if direction not in ("agreement", "raw"):
        raise ValueError(f"direction must be 'agreement' or 'raw'; got {direction!r}")
    if normalise not in _NORMALISERS:
        raise ValueError(f"normalise must be one of {_NORMALISERS}; got {normalise!r}")

    notes: List[str] = []
    df = pd.DataFrame(results).copy()
    missing_cols = [c for c in LONG_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"score frame is missing columns {missing_cols}; expected the long "
                         f"format from panelprep.scores.run_scores ({list(LONG_COLUMNS)})")
    if df.empty:
        empty = pd.DataFrame()
        return AgreementTable(empty, empty, empty, empty, empty,
                              pd.DataFrame(columns=["dataset", "mean_agreement", "n_scores_ok"]),
                              {}, {}, normalise, direction, ["no score rows"])

    dup = df.duplicated(subset=["dataset", "score"], keep="last")
    if dup.any():
        notes.append(f"{int(dup.sum())} duplicate (dataset, score) rows; kept the last of each")
        df = df.loc[~dup]

    meta = _registry_meta()
    orientation = _orientation_map(df, notes)
    descriptions = {s: str(meta.get(s, {}).get("description", "")) for s in orientation}

    raw = df.pivot(index="dataset", columns="score", values="value").astype(float)
    status = df.pivot(index="dataset", columns="score", values="status").fillna("missing")
    used = df.pivot(index="dataset", columns="score", values="n_genes_used") \
        if "n_genes_used" in df.columns else pd.DataFrame(index=raw.index, columns=raw.columns)
    req = df.pivot(index="dataset", columns="score", values="n_genes_requested") \
        if "n_genes_requested" in df.columns else pd.DataFrame(index=raw.index, columns=raw.columns)
    coverage = pd.DataFrame(
        [[_coverage_label(used.iat[i, j] if used.size else None,
                          req.iat[i, j] if req.size else None)
          for j in range(raw.shape[1])] for i in range(raw.shape[0])],
        index=raw.index, columns=raw.columns)

    # A value-less cell is one that is not 'ok' OR is 'ok' with a non-finite
    # number (a score may legitimately return NaN, e.g. a zero-norm correlation
    # vector). Both are rendered grey; the second is labelled 'nan'.
    status = status.where(~(status.eq("ok") & ~np.isfinite(raw)), "nan")
    valueless = ~status.eq("ok")
    n_grey = int(valueless.to_numpy().sum())
    if n_grey:
        notes.append(f"{n_grey} of {raw.size} cells carry no value and are drawn grey")

    signed = raw.copy()
    for s in signed.columns:
        if orientation.get(s) == "lower_is_better":
            signed[s] = -signed[s]
    signed = signed.mask(valueless)

    agreement = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    colour = pd.DataFrame(np.nan, index=raw.index, columns=raw.columns)
    for s in raw.columns:
        src = signed[s] if direction == "agreement" else raw[s].mask(valueless[s])
        a, p = _normalise_column(src.to_numpy(dtype=float), normalise)
        agreement[s], colour[s] = a, p

    n_ok = (~valueless).sum(axis=1)
    mean_agr = agreement.mean(axis=1, skipna=True)
    ranking = pd.DataFrame({
        "dataset": raw.index,
        "mean_agreement": mean_agr.to_numpy(),
        "n_scores_ok": n_ok.to_numpy().astype(int),
        "n_scores_total": int(raw.shape[1]),
    }).sort_values("mean_agreement", ascending=False, na_position="last").reset_index(drop=True)
    ranking["agreement_rank"] = np.arange(1, len(ranking) + 1)

    cols = _order_columns(list(raw.columns), score_order, meta)
    rows = _order_rows(list(raw.index), dataset_order, sort_datasets, ranking)
    raw, status, agreement, colour, coverage = (
        x.loc[rows, cols] for x in (raw, status, agreement, colour, coverage))

    return AgreementTable(raw=raw, agreement=agreement, colour=colour, status=status,
                          coverage=coverage, ranking=ranking, orientation=orientation,
                          descriptions=descriptions, normalise=normalise,
                          direction=direction, notes=notes)


def _coverage_label(u, r) -> str:
    try:
        u_i, r_i = int(u), int(r)
    except (TypeError, ValueError):
        return ""
    return f"{u_i}/{r_i}"


def _order_columns(cols: List[str], score_order, meta: Dict[str, dict]) -> List[str]:
    if score_order is not None:
        keep = [c for c in score_order if c in cols]
        return keep + [c for c in cols if c not in keep]
    known = [c for c in sorted(meta) if c in cols]        # registry order (sorted names)
    return known + sorted(c for c in cols if c not in known)


def _order_rows(rows: List[str], dataset_order, sort_datasets: str,
                ranking: pd.DataFrame) -> List[str]:
    if dataset_order is not None:
        keep = [r for r in dataset_order if r in rows]
        return keep + [r for r in rows if r not in keep]
    if sort_datasets == "agreement":
        return [d for d in ranking["dataset"] if d in rows]
    if sort_datasets == "name":
        return sorted(rows)
    if sort_datasets == "none":
        return rows
    raise ValueError("sort_datasets must be 'agreement', 'name' or 'none'")


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def _fmt(v: float, sig: int = 3) -> str:
    if v is None or not np.isfinite(v):
        return ""
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e5):
        return f"{v:.{sig - 1}e}"
    return f"{v:.{sig}g}"


def _text_colour(rgba) -> str:
    """Black or white, whichever has more contrast against the cell."""
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if lum > 0.55 else "#ffffff"


def _style_rc() -> dict:
    """Self-contained figure style (three font sizes, no global state)."""
    return {
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif",
    }


def score_heatmap(
    results: pd.DataFrame,
    *,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    annotate: bool = True,
    normalise: str = "rank",
    direction: str = "agreement",
    sort_datasets: str = "agreement",
    score_order: Optional[Sequence[str]] = None,
    dataset_order: Optional[Sequence[str]] = None,
    cmap: str = "RdBu_r",
    show_coverage: bool = True,
    figsize: Optional[tuple] = None,
    dpi: int = 200,
    close: bool = True,
) -> AgreementTable:
    """Datasets x scores agreement heatmap; red = strong agreement.

    Observation unit: one cell is one ``(dataset, score)`` pair. Normalisation
    assumption: see :func:`agreement_frame` -- colours are column-normalised
    agreement, the printed numbers are the untouched raw values. Missing-data
    behaviour: value-less cells are grey with their status label, and partial
    gene coverage (``n_genes_used < n_genes_requested``) is printed under the
    value when ``show_coverage``.

    Parameters
    ----------
    results
        Long-format frame from ``panelprep.scores.run_scores``.
    out_path
        Written if given; the parent directory is created.
    direction
        ``"agreement"`` (default, red = agrees with the query) or ``"raw"``
        (red = numerically large, which means opposite things per column -- the
        colourbar label says so).

    Returns
    -------
    AgreementTable
        The matrices behind the picture, with the figure on ``.fig``, so the
        numbers can be inspected rather than eyeballed.

    MODIFY HERE: ``normalise``, ``direction``, ``cmap``, ``sort_datasets``.
    Adding a *score* requires no change here -- columns, directions and
    descriptions come from the frame and the registry.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tab = agreement_frame(results, normalise=normalise, direction=direction,
                          sort_datasets=sort_datasets, score_order=score_order,
                          dataset_order=dataset_order)
    if tab.raw.empty:
        raise ValueError("no score rows to plot")

    n_rows, n_cols = tab.raw.shape
    figsize = figsize or (max(4.2, 1.35 * n_cols + 2.6), max(2.4, 0.62 * n_rows + 1.9))

    with mpl.rc_context(_style_rc()):
        fig, ax = plt.subplots(figsize=figsize)
        # value-less cells arrive as masked -> painted STATUS_COLOUR, never a data colour
        cm = mpl.colormaps[cmap].with_extremes(bad=STATUS_COLOUR)
        data = np.ma.masked_invalid(tab.colour.to_numpy(dtype=float))
        im = ax.imshow(data, cmap=cm, vmin=0.0, vmax=1.0, aspect="auto")

        ax.set_xticks(range(n_cols), labels=tab.raw.columns, rotation=30, ha="right")
        ax.set_yticks(range(n_rows), labels=tab.raw.index)
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)

        # Direction cue as a header row above each column (not under the rotated
        # tick labels, where it would collide): the printed numbers are raw, so
        # a reader needs to know which way each column runs.
        arrows = {"higher_is_better": "\u2191 high = agrees",
                  "lower_is_better": "\u2193 low = agrees"}
        ax.set_xlabel("")
        for j, s in enumerate(tab.raw.columns):
            ax.text(j, -0.56, arrows.get(tab.orientation.get(s, ""), ""),
                    ha="center", va="bottom", fontsize=6, color="#555555",
                    transform=ax.transData)

        if annotate:
            for i in range(n_rows):
                for j in range(n_cols):
                    st = str(tab.status.iat[i, j])
                    if st != "ok":
                        ax.text(j, i, STATUS_LABELS.get(st, st[:6]), ha="center", va="center",
                                fontsize=7, color="#333333", style="italic")
                        continue
                    rgba = cm(float(tab.colour.iat[i, j]))
                    txt = _fmt(float(tab.raw.iat[i, j]))
                    cov = str(tab.coverage.iat[i, j])
                    partial = bool(cov) and len(set(cov.split("/"))) > 1
                    if show_coverage and partial:
                        ax.text(j, i - 0.13, txt, ha="center", va="center", fontsize=7.5,
                                color=_text_colour(rgba))
                        ax.text(j, i + 0.22, f"{cov} genes", ha="center", va="center",
                                fontsize=6, color=_text_colour(rgba))
                    else:
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                                color=_text_colour(rgba))

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        if direction == "agreement":
            lab = {"rank": "rank within column", "minmax": "min-max within column",
                   "zscore": "z-score within column"}[normalise]
            cbar.set_label(f"agreement with the query panel\n({lab}; red = agrees most)")
        else:
            cbar.set_label("raw value within column\n(red = numerically large; "
                           "direction differs per column)")
        cbar.outline.set_visible(False)

        present = [s for s in STATUS_LABELS if (tab.status == s).to_numpy().any()]
        if present:
            handles = [Patch(facecolor=STATUS_COLOUR, edgecolor="none",
                             label=f"{STATUS_LABELS[s]} = {s.replace('_', ' ')}")
                       for s in present]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.18, 1.0),
                      frameon=False, fontsize=7, title="no value", title_fontsize=7,
                      handlelength=1.0, borderpad=0.2)

        head = title or "Panel-gene agreement across candidate datasets"
        sub = (f"{n_rows} datasets x {n_cols} scores; printed numbers are raw values; "
               f"{int((tab.status != 'ok').to_numpy().sum())} cells have no value")
        ax.set_title(f"{head}\n{sub}", fontsize=8, loc="left", color="#111111", pad=16)

        if out_path:
            from pathlib import Path
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            tab.out_path = str(out_path)
        tab.fig = fig
        if close and out_path:
            plt.close(fig)
    return tab


def score_table_styled(
    results: pd.DataFrame,
    *,
    normalise: str = "rank",
    direction: str = "agreement",
    sort_datasets: str = "agreement",
    cmap: str = "RdBu_r",
    sig: int = 3,
):
    """The same table as a pandas ``Styler`` for notebook display.

    Observation unit, normalisation and missing-data behaviour are identical to
    :func:`score_heatmap` -- the identical ``agreement_frame`` call drives both,
    so the notebook table and the saved figure can never disagree. Cell text is
    the raw value; value-less cells are grey with the status label.

    MODIFY HERE: ``sig`` (printed significant figures) and ``cmap``.
    """
    import matplotlib as mpl

    tab = agreement_frame(results, normalise=normalise, direction=direction,
                          sort_datasets=sort_datasets)
    cm = mpl.colormaps[cmap]
    text = tab.raw.copy().astype(object)
    for i in range(tab.raw.shape[0]):
        for j in range(tab.raw.shape[1]):
            st = str(tab.status.iat[i, j])
            text.iat[i, j] = _fmt(float(tab.raw.iat[i, j]), sig) if st == "ok" \
                else STATUS_LABELS.get(st, st)

    def _css(_):
        out = pd.DataFrame("", index=tab.raw.index, columns=tab.raw.columns)
        for i in range(tab.raw.shape[0]):
            for j in range(tab.raw.shape[1]):
                p = tab.colour.iat[i, j]
                if not np.isfinite(p):
                    out.iat[i, j] = f"background-color: {STATUS_COLOUR}; color: #333333; " \
                                    f"font-style: italic"
                else:
                    rgba = cm(float(p))
                    out.iat[i, j] = (f"background-color: {mpl.colors.to_hex(rgba)}; "
                                     f"color: {_text_colour(rgba)}")
        return out

    cap = (f"Red = strongest agreement with the query panel "
           f"({tab.normalise} normalisation within each column, "
           f"{tab.direction} direction); printed numbers are raw values; "
           f"grey = no value.")
    return text.style.apply(_css, axis=None).set_caption(cap)
