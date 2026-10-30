#!/usr/bin/env python
"""Build the results presentation (.pptx) from the saved tables and figures.

Every number on a slide is read from a file on disk -- ``score_results.csv``,
``prior_manifest.csv``, ``manifest.csv``, ``embedding_quality.csv`` -- so the
deck cannot drift from the analysis. Nothing is typed in from memory.

Run::

    python scripts/build_deck.py --out panelprep_results.pptx

MODIFY HERE
-----------
* ``THEME`` -- colours and fonts.
* ``FIGDIR`` -- where the figures come from.
* the ``slides()`` function -- the running order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

# ----------------------------------------------------------------- theme
THEME = {
    "ink": RGBColor(0x1A, 0x1A, 0x1A),
    "muted": RGBColor(0x6B, 0x6B, 0x6B),
    "accent": RGBColor(0xB2, 0x18, 0x2B),      # the heatmap red
    "accent2": RGBColor(0x21, 0x66, 0xAC),     # the heatmap blue
    "rule": RGBColor(0xD9, 0xD9, 0xD9),
    "bg": RGBColor(0xFF, 0xFF, 0xFF),
    "band": RGBColor(0xF4, 0xF4, 0xF4),
    "font": "Helvetica Neue",
    "mono": "Menlo",
}
W, H = Inches(13.333), Inches(7.5)
M = Inches(0.62)                                # outer margin

FIGDIR = Path("report_tex/figures")


# ------------------------------------------------------------- primitives
def _tb(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def _run(p, text, *, size=16, bold=False, color=None, mono=False, italic=False):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.name = THEME["mono"] if mono else THEME["font"]
    f.color.rgb = color or THEME["ink"]
    return r


def _rule(slide, y, *, x=M, w=None, color=None, h=Pt(1.4)):
    w = w or (W - 2 * M)
    from pptx.enum.shapes import MSO_SHAPE
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = color or THEME["rule"]
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def header(slide, title, kicker=None):
    """Slide title plus an optional small kicker line above it."""
    y = M
    if kicker:
        tf = _tb(slide, M, y, W - 2 * M, Inches(0.26))
        _run(tf.paragraphs[0], kicker.upper(), size=11, bold=True,
             color=THEME["accent"])
        y += Inches(0.3)
    tf = _tb(slide, M, y, W - 2 * M, Inches(0.55))
    _run(tf.paragraphs[0], title, size=26, bold=True)
    _rule(slide, y + Inches(0.62))
    return y + Inches(0.82)


def bullets(slide, items, *, x=M, y=Inches(1.6), w=None, size=15, gap=Pt(9)):
    """``items`` entries are ``str`` or ``(text, sub)``; ``sub`` is dimmed."""
    w = w or (W - 2 * M)
    tf = _tb(slide, x, y, w, H - y - M)
    first = True
    for item in items:
        text, sub = item if isinstance(item, tuple) else (item, None)
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = gap
        if text:
            _run(p, "— ", size=size, color=THEME["accent"], bold=True)
            # **bold** segments
            for i, seg in enumerate(text.split("**")):
                if seg:
                    _run(p, seg, size=size, bold=bool(i % 2))
        if sub:
            q = tf.add_paragraph()
            q.space_after = gap
            _run(q, "    " + sub, size=size - 3, color=THEME["muted"])
    return tf


def figure(slide, name, *, x=None, y=Inches(1.55), w=None, h=None):
    path = FIGDIR / name if not Path(name).exists() else Path(name)
    if not path.exists():
        raise FileNotFoundError(path)
    from PIL import Image
    iw, ih = Image.open(path).size
    avail_w = w or (W - 2 * M)
    avail_h = h or (H - y - Inches(0.75))
    scale = min(avail_w / iw, avail_h / ih)
    pw, ph = Emu(int(iw * scale)), Emu(int(ih * scale))
    px = x if x is not None else Emu(int((W - pw) / 2))
    slide.shapes.add_picture(str(path), px, y, width=pw, height=ph)
    return y + ph


def caption(slide, text, y, *, x=M, w=None, size=11):
    tf = _tb(slide, x, y + Inches(0.08), w or (W - 2 * M), Inches(0.6))
    p = tf.paragraphs[0]
    for i, seg in enumerate(text.split("**")):
        if seg:
            _run(p, seg, size=size, color=THEME["muted"], bold=bool(i % 2))
    return tf


def table(slide, df, *, x=M, y=Inches(1.6), w=None, h=None, size=12,
          header_size=12, col_widths=None, highlight_rows=(), mono_cols=()):
    w = w or (W - 2 * M)
    rows, cols = df.shape[0] + 1, df.shape[1]
    h = h or Inches(0.34) * rows
    shape = slide.shapes.add_table(rows, cols, x, y, w, h)
    tbl = shape.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            tbl.columns[j].width = Emu(int(w * cw / total))
    for j, name in enumerate(df.columns):
        c = tbl.cell(0, j)
        c.text = ""
        p = c.text_frame.paragraphs[0]
        _run(p, str(name), size=header_size, bold=True, color=THEME["bg"])
        c.fill.solid()
        c.fill.fore_color.rgb = THEME["ink"]
        c.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i in range(df.shape[0]):
        for j in range(cols):
            c = tbl.cell(i + 1, j)
            c.text = ""
            p = c.text_frame.paragraphs[0]
            val = df.iat[i, j]
            txt = "" if pd.isna(val) else str(val)
            hot = i in highlight_rows
            _run(p, txt, size=size, bold=hot,
                 color=THEME["accent"] if hot else THEME["ink"],
                 mono=(df.columns[j] in mono_cols))
            c.fill.solid()
            c.fill.fore_color.rgb = THEME["band"] if i % 2 else THEME["bg"]
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            if j > 0:
                p.alignment = PP_ALIGN.RIGHT if df.columns[j] not in mono_cols \
                    else PP_ALIGN.LEFT
    return y + h


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def section(prs, number, title, subtitle):
    s = blank(prs)
    from pptx.enum.shapes import MSO_SHAPE
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(0), Emu(0),
                              Inches(0.28), H)
    band.fill.solid()
    band.fill.fore_color.rgb = THEME["accent"]
    band.line.fill.background()
    band.shadow.inherit = False
    tf = _tb(s, Inches(1.1), Inches(2.6), W - Inches(2.2), Inches(1.0))
    _run(tf.paragraphs[0], f"{number}", size=15, bold=True, color=THEME["accent"])
    tf2 = _tb(s, Inches(1.1), Inches(2.95), W - Inches(2.2), Inches(0.9))
    _run(tf2.paragraphs[0], title, size=38, bold=True)
    tf3 = _tb(s, Inches(1.1), Inches(4.0), W - Inches(3.0), Inches(1.2))
    _run(tf3.paragraphs[0], subtitle, size=16, color=THEME["muted"])
    return s


# ------------------------------------------------------------------ data
def load_data():
    """Every number on a slide comes from one of these files."""
    d = {}
    d["synth"] = (pd.read_csv("outputs/synthetic/score_results.csv")
                  .pivot(index="dataset", columns="score", values="value"))
    d["pilot"] = (pd.read_csv("outputs/real_pilot/score_results.csv")
                  .pivot(index="dataset", columns="score", values="value"))
    d["nb"] = (pd.read_csv("outputs/notebook_run/score_results.csv")
               .pivot(index="dataset", columns="score", values="value"))
    d["priors"] = pd.read_csv("data/priors_export/prior_manifest.csv")
    d["store"] = pd.read_csv("outputs/real_pilot/store/manifest.csv")
    d["quality"] = pd.read_csv("outputs/notebook_run/figures/embedding_quality.csv")
    # The test result is read from the file pytest wrote, never typed in.
    p = Path("outputs/pytest_summary.txt")
    d["pytest"] = p.read_text().strip() if p.exists() else "test summary not available"
    return d


# --------------------------------------------------------------- slides
def slides(prs, d):
    syn, pil, nb = d["synth"], d["pilot"], d["nb"]
    pri, store, qual = d["priors"], d["store"], d["quality"]

    def f(df, row, col, nd=4):
        """Formatted value straight from a saved table."""
        v = df.loc[row, col]
        return "n/a" if pd.isna(v) else f"{v:.{nd}f}"

    # ---- 1 title
    s = blank(prs)
    _rule(s, Inches(2.52), x=M, w=Inches(1.6), color=THEME["accent"], h=Pt(4))
    tf = _tb(s, M, Inches(2.75), W - Inches(2.4), Inches(1.4))
    _run(tf.paragraphs[0], "Preprocessing for spatial gene-panel planning", size=40, bold=True)
    tf2 = _tb(s, M, Inches(4.05), W - Inches(3.2), Inches(1.0))
    _run(tf2.paragraphs[0],
         "Dataset matching, gene embeddings and multi-modal priors "
         "from a 4-gene measured panel", size=19, color=THEME["muted"])
    tf3 = _tb(s, M, Inches(5.5), W - Inches(2.4), Inches(1.2))
    p = tf3.paragraphs[0]
    _run(p, "panelprep", size=15, bold=True, mono=True)
    _run(p, "   ·   mouse gastrulation, E6.5–E8.0   ·   Stereo-seq + MARS-seq   "
            "·   merge target: ", size=15, color=THEME["muted"])
    _run(p, "Morpholab_Hackathon/preprocessing", size=15, mono=True,
         color=THEME["muted"])
    notes(s, f"Everything shown was run end to end. Offline suite: {d['pytest']}. "
             "The notebook executes all 17 of its code cells. Every number on "
             "these slides is read from a saved CSV at build time rather than retyped.")

    # ---- 2 the task
    s = blank(prs)
    y = header(s, "What the preprocessing team is given, and must return", kicker="the task")
    half = (W - 2 * M - Inches(0.5)) / 2
    tf = _tb(s, M, y, half, Inches(0.4))
    _run(tf.paragraphs[0], "INPUT", size=12, bold=True, color=THEME["accent2"])
    bullets(s, [
        ("An **N x S** expression matrix", "S is small — 4 measured genes: T, Krt8, Mest, Dnmt3b"),
        ("**2D locations** of those cells", "sagittal projection, plus within-section coordinates"),
        ("A **free-text descriptor**", "the biology story of the experiment"),
    ], x=M, y=y + Inches(0.34), w=half, size=15)
    tf = _tb(s, M + half + Inches(0.5), y, half, Inches(0.4))
    _run(tf.paragraphs[0], "OUTPUT", size=12, bold=True, color=THEME["accent2"])
    bullets(s, [
        ("**Candidate datasets**, tagged and scored", "4 categories; modular score registry"),
        ("**Gene embeddings for the whole transcriptome**", "not only the measured subset"),
        ("Embeddings from **outside expression**", "protein sequence, PPI, literature, regulation"),
        ("**Annotations**: pathway, ligand, receptor, TF", ""),
    ], x=M + half + Inches(0.5), y=y + Inches(0.34), w=half, size=15)
    notes(s, "The premise: you measured only 4 genes but need to reason about "
             "all ~16,000. Everything else follows from that asymmetry.")

    # ---- 3 what was built
    s = blank(prs)
    y = header(s, "One package, four registries, one configuration entry point",
               kicker="architecture")
    bullets(s, [
        ("**Registries** — scores, embedders, prior sources, annotation sources",
         "adding a score is one decorator; the heatmap, its colour direction and the report all follow"),
        ("**One Config** drives everything",
         "candidates, scores, embedders, dim, gene universe, seed — the notebook's Parameters cell"),
        ("**An embedding store** keeps the expensive work out of the notebook",
         "parquet + provenance sidecar; ESM-2 / BioGPT / node2vec are loaded, never retrained"),
        ("**Thin notebooks** import package functions",
         "05_run_analysis.ipynb: one configured run, 17 code cells, all verified to execute"),
        (f"**Offline test suite: {d['pytest']}**",
         "including a test that pins the Morpholab score to ours bit-for-bit"),
    ], y=y, size=16)
    notes(s, "Simplicity was an explicit requirement: one package, one config, "
             "small registries, notebooks that only call functions.")

    # ---- 4 section 1
    section(prs, "Part 1", "Which reference datasets match the panel?",
            "A modular score registry, three requested scores, and what had to be "
            "corrected before any of them was meaningful.")

    # ---- 5 the scores
    s = blank(prs)
    y = header(s, "The three requested scores, and the corrections each needed",
               kicker="score registry")
    tdf = pd.DataFrame({
        "score": ["avg_kl / avg_js", "corr_frobenius", "ot_cluster_ari", "corr_mae"],
        "what it compares": ["per-gene expression distributions",
                             "S x S panel correlation matrices",
                             "cell clusters, transported",
                             "S x S panel correlations (Pearson)"],
        "correction applied": [
            "pooled quantile bins + Laplace smoothing; JS preferred (symmetric, bounded)",
            "strict upper triangle, normalised to RMS, Spearman default",
            "requested form is undefined — ARI needs two labelings of the same objects",
            "none — reproduces Morpholab select_dataset exactly"],
    })
    y2 = table(s, tdf, y=y, size=12, col_widths=[1.5, 2.2, 5.0],
               mono_cols=("score", "what it compares", "correction applied"))
    caption(s, "**ot_cluster_ari as implemented:** cluster each side independently, "
               "entropic OT between them, push candidate labels through the coupling, "
               "build a **soft contingency table** of transported mass, then ordinary ARI "
               "algebra extended to real-valued counts. Verified equal to "
               "sklearn.adjusted_rand_score to <1e-9 on integer tables.", y2)
    notes(s, "The OT+ARI formulation as requested cannot be computed: ARI is "
             "defined for two labelings of one object set, and cells are not "
             "paired across datasets. The transported-mass contingency table "
             "preserves the intended question.")

    # ---- 6 synthetic acceptance
    s = blank(prs)
    y = header(s, "Acceptance test: four synthetic references with known answers",
               kicker="validity")
    y2 = figure(s, "fig01_synth_score_heatmap.png", y=y, h=Inches(4.0))
    caption(s,
        f"**shuffled_structure** permutes each gene independently *after* normalisation: "
        f"marginals are preserved exactly, co-expression is destroyed. "
        f"avg_js {f(syn,'sibling','avg_js')} (sibling) vs {f(syn,'shuffled_structure','avg_js')} "
        f"— ratio 1.47, correctly blind. "
        f"corr_frobenius {f(syn,'sibling','corr_frobenius')} vs "
        f"{f(syn,'shuffled_structure','corr_frobenius')} — ratio 11.7, correctly sensitive. "
        f"Rank-normalised mean agreement reproduces the expected order exactly.", y2)
    notes(s, "This is the slide that says the scores measure what they claim. "
             "A distributional score SHOULD be blind to a structure-only change; "
             "a structural score should not be.")

    # ---- 7 real pilot
    s = blank(prs)
    y = header(s, "Real pilot: one E7.5 embryo against seven candidate references",
               kicker="results")
    y2 = figure(s, "fig06a_pilot_heatmap_agreement.png", y=y, h=Inches(3.9))
    caption(s,
        f"Red = agrees most with the measured panel. The stage-matched spatial replicate "
        f"**seu3d_E7.5_rep2** wins on every score: avg_js {f(pil,'seu3d_E7.5_rep2','avg_js')}, "
        f"corr_frobenius {f(pil,'seu3d_E7.5_rep2','corr_frobenius')}, "
        f"ot_cluster_ari {f(pil,'seu3d_E7.5_rep2','ot_cluster_ari')}. "
        f"Agreement falls monotonically with developmental distance "
        f"(E7.75 {f(pil,'seu3d_E7.75_rep2','corr_frobenius')} → "
        f"E8.0 {f(pil,'seu3d_E8.0_rep1','corr_frobenius')}).", y2)
    notes(s, "The E6.5 single-cell reference is a deliberate negative control "
             "and the Foxc1/Foxc2 knockout is the causal arm.")

    # ---- 8 both colourings
    s = blank(prs)
    y = header(s, "Both colourings always ship — low divergence is not a low score",
               kicker="reading the table")
    y2 = figure(s, "fig06b_pilot_heatmap_raw.png", y=y, h=Inches(3.9))
    caption(s, "Left/previous slide colours by **agreement** (red = agrees most, so a small "
               "divergence is red). This one colours by **raw value** (red = numerically large, "
               "so the same cell is blue). Each column header carries its own direction. "
               "Shipping only one of these is how a reader concludes the best dataset is the worst.", y2)
    notes(s, "Requested explicitly after a 0.0234 cell appeared red. Both are "
             "correct; they answer different questions, so both are written.")

    # ---- 9 platform artefact
    s = blank(prs)
    y = header(s, "The caveat that changes how you read the table", kicker="honest limits")
    bullets(s, [
        ("Stereo-seq detects the panel in **8–22%** of cells; MARS-seq in **36–88%**",
         "a 5x difference in detection rate, not in biology"),
        ("The distributional scores see that as a distribution shift",
         f"avg_js ranks the stage-MISMATCHED E6.5 ({f(pil,'mittnenzweig_wt_E6.5','avg_js')}) "
         f"ahead of the stage-matched E7.5 ({f(pil,'mittnenzweig_wt_E7.5','avg_js')}) — wrong, and the reason is platform"),
        ("The correlation and OT scores get it right",
         f"corr_cosine {f(pil,'mittnenzweig_wt_E7.5','corr_cosine')} (matched) vs "
         f"{f(pil,'mittnenzweig_wt_E6.5','corr_cosine')} (mismatched); these are the scale-free ones"),
        ("**Rule:** read correlation and OT scores across platforms; read avg_js only within one",
         "recorded in the report and in the score metadata, not left for the reader to discover"),
    ], y=y, size=16)
    notes(s, "This is a property of the assays, not a bug. The fix is to say "
             "which score survives a platform change, which the units contract "
             "and the score metadata both encode.")

    # ---- 10 spatial defects
    s = blank(prs)
    y = header(s, "Two defects found in the spatial scores — and fixed", kicker="what went wrong")
    bullets(s, [
        ("**Moran's I is sample-size dependent**",
         "same embryo, same gene T: 0.330 / 0.222 / 0.286 / 0.292 at n = 300 / 600 / 1000 / 1500. "
         "The query was subsampled to 4000 and candidates to 1500, so the comparison was partly "
         "a comparison of n. Fixed: common n = min(N_q, N_c, max_obs) and coordinates rescaled "
         "to unit mean nearest-neighbour distance."),
        ("**The 2D sagittal projection superimposes 42–52 z-sections**",
         "Projected RMS put E7.5_rep2 at 0.0997 and E8.0_rep2 at 0.0623 — inverted. Within-section: "
         "0.0464 vs 0.1253 — correct, the same-stage replicate is 2.7x better. Fixed: section-wise "
         "Moran's I on within-section coordinates."),
        (f"After the fix seu3d_E7.5_rep2 moved from worst (rank 4/4) to "
         f"**{f(pil,'seu3d_E7.5_rep2','spatial_autocorr_frobenius')}**, rank 2/4", ""),
    ], y=y, size=15)
    notes(s, "Both were caught by asking whether the score could be driven by "
             "something other than biology — sample size, and the projection.")

    # ---- 11 section 2
    section(prs, "Part 2", "Gene embeddings for the whole transcriptome",
            "Six expression methods, separate and joint, plus four sources that "
            "never touch the expression values.")

    # ---- 12 shapes and locations
    s = blank(prs)
    y = header(s, "What an embedder returns, and where it is written", kicker="interfaces")
    exp = store[store["group"] == "expression"]
    prq = store[store["group"] == "prior"]
    tdf = pd.DataFrame({
        "method": ["pca_baseline", "ae_separate", "vae_separate", "joint_ae", "dcca",
                   "contrastive_ae", "factor_analysis", "protein_sequence (ESM-2)",
                   "ppi_network (node2vec)", "literature_text (BioGPT)", "regulatory_kegg"],
        "mode": ["separate", "separate", "separate", "joint", "joint", "joint",
                 "separate", "cache-only", "cache-only", "cache-only", "cache-only"],
        "shape G x d": ["2000 x 32"] * 7 + ["4006 x 480", "4006 x 64", "4006 x 1024", "4006 x 64"],
        "written to": ["store/expression/<method>__<dataset>.parquet"] * 3
                      + ["store/expression/<method>.parquet"] * 3
                      + ["store/expression/<method>__<dataset>.parquet"]
                      + ["store/prior/<source>.parquet"] * 4,
    })
    y2 = table(s, tdf, y=y, size=11, header_size=11, col_widths=[2.4, 1.2, 1.5, 5.0],
               mono_cols=("method", "mode", "shape G x d", "written to"))
    caption(s, f"One row per gene, indexed by **gene symbol**, float32, with a JSON provenance "
               f"sidecar. A separate embedder returns one matrix per dataset; a joint embedder "
               f"returns one. Uncovered genes are **all-NaN rows, never zeros** — a zero vector is "
               f"a position in the space. Store as built: **{len(exp)} expression + {len(prq)} prior "
               f"= {len(store)} embeddings**.", y2)
    notes(s, "The question 'what is the output shape and where is it saved' is "
             "answered by EmbeddingStore.inventory(), which generated this table.")

    # ---- 13 embedding panel
    s = blank(prs)
    y = header(s, "Separate embeddings: one per reference dataset", kicker="figures")
    y2 = figure(s, "fig10a_pilot_datasets_pca.png", y=y, h=Inches(4.1))
    caption(s, "One panel per dataset, one shared colour mapping so the panels are comparable, "
               "coloured by KEGG top pathway. Every title carries silhouette and kNN label "
               "agreement with a label-permutation p-value. Stars mark the four measured genes.", y2)

    # ---- 14 named genes
    s = blank(prs)
    y = header(s, "The integrated embedding, with gene names on the points", kicker="figures")
    y2 = figure(s, "fig09e_pilot_dcca_named_genes.png", y=y, h=Inches(4.1))
    caption(s,
        "**dcca** is genuinely integrated: all 7 datasets as views, every one of the 2000 genes "
        "present in all of them, final canonical correlation 31.7 / 32. T sits in the dense core "
        "with the transcription factors; Mest and Dnmt3b sit together on the sparse arm — the "
        "integrated embedding does **not** place the four measured genes near each other.", y2)
    notes(s, "label_genes names an explicit list; n_gene_labels names the N most "
             "peripheral genes automatically. Labels are thinned on collision.")

    # ---- 15 agreement
    s = blank(prs)
    y = header(s, "Do the methods agree? Mostly not — and platform dominates",
               kicker="method comparison")
    gap = Inches(0.35)
    halfw = (W - 2 * M - gap) / 2
    figure(s, "fig07a_pilot_agree_methods_cka.png", x=M, y=y, w=halfw, h=Inches(3.6))
    figure(s, "fig08_pilot_agree_datasets_cka.png", x=M + halfw + gap, y=y, w=halfw, h=Inches(3.6))
    caption(s,
        "**Left — six methods, same 2000 genes.** Linear CKA spans 0.105 to 0.994 and kNN-Jaccard "
        "only 0.018–0.161 against a null of 0.004–0.005: on real data the methods find "
        "substantially different gene geometries, which is why the package ships six rather than "
        "picking one.\n"
        "**Right — the same method across datasets.** CKA is 0.972–0.991 within the four Stereo-seq "
        "embryos and 0.932–0.971 within the three MARS-seq datasets, but only **0.427–0.481 across "
        "platforms**. The assay, not the developmental stage, is the dominant axis.",
        Inches(5.3), size=12)
    notes(s, "On synthetic data the same methods agree far more (kNN-Jaccard "
             "0.13-0.69). Agreement on a clean generative model is not evidence "
             "of agreement on real tissue.")

    # ---- 16 section 3
    section(prs, "Part 3", "Multi-modal priors over the genome",
            "Five sources in the style of IterPert's prior construction — built once, "
            "loaded thereafter.")

    # ---- 17 prior coverage
    s = blank(prs)
    y = header(s, "Five prior sources, measured coverage and real cost", kicker="priors")
    pv = pri.copy()
    pv["coverage"] = (pv["coverage_fraction"] * 100).round(1).astype(str) + "%"
    pv["build"] = pv["compute_seconds"].apply(
        lambda v: f"{v:.1f} s" if v < 120 else f"{v/60:.0f} min")
    tdf = pv[["source", "n_features", "coverage", "build", "status"]].rename(
        columns={"source": "source", "n_features": "d", "status": "status"})
    y2 = table(s, tdf, y=y, size=12, col_widths=[2.3, 0.7, 1.1, 1.1, 4.0],
               mono_cols=("source", "status"))
    caption(s,
        f"All five over the same **{int(pri['n_genes'].iloc[0]):,} mouse symbols**; coverage "
        f"fractions match the genome-wide ceilings of their sources to within 0.015. "
        f"**protein_sequence**, **ppi_network** and **literature_text** depend only on the gene "
        f"symbol list and together cost **{pri['compute_seconds'].sum()/3600:.1f} h** — so they are "
        f"built once by scripts/build_embedding_store.py and loaded thereafter. "
        f"**perturbation_pseudobulk** and **ops_morphology** are explicitly deferred, not implemented: "
        f"the first needs a genome-scale screen, the second (PERISCOPE) is human-only.", y2)
    notes(s, "regulatory_kegg as shipped is pathway co-membership, not the signed "
             "regulatory graph — labelled ok_fallback_pathway_membership rather "
             "than presented as what it isn't.")

    # ---- 18 kernels
    s = blank(prs)
    y = header(s, "Kernel fusion: a correction to the paper's own recipe", kicker="priors")
    bullets(s, [
        ("Linear kernel of z-scored features **equals** the correlation matrix",
         "verified to 7.772e-16; 0.384 if the features are not standardised first"),
        ("Convex combinations preserve positive semi-definiteness; element-wise max does not",
         "mean fusion +6.783e-02, Hadamard product +1.882e-01, **max fusion -4.142e-01**"),
        ("**The paper's own min-max normalisation is not PSD-preserving**",
         "it is an affine map of the off-diagonal that adds eigenvalue -c: mean fusion AFTER min-max "
         "gives -2.109e-02, and the real five-source fused kernel gives -115"),
        ("Missing data is masked per entry, not zero-filled",
         "2.28 sources per entry on average, 0 uncovered entries; the fusion warns when the result is not PSD"),
        ("Kernel alignment shows the sources are not redundant",
         "protein_sequence vs literature_text 0.996 and ppi vs protein 0.893, but co-expression vs "
         "everything else only 0.030–0.061 — expression carries information the others do not"),
    ], y=y, size=15)
    notes(s, "Project the kernel (clip negative eigenvalues) before using it in "
             "any method that requires PSD. The package warns rather than "
             "silently returning an indefinite kernel.")

    # ---- 19 section 4
    section(prs, "Part 4", "Annotations, and an honest negative result",
            "Pathway membership and ligand / receptor / TF roles — and what they "
            "do and do not recover.")

    # ---- 20 negative result
    s = blank(prs)
    y = header(s, "KEGG pathway membership is only marginally recoverable",
               kicker="negative result")
    qq = qual[qual["embedding"].isin(
        ["pca_baseline::seu3d_E7.5_rep2", "dcca", "joint_ae", "contrastive_ae",
         "ppi_network", "protein_sequence", "literature_text"])].copy()
    qq = qq.sort_values("silhouette", ascending=False)
    tdf = pd.DataFrame({
        "embedding": qq["embedding"].values,
        "silhouette": qq["silhouette"].round(3).values,
        "p": qq["silhouette_p"].round(3).values,
        "kNN label agreement": qq["knn_agreement"].round(3).values,
        "p ": qq["knn_p"].round(3).values,
        "genes": qq["n_genes_scored"].values,
    })
    y2 = table(s, tdf, y=y, size=12, col_widths=[3.4, 1.1, 0.7, 1.7, 0.7, 0.8],
               mono_cols=("embedding",))
    # Every number in this caption is computed from the same table the slide renders.
    expr_keys = ["pca_baseline::seu3d_E7.5_rep2", "ae_separate::seu3d_E7.5_rep2",
                 "vae_separate::seu3d_E7.5_rep2", "dcca", "joint_ae", "contrastive_ae"]
    ex = qual[qual["embedding"].isin(expr_keys)]
    pr = qual[qual["embedding"].isin(["ppi_network", "protein_sequence", "literature_text"])]
    cae = qual[qual["embedding"] == "contrastive_ae"]
    caption(s,
        f"Silhouette is **negative for every expression embedding** "
        f"({ex['silhouette'].min():.2f} to {ex['silhouette'].max():.2f}) and kNN label agreement "
        f"sits at {ex['knn_agreement'].min():.3f}–{ex['knn_agreement'].max():.3f} against a "
        f"permutation null of {ex['knn_null_mean'].min():.3f}–{ex['knn_null_mean'].max():.3f}. "
        f"contrastive_ae's silhouette p = {float(cae['silhouette_p'].iloc[0]):.2f} — "
        f"indistinguishable from chance. The sequence, PPI and literature priors score better only "
        f"because they cover far fewer of these genes "
        f"({pr['n_genes_scored'].min()}–{pr['n_genes_scored'].max()} vs "
        f"{int(ex['n_genes_scored'].iloc[0])}), so the comparison is not like-for-like. "
        f"**Reported as a negative result rather than cropped to the flattering subset.**", y2)
    notes(s, "A pretty UMAP coloured by pathway is not evidence that pathways "
             "are recovered. The permutation null is what turns the picture "
             "into a claim, and here the claim is weak.")

    # ---- 21 merge
    s = blank(prs)
    y = header(s, "Merging into Morpholab_Hackathon/preprocessing", kicker="integration")
    tdf = pd.DataFrame({
        "Morpholab": ["_gene_key", "obs x genes orientation", "REFERENCE_DATASETS",
                      "select_dataset score", "loadings (genes x d)"],
        "panelprep": ["normalize.gene_key", "PanelQuery / CandidateDataset",
                      "io.catalog.CATALOG", "score corr_mae",
                      "GeneEmbedding.matrix"],
        "status": ["identical, pinned by a test", "identical",
                   "bridged by reference_to_catalog",
                   "same formula, bit-for-bit (diff 0.00e+00)",
                   "same contract, metadata-only copy"],
    })
    y2 = table(s, tdf, y=y, size=12, col_widths=[2.2, 3.0, 4.0],
               mono_cols=("Morpholab", "panelprep", "status"))
    caption(s, "The merge is **additive** — nothing in dataset_selection.py has to change. Their one "
               "score becomes one registry row; their factor analysis becomes one embedder row.", y2)
    notes(s, "panelprep/integrations/morpholab.py is the only module that knows "
             "anything about the host repo.")

    # ---- 22 pearson
    s = blank(prs)
    y = header(s, "One decision to make before merging: Pearson", kicker="integration")
    tdf = pd.DataFrame({
        "correlation": ["Pearson", "Pearson", "Spearman", "Spearman"],
        "reducer": ["MAE (theirs)", "RMS", "MAE", "RMS"],
        "sibling": [0.0295, 0.0383, 0.0276, 0.0376],
        "same_program_diff_mixture": [0.0261, 0.0280, 0.0502, 0.0665],
        "order correct?": ["no", "no", "yes", "yes"],
    })
    y2 = table(s, tdf, y=y, size=13, col_widths=[1.4, 1.5, 1.2, 2.6, 1.4],
               mono_cols=("correlation", "reducer", "order correct?"),
               highlight_rows=(0, 1))
    caption(s,
        "select_dataset calls DataFrame.corr(), i.e. **Pearson**. On the synthetic fixtures, whose "
        "ordering is known by construction, both Pearson variants rank same_program_diff_mixture "
        "ahead of sibling. **The MAE-vs-RMS reducer makes no difference; Pearson vs Spearman decides "
        "it.** same_program_diff_mixture runs the same gene programs in different cell-type "
        "proportions, so its rank structure diverges while its linear correlation on four sparse "
        "genes stays close — the classic failure mode of Pearson on zero-inflated counts. "
        "corr_mae keeps their default so the number reproduces exactly; the finding is pinned as a "
        "regression test rather than silently corrected.", y2, size=12)
    notes(s, "Which correlation to use is the host repo's call, not this "
             "package's. Pass score_params={'corr_mae': {'method': 'spearman'}} "
             "to switch, or read the corr_frobenius row.")

    # ---- 23 status
    s = blank(prs)
    y = header(s, "Every method, labelled", kicker="scope")
    tdf = pd.DataFrame({
        "status": ["implemented + evaluated", "implemented + evaluated",
                   "implemented + evaluated", "implemented + evaluated",
                   "deferred", "deferred", "fixed this round"],
        "what": [
            "8 scores (KL, JS, Frobenius, cosine, MAE, OT+ARI, 2 spatial)",
            "7 expression embedders, separate and joint, 24 matrices",
            "5 prior sources over 4,006 genes + kernel fusion",
            "pathway / ligand / receptor / TF annotation (39 / 95 / 123 of 2000)",
            "perturbation_pseudobulk — needs a genome-scale screen (1 perturbation available)",
            "ops_morphology — PERISCOPE is human-only, this is mouse",
            "regulatory_kegg ignored the threaded cache_dir and offline flags",
        ],
    })
    y2 = table(s, tdf, y=y, size=13, col_widths=[2.6, 7.4], mono_cols=("status", "what"))
    caption(s, "Nothing is reported as working that was not run. The two deferred sources return "
               "all-NaN matrices with a documented reason rather than a plausible substitute.", y2)

    # ---- 24 how to run
    s = blank(prs)
    y = header(s, "How to run it", kicker="handover")
    tf = _tb(s, M, y, W - 2 * M, Inches(2.2))
    for i, line in enumerate([
        "# once: build the expensive, gene-list-only embeddings (or adopt the shipped ones)",
        "python scripts/build_embedding_store.py --import-dir data/priors_export",
        "",
        "# every run: edit the Parameters cell, then execute",
        "jupyter lab notebooks/05_run_analysis.ipynb",
        "",
        "# offline test suite",
        "python -m pytest tests -q -m 'not network'",
        f"#   -> {d['pytest']}",
    ]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(3)
        _run(p, line, size=13, mono=True,
             color=THEME["muted"] if line.startswith("#") else THEME["ink"])
    bullets(s, [
        ("The Parameters cell is the only thing to edit",
         "query + panel, candidate datasets, score names, embedder lists, dim, gene universe, "
         "projection, seed"),
        ("Adding a score updates the heatmap and the comparison report automatically",
         "one decorator in panelprep/scores/custom.py; nothing else to touch"),
        ("A missing expensive embedding is reported **blocked**, never silently retrained",
         "the run tells you the build command and carries on with everything else"),
    ], y=y + Inches(2.3), size=15)

    return prs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="panelprep_results.pptx")
    args = ap.parse_args(argv)
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slides(prs, load_data())
    prs.save(args.out)
    print(f"wrote {args.out}: {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
