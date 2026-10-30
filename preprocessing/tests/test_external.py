"""Acceptance tests for the external (non-transcriptomic / causal) embedders
and the gene annotators.

Two kinds of test
-----------------
``@pytest.mark.network``
    Hits rest.uniprot.org, the STRING download server, rest.kegg.jp and
    huggingface.co, and fills ``data/external/``. Run once; everything is
    cached afterwards.
``offline``
    No marker. Reads the cache only (``offline=True``) and is **skipped** with
    a clear reason when the cache is cold, so a fresh checkout does not report
    red tests for a missing download.

The ``network`` marker is not registered in this repo's pytest config (that
file is not owned by this module); pytest emits a ``PytestUnknownMarkWarning``
but the marker still selects correctly. Add ``markers = ["network: hits
external services", "slow: minutes, not seconds"]`` to
``pyproject.toml``/``pytest.ini`` to silence it.

Run::

    pytest tests/test_external.py -m network -s      # first time, with downloads
    pytest tests/test_external.py -m "not network"   # cache-only
    python tests/test_external.py                    # coverage report, 200 genes

Every assertion here reports the number it checked (``-s`` prints them), because
"the figure looks right" is not evidence.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.getcwd())

from panelprep.annotate import annotate_genes                       # noqa: E402
from panelprep.annotate.roles import ROLE_QUERIES                   # noqa: E402
from panelprep.embed.external import (DEFAULT_ESM_CHECKPOINT,       # noqa: E402
                                      _symbol_lookup, esm2_embed_sequences,
                                      esm2_sequence, kegg_causal, kegg_relations,
                                      kegg_signed_graph, string_graph, string_ppi)

#: MODIFY HERE: point the tests at a different cache with
#: ``PANELPREP_CACHE=/path pytest ...``
CACHE = os.environ.get("PANELPREP_CACHE", "data/external")

#: Mouse gastrulation markers -- the biology of the synthetic fixture's
#: descriptor (E7.5 embryo, germ-layer patterning).
PANEL = ["Sox2", "T", "Foxa2", "Hand1", "Mesp1", "Gata4"]

#: Pool the "unrelated gene" for the ESM homology check is drawn from. None is
#: a SOX-family member or a developmental TF; the draw is seeded so the test is
#: reproducible but not cherry-picked.
UNRELATED_POOL = ["Actb", "Ldha", "Myh9", "Cox4i1", "Rpl13a", "Atp5f1a", "Hbb-bs"]


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _skip_if_cold(*paths: str) -> None:
    missing = [p for p in paths if not Path(CACHE, p).exists()]
    if missing:
        pytest.skip(f"cold cache: {missing} (run `pytest -m network` once)")


# ==========================================================================
# 1. ESM-2 on real mouse genes
# ==========================================================================

@pytest.mark.network
def test_esm2_panel_matrix_and_homology():
    """(6, d) matrix, no all-NaN row, and Sox2 closer to Sox17 than to a
    randomly drawn unrelated gene. Reports both cosines."""
    rng = np.random.default_rng(0)
    unrelated = UNRELATED_POOL[int(rng.integers(len(UNRELATED_POOL)))]
    genes = PANEL + ["Sox17", unrelated]
    emb = esm2_sequence({}, genes=genes, cache_dir=CACHE, offline=False)

    panel = emb.matrix.loc[PANEL]
    assert panel.shape[0] == 6, panel.shape
    assert panel.shape[1] == 480, f"expected ESM-2 35M hidden size 480, got {panel.shape[1]}"
    n_all_nan = int(panel.isna().all(axis=1).sum())
    assert n_all_nan == 0, f"{n_all_nan} all-NaN rows in the panel embedding"

    v = {g: emb.matrix.loc[g].to_numpy() for g in ("Sox2", "Sox17", unrelated)}
    cos_related = _cos(v["Sox2"], v["Sox17"])
    cos_unrelated = _cos(v["Sox2"], v[unrelated])
    print(f"\n[1] ESM-2 {emb.matrix.shape} from {DEFAULT_ESM_CHECKPOINT}; "
          f"cos(Sox2, Sox17) = {cos_related:.4f}, "
          f"cos(Sox2, {unrelated}) = {cos_unrelated:.4f}, "
          f"margin = {cos_related - cos_unrelated:+.4f}; "
          f"n_no_protein = {emb.provenance['n_no_protein']}, "
          f"n_truncated = {emb.provenance['n_truncated']}")
    assert cos_related > cos_unrelated, (
        f"cos(Sox2,Sox17)={cos_related:.4f} !> cos(Sox2,{unrelated})={cos_unrelated:.4f}")


def test_esm2_offline_from_cache():
    """The same call with ``offline=True`` returns the cached matrix."""
    _skip_if_cold("uniprot/mouse_reviewed_symbol_lookup.json",
                  f"esm2/{DEFAULT_ESM_CHECKPOINT.replace('/', '__')}/embeddings.npz")
    emb = esm2_sequence({}, genes=PANEL, cache_dir=CACHE, offline=True)
    assert emb.provenance.get("status") == "ok", emb.provenance
    assert emb.matrix.shape == (6, 480), emb.matrix.shape
    assert int(emb.matrix.isna().all(axis=1).sum()) == 0
    print(f"\n[1-offline] cached ESM-2 {emb.matrix.shape}, "
          f"n_from_cache = {emb.provenance['n_from_cache']}, "
          f"n_computed = {emb.provenance['n_computed']}")


# ==========================================================================
# 2. pooling: special tokens excluded, padding-invariant
# ==========================================================================

@pytest.mark.network
def test_pooling_excludes_special_tokens_and_is_padding_invariant():
    """The two failure modes of the naive ``last_hidden_state.mean(1)``.

    (a) the pooled vector must differ from the mean over the *full* padded
        tensor (which includes BOS, EOS and padding);
    (b) a sequence batched with a much longer one must get the same vector as
        when it is embedded alone.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    from panelprep.embed.external import _ensure_checkpoint

    lut = _symbol_lookup(["Sox2", "Fgfr1"], cache_dir=CACHE, offline=False)
    short = lut.loc["Sox2", "sequence"]
    long = lut.loc["Fgfr1", "sequence"]
    assert isinstance(short, str) and isinstance(long, str)

    local = _ensure_checkpoint(DEFAULT_ESM_CHECKPOINT, cache_dir=CACHE, offline=False)
    tok = AutoTokenizer.from_pretrained(str(local))
    model = AutoModel.from_pretrained(str(local), add_pooling_layer=False).eval()

    enc = tok([short, long], return_tensors="pt", padding=True)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state
    special = torch.tensor(sorted(set(tok.all_special_ids)))
    keep = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special)
    w = keep.unsqueeze(-1).to(hidden.dtype)
    correct = ((hidden * w).sum(1) / w.sum(1)).numpy()
    naive = hidden.mean(1).numpy()                     # the common, wrong shortcut

    delta = float(np.abs(correct[0] - naive[0]).max())
    n_pad = int((~enc["attention_mask"][0].bool()).sum())
    print(f"\n[2a] Sox2 ({len(short)} aa) batched with Fgfr1 ({len(long)} aa): "
          f"{n_pad} padding positions + 2 special tokens; "
          f"max|correct - naive_full_mean| = {delta:.5f}")
    assert delta > 1e-3, "pooled vector is indistinguishable from the naive mean"

    # Fresh embedding cache so this really recomputes every run: the checkpoint
    # is linked in from the warm cache, only `embeddings.npz` starts empty.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "esm2" / DEFAULT_ESM_CHECKPOINT.replace("/", "__")
        dst.mkdir(parents=True)
        for f in local.iterdir():
            if f.name == "embeddings.npz":
                continue
            try:
                os.link(f, dst / f.name)
            except OSError:
                shutil.copy2(f, dst / f.name)
        single, _ = esm2_embed_sequences({"A": short}, cache_dir=tmp, offline=False)
        batched, _ = esm2_embed_sequences({"A2": short, "B": long},
                                          cache_dir=tmp, offline=False, batch_size=2)
    diff = float(np.abs(single["A"] - batched["A2"]).max())
    print(f"[2b] max|batched - single| = {diff:.3e}  (tolerance 1e-4)")
    assert diff < 1e-4, f"padding changed the embedding by {diff:.3e}"


# ==========================================================================
# 3. STRING graph
# ==========================================================================

PLURIPOTENCY = ["Sox2", "Pou5f1", "Nanog"]


@pytest.mark.network
def test_string_pluripotency_triangle():
    """All three pairwise edges exist at threshold 400; reports the weights."""
    g, meta = string_graph(PLURIPOTENCY, threshold=400, cache_dir=CACHE, offline=False)
    edges = {frozenset((a, b)): d["combined_score"] for a, b, d in g.edges(data=True)}
    print(f"\n[3] STRING v{meta['string_version']} threshold 400: "
          f"{meta['n_nodes']} nodes, {meta['n_edges']} edges, "
          f"unmapped = {meta['unmapped']}, isolates = {meta['isolates']}; "
          + ", ".join(f"{'-'.join(sorted(k))} = {v}"
                      for k, v in sorted(edges.items(), key=lambda kv: sorted(kv[0]))))
    for a, b in (("Sox2", "Pou5f1"), ("Sox2", "Nanog"), ("Pou5f1", "Nanog")):
        assert frozenset((a, b)) in edges, f"missing STRING edge {a}-{b} at 400"
        assert edges[frozenset((a, b))] >= 400
    assert meta["unmapped"] == [] and meta["isolates"] == []


def test_string_embedding_offline_and_isolates():
    """Cache-only run; an unconnected gene gets a NaN row, not a zero row."""
    _skip_if_cold("string/10090.protein.links.v12.0.txt.gz",
                  "string/10090.protein.info.v12.0.txt.gz")
    genes = PLURIPOTENCY + ["Hbb-bs", "NotAGene9999"]
    emb = string_ppi({}, genes=genes, dim=3, cache_dir=CACHE, offline=True)
    nan_rows = emb.matrix.index[emb.matrix.isna().all(axis=1)].tolist()
    print(f"\n[3-offline] {emb.matrix.shape}; n_unmapped = {emb.provenance['n_unmapped']} "
          f"{emb.provenance['unmapped_examples']}, n_isolates = "
          f"{emb.provenance['n_isolates']} {emb.provenance['isolate_examples']}, "
          f"NaN rows = {nan_rows}, eigenvalues = "
          f"{emb.provenance['laplacian_eigenvalues']}")
    assert "NotAGene9999" in nan_rows
    assert emb.provenance["n_unmapped"] >= 1
    assert not emb.matrix.loc[PLURIPOTENCY].isna().any().any()
    assert emb.category == "non_transcriptomic"


# ==========================================================================
# 4. KEGG signed graph
# ==========================================================================

#: Genes chosen to contain curated activation *and* inhibition relations
#: (p53/Mdm2 and the Wnt destruction complex are the classic inhibitory edges).
SIGNED_SET = ["Tp53", "Mdm2", "Cdkn1a", "Ctnnb1", "Gsk3b", "Apc", "Axin1", "Myc",
              "Jun", "Fos", "Smad4", "Smad7", "Tgfb1", "Sox2", "Pou5f1", "Nanog"]


@pytest.mark.network
def test_kegg_signed_graph_has_both_signs():
    """At least one +1 and one -1 edge; reports the counts and examples."""
    g, meta = kegg_signed_graph(SIGNED_SET, cache_dir=CACHE, offline=False)
    pos = [(a, b, d["weight"], d["subtypes"]) for a, b, d in g.edges(data=True)
           if d["weight"] == 1.0]
    neg = [(a, b, d["weight"], d["subtypes"]) for a, b, d in g.edges(data=True)
           if d["weight"] == -1.0]
    print(f"\n[4] KEGG signed graph over {len(SIGNED_SET)} genes: "
          f"{meta['n_nodes']} nodes, {meta['n_edges']} directed edges, "
          f"{meta['n_positive_edges']} positive / {meta['n_negative_edges']} negative; "
          f"exactly +1: {len(pos)}, exactly -1: {len(neg)}; "
          f"absent genes = {len(meta['absent'])} {meta['absent']}\n"
          f"    +1 examples: {pos[:3]}\n    -1 examples: {neg[:3]}")
    assert len(pos) >= 1, "no +1 (activation/expression) edge"
    assert len(neg) >= 1, "no -1 (inhibition/repression) edge"


def test_kegg_embedding_offline_direction_blocks():
    """Cache-only; the embedding keeps separate in/out blocks and NaN-rows the
    genes with no curated relation."""
    _skip_if_cold("kegg/relations_mmu.tsv")
    emb = kegg_causal({}, genes=SIGNED_SET, dim=12, cache_dir=CACHE, offline=True)
    cols = list(emb.matrix.columns)
    print(f"\n[4-offline] {emb.matrix.shape} blocks = {emb.provenance['block_dims']}, "
          f"n_embedded = {emb.provenance['n_embedded']}, "
          f"n_absent = {emb.provenance['n_absent']}, "
          f"pos/neg edges = {emb.provenance['n_positive_edges']}/"
          f"{emb.provenance['n_negative_edges']}, columns = {cols}")
    assert emb.category == "causal"
    assert any(c.startswith("kegg_out") for c in cols)
    assert any(c.startswith("kegg_in") for c in cols)
    assert any(c.startswith("kegg_sym") for c in cols)
    assert emb.provenance["n_absent"] == int(emb.matrix.isna().all(axis=1).sum())
    assert "NOT causal effects" in emb.provenance["caveat"]


def test_kegg_relation_table_signs_are_present():
    """The parsed relation table itself carries both signed subtypes."""
    _skip_if_cold("kegg/relations_mmu.tsv")
    rel = kegg_relations(CACHE, offline=True)
    counts = rel["subtype"].value_counts()
    print(f"\n[4-table] {len(rel)} relations over {rel['pathway'].nunique()} pathways; "
          f"activation = {counts.get('activation', 0)}, "
          f"inhibition = {counts.get('inhibition', 0)}, "
          f"expression = {counts.get('expression', 0)}, "
          f"repression = {counts.get('repression', 0)}")
    assert counts.get("activation", 0) > 0 and counts.get("inhibition", 0) > 0


# ==========================================================================
# 5. annotations
# ==========================================================================

@pytest.mark.network
def test_annotate_marks_tfs_receptor_and_ligand():
    """T/Sox2/Gata4 are TFs, Fgfr1 is a receptor, Wnt3 is a ligand.
    Prints the exact rows."""
    genes = ["T", "Sox2", "Gata4", "Fgfr1", "Wnt3", "Actb"]
    ann = annotate_genes(genes, sources=("kegg", "roles"), cache_dir=CACHE, offline=False)
    cols = ["is_tf", "is_receptor", "is_ligand", "accession", "top_pathway", "n_pathways"]
    print("\n[5] annotate_genes rows:\n" + ann.table[cols].to_string())
    print(f"    coverage = {ann.sources['coverage']}")
    print(f"    role queries = {json.dumps(ann.sources['roles']['role_queries'], indent=6)}")
    for g in ("T", "Sox2", "Gata4"):
        assert bool(ann.table.loc[g, "is_tf"]), f"{g} not flagged is_tf"
    assert bool(ann.table.loc["Fgfr1", "is_receptor"]), "Fgfr1 not flagged is_receptor"
    assert bool(ann.table.loc["Wnt3", "is_ligand"]), "Wnt3 not flagged is_ligand"
    assert not bool(ann.table.loc["Fgfr1", "is_tf"])
    assert not bool(ann.table.loc["Actb", "is_tf"])
    assert set(ann.sources["roles"]["role_queries"]) == set(ROLE_QUERIES)


def test_annotate_offline_and_schema():
    """Cache-only run satisfies the ``GeneAnnotation`` schema, and the
    membership frame is gene x pathway."""
    _skip_if_cold("kegg/link_mmu_pathway.txt", "uniprot/roles/is_tf_go.tsv")
    genes = ["T", "Sox2", "Gata4", "Fgfr1", "Wnt3", "Actb"]
    ann = annotate_genes(genes, cache_dir=CACHE, offline=True)
    for col in ("is_ligand", "is_receptor", "is_tf", "pathways", "top_pathway"):
        assert col in ann.table.columns
    mem = ann.sources["kegg_membership"]
    assert list(mem.index) == genes
    print(f"\n[5-offline] table {ann.table.shape}, membership {mem.shape}, "
          f"coverage = {ann.sources['coverage']}, "
          f"top_pathways = {ann.table['top_pathway'].to_dict()}")
    assert ann.table.loc["Sox2", "is_tf"] and not ann.table.loc["Actb", "is_tf"]


def test_annotate_unknown_genes_are_false_not_missing():
    """A nonsense symbol is False/empty everywhere -- a status, not an error."""
    _skip_if_cold("kegg/link_mmu_pathway.txt", "uniprot/roles/is_tf_go.tsv")
    ann = annotate_genes(["Sox2", "NotAGene9999"], cache_dir=CACHE, offline=True)
    row = ann.table.loc["NotAGene9999"]
    print(f"\n[5-unknown] NotAGene9999 -> is_tf={row['is_tf']}, "
          f"is_receptor={row['is_receptor']}, is_ligand={row['is_ligand']}, "
          f"n_pathways={row['n_pathways']}, top_pathway={row['top_pathway']!r}")
    assert not (row["is_tf"] or row["is_receptor"] or row["is_ligand"])
    assert row["n_pathways"] == 0 and row["top_pathway"] == "none"


def test_blocked_sources_are_reported_not_silently_false():
    """With a cold cache and ``offline=True`` the roles come back all-False
    *with* an explicit note, so all-False can never be read as a negative."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ann = annotate_genes(["Sox2", "Gata4"], cache_dir=tmp, offline=True)
        roles, kegg = ann.sources["roles"], ann.sources["kegg"]
        print(f"\n[blocked] roles.status = {roles['status']}, "
              f"kegg.status = {kegg['status']}, note = {roles.get('note', '')[:90]}")
        assert roles["status"] == "blocked" and "note" in roles
        assert kegg["status"] == "blocked"
        assert not ann.table[["is_tf", "is_receptor", "is_ligand"]].any().any()


def test_blocked_embedders_return_nan_not_exception():
    """A cold cache makes the external embedders ``blocked``, never raising."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = {}
        for name, fn in (("string_ppi", string_ppi), ("kegg_causal", kegg_causal),
                         ("esm2_sequence", esm2_sequence)):
            emb = fn({}, genes=["Sox2", "Gata4"], dim=4, cache_dir=tmp, offline=True)
            out[name] = emb.provenance["status"]
            assert emb.matrix.isna().all().all(), f"{name} returned non-NaN from cold cache"
        print(f"\n[blocked-embed] statuses = {out}")
        assert set(out.values()) == {"blocked"}


# ==========================================================================
# coverage on a 200-gene mouse list -- a result, not a detail
# ==========================================================================

def sample_mouse_genes(n: int = 200, seed: int = 0, cache_dir: str = CACHE) -> list:
    """A seeded random sample of ``n`` protein-coding mouse gene symbols.

    Sampling frame: KEGG ``list/mmu`` restricted to ``CDS`` entries that carry a
    gene symbol -- i.e. the NCBI mouse protein-coding gene set, ~22k symbols.
    Chosen deliberately over "200 reviewed Swiss-Prot genes" (which would make
    UniProt coverage 100% by construction) and over a curated marker panel
    (which would be biased toward well-annotated genes). The sample is seeded,
    so the coverage numbers are reproducible.
    MODIFY HERE: ``seed``/``n``; or pass your own list to ``coverage_report``.
    """
    from panelprep.embed.external import KEGG_REST, _cached_text, cache_root

    txt = _cached_text(f"{KEGG_REST}/list/mmu",
                       cache_root(cache_dir) / "kegg" / "list_mmu.txt",
                       offline=True, min_bytes=1000)
    syms = []
    for line in txt.strip().split("\n"):
        p = line.split("\t")
        if len(p) >= 4 and p[1] == "CDS" and ";" in p[-1]:
            s = p[-1].split(";")[0].split(",")[0].strip()
            if s and not s.startswith("mmu:"):
                syms.append(s)
    syms = sorted(set(syms))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(syms), size=min(n, len(syms)), replace=False)
    return [syms[i] for i in sorted(idx)]


def coverage_report(genes=None, *, cache_dir: str = CACHE, offline: bool = False,
                    out_csv: str = "outputs/external_coverage_200.csv") -> pd.DataFrame:
    """Per-gene coverage across all four external sources, for a gene list.

    Observation unit: one row per gene symbol.
    Normalisation: not applicable.
    Missing data: the whole point -- every column is a presence flag, and the
    printed summary is the denominator-explicit count.
    MODIFY HERE: ``genes`` (defaults to ``sample_mouse_genes(200)``), ``offline``.
    """
    genes = list(genes) if genes is not None else sample_mouse_genes(200, cache_dir=cache_dir)
    lut = _symbol_lookup(genes, cache_dir=cache_dir, offline=offline)
    esm = esm2_sequence({}, genes=genes, cache_dir=cache_dir, offline=offline)
    ppi = string_ppi({}, genes=genes, dim=32, cache_dir=cache_dir, offline=offline)
    keg = kegg_causal({}, genes=genes, dim=32, cache_dir=cache_dir, offline=offline)
    ann = annotate_genes(genes, cache_dir=cache_dir, offline=offline)
    rel = kegg_relations(cache_dir, offline=True)

    # Two different questions, kept apart on purpose:
    #   *_in_source  -- does the source know this gene at all?
    #   *_embeddable -- does it have an edge to another gene *in this list*?
    # A random sample of 200 genes out of ~22k is sparse by construction, so the
    # embeddable fraction is a property of the requested gene set, not of STRING
    # or KEGG coverage. Reporting only the second number would understate both
    # sources badly; reporting only the first would overstate what the
    # embeddings actually cover.
    kegg_known = set(rel["source"]).union(rel["target"]) if len(rel) else set()
    unmapped = set(ppi.provenance.get("unmapped", []))        # full list, not the preview

    df = pd.DataFrame(index=pd.Index(genes, name="gene"))
    df["uniprot_hit"] = lut["accession"].notna().reindex(genes).fillna(False).values
    df["esm_embedding"] = ~esm.matrix.isna().all(axis=1).reindex(genes).values
    df["string_in_source"] = [g not in unmapped for g in genes]
    df["string_node"] = ~ppi.matrix.isna().all(axis=1).reindex(genes).values
    df["kegg_in_source"] = [g in kegg_known for g in genes]
    df["kegg_node"] = ~keg.matrix.isna().all(axis=1).reindex(genes).values
    df["pathway"] = (ann.table["n_pathways"] > 0).reindex(genes).fillna(False).values
    df["role_flag"] = (ann.table[["is_tf", "is_receptor", "is_ligand"]]
                       .any(axis=1).reindex(genes).fillna(False).values)
    n = len(genes)
    print(f"\n=== external-source coverage, {n} mouse genes "
          f"(seeded sample of KEGG CDS symbols) ===")
    for c in df.columns:
        print(f"    {c:17s} {int(df[c].sum())}/{n} ({100 * df[c].mean():.1f}%)")
    print(f"    STRING: n_unmapped = {ppi.provenance['n_unmapped']}, "
          f"n_isolates = {ppi.provenance['n_isolates']} at threshold "
          f"{ppi.provenance['threshold']} "
          f"(isolate = mapped but no edge to another gene in this list)")
    print(f"    KEGG:   {keg.provenance['n_nodes']} nodes, "
          f"{keg.provenance['n_edges']} edges "
          f"({keg.provenance['n_positive_edges']}+ / "
          f"{keg.provenance['n_negative_edges']}-)")
    print(f"    ESM-2:  n_computed = {esm.provenance['n_computed']}, "
          f"n_from_cache = {esm.provenance['n_from_cache']}, "
          f"n_truncated = {esm.provenance['n_truncated']}, "
          f"seconds = {esm.provenance.get('seconds', 0)}")
    print(f"    roles:  {ann.sources['coverage']}")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv)
        print(f"    written: {out_csv}")
    return df


@pytest.mark.network
@pytest.mark.slow
def test_coverage_200_genes():
    """Coverage of a seeded 200-gene mouse sample across all four sources."""
    df = coverage_report()
    assert len(df) == 200
    # What is asserted here are *invariants*, not magnitudes. The coverage
    # fractions themselves are a result to be read, not a target to be met: a
    # seeded random sample of mouse genes contains Riken clones and predicted
    # loci, and tuning a threshold until it passes would turn the number into
    # decoration. The only thing a test can legitimately check is that each
    # source answered at all and that the columns are mutually consistent.
    for col in ("uniprot_hit", "string_in_source", "pathway"):
        assert df[col].sum() > 0, f"{col} is zero for all 200 genes -- source silent?"
    assert (df["esm_embedding"] == df["uniprot_hit"]).all(), \
        "ESM coverage must equal UniProt coverage (sequence comes from UniProt)"
    assert (df["string_node"] <= df["string_in_source"]).all(), \
        "a gene cannot be an embeddable STRING node without being in STRING"
    assert (df["kegg_node"] <= df["kegg_in_source"]).all(), \
        "a gene cannot have a within-set KEGG edge without being in the relation table"
    assert (df["role_flag"] <= df["uniprot_hit"]).all(), \
        "role flags come from UniProt, so they cannot exceed UniProt coverage"


if __name__ == "__main__":
    coverage_report(offline=False)
