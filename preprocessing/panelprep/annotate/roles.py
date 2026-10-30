"""Functional role flags: transcription factor, receptor, ligand (item 4).

Evidence actually used
----------------------
Every flag comes from a UniProtKB query over the **reviewed (Swiss-Prot) mouse
proteome**, run against ``rest.uniprot.org`` and cached. The exact query
strings are constants in this module and are copied verbatim into
``GeneAnnotation.sources["role_queries"]``, so a reader can paste them into the
UniProt website and get the same set back. Nothing here is a hand-typed gene
list: a hardcoded list of "known" TFs would be an opinion wearing a database's
clothes, and it would silently never change when the ontology does.

    is_tf        GO:0003700  DNA-binding transcription factor activity
    is_receptor  GO:0038023  signaling receptor activity      OR  keyword KW-0675 (Receptor)
    is_ligand    GO:0048018  receptor ligand activity

Three deliberate departures from the loosest reading of the brief, and why
-------------------------------------------------------------------------
1. **UniProt keywords KW-0804 ("Transcription") and KW-0805 ("Transcription
   regulation") are computed but do not drive ``is_tf`` by default.** They are
   process keywords, not molecular-function ones: they are carried by RNA
   polymerase subunits, mediator components, chromatin remodellers and
   coactivators that do not bind DNA sequence-specifically. The reviewed mouse
   set for KW-0805 is 1871 entries against 1019 for GO:0003700. Calling all of
   those "transcription factors" would inflate the flag and make a
   colour-by-role plot meaningless. Both are returned as the separate columns
   ``is_tf_go`` / ``is_tf_keyword`` so the union is one line away:
   ``tf_evidence=("go","keyword")``.
2. **GO:0005102 (signaling receptor binding) is computed but is not the default
   for ``is_ligand``.** It is a superset of ligand activity that also contains
   receptor-binding adaptors, antagonists and decoys. It is exposed as
   ``is_ligand_broad`` and selectable via ``ligand_evidence``.
3. **A gene is flagged via its accession, not its symbol, whenever possible.**
   Roles are a property of a protein; matching role sets by gene symbol invites
   synonym collisions (mouse ``T`` is a documented example: UniProt's loose
   ``gene:`` field returns Lck for it). ``panelprep.embed.external._symbol_lookup``
   resolves the symbol to an accession first; symbol matching against *primary*
   symbols only is the fallback and is counted separately in the meta.

Missing-data behaviour
----------------------
If a query cannot be made at all (offline with a cold cache, or the service is
unreachable) every flag is returned ``False`` **with an explicit note** in the
returned meta -- ``status="blocked"`` plus the reason -- so a table of all-False
roles can never be mistaken for "mouse has no transcription factors". Absence
of evidence is reported as absence of evidence. A gene that resolves to no
reviewed protein is likewise ``False`` everywhere, counted in
``n_without_accession``.

MODIFY HERE
-----------
* ``ROLE_QUERIES`` -- add a role (e.g. GPCRs via keyword KW-0297) and it flows
  straight into the returned table.
* ``tf_evidence`` / ``receptor_evidence`` / ``ligand_evidence`` -- which
  evidence sources define each boolean.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests

from ..embed.external import (DEFAULT_CACHE_DIR, MOUSE_TAXON, OfflineCacheMiss,
                              UNIPROT_SEARCH, _session, _symbol_lookup, cache_root)
from ..normalize import gene_key

_REVIEWED_MOUSE = f"(organism_id:{MOUSE_TAXON}) AND (reviewed:true)"

#: The exact UniProtKB query per evidence source. Keys are the column names
#: produced in the role table.
#: MODIFY HERE: add an entry to add a role column.
ROLE_QUERIES: Dict[str, str] = {
    "is_tf_go": f"{_REVIEWED_MOUSE} AND (go:0003700)",
    "is_tf_keyword": f"{_REVIEWED_MOUSE} AND (keyword:KW-0804 OR keyword:KW-0805)",
    "is_receptor_go": f"{_REVIEWED_MOUSE} AND (go:0038023)",
    "is_receptor_keyword": f"{_REVIEWED_MOUSE} AND (keyword:KW-0675)",
    "is_ligand_go": f"{_REVIEWED_MOUSE} AND (go:0048018)",
    "is_ligand_broad": f"{_REVIEWED_MOUSE} AND (go:0005102)",
}

#: Which evidence columns combine (logical OR) into each public flag.
DEFAULT_EVIDENCE: Dict[str, Tuple[str, ...]] = {
    "is_tf": ("go",),                       # GO:0003700 only -- see module docstring
    "is_receptor": ("go", "keyword"),       # GO:0038023 OR KW-0675
    "is_ligand": ("go",),                   # GO:0048018 only; "broad" adds GO:0005102
}

_EVIDENCE_COLUMNS: Dict[str, Dict[str, str]] = {
    "is_tf": {"go": "is_tf_go", "keyword": "is_tf_keyword"},
    "is_receptor": {"go": "is_receptor_go", "keyword": "is_receptor_keyword"},
    "is_ligand": {"go": "is_ligand_go", "broad": "is_ligand_broad"},
}

_FIELDS = "accession,gene_primary"
_PAGE = 500


def _slug(name: str) -> str:
    return name.replace(":", "_").replace(" ", "_")


def _uniprot_paged(query: str, dest: Path, *, offline: bool,
                   fields: str = _FIELDS, page: int = _PAGE) -> pd.DataFrame:
    """Run a UniProt search to exhaustion, following the ``link`` cursor.

    Observation unit: one UniProt entry per row.
    Normalisation: not applicable.
    Missing data: ``offline=True`` with no cache raises ``OfflineCacheMiss``.
    Caching: the concatenated TSV is written to ``dest``; a non-empty cache is
    reused without touching the network.
    MODIFY HERE: ``page`` (UniProt caps ``size`` at 500 for TSV).
    """
    if dest.exists() and dest.stat().st_size > 20:
        return pd.read_csv(dest, sep="\t")
    if offline:
        raise OfflineCacheMiss(f"offline=True and cache miss: {dest}")
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    url, params = UNIPROT_SEARCH, {"query": query, "format": "tsv",
                                   "size": page, "fields": fields}
    for _ in range(200):                              # hard stop: 100k entries
        r = _session().get(url, params=params, timeout=120)
        r.raise_for_status()
        lines = r.text.strip().split("\n")
        if not lines or not lines[0]:
            break
        if header is None:
            header = lines[0].split("\t")
        rows += [ln.split("\t") for ln in lines[1:] if ln]
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            break
        url, params = nxt, None
    df = pd.DataFrame(rows, columns=header or ["Entry", "Gene Names (primary)"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, sep="\t", index=False)
    return df


def _evidence_sets(cache_dir: str, offline: bool
                   ) -> Tuple[Dict[str, Tuple[Set[str], Set[str]]], dict]:
    """``{column: (accessions, primary_symbol_keys)}`` for every role query."""
    out: Dict[str, Tuple[Set[str], Set[str]]] = {}
    notes: Dict[str, object] = {}
    root = cache_root(cache_dir) / "uniprot" / "roles"
    for col, query in ROLE_QUERIES.items():
        try:
            df = _uniprot_paged(query, root / f"{_slug(col)}.tsv", offline=offline)
        except (OfflineCacheMiss, requests.RequestException, ValueError) as exc:
            out[col] = (set(), set())
            notes[col] = f"unavailable: {type(exc).__name__}: {exc}"
            continue
        accs = set(df.iloc[:, 0].astype(str)) if len(df) else set()
        syms = ({gene_key(s) for s in df.iloc[:, 1].astype(str) if s and s != "nan"}
                if df.shape[1] > 1 and len(df) else set())
        out[col] = (accs, syms)
        notes[col] = {"n_entries": int(len(df))}
    return out, notes


def role_flags(
    genes: Sequence[str],
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
    tf_evidence: Sequence[str] = DEFAULT_EVIDENCE["is_tf"],
    receptor_evidence: Sequence[str] = DEFAULT_EVIDENCE["is_receptor"],
    ligand_evidence: Sequence[str] = DEFAULT_EVIDENCE["is_ligand"],
) -> Tuple[pd.DataFrame, dict]:
    """Ligand / receptor / TF flags for ``genes`` from UniProtKB.

    Observation unit: one row per *requested gene symbol*, resolved to one
    reviewed mouse protein.
    Normalisation: not applicable (no expression involved).
    Missing data: a gene with no reviewed protein, or any gene at all when the
    queries could not be made, is ``False`` on every flag; the meta carries
    ``status`` (``ok``/``blocked``/``partial``), the per-query entry counts and
    the number of genes matched by accession versus by primary symbol.
    Determinism: set membership only; no sampling, no thresholds.

    MODIFY HERE: ``tf_evidence=("go", "keyword")`` to widen ``is_tf`` to
    UniProt's transcription keywords (see module docstring for why that is not
    the default); ``ROLE_QUERIES`` to add a role.

    Returns
    -------
    (table, meta)
        ``table`` indexed by requested symbol with the public booleans
        ``is_tf``, ``is_receptor``, ``is_ligand`` plus every raw evidence
        column and ``accession`` / ``role_match`` provenance.
    """
    genes = [str(g) for g in genes]
    ev, notes = _evidence_sets(cache_dir, offline)
    try:
        lut = _symbol_lookup(genes, cache_dir=cache_dir, offline=offline)
        accessions = {g: lut.loc[g, "accession"] for g in genes}
    except (OfflineCacheMiss, RuntimeError) as exc:
        accessions = {g: None for g in genes}
        notes["symbol_lookup"] = f"unavailable: {exc}"

    all_empty = all(not accs and not syms for accs, syms in ev.values())
    rows = []
    for g in genes:
        acc = accessions.get(g)
        acc = acc if isinstance(acc, str) else None
        key = gene_key(g)
        rec: Dict[str, object] = {"gene": g, "accession": acc,
                                  "role_match": "accession" if acc else
                                  ("symbol" if not all_empty else "none")}
        for col, (accs, syms) in ev.items():
            rec[col] = bool(acc in accs) if acc else bool(key in syms)
        rows.append(rec)
    table = pd.DataFrame(rows).set_index("gene")

    for flag, choices in (("is_tf", tf_evidence), ("is_receptor", receptor_evidence),
                          ("is_ligand", ligand_evidence)):
        cols = [_EVIDENCE_COLUMNS[flag][c] for c in choices
                if c in _EVIDENCE_COLUMNS[flag]]
        table[flag] = table[cols].any(axis=1) if cols else False

    ordered = ["is_tf", "is_receptor", "is_ligand", "accession", "role_match"] + \
              [c for c in ROLE_QUERIES if c in table.columns]
    table = table[ordered]

    meta = {
        "role_queries": dict(ROLE_QUERIES),
        "evidence_used": {"is_tf": list(tf_evidence),
                          "is_receptor": list(receptor_evidence),
                          "is_ligand": list(ligand_evidence)},
        "query_results": notes,
        "n_genes": len(genes),
        "n_without_accession": int(sum(1 for g in genes
                                       if not isinstance(accessions.get(g), str))),
        "n_matched_by_accession": int((table["role_match"] == "accession").sum()),
        "n_matched_by_symbol": int((table["role_match"] == "symbol").sum()),
        "n_tf": int(table["is_tf"].sum()),
        "n_receptor": int(table["is_receptor"].sum()),
        "n_ligand": int(table["is_ligand"].sum()),
        "status": "blocked" if all_empty else
                  ("partial" if any(isinstance(v, str) for v in notes.values()) else "ok"),
    }
    if all_empty:
        meta["note"] = ("no UniProt role query could be executed; every flag is False "
                        "because no evidence was retrievable, NOT because the genes "
                        "lack these roles")
    return table, meta


__all__ = ["role_flags", "ROLE_QUERIES", "DEFAULT_EVIDENCE"]
