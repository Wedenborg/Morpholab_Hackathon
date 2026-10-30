"""Item 4: per-gene pathway membership and functional role flags.

One public entry point::

    from panelprep.annotate import annotate_genes
    ann = annotate_genes(genes, sources=("kegg", "roles"),
                         cache_dir="data/external", offline=False)
    ann.table          # indexed by gene symbol
    ann.sources        # every query string, coverage count and caveat

``panelprep.pipeline.annotate`` calls exactly this with
``Config.annotation_sources``, ``Config.data_dir / "external"`` and
``Config.offline``, so the notebooks never build the table themselves.

The returned ``GeneAnnotation.table`` always carries the schema's required
columns -- ``is_ligand``, ``is_receptor``, ``is_tf``, ``pathways``,
``top_pathway`` -- whatever subset of sources ran, plus ``n_pathways``,
``top_pathway_id`` and the raw per-evidence role columns. A source that was not
requested, or that could not be reached, produces its columns' *neutral* values
(``False`` / ``""`` / ``"none"``) and an explicit note in ``sources``; it never
raises, and it is never silently indistinguishable from a real negative.

``sources`` keys
----------------
``kegg``             queries, coverage counts, the largest pathway in the release
``kegg_membership``  the gene x pathway boolean frame (for custom colouring)
``roles``            the exact UniProt query strings and per-query entry counts
``coverage``         one dict of the counts a reviewer will ask for first
``caveats``          what these annotations do and do not license

MODIFY HERE
-----------
* ``sources=("kegg",)`` to skip the UniProt role queries (faster, no ESM-style
  symbol resolution needed), or ``("roles",)`` to skip KEGG.
* ``annotate_genes(..., pathway_kwargs={"exclude_pathways": GLOBAL_MAPS})`` and
  ``role_kwargs={"tf_evidence": ("go", "keyword")}`` to retune either half
  without touching this file.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import pandas as pd

from ..schemas import GeneAnnotation
from .pathways import GLOBAL_MAPS, pathway_membership
from .roles import ROLE_QUERIES, role_flags

DEFAULT_SOURCES = ("kegg", "roles")

CAVEATS = {
    "kegg": "KEGG pathway membership is curated and incomplete; a gene with no "
            "pathway is unannotated, not uninvolved. Membership is organism-level, "
            "not tissue- or stage-specific, so an E7.5 embryo panel inherits "
            "annotations curated in adult tissues.",
    "roles": "Role flags come from reviewed (Swiss-Prot) UniProt entries only. "
             "Coverage is therefore biased toward well-studied proteins, and a "
             "False flag means 'no reviewed evidence found', not 'not a TF'.",
}


def annotate_genes(
    genes: Sequence[str],
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    cache_dir: str = "data/external",
    offline: bool = False,
    pathway_kwargs: Optional[dict] = None,
    role_kwargs: Optional[dict] = None,
) -> GeneAnnotation:
    """Build the item-4 annotation table for ``genes``.

    Observation unit: one row per requested gene symbol, in the order given.
    Duplicated symbols are collapsed to their first occurrence (the schema
    forbids a duplicated index) and the drop is recorded in
    ``sources["coverage"]["n_duplicate_symbols_dropped"]``.
    Normalisation: not applicable -- no expression data is touched. Symbols are
    matched case-insensitively via ``panelprep.normalize.gene_key``; no alias
    table is applied beyond what KEGG and UniProt themselves list.
    Missing data: never raises for a data reason. A gene absent from a source
    gets that source's neutral value and is counted; an unreachable source sets
    ``sources[<name>]["status"] = "blocked"`` with the reason.
    Offline: ``offline=True`` reads the cache under ``cache_dir`` only.

    MODIFY HERE: ``sources``, plus ``pathway_kwargs`` / ``role_kwargs`` to
    forward options to ``pathways.pathway_membership`` /
    ``roles.role_flags`` (see those modules for the knobs).
    """
    genes = [str(g) for g in genes]
    seen, uniq = set(), []
    for g in genes:
        if g not in seen:
            seen.add(g)
            uniq.append(g)
    n_dropped = len(genes) - len(uniq)
    genes = uniq

    index = pd.Index(genes, name="gene")
    table = pd.DataFrame(index=index)
    srcs: Dict[str, object] = {"requested_sources": list(sources),
                               "offline": bool(offline), "cache_dir": str(cache_dir)}

    if "kegg" in sources:
        ptab, membership, pmeta = pathway_membership(
            genes, cache_dir=cache_dir, offline=offline, **(pathway_kwargs or {}))
        table = table.join(ptab)
        srcs["kegg"] = pmeta
        srcs["kegg_membership"] = membership
    else:
        table["pathways"] = ""
        table["top_pathway"] = "none"
        table["top_pathway_id"] = ""
        table["n_pathways"] = 0
        srcs["kegg"] = {"status": "not_requested"}

    if "roles" in sources:
        rtab, rmeta = role_flags(genes, cache_dir=cache_dir, offline=offline,
                                 **(role_kwargs or {}))
        table = table.join(rtab)
        srcs["roles"] = rmeta
    else:
        for col in ("is_tf", "is_receptor", "is_ligand"):
            table[col] = False
        srcs["roles"] = {"status": "not_requested"}

    for col in GeneAnnotation.REQUIRED:                 # schema guarantee
        if col not in table.columns:
            table[col] = False if col.startswith("is_") else (
                "none" if col == "top_pathway" else "")

    front = ["is_tf", "is_receptor", "is_ligand", "top_pathway", "n_pathways", "pathways"]
    table = table[[c for c in front if c in table.columns] +
                  [c for c in table.columns if c not in front]]

    srcs["coverage"] = {
        "n_genes": len(genes),
        "n_duplicate_symbols_dropped": int(n_dropped),
        "n_with_pathway": int((table.get("n_pathways", pd.Series(0, index=index)) > 0).sum()),
        "n_tf": int(table["is_tf"].sum()),
        "n_receptor": int(table["is_receptor"].sum()),
        "n_ligand": int(table["is_ligand"].sum()),
        "n_any_role": int((table["is_tf"] | table["is_receptor"] | table["is_ligand"]).sum()),
        "n_unannotated": int((~(table["is_tf"] | table["is_receptor"] | table["is_ligand"])
                             & (table.get("n_pathways",
                                          pd.Series(0, index=index)) == 0)).sum()),
    }
    srcs["caveats"] = {k: v for k, v in CAVEATS.items() if k in sources}
    return GeneAnnotation(table=table, sources=srcs)


__all__ = ["annotate_genes", "pathway_membership", "role_flags", "ROLE_QUERIES",
           "GLOBAL_MAPS", "DEFAULT_SOURCES", "CAVEATS"]

