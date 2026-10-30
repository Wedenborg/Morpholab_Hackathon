"""KEGG pathway membership for mouse genes (item 4, pathway half).

Source
------
``rest.kegg.jp/link/mmu/pathway``  -- every (pathway, gene) pair for mouse
``rest.kegg.jp/list/pathway/mmu``  -- pathway id -> human-readable name
``rest.kegg.jp/list/mmu``          -- KEGG gene id -> gene symbol

All three are cached under ``<cache_dir>/kegg/`` by
``panelprep.embed.external``'s shared cache helpers, so the pathway annotator
and the ``kegg_causal`` embedder read the same files and cannot disagree about
which release they are using.

Why ``top_pathway`` is the *smallest* pathway
---------------------------------------------
Colouring an embedding plot by "a pathway the gene is in" is useless if the
chosen pathway is ``mmu01100 Metabolic pathways``: that single map contains
1671 mouse genes (measured, not assumed -- it is the largest in the release),
so most points land in one colour and the plot shows nothing. ``top_pathway``
is therefore the pathway with the **fewest member genes** among those the gene
belongs to, i.e. the most specific annotation available, with ties broken by
ascending pathway id so the result is deterministic across runs and machines.
``pathways`` keeps the full semicolon-joined list, and the gene x pathway
membership matrix is returned in ``GeneAnnotation.sources["kegg_membership"]``
for anyone who wants to do this differently.

MODIFY HERE
-----------
* ``top_pathway_rule`` -- ``"smallest"`` (default) or ``"largest"``; or pass
  ``exclude_pathways`` to blacklist the global maps (``mmu01100``, ``mmu01110``
  ...) from both the list and the ``top_pathway`` choice.
* ``min_pathway_size`` -- ignore pathways with fewer than this many mouse genes
  when picking ``top_pathway`` (a 2-gene map is specific but uninformative).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..embed.external import (DEFAULT_CACHE_DIR, KEGG_REST, OfflineCacheMiss,
                              _cached_text, cache_root, kegg_gene_symbols,
                              kegg_pathway_ids)
from ..normalize import gene_key

#: KEGG "global/overview" maps. Not excluded by default (they are real
#: annotations), but they are the ones that make a colour-by-pathway plot
#: useless, so they are listed here for ``exclude_pathways=GLOBAL_MAPS``.
GLOBAL_MAPS: Tuple[str, ...] = ("mmu01100", "mmu01110", "mmu01120", "mmu01200",
                                "mmu01210", "mmu01212", "mmu01230", "mmu01232",
                                "mmu01240", "mmu01250")

KEGG_LINK_QUERY = f"{KEGG_REST}/link/mmu/pathway"
KEGG_PATHWAY_LIST_QUERY = f"{KEGG_REST}/list/pathway/mmu"
KEGG_GENE_LIST_QUERY = f"{KEGG_REST}/list/mmu"


def kegg_gene_pathway_links(cache_dir: str = DEFAULT_CACHE_DIR, *,
                            offline: bool = False) -> pd.DataFrame:
    """Every (pathway, KEGG gene id) pair for mouse, as a long table.

    Observation unit: one (pathway, gene) membership statement.
    Normalisation: not applicable.
    Missing data: raises ``OfflineCacheMiss`` when ``offline=True`` and the
    cache file is absent; the caller turns that into an explicit source note.
    MODIFY HERE: nothing -- change the organism prefix in
    ``KEGG_LINK_QUERY`` if you move off mouse.
    """
    txt = _cached_text(KEGG_LINK_QUERY,
                       cache_root(cache_dir) / "kegg" / "link_mmu_pathway.txt",
                       offline=offline, min_bytes=10_000)
    rows = [ln.split("\t") for ln in txt.strip().split("\n") if "\t" in ln]
    df = pd.DataFrame(rows, columns=["pathway", "kegg_gene"])
    df["pathway"] = df["pathway"].str.replace("path:", "", regex=False)
    return df


def pathway_membership(
    genes: Sequence[str],
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
    exclude_pathways: Sequence[str] = (),
    min_pathway_size: int = 0,
    top_pathway_rule: str = "smallest",
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """KEGG pathway annotation for ``genes``.

    Observation unit: one row per *requested gene symbol*; a gene belongs to
    zero or more pathways.
    Normalisation: symbols are matched through
    ``panelprep.normalize.gene_key`` (case-insensitive) against KEGG's own
    primary symbols. KEGG symbols for mouse are Title case (``Sox2``); no alias
    expansion beyond KEGG's primary name is attempted, so a gene annotated only
    under an older symbol is reported as having no pathway rather than guessed
    at.
    Missing data: a gene with no KEGG entry, or an entry in no pathway, gets
    ``pathways=""``, ``top_pathway="none"`` and ``n_pathways=0``. These are
    counted in the returned meta, not hidden.
    Determinism: pathway lists are sorted by name; ``top_pathway`` ties
    (equal pathway size) are broken by ascending pathway id.

    MODIFY HERE: ``exclude_pathways=GLOBAL_MAPS`` to drop KEGG's global maps,
    ``min_pathway_size`` to ignore tiny maps when choosing ``top_pathway``,
    ``top_pathway_rule="largest"`` for the opposite convention.

    Returns
    -------
    (table, membership, meta)
        ``table``      indexed by requested symbol, columns ``pathways``,
                       ``top_pathway``, ``top_pathway_id``, ``n_pathways``.
        ``membership`` gene x pathway boolean frame (columns = pathway names
                       present among these genes).
        ``meta``       the exact queries used plus coverage counts.
    """
    genes = [str(g) for g in genes]
    meta: Dict[str, object] = {
        "queries": {"gene_pathway_links": KEGG_LINK_QUERY,
                    "pathway_names": KEGG_PATHWAY_LIST_QUERY,
                    "gene_symbols": KEGG_GENE_LIST_QUERY},
        "top_pathway_rule": top_pathway_rule,
        "min_pathway_size": int(min_pathway_size),
        "excluded_pathways": list(exclude_pathways),
    }
    empty_table = pd.DataFrame(
        {"pathways": "", "top_pathway": "none", "top_pathway_id": "",
         "n_pathways": 0}, index=pd.Index(genes, name="gene"))

    try:
        links = kegg_gene_pathway_links(cache_dir, offline=offline)
        names = kegg_pathway_ids(cache_dir, offline=offline)
        id2sym = kegg_gene_symbols(cache_dir, offline=offline)
    except (OfflineCacheMiss, RuntimeError) as exc:
        meta.update({"status": "blocked", "note": f"KEGG unavailable: {exc}",
                     "n_genes": len(genes), "n_with_pathway": 0})
        return empty_table, pd.DataFrame(index=pd.Index(genes, name="gene")), meta

    excl = set(exclude_pathways)
    links = links[~links["pathway"].isin(excl)]
    #: pathway size = number of distinct mouse genes KEGG puts in that map.
    sizes = links.groupby("pathway")["kegg_gene"].nunique().to_dict()
    links = links.assign(symbol=links["kegg_gene"].map(id2sym))
    links = links[links["symbol"].notna()]
    links["key"] = links["symbol"].map(gene_key)

    by_key: Dict[str, List[str]] = {}
    for key, pid in zip(links["key"], links["pathway"]):
        by_key.setdefault(key, []).append(pid)

    rows, mem_rows = [], {}
    for g in genes:
        pids = sorted(set(by_key.get(gene_key(g), ())))
        pnames = sorted({names.get(p, p) for p in pids})
        if pids:
            eligible = [p for p in pids if sizes.get(p, 0) >= min_pathway_size] or pids
            reverse = top_pathway_rule == "largest"
            # deterministic: (size, pathway id); id ascending breaks size ties
            top = sorted(eligible, key=lambda p: ((-sizes.get(p, 0)) if reverse
                                                  else sizes.get(p, 0), p))[0]
            top_name = names.get(top, top)
        else:
            top, top_name = "", "none"
        rows.append({"gene": g, "pathways": ";".join(pnames),
                     "top_pathway": top_name, "top_pathway_id": top,
                     "n_pathways": len(pids)})
        mem_rows[g] = {names.get(p, p): True for p in pids}

    table = pd.DataFrame(rows).set_index("gene")
    membership = (pd.DataFrame.from_dict(mem_rows, orient="index")
                  .reindex(genes).fillna(False).astype(bool))
    membership.index.name = "gene"
    membership = membership.reindex(sorted(membership.columns), axis=1)
    meta.update({
        "status": "ok",
        "n_genes": len(genes),
        "n_with_pathway": int((table["n_pathways"] > 0).sum()),
        "n_pathways_seen": int(membership.shape[1]),
        "largest_pathway": max(sizes.items(), key=lambda kv: kv[1]) if sizes else None,
        "kegg_release_pathways": len(names),
        "top_pathway_size_rule": "fewest member genes; ties by ascending pathway id",
    })
    return table, membership, meta


__all__ = ["pathway_membership", "kegg_gene_pathway_links", "GLOBAL_MAPS"]
