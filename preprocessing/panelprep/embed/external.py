"""Non-transcriptomic and causal gene embeddings (items 3 and, for KEGG, the
brief's fourth "causal" category).

All three embedders in this module are registered with ``mode="external"``:
they ignore every expression matrix and embed the ``genes`` list itself. That
is the whole point of the category -- a protein language model, a protein
interaction network and a curated regulatory network say something about a gene
that was never measured in the panel, so these are the only embedders that can
cover the transcriptome-wide gene universe when a candidate dataset does not
contain the gene.

What each one is, and what it is not
------------------------------------
``esm2_sequence``  (category ``non_transcriptomic``)
    Mean-pooled residue representations from ESM-2. The observation unit is a
    *protein*, one per gene, so this embedding has nothing to do with cells,
    normalisation or the panel. Two genes are close when their amino-acid
    sequences are close in the language model's representation, which largely
    means homology and shared domain architecture.

``string_ppi``  (category ``non_transcriptomic``)
    Spectral embedding of the STRING mouse protein-protein association network.
    Observation unit is a *protein pair*; the edge weight is STRING's combined
    score, which mixes experimental, database, co-expression, text-mining and
    genomic-context evidence. It is an association score, not a measured binding
    affinity.

``kegg_causal``  (category ``causal``)
    Signed spectral embedding of KEGG's curated regulatory relations.
    **These are literature-curated regulatory relations, not causal effects
    estimated from data.** Nothing here is identified, nothing is an effect
    size, and no intervention was performed. Treating a KEGG ``activation``
    edge as an identified causal effect -- plugging these numbers into a
    do-calculus argument, or reporting them as an estimated perturbation
    response -- would be wrong. The honest reading is "the literature says A
    activates B somewhere in mouse", which is a prior over causal structure,
    not an estimate of it.

Shared machinery
----------------
``_symbol_lookup`` resolves mouse gene symbols to reviewed (Swiss-Prot)
UniProt accessions and canonical sequences, and is used by ``esm2_sequence``
and by ``panelprep.annotate.roles``. All three embedders write under one
on-disk cache rooted at ``cache_dir`` (default ``data/external``):

    data/external/uniprot/mouse_reviewed_symbol_lookup.json
    data/external/esm2/<checkpoint>/{config.json,...,model.safetensors}
    data/external/esm2/<checkpoint>/embeddings.npz      (keyed by accession)
    data/external/string/10090.protein.{info,links}.v12.0.txt.gz
    data/external/kegg/{list_mmu.txt,list_pathway_mmu.txt,kgml/mmu*.xml}
    data/external/kegg/relations_mmu.tsv                (parsed relation table)

With ``offline=True`` nothing is fetched: a missing cache entry is a *status*,
not an exception -- the embedder returns an all-NaN ``GeneEmbedding`` whose
``provenance["status"]`` is ``"blocked"`` with a reason, so one unavailable
source cannot take down a run.

Missing data, uniformly
-----------------------
A gene that cannot be placed in the source (no reviewed mouse protein, no
STRING node, no KEGG relation, degree zero) gets an **all-NaN row** rather than
a zero row. A zero row is a point in the embedding space and would be clustered
with other zero rows as if those genes were similar; NaN is not. Every
embedder reports the counts in ``provenance``.

MODIFY HERE (quick index)
-------------------------
* different protein language model  -> ``esm2_sequence(checkpoint=...)``
* reduce ESM-2's native 480-d to d  -> ``esm2_sequence(pca_dim=32)``
* STRING confidence cut             -> ``string_ppi(threshold=400)``  (700 = high)
* node2vec instead of spectral      -> see ``_spectral_embedding`` note
* which KEGG relations are signed   -> ``SIGN_BY_SUBTYPE`` / ``unsigned_weight``
* where the cache lives             -> ``cache_dir`` (``Config.embedder_params``)
"""
from __future__ import annotations

import gzip
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from ..normalize import gene_key
from ..schemas import GeneEmbedding
from . import EMBEDDERS

# --------------------------------------------------------------------------
# constants. MODIFY HERE: endpoints and the organism.
# --------------------------------------------------------------------------
DEFAULT_CACHE_DIR = "data/external"
MOUSE_TAXON = 10090
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
KEGG_REST = "https://rest.kegg.jp"
STRING_DOWNLOAD = "https://stringdb-downloads.org/download"
HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{filename}"
STRING_VERSION = "12.0"
DEFAULT_ESM_CHECKPOINT = "facebook/esm2_t12_35M_UR50D"

#: KEGG relation subtypes that carry a direction of effect. Everything else is
#: kept as an *unsigned structural* edge at ``unsigned_weight``.
#: MODIFY HERE: move a subtype between the dicts to change what counts as
#: activating / repressing (e.g. treat "phosphorylation" as activation).
SIGN_BY_SUBTYPE: Dict[str, int] = {
    "activation": +1,
    "expression": +1,
    "inhibition": -1,
    "repression": -1,
}

#: Relation ``type`` attributes kept by default. PPrel = protein-protein,
#: GErel = gene expression (transcriptional), PCrel = protein-compound,
#: ECrel = enzyme-enzyme (shared metabolite, not regulation).
DEFAULT_RELATION_TYPES: Tuple[str, ...] = ("PPrel", "GErel")

_HTTP_TIMEOUT = 120
_SESSION: Optional[requests.Session] = None


class OfflineCacheMiss(RuntimeError):
    """Raised internally when ``offline=True`` and the cache lacks a file.

    Never escapes an embedder: it is converted into a ``blocked`` provenance
    status so a missing cache degrades the run instead of killing it.
    """


# ==========================================================================
# cache plumbing  (one on-disk cache, shared by all three embedders)
# ==========================================================================

def cache_root(cache_dir: str | os.PathLike = DEFAULT_CACHE_DIR) -> Path:
    """The single external-data cache directory, created if needed.

    Observation unit: not applicable (file-system helper).
    Normalisation: not applicable.
    Missing data: creates the directory; never deletes anything.
    MODIFY HERE: pass ``cache_dir`` through ``Config.embedder_params`` (the
    pipeline already passes ``Config.data_dir / "external"`` to the annotators).
    """
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION


def _fetch_to_file(url: str, dest: Path, *, params: Optional[dict] = None,
                   binary: bool = False, min_bytes: int = 1,
                   retries: int = 3, sleep: float = 0.34) -> Path:
    """GET ``url`` into ``dest`` unless ``dest`` already looks complete.

    ``min_bytes`` guards against a truncated or error-page cache entry being
    trusted forever. ``sleep`` throttles consecutive calls (KEGG asks for <= 3
    requests/second).
    """
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            r = _session().get(url, params=params, timeout=_HTTP_TIMEOUT, stream=binary)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}"
                time.sleep(1.0 + attempt)
                continue
            tmp = dest.with_suffix(dest.suffix + ".part")
            if binary:
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            else:
                tmp.write_text(r.text)
            os.replace(tmp, dest)
            if sleep:
                time.sleep(sleep)
            return dest
        except requests.RequestException as exc:   # network / proxy failure
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"could not fetch {url}: {last}")


def _cached_text(url: str, dest: Path, *, offline: bool, params: Optional[dict] = None,
                 min_bytes: int = 1, sleep: float = 0.34) -> str:
    """Cached text GET. ``offline=True`` -> cache only, else ``OfflineCacheMiss``."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return dest.read_text()
    if offline:
        raise OfflineCacheMiss(f"offline=True and cache miss: {dest}")
    _fetch_to_file(url, dest, params=params, min_bytes=min_bytes, sleep=sleep)
    return dest.read_text()


def _blocked(method: str, category: str, genes: Sequence[str], dim: int,
             reason: str, **extra) -> GeneEmbedding:
    """An all-NaN ``GeneEmbedding`` carrying ``status="blocked"`` and a reason."""
    mat = pd.DataFrame(np.full((len(genes), max(int(dim), 1)), np.nan, dtype=np.float32),
                       index=list(genes),
                       columns=[f"{method}_{i}" for i in range(max(int(dim), 1))])
    return GeneEmbedding(matrix=mat, method=method, category=category,
                         provenance={"status": "blocked", "reason": reason,
                                     "n_genes": len(genes), "n_embedded": 0,
                                     "n_nan_rows": len(genes), **extra})


# ==========================================================================
# shared symbol -> UniProt resolution
# ==========================================================================

_UNIPROT_FIELDS = "accession,gene_primary,gene_names,protein_name,length,sequence"


def _uniprot_lookup_path(cache_dir) -> Path:
    return cache_root(cache_dir) / "uniprot" / "mouse_reviewed_symbol_lookup.json"


def uniprot_query_string(symbols: Sequence[str], taxon: int = MOUSE_TAXON) -> str:
    """The exact UniProt query used for a batch of symbols.

    ``gene_exact:`` is deliberate. The looser ``gene:`` field matches synonyms
    *substring-wise* and a one-letter mouse symbol such as ``T`` (brachyury)
    then also returns unrelated proteins -- verified: ``gene:T`` returns Lck.
    ``gene_exact:T`` returns exactly P20293 (Tbxt, synonym ``T``).

    Observation unit: one UniProt entry (a protein) per returned row.
    Normalisation: not applicable.
    Missing data: a symbol with no reviewed entry simply does not appear.
    MODIFY HERE: drop ``reviewed:true`` to include TrEMBL (far more coverage,
    far lower annotation quality), or change ``taxon`` for another organism.
    """
    ors = " OR ".join(f"gene_exact:{s}" for s in symbols)
    return f"(organism_id:{taxon}) AND (reviewed:true) AND ({ors})"


def _symbol_lookup(
    genes: Sequence[str],
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
    batch_size: int = 40,
    taxon: int = MOUSE_TAXON,
) -> pd.DataFrame:
    """Resolve mouse gene symbols to reviewed UniProt accessions and sequences.

    Shared by ``esm2_sequence`` and ``panelprep.annotate.roles`` so there is a
    single place where a symbol becomes a protein.

    Observation unit: one row per *requested gene symbol* (not per UniProt
    entry). Index is the requested symbol exactly as given.
    Normalisation: symbols are compared with ``panelprep.normalize.gene_key``
    (upper-case), so ``sox2``/``Sox2``/``SOX2`` resolve identically; UniProt's
    own primary symbol is returned in ``uniprot_symbol`` so a rename such as
    ``T -> Tbxt`` is visible rather than hidden.
    Missing data: a symbol with no reviewed mouse entry gets
    ``status="no_reviewed_entry"`` and ``accession=None``; nothing is guessed
    and no alias table is applied beyond UniProt's own synonym list.
    Match provenance: ``match_type`` is ``primary`` (UniProt's primary gene
    name equals the request), ``synonym`` (matched a listed synonym only) or
    ``none``. When several reviewed entries match, the lowest accession is
    taken, deterministically, and ``n_candidates`` records the ambiguity.

    MODIFY HERE: ``batch_size`` (URL length vs. number of requests),
    ``taxon`` for a different organism, or ``uniprot_query_string`` to relax
    ``reviewed:true``.

    Returns
    -------
    DataFrame indexed by requested symbol with columns
    ``gene_key, accession, uniprot_symbol, protein_name, length, sequence,
    match_type, n_candidates, status``.
    """
    genes = [str(g) for g in genes]
    path = _uniprot_lookup_path(cache_dir)
    cache: Dict[str, dict] = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text())
        except json.JSONDecodeError:
            cache = {}

    wanted = {gene_key(g) for g in genes}
    todo = sorted(k for k in wanted if k not in cache)
    if todo and not offline:
        path.parent.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            query = uniprot_query_string(batch, taxon=taxon)
            try:
                r = _session().get(UNIPROT_SEARCH, timeout=_HTTP_TIMEOUT,
                                   params={"query": query, "format": "tsv",
                                           "size": 500, "fields": _UNIPROT_FIELDS})
                r.raise_for_status()
                rows = [ln.split("\t") for ln in r.text.strip().split("\n")[1:] if ln]
            except (requests.RequestException, ValueError):
                rows = []            # leave this batch unresolved; retried next call
                continue
            hits: Dict[str, List[dict]] = {k: [] for k in batch}
            for row in rows:
                if len(row) < 6:
                    continue
                acc, primary, names, pname, length, seq = row[:6]
                keys = [gene_key(n) for n in names.split()] if names else []
                pkey = gene_key(primary) if primary else None
                rec = {"accession": acc, "uniprot_symbol": primary,
                       "protein_name": pname,
                       "length": int(length) if str(length).isdigit() else None,
                       "sequence": seq}
                for k in batch:
                    if k == pkey:
                        hits[k].append({**rec, "match_type": "primary"})
                    elif k in keys:
                        hits[k].append({**rec, "match_type": "synonym"})
            for k in batch:
                cand = hits.get(k, [])
                if not cand:
                    cache[k] = {"accession": None, "uniprot_symbol": None,
                                "protein_name": None, "length": None, "sequence": None,
                                "match_type": "none", "n_candidates": 0,
                                "status": "no_reviewed_entry", "query": query}
                    continue
                prim = [c for c in cand if c["match_type"] == "primary"]
                pick = sorted(prim or cand, key=lambda c: c["accession"])[0]
                cache[k] = {**pick, "n_candidates": len(cand), "status": "ok",
                            "query": query}
            path.write_text(json.dumps(cache))

    recs = []
    for g in genes:
        k = gene_key(g)
        c = cache.get(k)
        if c is None:
            c = {"accession": None, "uniprot_symbol": None, "protein_name": None,
                 "length": None, "sequence": None, "match_type": "none",
                 "n_candidates": 0,
                 "status": "offline_cache_miss" if offline else "unresolved"}
        recs.append({"gene": g, "gene_key": k,
                     "accession": c.get("accession"),
                     "uniprot_symbol": c.get("uniprot_symbol"),
                     "protein_name": c.get("protein_name"),
                     "length": c.get("length"),
                     "sequence": c.get("sequence"),
                     "match_type": c.get("match_type", "none"),
                     "n_candidates": c.get("n_candidates", 0),
                     "status": c.get("status", "unresolved")})
    return pd.DataFrame(recs).set_index("gene")


# ==========================================================================
# (1) ESM-2 sequence embedding
# ==========================================================================

_HF_FILES_REQUIRED = ("config.json", "vocab.txt", "tokenizer_config.json")
_HF_FILES_OPTIONAL = ("special_tokens_map.json",)
_HF_WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def _ensure_checkpoint(checkpoint: str, *, cache_dir: str, offline: bool,
                       revision: str = "main") -> Path:
    """Materialise a Hugging Face checkpoint as a local directory.

    Files are pulled with ``requests`` rather than ``huggingface_hub`` on
    purpose: the hub client's HTTP stack fails through this sandbox's proxy,
    and a plain directory is what makes ``offline=True`` reproducible -- the
    model is a cache entry like any other download.
    MODIFY HERE: ``revision`` to pin a commit instead of ``main``.
    """
    d = cache_root(cache_dir) / "esm2" / checkpoint.replace("/", "__")
    have_weights = any((d / w).exists() and (d / w).stat().st_size > 1_000_000
                       for w in _HF_WEIGHTS)
    if have_weights and all((d / f).exists() for f in _HF_FILES_REQUIRED):
        return d
    if offline:
        raise OfflineCacheMiss(f"offline=True and no cached checkpoint at {d}")
    d.mkdir(parents=True, exist_ok=True)
    for fn in _HF_FILES_REQUIRED:
        _fetch_to_file(HF_RESOLVE.format(repo=checkpoint, revision=revision, filename=fn),
                       d / fn, sleep=0.0)
    for fn in _HF_FILES_OPTIONAL:
        try:
            _fetch_to_file(HF_RESOLVE.format(repo=checkpoint, revision=revision, filename=fn),
                           d / fn, sleep=0.0)
        except RuntimeError:
            pass
    if not any((d / w).exists() for w in _HF_WEIGHTS):
        err = None
        for w in _HF_WEIGHTS:
            try:
                _fetch_to_file(HF_RESOLVE.format(repo=checkpoint, revision=revision, filename=w),
                               d / w, binary=True, min_bytes=1_000_000, sleep=0.0)
                break
            except RuntimeError as exc:
                err = exc
        else:
            raise RuntimeError(f"no weights file for {checkpoint}: {err}")
    return d


def _esm_cache_path(checkpoint: str, cache_dir: str) -> Path:
    return (cache_root(cache_dir) / "esm2" / checkpoint.replace("/", "__")
            / "embeddings.npz")


def _load_esm_cache(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def esm2_embed_sequences(
    sequences: Dict[str, str],
    *,
    checkpoint: str = DEFAULT_ESM_CHECKPOINT,
    max_len: int = 1022,
    batch_size: int = 8,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
) -> Tuple[Dict[str, np.ndarray], dict]:
    """Mean-pooled ESM-2 representations for ``{accession: sequence}``.

    Pooling -- the part that is easy to get wrong
    ---------------------------------------------
    The per-protein vector is the mean of the **last hidden layer over residue
    positions only**: the BOS/``<cls>`` token, the EOS token and every padding
    position are excluded by masking on ``input_ids`` against the tokenizer's
    special-token ids. Taking ``last_hidden_state.mean(dim=1)`` over the padded
    tensor -- the common shortcut -- mixes in two special tokens plus however
    much padding the batch happened to need, so the *same* protein gets a
    *different* vector depending on which other proteins shared its batch.
    ``tests/test_external.py`` asserts both properties: pooled != naive mean,
    and batched == single-sequence to < 1e-4.

    Observation unit: one protein (one canonical Swiss-Prot sequence).
    Normalisation: none; ESM-2 representations are used raw. No L2
    normalisation is applied, so use cosine distance downstream if you want
    scale invariance.
    Missing data: an empty or non-string sequence is skipped and reported in
    the returned provenance under ``skipped``.
    Truncation: sequences longer than ``max_len`` residues are truncated to the
    first ``max_len`` (ESM-2's positional limit is 1024 tokens = 1022 residues
    plus BOS/EOS); truncated accessions are listed in provenance.

    MODIFY HERE: ``checkpoint`` (``facebook/esm2_t33_650M_UR50D`` is the usual
    step up, ~20x slower on CPU), ``batch_size``, ``max_len``.
    """
    import torch                                   # imported lazily: CPU-only, heavy
    from transformers import AutoModel, AutoTokenizer

    path = _esm_cache_path(checkpoint, cache_dir)
    cached = _load_esm_cache(path)
    todo = {a: s for a, s in sequences.items()
            if a not in cached and isinstance(s, str) and len(s) > 0}
    skipped = sorted(a for a, s in sequences.items()
                     if not (isinstance(s, str) and len(s) > 0))
    prov = {"checkpoint": checkpoint, "max_len": int(max_len),
            "n_requested": len(sequences), "n_from_cache": len(sequences) - len(todo),
            "n_computed": 0, "truncated": [], "skipped": skipped,
            "pooling": "mean over residue positions; BOS/EOS/pad excluded"}
    if todo:
        if offline:
            raise OfflineCacheMiss(
                f"offline=True and {len(todo)} accessions are not in {path}")
        local = _ensure_checkpoint(checkpoint, cache_dir=cache_dir, offline=offline)
        tok = AutoTokenizer.from_pretrained(str(local))
        try:
            model = AutoModel.from_pretrained(str(local), add_pooling_layer=False)
        except TypeError:                           # model class without that kwarg
            model = AutoModel.from_pretrained(str(local))
        model.eval()
        special = torch.tensor(sorted(set(tok.all_special_ids)), dtype=torch.long)
        # Sort by length so a batch pads as little as possible. Pooling is
        # padding-invariant by construction, so this only affects speed.
        order = sorted(todo, key=lambda a: len(todo[a]))
        t0 = time.time()
        for i in range(0, len(order), batch_size):
            accs = order[i:i + batch_size]
            seqs = [todo[a][:max_len] for a in accs]
            prov["truncated"] += [a for a in accs if len(todo[a]) > max_len]
            enc = tok(seqs, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_len + 2)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state          # (B, L, d)
            keep = enc["attention_mask"].bool() & ~torch.isin(enc["input_ids"], special)
            w = keep.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * w).sum(1) / w.sum(1).clamp(min=1.0)
            for j, a in enumerate(accs):
                cached[a] = pooled[j].numpy().astype(np.float32)
            prov["n_computed"] += len(accs)
        prov["seconds"] = round(time.time() - t0, 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **cached)
    prov["truncated"] = sorted(set(prov["truncated"]))
    prov["n_truncated"] = len(prov["truncated"])
    return {a: cached[a] for a in sequences if a in cached}, prov


@EMBEDDERS.register(
    "esm2_sequence",
    category="non_transcriptomic", mode="external", needs_expression=False,
    defaults=dict(checkpoint=DEFAULT_ESM_CHECKPOINT, max_len=1022, batch_size=8,
                  cache_dir=DEFAULT_CACHE_DIR, offline=False, pca_dim=None),
    description="ESM-2 protein language-model embedding: mean of last-layer residue "
                "representations (BOS/EOS/padding excluded) of each gene's reviewed "
                "mouse Swiss-Prot protein",
)
def esm2_sequence(datasets, *, genes, dim: int = 32, rng=None,
                  checkpoint: str = DEFAULT_ESM_CHECKPOINT, max_len: int = 1022,
                  batch_size: int = 8, cache_dir: str = DEFAULT_CACHE_DIR,
                  offline: bool = False, pca_dim: Optional[int] = None,
                  **_) -> GeneEmbedding:
    """Gene embedding from protein sequence (``mode="external"``).

    Observation unit: one protein per gene -- **no cells are involved**.
    ``datasets`` is accepted to satisfy the embedder contract and ignored.
    Normalisation: irrelevant; this embedder never touches an expression
    matrix, so it is immune to the units mismatch that constrains the
    expression embedders.
    Missing data: genes without a reviewed mouse Swiss-Prot entry (or whose
    sequence is unavailable) get all-NaN rows, counted in
    ``provenance["n_no_protein"]`` and listed in ``provenance["genes_no_protein"]``.
    Determinism: ``model.eval()`` with no dropout and no sampling, so repeated
    runs are bit-identical; ``rng`` is unused.
    Output dimension: ESM-2's native hidden size (480 for the default 35M
    checkpoint), **not** the runner's ``dim``. ``dim`` is ignored unless
    ``pca_dim`` is set, in which case the embedding is PCA-reduced to
    ``pca_dim`` columns (PCA is fit on the non-NaN rows only).

    MODIFY HERE: ``checkpoint``, ``pca_dim`` (set to ``Config.embedding_dim``
    to make this comparable with the expression embedders), ``max_len``.
    Set through ``Config.embedder_params["esm2_sequence"]``.
    """
    genes = [str(g) for g in genes]
    try:
        lut = _symbol_lookup(genes, cache_dir=cache_dir, offline=offline)
        if offline and bool((lut["status"] == "offline_cache_miss").all()):
            # Cold symbol cache: "no protein found" would be a lie, the truth is
            # "we were not allowed to look". Report it as blocked, not as a
            # biological negative.
            raise OfflineCacheMiss(
                f"offline=True and none of the {len(genes)} symbols are in "
                f"{_uniprot_lookup_path(cache_dir)}")
        seqs = {a: s for a, s in zip(lut["accession"], lut["sequence"])
                if isinstance(a, str) and isinstance(s, str) and s}
        vecs, prov = esm2_embed_sequences(
            seqs, checkpoint=checkpoint, max_len=max_len, batch_size=batch_size,
            cache_dir=cache_dir, offline=offline)
    except OfflineCacheMiss as exc:
        return _blocked("esm2_sequence", "non_transcriptomic", genes,
                        pca_dim or 480, str(exc), checkpoint=checkpoint)
    except RuntimeError as exc:
        return _blocked("esm2_sequence", "non_transcriptomic", genes,
                        pca_dim or 480, f"fetch failed: {exc}", checkpoint=checkpoint)

    d = len(next(iter(vecs.values()))) if vecs else int(pca_dim or 480)
    mat = np.full((len(genes), d), np.nan, dtype=np.float32)
    no_protein: List[str] = []
    for i, g in enumerate(genes):
        acc = lut.loc[g, "accession"]
        if isinstance(acc, str) and acc in vecs:
            mat[i] = vecs[acc]
        else:
            no_protein.append(g)
    cols = [f"esm2_{i}" for i in range(d)]

    if pca_dim is not None and len(genes) - len(no_protein) > 1:
        from sklearn.decomposition import PCA

        ok = ~np.isnan(mat).any(axis=1)
        k = int(min(pca_dim, ok.sum(), d))
        red = np.full((len(genes), k), np.nan, dtype=np.float32)
        red[ok] = PCA(n_components=k, random_state=0).fit_transform(mat[ok])
        mat, cols = red, [f"esm2_pc{i}" for i in range(k)]

    prov.update({
        "source": "UniProtKB reviewed (Swiss-Prot) mouse proteome + ESM-2",
        "uniprot_query_example": uniprot_query_string(
            [g for g in genes[:3]] or ["Sox2"]),
        "n_genes": len(genes),
        "n_embedded": int(len(genes) - len(no_protein)),
        "n_no_protein": len(no_protein),
        "genes_no_protein": no_protein[:200],
        "n_nan_rows": len(no_protein),
        "native_dim": d if pca_dim is None else int(prov.get("native_dim", 480)),
        "pca_dim": pca_dim,
        "status": "ok" if len(no_protein) < len(genes) else "no_coverage",
    })
    return GeneEmbedding(matrix=pd.DataFrame(mat, index=genes, columns=cols),
                         method="esm2_sequence", category="non_transcriptomic",
                         provenance=prov)


# ==========================================================================
# (2) STRING protein-protein association network
# ==========================================================================

def _string_paths(cache_dir: str, version: str = STRING_VERSION,
                  taxon: int = MOUSE_TAXON) -> Tuple[Path, Path]:
    d = cache_root(cache_dir) / "string"
    return (d / f"{taxon}.protein.info.v{version}.txt.gz",
            d / f"{taxon}.protein.links.v{version}.txt.gz")


def string_files(cache_dir: str = DEFAULT_CACHE_DIR, *, offline: bool = False,
                 version: str = STRING_VERSION, taxon: int = MOUSE_TAXON
                 ) -> Tuple[Path, Path]:
    """Ensure the STRING ``protein.info`` and ``protein.links`` files are cached.

    Observation unit: ``info`` = one protein per row, ``links`` = one protein
    *pair* per row with a combined score in 0-1000.
    Normalisation: none. The combined score is STRING's own probabilistic
    integration of evidence channels, already comparable across pairs.
    Missing data: raises ``OfflineCacheMiss`` when ``offline=True`` and the
    files are absent -- callers turn that into a ``blocked`` status.
    MODIFY HERE: ``version`` to pin a STRING release, ``taxon`` for another
    organism. Mouse ``links`` is ~75 MB gzipped and is downloaded once.
    """
    info, links = _string_paths(cache_dir, version, taxon)
    for p, sub in ((info, "protein.info"), (links, "protein.links")):
        if p.exists() and p.stat().st_size > 10_000:
            continue
        if offline:
            raise OfflineCacheMiss(f"offline=True and cache miss: {p}")
        _fetch_to_file(f"{STRING_DOWNLOAD}/{sub}.v{version}/{p.name}", p,
                       binary=True, min_bytes=10_000, sleep=0.0)
    return info, links


def string_graph(
    genes: Sequence[str],
    *,
    threshold: int = 400,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
    version: str = STRING_VERSION,
    taxon: int = MOUSE_TAXON,
    chunksize: int = 2_000_000,
):
    """Weighted ``networkx`` graph over ``genes`` from STRING combined scores.

    Observation unit: a protein pair. Edge attribute ``weight`` is
    ``combined_score / 1000`` in (0, 1]; ``combined_score`` keeps the raw value.
    Normalisation: STRING scores are already on a fixed 0-1000 scale.
    Missing data: a requested symbol absent from ``protein.info`` is not added
    as a node and is returned in ``unmapped``; a mapped symbol with no edge
    above ``threshold`` is a degree-0 node (an *isolate*), returned separately,
    because an isolate has no spectral embedding.
    Thresholds: 150 = low, 400 = medium (default), 700 = high, 900 = highest
    confidence. The threshold is applied to the combined score, which includes
    the text-mining channel; raise it to 700 if you want mostly experimental
    and database evidence.

    MODIFY HERE: ``threshold``; or filter on a specific evidence channel by
    switching to the ``protein.links.detailed`` file (it adds ``escore``,
    ``dscore``, ... columns), which is a one-line change to ``string_files``.
    """
    import networkx as nx

    info_p, links_p = string_files(cache_dir, offline=offline, version=version, taxon=taxon)
    info = pd.read_csv(info_p, sep="\t", compression="gzip",
                       usecols=["#string_protein_id", "preferred_name"])
    info.columns = ["string_id", "preferred_name"]
    info["key"] = info["preferred_name"].map(gene_key)
    # Deterministic collapse of duplicate preferred names: lowest string id.
    sym2id = (info.sort_values("string_id").drop_duplicates("key")
              .set_index("key")["string_id"].to_dict())

    keys = {gene_key(g): str(g) for g in genes}
    id2gene = {sym2id[k]: g for k, g in keys.items() if k in sym2id}
    unmapped = sorted(g for k, g in keys.items() if k not in sym2id)

    g = nx.Graph()
    g.add_nodes_from(id2gene.values())
    ids = set(id2gene)
    if ids:
        reader = pd.read_csv(links_p, sep=" ", compression="gzip", chunksize=chunksize,
                             usecols=["protein1", "protein2", "combined_score"])
        for chunk in reader:
            sel = chunk[chunk["protein1"].isin(ids) & chunk["protein2"].isin(ids)
                        & (chunk["combined_score"] >= threshold)]
            for p1, p2, sc in sel.itertuples(index=False):
                a, b = id2gene[p1], id2gene[p2]
                if a != b:
                    g.add_edge(a, b, weight=float(sc) / 1000.0, combined_score=int(sc))
    isolates = sorted(n for n in g.nodes if g.degree(n) == 0)
    return g, {"unmapped": unmapped, "isolates": isolates,
               "n_nodes": g.number_of_nodes(), "n_edges": g.number_of_edges(),
               "threshold": int(threshold), "string_version": version,
               "source": f"STRING v{version} {taxon}.protein.links / .protein.info"}


def _spectral_embedding(adj: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised-Laplacian spectral embedding of a non-negative weighted graph.

    ``L = I - D^{-1/2} A D^{-1/2}``; the ``dim`` eigenvectors of smallest
    eigenvalue are taken (equivalently the *top* eigenvectors of the normalised
    adjacency, whose eigenvalues are ``1 - lambda``) and each is scaled by
    ``1 - lambda`` so smooth, low-frequency directions dominate and
    high-frequency ones are damped.

    Deterministic by construction: ``scipy.linalg.eigh`` plus a fixed sign
    convention (the largest-magnitude entry of each eigenvector is made
    positive), so repeated runs give identical matrices -- unlike a random-walk
    method with a sampled corpus.

    MODIFY HERE: swap this for node2vec (random-walk + skip-gram) if you want
    an embedding that captures higher-order structural roles. It is not wired
    in here because it would add a dependency (``node2vec``/``gensim``) and
    lose exact reproducibility unless the walk seed is pinned; the drop-in
    point is this function -- it only needs ``(n, n) -> (n, dim)``.
    """
    from scipy.linalg import eigh

    n = adj.shape[0]
    deg = adj.sum(1)
    inv = np.zeros(n)
    nz = deg > 0
    inv[nz] = 1.0 / np.sqrt(deg[nz])
    lap = np.eye(n) - (inv[:, None] * adj * inv[None, :])
    lap = (lap + lap.T) / 2.0
    k = int(min(dim, n))
    vals, vecs = eigh(lap, subset_by_index=[0, k - 1])
    for j in range(vecs.shape[1]):                       # fixed sign convention
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    return vecs * (1.0 - vals)[None, :], vals


@EMBEDDERS.register(
    "string_ppi",
    category="non_transcriptomic", mode="external", needs_expression=False,
    defaults=dict(threshold=400, cache_dir=DEFAULT_CACHE_DIR, offline=False,
                  version=STRING_VERSION),
    description="spectral embedding of the STRING mouse protein-protein association "
                "network (combined score >= threshold); deterministic normalised-Laplacian "
                "eigenvectors scaled by (1 - eigenvalue)",
)
def string_ppi(datasets, *, genes, dim: int = 32, rng=None, threshold: int = 400,
               cache_dir: str = DEFAULT_CACHE_DIR, offline: bool = False,
               version: str = STRING_VERSION, **_) -> GeneEmbedding:
    """Gene embedding from protein-protein association structure.

    Observation unit: a protein pair (an edge), not a cell. ``datasets`` is
    ignored.
    Normalisation: STRING combined scores (0-1000) scaled to (0, 1]; no
    expression normalisation applies.
    Missing data: symbols with no STRING node and nodes with degree 0 at the
    chosen threshold both get all-NaN rows. Both counts are in provenance
    (``n_unmapped``, ``n_isolates``); an isolate is *not* given a zero vector,
    which would place every unconnected gene at the same point.
    Interpretation: closeness means "sits in the same region of the
    association network". Because STRING's combined score includes a
    text-mining channel, well-studied genes are better connected than
    equally-interacting obscure ones -- a literature bias, not a biological
    one.

    MODIFY HERE: ``threshold`` (700 for high confidence), ``dim``, or replace
    ``_spectral_embedding`` with node2vec (see its docstring).
    """
    genes = [str(g) for g in genes]
    try:
        g, meta = string_graph(genes, threshold=threshold, cache_dir=cache_dir,
                               offline=offline, version=version)
    except OfflineCacheMiss as exc:
        return _blocked("string_ppi", "non_transcriptomic", genes, dim, str(exc))
    except RuntimeError as exc:
        return _blocked("string_ppi", "non_transcriptomic", genes, dim,
                        f"fetch failed: {exc}")

    import networkx as nx

    connected = [n for n in genes if n in g and g.degree(n) > 0]
    # Column count: the spectral rank is capped by the number of usable nodes,
    # but when there are none we still emit ``dim`` NaN columns so a failed
    # source has the same shape as a successful one (downstream concatenation
    # does not need a special case).
    n_cols = int(min(dim, len(connected))) if len(connected) >= 2 else int(dim)
    cols = [f"string_{i}" for i in range(max(n_cols, 1))]
    mat = np.full((len(genes), len(cols)), np.nan, dtype=np.float32)
    eigenvalues: List[float] = []
    if len(connected) >= 2:
        adj = nx.to_numpy_array(g, nodelist=connected, weight="weight")
        emb, vals = _spectral_embedding(adj, len(cols))
        pos = {gname: i for i, gname in enumerate(genes)}
        for i, gname in enumerate(connected):
            mat[pos[gname], :emb.shape[1]] = emb[i]
        eigenvalues = [float(v) for v in vals]
    prov = {**meta,
            "n_genes": len(genes),
            "n_embedded": len(connected) if len(connected) >= 2 else 0,
            "n_unmapped": len(meta["unmapped"]),
            "n_isolates": len(meta["isolates"]),
            "unmapped_examples": meta["unmapped"][:50],
            "isolate_examples": meta["isolates"][:50],
            "n_nan_rows": int(np.isnan(mat).all(axis=1).sum()),
            "laplacian_eigenvalues": [round(v, 5) for v in eigenvalues[:10]],
            "scaling": "eigenvector_j * (1 - eigenvalue_j)",
            "status": "ok" if len(connected) >= 2 else "insufficient_nodes"}
    return GeneEmbedding(matrix=pd.DataFrame(mat, index=genes, columns=cols),
                         method="string_ppi", category="non_transcriptomic",
                         provenance=prov)


# ==========================================================================
# (3) KEGG curated signed regulatory network  (category "causal")
# ==========================================================================

def kegg_gene_symbols(cache_dir: str = DEFAULT_CACHE_DIR, *, offline: bool = False
                      ) -> Dict[str, str]:
    """``{kegg_gene_id: primary symbol}`` from ``rest.kegg.jp/list/mmu``.

    Observation unit: one KEGG gene entry per row.
    Normalisation: not applicable. The symbol is the first comma-separated
    token of the description field before ``;``; entries with no symbol keep
    their KEGG id so an edge is never silently dropped.
    Missing data: raises ``OfflineCacheMiss`` under ``offline=True`` with no
    cache.
    MODIFY HERE: nothing organism-specific except the ``mmu`` prefix.
    """
    txt = _cached_text(f"{KEGG_REST}/list/mmu",
                       cache_root(cache_dir) / "kegg" / "list_mmu.txt",
                       offline=offline, min_bytes=1000)
    out: Dict[str, str] = {}
    for line in txt.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        kid, desc = parts[0], parts[-1]
        sym = desc.split(";")[0].split(",")[0].strip() if ";" in desc else kid
        out[kid] = sym
    return out


def kegg_pathway_ids(cache_dir: str = DEFAULT_CACHE_DIR, *, offline: bool = False
                     ) -> Dict[str, str]:
    """``{pathway_id: name}`` for mouse from ``rest.kegg.jp/list/pathway/mmu``."""
    txt = _cached_text(f"{KEGG_REST}/list/pathway/mmu",
                       cache_root(cache_dir) / "kegg" / "list_pathway_mmu.txt",
                       offline=offline, min_bytes=1000)
    out: Dict[str, str] = {}
    for line in txt.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = parts[0].replace("path:", "")
            out[pid] = parts[1].split(" - Mus musculus")[0].strip()
    return out


def kegg_relations(cache_dir: str = DEFAULT_CACHE_DIR, *, offline: bool = False,
                   max_pathways: Optional[int] = None, rebuild: bool = False
                   ) -> pd.DataFrame:
    """Parse every mouse KGML into a table of curated ``relation`` entries.

    Observation unit: one KGML ``<relation>`` + ``<subtype>`` pair, i.e. one
    curated statement in one pathway map. A gene pair appearing in several
    pathways contributes several rows, which is why the embedder aggregates.
    Normalisation: not applicable.
    Missing data: a pathway whose KGML cannot be fetched is skipped and
    counted; group entries (KEGG complexes) are expanded to their components,
    and multi-gene entries (``name="mmu:1 mmu:2"``) expand to every gene, so a
    relation between two complexes becomes the product of their members.
    Caching: the parsed table is written to ``kegg/relations_mmu.tsv`` and the
    ~368 KGML files to ``kegg/kgml/``; set ``rebuild=True`` to re-parse.

    MODIFY HERE: ``max_pathways`` for a fast smoke run (pathways are visited in
    sorted id order, so a prefix is reproducible).

    Returns
    -------
    DataFrame with ``pathway, source, target, rel_type, subtype``; ``source``
    and ``target`` are mouse gene symbols.
    """
    root = cache_root(cache_dir) / "kegg"
    table_p = root / ("relations_mmu.tsv" if max_pathways is None
                      else f"relations_mmu_top{max_pathways}.tsv")
    if table_p.exists() and not rebuild:
        return pd.read_csv(table_p, sep="\t")

    pathways = sorted(kegg_pathway_ids(cache_dir, offline=offline))
    if max_pathways is not None:
        pathways = pathways[:max_pathways]
    id2sym = kegg_gene_symbols(cache_dir, offline=offline)

    rows, failed = [], []
    for pid in pathways:
        dest = root / "kgml" / f"{pid}.xml"
        try:
            xml = _cached_text(f"{KEGG_REST}/get/{pid}/kgml", dest,
                               offline=offline, min_bytes=50)
        except (OfflineCacheMiss, RuntimeError) as exc:
            failed.append((pid, str(exc)[:80]))
            continue
        try:
            tree = ET.fromstring(xml)
        except ET.ParseError:
            failed.append((pid, "unparseable KGML"))
            continue
        entry_genes: Dict[str, List[str]] = {}
        groups: Dict[str, List[str]] = {}
        for e in tree.findall("entry"):
            eid, etype = e.get("id"), e.get("type")
            if etype == "gene":
                entry_genes[eid] = [id2sym.get(n, n) for n in (e.get("name") or "").split()
                                    if n.startswith("mmu:")]
            elif etype == "group":
                groups[eid] = [c.get("id") for c in e.findall("component")]
        for eid, comps in groups.items():            # expand complexes
            entry_genes[eid] = [s for c in comps for s in entry_genes.get(c, [])]
        for rel in tree.findall("relation"):
            src = entry_genes.get(rel.get("entry1"), [])
            tgt = entry_genes.get(rel.get("entry2"), [])
            if not src or not tgt:
                continue
            for st in rel.findall("subtype"):
                for a in src:
                    for b in tgt:
                        if a != b:
                            rows.append({"pathway": pid, "source": a, "target": b,
                                         "rel_type": rel.get("type"),
                                         "subtype": st.get("name")})
    df = pd.DataFrame(rows, columns=["pathway", "source", "target", "rel_type", "subtype"])
    df.attrs["failed_pathways"] = failed
    df.attrs["n_pathways"] = len(pathways)
    table_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_p, sep="\t", index=False)
    (root / "relations_mmu_failures.json").write_text(json.dumps(failed, indent=1))
    return df


def kegg_signed_graph(
    genes: Sequence[str],
    *,
    cache_dir: str = DEFAULT_CACHE_DIR,
    offline: bool = False,
    unsigned_weight: float = 0.25,
    relation_types: Sequence[str] = DEFAULT_RELATION_TYPES,
    max_pathways: Optional[int] = None,
):
    """Signed, directed ``networkx.DiGraph`` over ``genes`` from KEGG relations.

    Signing: ``activation``/``expression`` -> ``+1``,
    ``inhibition``/``repression`` -> ``-1`` (see ``SIGN_BY_SUBTYPE``). Every
    other subtype -- ``binding/association``, ``phosphorylation``,
    ``dissociation``, ``indirect effect``, ... -- is kept as an **unsigned
    structural edge** with weight ``+unsigned_weight``: it records that the two
    products interact without claiming a direction of effect. Set
    ``unsigned_weight=0`` to drop them entirely.

    Observation unit: an ordered gene pair. Multiple curated statements for the
    same pair are aggregated to the **mean signed weight**, so a pair that is
    activating in three pathways and inhibiting in one gets ``+0.5`` rather
    than a count; ``n_relations`` and ``n_pathways`` keep the raw support.
    Normalisation: weights lie in ``[-1, 1]`` by construction.
    Missing data: genes with no curated relation are absent from the graph and
    returned in ``absent``.
    MODIFY HERE: ``relation_types`` (add ``"PCrel"`` for protein-compound,
    ``"ECrel"`` for enzyme-enzyme -- the latter is metabolite sharing, not
    regulation), ``unsigned_weight``, ``SIGN_BY_SUBTYPE``.
    """
    import networkx as nx

    rel = kegg_relations(cache_dir, offline=offline, max_pathways=max_pathways)
    keys = {gene_key(g): str(g) for g in genes}
    if len(rel):
        rel = rel[rel["rel_type"].isin(list(relation_types))].copy()
        rel["skey"] = rel["source"].map(gene_key)
        rel["tkey"] = rel["target"].map(gene_key)
        rel = rel[rel["skey"].isin(keys) & rel["tkey"].isin(keys)]
        rel["signed"] = rel["subtype"].map(SIGN_BY_SUBTYPE).astype("float64")
        rel["signed"] = rel["signed"].fillna(float(unsigned_weight))
        rel = rel[rel["signed"] != 0.0]

    g = nx.DiGraph()
    counts = {"activation": 0, "inhibition": 0, "unsigned": 0}
    if len(rel):
        grouped = rel.groupby(["skey", "tkey"], sort=True)
        for (sk, tk), sub in grouped:
            w = float(sub["signed"].mean())
            g.add_edge(keys[sk], keys[tk], weight=w,
                       sign=int(np.sign(w)),
                       n_relations=int(len(sub)),
                       n_pathways=int(sub["pathway"].nunique()),
                       subtypes=";".join(sorted(set(sub["subtype"]))))
            if w > 0 and set(sub["subtype"]) & {"activation", "expression"}:
                counts["activation"] += 1
            elif w < 0:
                counts["inhibition"] += 1
            else:
                counts["unsigned"] += 1
    absent = sorted(gname for gname in keys.values() if gname not in g)
    pos = sum(1 for _, _, d in g.edges(data=True) if d["sign"] > 0)
    neg = sum(1 for _, _, d in g.edges(data=True) if d["sign"] < 0)
    return g, {"absent": absent, "n_nodes": g.number_of_nodes(),
               "n_edges": g.number_of_edges(),
               "n_positive_edges": pos, "n_negative_edges": neg,
               "edge_subtype_counts": counts,
               "unsigned_weight": float(unsigned_weight),
               "relation_types": list(relation_types),
               "source": "KEGG KGML relation entries for mmu pathways "
                         "(literature-curated, not estimated from data)"}


def _signed_spectral_embedding(adj_dir: np.ndarray, dim: int
                               ) -> Tuple[np.ndarray, List[str], dict]:
    """Signed-graph spectral embedding that keeps edge direction.

    Three blocks are concatenated:

    ``sym``  eigenvectors of the **signed Laplacian** ``L = D_bar - A_sym``
             where ``A_sym = (A + A^T) / 2`` and ``D_bar = diag(sum_j |A_sym|)``.
             ``L`` is positive semi-definite for signed graphs and its small
             eigenvectors separate groups that are internally activating and
             mutually repressing -- the signed analogue of a graph cut.
    ``out``  left singular vectors of the *directed* ``A``, scaled by their
             singular values: a gene's profile **as a regulator**.
    ``in``   right singular vectors, likewise: a gene's profile **as a target**.

    The symmetric Laplacian alone would discard direction entirely (A -> A_sym),
    which for a regulatory network throws away the only thing that makes it
    regulatory; hence the in/out blocks. Column budget: ``dim // 2`` symmetric,
    the rest split evenly between out and in.
    MODIFY HERE: the 50/25/25 split below.
    """
    from scipy.linalg import eigh, svd

    n = adj_dir.shape[0]
    n_sym = max(1, int(dim) // 2)
    n_dir = max(0, int(dim) - n_sym)
    n_out = n_dir // 2
    n_in = n_dir - n_out

    a_sym = (adj_dir + adj_dir.T) / 2.0
    d_bar = np.diag(np.abs(a_sym).sum(1))
    lap = d_bar - a_sym
    lap = (lap + lap.T) / 2.0
    k = int(min(n_sym, n))
    vals, vecs = eigh(lap, subset_by_index=[0, k - 1])
    for j in range(vecs.shape[1]):
        if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
            vecs[:, j] *= -1
    blocks = [vecs]
    cols = [f"kegg_sym{j}" for j in range(vecs.shape[1])]

    if n_dir > 0:
        u, s, vt = svd(adj_dir, full_matrices=False)
        for j in range(u.shape[1]):                   # fixed sign convention
            if u[np.argmax(np.abs(u[:, j])), j] < 0:
                u[:, j] *= -1
                vt[j, :] *= -1
        ko, ki = int(min(n_out, len(s))), int(min(n_in, len(s)))
        if ko:
            blocks.append(u[:, :ko] * s[:ko][None, :])
            cols += [f"kegg_out{j}" for j in range(ko)]
        if ki:
            blocks.append(vt[:ki, :].T * s[:ki][None, :])
            cols += [f"kegg_in{j}" for j in range(ki)]
        sv = [float(x) for x in s[:10]]
    else:
        sv = []
    emb = np.concatenate(blocks, axis=1)
    return emb, cols, {"signed_laplacian_eigenvalues": [round(float(v), 5) for v in vals[:10]],
                       "directed_singular_values": [round(v, 5) for v in sv],
                       "block_dims": {"sym": int(vecs.shape[1]),
                                      "out": int(n_out), "in": int(n_in)}}


@EMBEDDERS.register(
    "kegg_causal",
    category="causal", mode="external", needs_expression=False,
    defaults=dict(cache_dir=DEFAULT_CACHE_DIR, offline=False, unsigned_weight=0.25,
                  relation_types=DEFAULT_RELATION_TYPES, max_pathways=None),
    description="signed spectral embedding of KEGG's curated mouse regulatory relations "
                "(activation/expression +1, inhibition/repression -1); literature-curated "
                "structure, NOT causal effects estimated from data",
)
def kegg_causal(datasets, *, genes, dim: int = 32, rng=None,
                cache_dir: str = DEFAULT_CACHE_DIR, offline: bool = False,
                unsigned_weight: float = 0.25,
                relation_types: Sequence[str] = DEFAULT_RELATION_TYPES,
                max_pathways: Optional[int] = None, **_) -> GeneEmbedding:
    """Gene embedding from curated regulatory structure (category ``causal``).

    **Scientific caveat, stated plainly.** KEGG ``relation`` entries are
    *literature-curated regulatory relations*: a curator read papers and
    recorded that gene A activates or inhibits gene B in some context. They are
    **not** causal effects estimated from data. There is no estimand here, no
    identification argument, no effect size, no context and no uncertainty.
    Using this embedding as if it encoded identified causal effects -- e.g.
    reading a coordinate as a perturbation response, or claiming a
    CRISPR-screen prediction is "validated" because the edges agree -- would be
    wrong. What it legitimately provides is a *prior over causal structure*: a
    tagged, independent view of which genes are believed to regulate which,
    useful exactly because it was not derived from the expression matrices
    being scored. The brief's ``causal`` category is populated by this prior;
    a genuinely estimated causal embedding would need interventional data
    (Perturb-seq and similar), which this pipeline does not have.

    Observation unit: an ordered gene pair (one curated statement, aggregated
    across pathways). ``datasets`` is ignored.
    Normalisation: edge weights in ``[-1, 1]``; no expression normalisation.
    Missing data: genes with no curated relation get all-NaN rows, counted in
    ``provenance["n_absent"]``. The sign and direction counts are reported so
    "the graph had no negative edges" cannot hide inside a plot.
    Determinism: ``eigh``/``svd`` with a fixed sign convention; ``rng`` unused.

    MODIFY HERE: ``unsigned_weight`` (0 drops binding/phosphorylation edges),
    ``relation_types``, ``SIGN_BY_SUBTYPE``, ``max_pathways``.
    """
    genes = [str(g) for g in genes]
    try:
        g, meta = kegg_signed_graph(genes, cache_dir=cache_dir, offline=offline,
                                    unsigned_weight=unsigned_weight,
                                    relation_types=relation_types,
                                    max_pathways=max_pathways)
    except OfflineCacheMiss as exc:
        return _blocked("kegg_causal", "causal", genes, dim, str(exc))
    except RuntimeError as exc:
        return _blocked("kegg_causal", "causal", genes, dim, f"fetch failed: {exc}")

    import networkx as nx

    present = [n for n in genes if n in g]
    if len(present) < 2:
        return _blocked("kegg_causal", "causal", genes, dim,
                        f"only {len(present)} genes have curated KEGG relations",
                        **{k: v for k, v in meta.items() if k != "absent"})
    adj = nx.to_numpy_array(g, nodelist=present, weight="weight")
    emb, cols, spec = _signed_spectral_embedding(adj, dim)
    mat = np.full((len(genes), emb.shape[1]), np.nan, dtype=np.float32)
    pos = {gname: i for i, gname in enumerate(genes)}
    for i, gname in enumerate(present):
        mat[pos[gname]] = emb[i]
    prov = {**{k: v for k, v in meta.items() if k != "absent"}, **spec,
            "n_genes": len(genes), "n_embedded": len(present),
            "n_absent": len(meta["absent"]), "absent_examples": meta["absent"][:50],
            "n_nan_rows": int(np.isnan(mat).all(axis=1).sum()),
            "caveat": "literature-curated regulatory relations, NOT causal effects "
                      "estimated from data; a prior over structure, not an estimate",
            "status": "ok"}
    return GeneEmbedding(matrix=pd.DataFrame(mat, index=genes, columns=cols),
                         method="kegg_causal", category="causal", provenance=prov)


__all__ = [
    "esm2_sequence", "string_ppi", "kegg_causal",
    "esm2_embed_sequences", "string_graph", "string_files", "kegg_signed_graph",
    "kegg_relations", "kegg_gene_symbols", "kegg_pathway_ids",
    "uniprot_query_string", "cache_root", "OfflineCacheMiss",
    "DEFAULT_CACHE_DIR", "DEFAULT_ESM_CHECKPOINT", "SIGN_BY_SUBTYPE",
]
