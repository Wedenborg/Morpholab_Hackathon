"""Matched genomic sequences, motifs, and cached local pretrained embeddings."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .core import digest, make_block, open_text, table


def fasta(path):
    from Bio import SeqIO
    with open_text(path) as handle:
        first = next((line for line in handle if line.strip()), "")
        if first and not first.startswith(">"):
            raise ValueError("FASTA sequence precedes its header")
        handle.seek(0)
        for record in SeqIO.parse(handle, "fasta"):
            yield record.id, str(record.seq).upper()


def cache_embedding(spec, payload, encode):
    """Content-addressed cache includes exact input, model, revision and pooling."""
    cache = Path(spec["cache_dir"]) / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    identity = {key: spec.get(key) for key in ("model", "model_revision", "kind", "pooling", "isoform_rule", "max_tokens")}
    key = hashlib.sha256(json.dumps([identity, payload], sort_keys=True).encode()).hexdigest()
    path = cache / (key + ".npy")
    if path.exists():
        value = np.load(path, allow_pickle=False)
    else:
        value = np.asarray(encode(payload), dtype=np.float64)
        np.save(path, value, allow_pickle=False)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Invalid cached embedding")
    return value


def require_model(spec):
    for key in ("model", "model_revision", "model_audit", "pooling"):
        if not spec.get(key):
            raise ValueError(f"Local inference requires {key}")
    # Mutable 'main'/'latest' revisions cannot reproduce an embedding.
    if spec["model_revision"] in {"main", "latest", "master"}:
        raise ValueError("Use an immutable model commit/revision")


def descriptions(spec, ids, provenance, mapper):
    require_model(spec)
    if spec["evidence_basis"] != "curated_description":
        raise ValueError("Text input must be curated descriptions")
    rows = table(spec["path"], ("gene_id", "description", "reference"))
    values, exact = {}, []
    model = None
    if spec["pooling"] != "model_default":
        raise ValueError("Sentence model pooling must be model_default")

    def encode(text):
        nonlocal model
        if model is None:
            from sentence_transformers import SentenceTransformer
            try:
                model = SentenceTransformer(spec["model"], revision=spec["model_revision"],
                                            cache_folder=spec["cache_dir"], local_files_only=True,
                                            trust_remote_code=False)
            except OSError as error:
                raise FileNotFoundError(f"Local text-model assets unavailable: {spec['model']}@{spec['model_revision']}: {error}") from error
        return model.encode(text, normalize_embeddings=False)

    seen = {}
    for row in rows:
        if not row["description"].strip():
            continue
        if not row["reference"]:
            raise ValueError("Every description needs a source reference")
        for gene in mapper(row["gene_id"]):
            if gene in seen and seen[gene] != row["description"]:
                raise ValueError("Conflicting descriptions; choose one curated source per block")
            seen[gene] = row["description"]
            values[gene] = cache_embedding(spec, row["description"], encode)
            exact.append(dict(row, mouse_gene=gene))
    n = len(next(iter(values.values()))) if values else 0
    return [make_block(spec["name"] + ".text", values, [f"text_dim_{i:04d}" for i in range(n)], ids,
                       dict(provenance, exact_descriptions=exact, construction="local text model; cosine Gram",
                            interpretation="functional-semantic similarity, not expression predictability"))]


def proteins(spec, ids, provenance, mapper):
    require_model(spec)
    if spec["evidence_basis"] != "sequence_only_model":
        raise ValueError("Protein models require sequence-only pretraining")
    if spec.get("isoform_rule") != "longest_sequence_then_identifier" or spec["pooling"] != "mean_residues":
        raise ValueError("ESM uses longest_sequence_then_identifier and mean_residues pooling")
    # Restrict inference to the verified ESM-2 architecture; no arbitrary remote model code.
    if not re.fullmatch(r"facebook/esm2_t\d+_\w+_UR50D", spec["model"]):
        raise ValueError("Local protein inference currently supports official facebook/esm2_*_UR50D models")
    selected = {}
    for protein, sequence in fasta(spec["path"]):
        if not sequence or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYBXZUO"):
            raise ValueError(f"Invalid protein sequence for {protein}")
        for gene in mapper(protein):
            if ids.records[gene]["biotype"] != "protein_coding":
                raise ValueError("Protein FASTA maps to noncoding gene")
            current = selected.get(gene)
            if current is None or (-len(sequence), protein) < (-len(current[1]), current[0]):
                selected[gene] = (protein, sequence)
    model, tokenizer = None, None
    max_residues = int(spec.get("max_tokens", 1022))
    if not 1 <= max_residues <= 1022:
        raise ValueError("ESM-2 chunk length must be 1..1022 residues")

    def encode(sequence):
        nonlocal model, tokenizer
        import torch
        from transformers import AutoTokenizer, EsmModel
        if model is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["model_revision"],
                                                           cache_dir=spec["cache_dir"], local_files_only=True)
                model = EsmModel.from_pretrained(spec["model"], revision=spec["model_revision"],
                                                 cache_dir=spec["cache_dir"], local_files_only=True).eval()
            except OSError as error:
                raise FileNotFoundError(f"Local protein-model assets unavailable: {spec['model']}@{spec['model_revision']}: {error}") from error
        total, length = None, 0
        with torch.inference_mode():
            for start in range(0, len(sequence), max_residues):
                chunk = sequence[start:start + max_residues]
                tokens = tokenizer(chunk, return_tensors="pt")
                # Exclude BOS and EOS; chunk pooling weighted by residue count.
                vector = model(**tokens).last_hidden_state[0, 1:1 + len(chunk)].sum(dim=0).cpu().numpy()
                total = vector if total is None else total + vector
                length += len(chunk)
        return total / length

    values = {gene: cache_embedding(spec, sequence, encode) for gene, (_, sequence) in selected.items()}
    n = len(next(iter(values.values()))) if values else 0
    return [make_block(spec["name"] + ".esm", values, [f"esm_dim_{i:04d}" for i in range(n)], ids,
                       dict(provenance, construction="final-layer mean residues, BOS/EOS excluded; length-weighted nonoverlapping chunks",
                            max_residues=max_residues,
                            selected_isoforms={gene: {"protein_id": protein, "length": len(seq),
                                                       "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest()}
                                               for gene, (protein, seq) in selected.items()}))]


def genome_records(spec, mapper):
    from gffutils.feature import feature_from_line
    from intervaltree import IntervalTree
    for key in ("assembly", "fasta_assembly", "gtf_assembly"):
        if not spec.get(key):
            raise ValueError(f"Genomic source needs {key}")
    if len({spec["assembly"], spec["fasta_assembly"], spec["gtf_assembly"]}) != 1:
        raise ValueError("Genome and annotation assemblies must match")
    if spec.get("transcript_rule", "longest_exonic_then_id") != "longest_exonic_then_id":
        raise ValueError("Only expression-independent longest_exonic_then_id is supported")
    transcripts = defaultdict(lambda: defaultdict(list))
    loci = {}
    with open_text(spec["gtf_path"]) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) != 9:
                raise ValueError("GTF requires nine columns")
            if row[2] != "exon":
                continue
            feature = feature_from_line(line)
            attrs = feature.attributes
            if any(len(attrs.get(key, [])) != 1 for key in ("gene_id", "transcript_id")) or feature.strand not in {"+", "-"}:
                raise ValueError("Exons need gene_id, transcript_id and known strand")
            start, end = feature.start - 1, feature.end
            if start < 0 or end <= start:
                raise ValueError("Invalid GTF interval")
            for gene in mapper(attrs["gene_id"][0]):
                if gene in loci and (loci[gene][0], loci[gene][3]) != (row[0], row[6]):
                    raise ValueError("Gene has inconsistent chromosome/strand")
                old = loci.get(gene, (row[0], start, end, row[6]))
                loci[gene] = (row[0], min(start, old[1]), max(end, old[2]), row[6])
                transcripts[gene][attrs["transcript_id"][0]].append((start, end))

    def union(intervals):
        tree = IntervalTree.from_tuples(intervals)
        tree.merge_overlaps(strict=False)
        return [(item.begin, item.end) for item in sorted(tree)]

    selected = {}
    for gene, choices in transcripts.items():
        choices = {tid: union(exons) for tid, exons in choices.items()}
        transcript = min(choices, key=lambda tid: (-sum(b - a for a, b in choices[tid]), tid))
        selected[gene] = {"locus": loci[gene], "transcript": transcript, "exons": choices[transcript],
                          "transcript_count": len(choices)}
    return selected


def reverse_complement(sequence):
    from Bio.Seq import reverse_complement as complement
    return complement(sequence)


def genomic_sequences(spec, mapper):
    records = genome_records(spec, mapper)
    # FASTA index lives in the writable cache, never alongside a read-only resource.
    from pyfaidx import Fasta
    if str(spec["fasta_path"]).endswith(".gz"):
        raise ValueError("Genomic FASTA must be uncompressed for indexed access")
    index = Path(spec["cache_dir"]) / (digest(spec["fasta_path"]) + ".fai")
    index.parent.mkdir(parents=True, exist_ok=True)
    upstream, downstream = spec.get("promoter_window", [2000, 200])
    if not isinstance(upstream, int) or not isinstance(downstream, int) or upstream < 0 or downstream < 1:
        raise ValueError("promoter_window must be [nonnegative upstream, positive downstream] bases")
    genome = Fasta(spec["fasta_path"], indexname=str(index), as_raw=True, sequence_always_upper=True)
    try:
        for gene, record in records.items():
            chrom, start, end, strand = record["locus"]
            if chrom not in genome or end > len(genome[chrom]):
                raise ValueError(f"GTF/FASTA contig mismatch for {gene}")
            exons = record["exons"]
            tss = exons[0][0] if strand == "+" else exons[-1][1] - 1
            left, right = (tss - upstream, tss + downstream) if strand == "+" else (tss - downstream + 1, tss + upstream + 1)
            left, right = max(0, left), min(len(genome[chrom]), right)
            promoter = str(genome[chrom][left:right])
            record.update(tss=tss, promoter_interval=[left, right],
                          promoter=promoter if strand == "+" else reverse_complement(promoter),
                          sequence=str(genome[chrom][start:end]))
    finally:
        genome.close()
    return records


def composition(sequence):
    from Bio.SeqUtils import gc_fraction
    n = sum(sequence.count(base) for base in "ACGT")
    if not n:
        return None
    cpg = sequence.count("CG")
    # Exclude all ambiguous bases to preserve the original ACGT-only denominator.
    return [gc_fraction("".join(base for base in sequence if base in "ACGT")),
            cpg / max(1, len(sequence) - 1), (len(sequence) - n) / len(sequence)]


def genome(spec, ids, provenance, mapper):
    if spec["evidence_basis"] != "sequence":
        raise ValueError("Genomic features must use sequence/structural annotations")
    records = genomic_sequences(spec, mapper)
    structure, gene_comp, promoter_comp, position, chromosomes = {}, {}, {}, {}, {}
    chroms = sorted({record["locus"][0] for record in records.values()})
    for gene, record in records.items():
        chrom, start, end, strand = record["locus"]
        structure[gene] = np.log1p([end - start, record["transcript_count"], len(record["exons"]),
                                   sum(b - a for a, b in record["exons"])])
        comp = composition(record["sequence"])
        if comp is not None:
            gene_comp[gene] = comp
        comp = composition(record["promoter"])
        if comp is not None:
            promoter_comp[gene] = comp
        position[gene] = [np.log1p(start), np.log1p(end), 1 if strand == "+" else -1]
        chromosomes[gene] = [float(value == chrom) for value in chroms]
    details = {gene: {key: value for key, value in record.items() if key not in {"sequence", "promoter"}} for gene, record in records.items()}
    p = dict(provenance, selected_transcripts=details,
             coordinates="0-based half-open; strand-relative window; contig edges clipped",
             tss_selection="longest union-exonic transcript, lexical ID tie-break; strand-specific exon boundary",
             composition="GC among ACGT; CpG dinucleotides/(length-1); ambiguous base fraction")
    return [make_block(spec["name"] + "." + name, values, labels, ids, dict(p, construction=transform))
            for name, values, labels, transform in (
                ("structure", structure, ["log1p_gene_span", "log1p_transcript_count", "log1p_exon_count", "log1p_exonic_bases"], "log1p counts; no expression-based transcript selection"),
                ("gene_composition", gene_comp, ["gc_fraction", "cpg_frequency", "ambiguous_fraction"], "sequence composition; no scaling"),
                ("promoter_composition", promoter_comp, ["gc_fraction", "cpg_frequency", "ambiguous_fraction"], "strand-oriented promoter sequence composition; no scaling"),
                ("location", position, ["log1p_start", "log1p_end", "strand"], "location only; log1p base positions; strand +/-1"),
                ("chromosome", chromosomes, ["chromosome:" + chrom for chrom in chroms], "location only; chromosome one-hot"))]


def motifs(spec, ids, provenance, mapper):
    from Bio import motifs as bio_motifs
    if spec["evidence_basis"] != "sequence":
        raise ValueError("Motif features require sequence evidence")
    for key in ("motif_resource", "motif_version", "motif_url", "score_threshold"):
        if key not in spec:
            raise ValueError(f"Motif input needs {key}")
    records = genomic_sequences(spec, mapper)
    pwms = {}
    motif_format = spec.get("motif_format", "json")
    with open_text(spec["path"]) as handle:
        if motif_format == "json":
            parsed = []
            for label, matrix in json.load(handle).items():
                counts = np.array(matrix, dtype=float)
                if counts.ndim != 2 or counts.shape[1] != 4 or not len(counts) or not np.isfinite(counts).all() or (counts < 0).any():
                    raise ValueError("Motif PFM must be nonnegative finite length x 4 in A,C,G,T order")
                motif = bio_motifs.Motif(counts={base: counts[:, i] for i, base in enumerate("ACGT")})
                parsed.append((label, motif))
        elif motif_format == "jaspar":
            parsed = [(motif.matrix_id, motif) for motif in bio_motifs.parse(handle, "jaspar")]
        else:
            raise ValueError("motif_format must be json or jaspar")
    for label, motif in parsed:
        if label in pwms or not len(motif):
            raise ValueError("Motif IDs must be unique and matrices nonempty")
        counts = np.asarray([motif.counts[base] for base in "ACGT"], dtype=float)
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Motif counts must be nonnegative and finite")
        pwms[label] = motif.counts.normalize(pseudocounts=.5).log_odds(background={base: .25 for base in "ACGT"})
    labels = sorted(pwms)
    if not labels:
        raise ValueError("No motif matrices supplied")
    threshold = float(spec["score_threshold"])
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("Motif threshold must be finite and nonnegative")
    values, observed = {}, set()
    for gene, record in records.items():
        promoter = record["promoter"]
        profile = []
        # Require at least one unambiguous window for every motif; otherwise whole block row is missing.
        complete = True
        for label in labels:
            pwm = pwms[label]
            if len(promoter) < pwm.length:
                complete = False
                break
            scores = np.concatenate([np.atleast_1d(matrix.calculate(promoter))
                                     for matrix in (pwm, pwm.reverse_complement())])
            finite = scores[np.isfinite(scores)]
            if not len(finite):
                complete = False
                break
            best = float(finite.max())
            profile.append(best if best >= threshold else 0.)
        if complete:
            observed.add(gene)
            values[gene] = profile
    return [make_block(spec["name"] + ".motifs", values, labels, ids,
                       dict(provenance, construction="max log2-odds across both strands, retain score >= threshold else zero",
                            implementation="Biopython Bio.motifs PSSM.calculate (float32 scores)",
                            background=[.25] * 4, pseudocount=.5,
                            limitation="motif matches indicate sequence compatibility, not TF occupancy or causal regulation",
                            selected_transcripts={gene: {key: value for key, value in record.items() if key not in {"sequence", "promoter"}}
                                                  for gene, record in records.items()}), observed=observed)]


LOADERS = {"protein_esm": proteins, "descriptions": descriptions, "genome": genome, "motifs": motifs}
