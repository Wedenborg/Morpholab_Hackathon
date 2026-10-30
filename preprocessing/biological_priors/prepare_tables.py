"""Provider-format adapters for retrieved mouse-prior tables.

Biopython reads FASTA; CSV/JSON readers preserve native table evidence. These
adapters normalize identifiers and columns, not infer biological relationships.
"""

from collections import defaultdict
import csv
import json
import re
from pathlib import Path
from uuid import uuid4

from .core import Identifiers, open_text, table
from .core import digest
from .retrieval import MEMBERSHIP_COLUMNS


def uniprot_rows(path):
    return table(path, ("Entry", "Entry Name", "Reviewed", "Length", "MGI", "Function [CC]", "Protein families"))


def enrich_identifiers(store, mapping_path, uniprot_path=None, peptide_path=None):
    records = {r["ensembl_gene_id"]: dict(r) for r in table(mapping_path)}
    lookup = Identifiers(mapping_path, [])
    additions = defaultdict(lambda: defaultdict(set))
    if uniprot_path:
        for row in uniprot_rows(uniprot_path):
            for mgi in re.findall(r"MGI:\d+", row["MGI"]):
                for gene in lookup.identifiers.get(mgi, ()):
                    additions[gene]["uniprot_id"].update((row["Entry"], row["Entry Name"]))
    if peptide_path:
        from Bio import SeqIO
        with open_text(peptide_path) as handle:
            for protein in SeqIO.parse(handle, "fasta"):
                match = re.search(r"\bgene:(ENSMUSG\d+)(?:\.\d+)?\b", protein.description)
                if not match:
                    raise ValueError(f"Ensembl peptide lacks a gene cross-reference: {protein.id}")
                gene = lookup.resolve(match.group(1))
                if not gene:
                    # Retain an explicit stable gene identifier rather than
                    # guessing its current symbol or substituting an ortholog.
                    gene = match.group(1)
                    records.setdefault(gene, dict(ensembl_gene_id=gene, symbol="", taxon_id="10090",
                                                  biotype="protein_coding", mapping_source="Ensembl release 102 peptide gene cross-reference"))
                base = protein.id.split(".")[0]
                additions[gene]["protein_id"].update((protein.id, base))
                additions[gene]["string_id"].add("10090." + base)
    for gene, columns in additions.items():
        for column, values in columns.items():
            records[gene][column] = "|".join(sorted((set(records[gene].get(column, "").split("|")) | values) - {""}))
    columns = sorted({key for record in records.values() for key in record})
    return store.table("mouse_identifiers.tsv", (records[gene] for gene in sorted(records)), columns)


def member(gene, feature, evidence, reference, **extra):
    return dict(gene_id=gene, feature=feature, evidence=evidence,
                association="positive", reference=reference, **extra)


def prepare_complexes(store, path, ids):
    def rows():
        for row in table(path, ("#Complex ac", "Taxonomy identifier", "Identifiers (and stoichiometry) of molecules in complex")):
            if row["Taxonomy identifier"] != "10090":
                continue
            participants = row.get("Expanded participant list") or row["Identifiers (and stoichiometry) of molecules in complex"]
            for participant in participants.split("|"):
                accession = participant.split("(", 1)[0]
                gene = ids.resolve(accession)
                if not gene and re.fullmatch(r"[A-Z0-9]+-\d+", accession):
                    gene = ids.resolve(accession.split("-", 1)[0])
                if gene:
                    yield member(gene, row["#Complex ac"], "complex_portal_curated", row.get("Cross references", ""),
                                 raw_evidence=row.get("Evidence Code", ""), participant=participant)
    return store.table("complexes.tsv", rows(), MEMBERSHIP_COLUMNS + ("raw_evidence", "participant"))


def prepare_domains(store, path, ids):
    def rows():
        with open_text(path) as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 4 or not re.fullmatch(r"IPR\d+", row[0]):
                    raise ValueError("Unexpected MGI InterPro report schema")
                gene = ids.resolve(row[2])
                if gene and ids.records[gene]["biotype"] == "protein_coding":
                    yield member(gene, row[0], "interpro_sequence_annotation", row[0], entry_name=row[1])
    return store.table("domains.tsv", rows(), MEMBERSHIP_COLUMNS + ("entry_name",))


def prepare_proteins(store, path, ids):
    from Bio import SeqIO
    from Bio.Seq import Seq

    output = store.root / "prepared" / "mouse_proteins.fa"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + "." + uuid4().hex + ".partial")
    counts = dict(retained=0, excluded_invalid_or_unmapped_or_noncoding=0, terminal_stops_removed=0)

    def proteins():
        with open_text(path) as handle:
            for record in SeqIO.parse(handle, "fasta"):
                sequence = str(record.seq).upper()
                clean = sequence.rstrip("*")
                gene = ids.resolve(record.id)
                if not gene or ids.records[gene]["biotype"] != "protein_coding" or not clean or not set(clean) <= set("ACDEFGHIKLMNPQRSTVWYBXZUO"):
                    counts["excluded_invalid_or_unmapped_or_noncoding"] += 1
                    continue
                counts["terminal_stops_removed"] += int(clean != sequence)
                counts["retained"] += 1
                record.seq = Seq(clean)
                yield record

    try:
        with temporary.open("w") as handle:
            SeqIO.write(proteins(), handle, "fasta")
        if not counts["retained"]:
            raise ValueError("No valid mapped protein-coding sequences in Ensembl peptide FASTA")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    store.record("prepared/mouse_proteins.fa", dict(path=str(output), sha256=digest(output), **counts))
    return output


def prepare_descriptions(store, path, ids):
    selected = {}
    for row in uniprot_rows(path):
        if row["Reviewed"].lower() != "reviewed" or not row["Function [CC]"].strip():
            continue
        gene = ids.resolve(row["Entry"])
        if not gene:
            continue
        rank = (-int(row["Length"]), row["Entry"])
        if gene not in selected or rank < selected[gene][0]:
            selected[gene] = (rank, row)
    return store.table("descriptions.tsv", (
        dict(gene_id=gene, description=row["Function [CC]"], reference="UniProtKB:" + row["Entry"],
             protein_accession=row["Entry"], sequence_length=row["Length"])
        for gene, (_, row) in sorted(selected.items())),
        ("gene_id", "description", "reference", "protein_accession", "sequence_length"))


def prepare_phenotypes(store, path, ids):
    def rows():
        with open_text(path) as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 8 or not row[4].startswith("MP:"):
                    raise ValueError("Unexpected MGI GenePheno schema: expected eight columns")
                genes = set(re.findall(r"MGI:\d+", row[6]))
                # Do not assign a multigene genotype effect to every gene.
                if len(genes) != 1:
                    continue
                gene = ids.resolve(next(iter(genes)))
                if gene:
                    yield member(gene, row[4], "mgi_genotype_phenotype", row[5], allele=row[2],
                                 genotype=row[7], context=row[3], allelic_composition=row[0], allele_symbols=row[1])
    return store.table("phenotypes.tsv", rows(), MEMBERSHIP_COLUMNS +
                       ("allele", "genotype", "context", "allelic_composition", "allele_symbols"))


def prepare_oma(store, page_paths, group_path, ids):
    families, orthogroups = defaultdict(set), defaultdict(set)
    mapped = []
    seen = set()
    for path in page_paths:
        entries = json.loads(path.read_text())
        for entry in entries:
            if entry["species"]["taxon_id"] != 10090:
                raise ValueError("Non-mouse entry in OMA mouse genome endpoint")
            if entry["omaid"] in seen:
                raise ValueError("OMA pagination repeated an entry; refusing a truncated/repeated download")
            seen.add(entry["omaid"])
            gene = ids.resolve(entry["canonicalid"])
            if not gene:
                continue
            hog = entry.get("oma_hog_id", "")
            group = str(entry.get("oma_group", 0))
            if hog:
                families[hog.split(".", 1)[0]].add(gene)
            if group != "0":
                orthogroups[group].add(gene)
            mapped.append(dict(gene_id=gene, oma_id=entry["omaid"], canonical_id=entry["canonicalid"],
                               hog_id=hog, oma_group=group, is_main_isoform=entry.get("is_main_isoform")))
    reference = "https://omabrowser.org/oma/current/"
    paths = {}
    for name, groups in (("families", families), ("orthogroups", orthogroups),
                         ("paralogs", {key: genes for key, genes in families.items() if len(genes) >= 2})):
        paths[name] = store.table(name + ".tsv", (
            member(gene, ("OMA:" + key) if name == "orthogroups" else key,
                   "oma_sequence_homology", reference)
            for key, genes in sorted(groups.items()) for gene in sorted(genes)), MEMBERSHIP_COLUMNS)
    # Genome breadth of a strict OMA group is a sequence-homology conservation
    # descriptor, not an alignment identity or a new orthology inference.
    breadth = {}
    with open_text(group_path) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if fields[0] not in orthogroups:
                continue
            genomes = {token[:5] for token in fields[1:] if re.fullmatch(r"[A-Z0-9]{5}\d+", token)}
            if not genomes:
                raise ValueError("OMA group row has no parseable member identifiers")
            for gene in orthogroups[fields[0]]:
                breadth[gene] = max(len(genomes), breadth.get(gene, 0))
    paths["conservation"] = store.table("conservation.tsv", (
        dict(gene_id=gene, oma_group_genome_count=value) for gene, value in sorted(breadth.items())),
        ("gene_id", "oma_group_genome_count"))
    store.table("oma_identifier_links.tsv", mapped,
                ("gene_id", "oma_id", "canonical_id", "hog_id", "oma_group", "is_main_isoform"))
    return paths
