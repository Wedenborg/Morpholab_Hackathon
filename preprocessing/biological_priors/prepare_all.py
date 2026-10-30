"""Retrieve and prepare public mouse resources for all biological-prior families."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import re
import shutil
from uuid import uuid4
import zipfile

import requests

from .core import FAMILIES, Identifiers, digest, open_text
from .prepare_tables import (enrich_identifiers, prepare_complexes, prepare_descriptions,
                             prepare_domains, prepare_oma, prepare_phenotypes, prepare_proteins)
from .retrieval import ResourceStore, atomic_json, source


DEFAULT_OUTPUT_CONFIG = Path(__file__).resolve().parents[1] / "results/priors.mouse.json"

FILES = {
    "uniprot/mouse.tsv": "https://rest.uniprot.org/uniprotkb/stream?query=organism_id:10090&format=tsv&fields=accession,id,reviewed,length,xref_mgi,cc_function,protein_families",
    "ensembl/Mus_musculus.GRCm38.pep.all.fa.gz": "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/pep/Mus_musculus.GRCm38.pep.all.fa.gz",
    "ensembl/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz": "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
    "ensembl/Mus_musculus.GRCm38.102.gtf.gz": "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz",
    "reactome/Ensembl2Reactome_All_Levels.txt": "https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt",
    "complexportal/10090.tsv": "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/10090.tsv",
    "string/10090.protein.links.full.v12.0.txt.gz": "https://stringdb-downloads.org/download/protein.links.full.v12.0/10090.protein.links.full.v12.0.txt.gz",
    "biogrid/BIOGRID-ORGANISM-5.0.262.tab3.zip": "https://biogrid-downloads.nyc3.digitaloceanspaces.com/BioGRID/Release-Archive/BIOGRID-5.0.262/BIOGRID-ORGANISM-5.0.262.tab3.zip",
    "trrust/trrust_rawdata.mouse.tsv": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
    "jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt": "https://jaspar2024.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "mgi/MGI_InterProDomains.rpt": "https://www.informatics.jax.org/downloads/reports/MGI_InterProDomains.rpt",
    "mgi/MGI_GenePheno.rpt": "https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt",
    "oma/oma-groups.txt.gz": "https://omabrowser.org/All/oma-groups.txt.gz",
}

GENETIC_ASSAYS = ("Synthetic Lethality", "Synthetic Growth Defect", "Synthetic Rescue",
                  "Phenotypic Enhancement", "Phenotypic Suppression")
PHYSICAL_ASSAYS = ("Two-hybrid", "Affinity Capture-MS", "Reconstituted Complex", "Co-crystal Structure")


class MousePreparation:
    def __init__(self, store, ids, go_config, download_models):
        self.store, self.ids, self.go_config = store, ids, go_config
        self.download_models = download_models

    def get(self, key):
        return self.store.fetch(key, FILES[key])

    def spec(self, name, family, kind, path, basis, key, filters, **options):
        record = self.store.records[key]
        return source(name, family, kind, path, basis, FILES[key],
                      "Downloaded snapshot SHA256 " + record["sha256"],
                      [FILES[key], "Public native-mouse source; raw file retained with hash"], filters,
                      retrieval=dict(raw_path=record["path"], raw_sha256=record["sha256"]), **options)

    def genome_inputs(self):
        fasta = self.get("ensembl/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz")
        import pooch
        # The genome reader needs plain FASTA. Pooch reuses the expanded cache.
        expanded = Path(pooch.Decompress(method="gzip")(str(fasta), "fetch", None))
        # Reuse the already downloaded, explicitly matched release-102 GTF.
        snapshot = self.go_config.get("resource_snapshot", {}).get("files", {})
        gtf = Path(snapshot.get("Mus_musculus.GRCm38.102.gtf.gz", {}).get("path", ""))
        if not gtf.is_file():
            gtf = self.get("ensembl/Mus_musculus.GRCm38.102.gtf.gz")
        return dict(gtf_path=str(gtf), fasta_path=str(expanded), assembly="GRCm38",
                    fasta_assembly="GRCm38", gtf_assembly="GRCm38", promoter_window=[2000, 200],
                    transcript_rule="longest_exonic_then_id")

    def pathways(self):
        key = "reactome/Ensembl2Reactome_All_Levels.txt"
        path = self.get(key)
        with open_text(path) as handle:
            first = next(csv.reader(handle, delimiter="\t"))
        if len(first) != 6:
            raise ValueError("Reactome returned an unexpected Ensembl mapping format")
        specs = [self.spec("mouse_reactome", "pathways", "reactome", path, "curated_pathway", key,
                           ["Native mouse rows only; retain evidence and pathway IDs", "Curated/inferred mouse pathways remain identified in the raw file; no expression gene sets"])]
        key = "complexportal/10090.tsv"
        path = prepare_complexes(self.store, self.get(key), self.ids)
        specs.append(self.spec("mouse_complexes", "pathways", "membership", path, "curated_annotation", key,
                               ["Curated mouse ComplexTab file, excluding the separate predicted-complex download",
                                "Map declared UniProt participants; preserve stoichiometry, references and original evidence; omit unmapped/nonprotein participants"],
                               allowed_evidence=["complex_portal_curated"]))
        return specs

    def biogrid(self):
        key = "biogrid/BIOGRID-ORGANISM-5.0.262.tab3.zip"
        archive = self.get(key)
        name = "BIOGRID-ORGANISM-Mus_musculus-5.0.262.tab3.txt"
        output = self.store.root / "prepared" / name
        with zipfile.ZipFile(archive) as bundle:
            matches = [item for item in bundle.namelist() if Path(item).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} member in the BioGRID organism archive")
            # Copy the selected member; never extract arbitrary archive paths.
            temporary = output.with_name(output.name + "." + uuid4().hex + ".partial")
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                with bundle.open(matches[0]) as source_handle, temporary.open("wb") as destination:
                    shutil.copyfileobj(source_handle, destination)
                temporary.replace(output)
            finally:
                temporary.unlink(missing_ok=True)
        counts = Counter()
        with output.open() as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row["Organism ID Interactor A"] == row["Organism ID Interactor B"] == "10090":
                    counts[row["Experimental System Type"].lower(), row["Experimental System"]] += 1
        self.store.record("prepared/" + name, dict(path=str(output), sha256=digest(output),
                          mouse_assay_counts={"/".join(key): value for key, value in sorted(counts.items())}))
        return key, output, counts

    def interactions(self):
        key = "string/10090.protein.links.full.v12.0.txt.gz"
        path = self.get(key)
        with open_text(path) as handle:
            if not {"protein1", "protein2", "neighborhood", "fusion", "cooccurence"} <= set(next(handle).split()):
                raise ValueError("STRING full links lacks required independent channel columns")
        specs = [self.spec("mouse_string_sequence", "interactions", "string", path, "sequence", key,
                          ["Mouse proteins; use only neighborhood, fusion and cooccurence", "Exclude coexpression, transferred expression, experiments, database, text mining and combined scores"],
                          channels=["neighborhood", "fusion", "cooccurence"], min_score=150, two_hop_weight=.5)]
        key, path, counts = self.biogrid()
        allowed = [assay for assay in PHYSICAL_ASSAYS if counts["physical", assay]]
        if allowed:
            specs.append(self.spec("mouse_biogrid_physical", "interactions", "biogrid", path, "physical_interaction", key,
                                   ["Both interactors mouse; physical category only; direct physical-assay allowlist",
                                    "Exclude expression/coexpression systems; retain publication and assay"],
                                   allowed_experimental_systems=allowed))
        return specs

    def regulation(self):
        key = "trrust/trrust_rawdata.mouse.tsv"
        path = self.get(key)
        with path.open() as handle:
            if any(len(row) != 4 for row in csv.reader(handle, delimiter="\t")):
                raise ValueError("TRRUST mouse file must contain exactly four columns")
        specs = [self.spec("mouse_trrust", "regulation", "trrust", path, "curated_regulation", key,
                           ["Use literature-curated regulatory relationships, directions, signs and PubMed references; no expression numerical profiles or inferred coexpression networks",
                            "The four-column download does not expose individual experimental readouts; curation is not a claim that every underlying publication lacked expression measurements"],
                           allowed_evidence=["literature_curated"])]
        key = "jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt"
        path = self.get(key)
        from Bio import motifs
        with path.open() as handle:
            count = sum(1 for _ in motifs.parse(handle, "jaspar"))
        if not count:
            raise ValueError("No JASPAR motifs in provider download")
        specs.append(self.spec("mouse_motifs", "regulation", "motifs", path, "sequence", key,
                               ["JASPAR 2024 CORE vertebrate nonredundant PFMs; sequence compatibility, not binding or expression prediction",
                                "Matched Ensembl GRCm38 release-102 annotation and primary assembly; longest-exonic transcript selection; both strands"],
                               motif_format="jaspar", motif_resource="JASPAR CORE vertebrates", motif_version="2024",
                               motif_url=FILES[key], score_threshold=8., **self.genome_inputs()))
        return specs

    def evolution(self):
        pages = []
        page = 1
        page_hashes = set()
        metadata = self.store.fetch("oma/mouse-genome.json", "https://omabrowser.org/api/genome/MOUSE/")
        expected = json.loads(metadata.read_text()).get("nr_entries")
        received = 0
        while True:
            url = f"https://omabrowser.org/api/genome/MOUSE/proteins/?per_page=1000&page={page}"
            path = self.store.fetch(f"oma/mouse-proteins-{page:04d}.json", url)
            entries = json.loads(path.read_text())
            if not isinstance(entries, list):
                raise ValueError("OMA genome endpoint no longer returns an entry list")
            if not entries:
                if expected and received < int(expected):
                    raise ValueError(f"OMA pagination ended after {received} of {expected} entries")
                break
            checksum = digest(path)
            if checksum in page_hashes:
                raise ValueError("OMA endpoint repeated a page instead of advancing pagination")
            page_hashes.add(checksum)
            pages.append(path)
            received += len(entries)
            if expected and received >= int(expected):
                break
            if not expected and len(entries) < 1000:
                break
            page += 1
        paths = prepare_oma(self.store, pages, self.get("oma/oma-groups.txt.gz"), self.ids)
        filters = ["Native mouse protein cross-references mapped without human substitution; retain all explicit isoform links",
                   "Use OMA's published sequence-derived strict groups and root HOG memberships; no new homology inference",
                   "Paralog-family features are root HOGs with at least two distinct mapped mouse genes; not pairwise duplication dates or confidence estimates",
                   "Conservation descriptor counts distinct OMA genomes in strict groups; maximum across a gene's mapped groups, not alignment identity"]
        specs = []
        for name, path in paths.items():
            options = dict(allowed_evidence=["oma_sequence_homology"]) if name != "conservation" else {}
            specs.append(source("mouse_oma_" + name, "evolution", "numeric" if name == "conservation" else "membership",
                                path, "sequence", "https://omabrowser.org/oma/current/",
                                "Pinned OMA mouse API and group-file snapshots; see downloads.json",
                                ["OMA mouse genome API pages and official strict-group flat file", "Raw responses, hashes and normalized identifier links retained"], filters, **options))
        return specs

    def model(self, repo, kind, pooling):
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        from huggingface_hub import HfApi, snapshot_download
        key = "models/" + repo
        previous = self.store.records.get(key)
        revision = previous["revision"] if previous else HfApi().model_info(repo).sha
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Hugging Face did not return an immutable model commit")
        # Record the chosen commit before downloading, so retries keep the pin.
        self.store.record(key, dict(repo=repo, revision=revision))
        if self.download_models:
            print(f"Retrieving model assets: {repo}@{revision}", flush=True)
            snapshot = snapshot_download(repo_id=repo, revision=revision, cache_dir=str(self.store.root / "cache"),
                                         ignore_patterns=["*.h5", "*.msgpack", "*.onnx", "*.tflite", "*.ot", "*.pt", "onnx/*", "openvino/*"], max_workers=4)
            self.store.record(key, dict(repo=repo, revision=revision, snapshot=str(snapshot)))
        return dict(model=repo, model_revision=revision, pooling=pooling,
                    model_audit=("Official ESM-2 masked protein-sequence language model; features use only supplied protein sequences; no expression-based pretraining or supervision."
                                 if kind == "protein_esm" else
                                 "General sentence encoder pretrained/distilled on text and sentence-pair objectives; no expression matrices or expression-supervised gene embedding model. Source inputs are reviewed UniProt function comments."))

    def protein(self):
        key = "mgi/MGI_InterProDomains.rpt"
        path = prepare_domains(self.store, self.get(key), self.ids)
        specs = [self.spec("mouse_domains", "protein", "membership", path, "sequence", key,
                           ["Native MGI-to-InterPro sequence annotation links; protein-coding genes only",
                            "Entry presence includes domains/families/superfamilies from the provider report; no inferred architecture or expression weights"],
                           allowed_evidence=["interpro_sequence_annotation"])]
        key = "ensembl/Mus_musculus.GRCm38.pep.all.fa.gz"
        path = prepare_proteins(self.store, self.get(key), self.ids)
        specs.append(self.spec("mouse_esm", "protein", "protein_esm", path, "sequence_only_model", key,
                               ["All valid mapped protein-coding isoforms; remove terminal stop symbols, report discarded invalid/unmapped/noncoding entries",
                                "Runtime selects longest sequence then lexical protein ID and pools mean residue states; no expression-ranked isoform selection"],
                               isoform_rule="longest_sequence_then_identifier", max_tokens=1022,
                               **self.model("facebook/esm2_t6_8M_UR50D", "protein_esm", "mean_residues")))
        return specs

    def genomic(self):
        inputs = self.genome_inputs()
        return [source("mouse_genome", "genomic", "genome", inputs["gtf_path"], "sequence",
                       "https://ftp.ensembl.org/pub/release-102/", "Ensembl 102; GRCm38",
                       ["Matched Ensembl GRCm38 primary genome assembly and release-102 GTF"],
                       ["Gene structure, base composition and explicit location features only; longest-exonic transcript selection",
                        "Reference source assembly is explicit and is not inferred from the sample"], **inputs)]

    def literature(self):
        key = "uniprot/mouse.tsv"
        path = prepare_descriptions(self.store, self.get(key), self.ids)
        return [self.spec("mouse_descriptions", "literature", "descriptions", path, "curated_description", key,
                          ["Reviewed UniProt mouse entries with nonempty curated function comments; preserve exact text and accession reference",
                           "For multiple descriptions per gene choose longest reviewed entry, then lexical accession; no generated summaries or gene-name substitutes"],
                          **self.model("sentence-transformers/all-MiniLM-L6-v2", "descriptions", "model_default"))]

    def phenotypes(self):
        key = "mgi/MGI_GenePheno.rpt"
        path = prepare_phenotypes(self.store, self.get(key), self.ids)
        return [self.spec("mouse_phenotypes", "phenotypes", "membership", path, "genetic_phenotype", key,
                          ["MGI gene-genotype MP annotations excluding conditional mutations; only single-gene genotypes",
                           "Retain allele IDs, genotype, genetic background, allelic composition and publication IDs; no expression-atlas annotations"],
                          allowed_evidence=["mgi_genotype_phenotype"],
                          context="Native mouse single-gene nonconditional genetic perturbations; contexts retained per source row")]

    def perturbation(self):
        key, path, counts = self.biogrid()
        allowed = [assay for assay in GENETIC_ASSAYS if counts["genetic", assay]]
        if not allowed:
            raise ValueError("BioGRID mouse archive has no eligible genetic interaction assays")
        return [self.spec("mouse_genetic_interactions", "perturbation", "biogrid", path, "genetic_interaction", key,
                          ["Both interactors mouse; genetic category only; positive allowlist of synthetic lethality/growth/rescue and genetic phenotype enhancement/suppression",
                           "Exclude expression/coexpression experimental systems; retain assay and publication; genetic interactions are not quantitative fitness measurements"],
                          allowed_experimental_systems=allowed, assay="BioGRID curated mouse genetic interactions",
                          context="Native mouse genetic perturbation contexts; publication-specific, not assumed embryonic")]


PROBE_URLS = {
    "oma": "https://omabrowser.org/api/genome/MOUSE/proteins/?per_page=1000&page=1",
    "compara": "https://ftp.ensembl.org/pub/release-102/tsv/ensembl-compara/homologies/mus_musculus/",
    "domains": "https://www.informatics.jax.org/downloads/reports/MGI_InterProDomains.rpt",
    "uniprot": "https://rest.uniprot.org/uniprotkb/search?query=organism_id:10090&format=tsv&fields=accession,id,reviewed,length,xref_mgi,cc_function,protein_families&size=1",
    "complexes": "https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/10090.tsv",
    "phenotypes": "https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt",
    "trrust": "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
    "jaspar": "https://jaspar2024.elixir.no/download/data/2024/CORE/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
    "biogrid": "https://biogrid-downloads.nyc3.digitaloceanspaces.com/BioGRID/Release-Archive/BIOGRID-5.0.262/BIOGRID-ORGANISM-5.0.262.tab3.zip",
    "homology": "https://www.informatics.jax.org/downloads/reports/HOM_AllOrganism.rpt",
    "ensembl_peptides": "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/pep/Mus_musculus.GRCm38.pep.all.fa.gz",
    "ensembl_genome": "https://ftp.ensembl.org/pub/release-102/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",
    "string": "https://stringdb-downloads.org/download/protein.links.full.v12.0/10090.protein.links.full.v12.0.txt.gz",
}


def probe():
    from concurrent.futures import ThreadPoolExecutor

    def inspect(item):
        name, url = item
        try:
            with requests.get(url, stream=True, timeout=(20, 45)) as response:
                prefix = next(response.iter_content(2200), b"")
                return dict(name=name, status=response.status_code, url=response.url,
                            preview=(prefix[:24].hex() if prefix.startswith((b"\x1f\x8b", b"PK"))
                                     else prefix.decode(errors="replace")[:1800]))
        except requests.RequestException as error:
            return dict(name=name, error=str(error))

    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(inspect, PROBE_URLS.items()):
            print(json.dumps(result), flush=True)


FAMILY_FILES = {
    "go": (),
    "pathways": ("reactome/Ensembl2Reactome_All_Levels.txt", "complexportal/10090.tsv", "uniprot/mouse.tsv"),
    "interactions": ("string/10090.protein.links.full.v12.0.txt.gz", "biogrid/BIOGRID-ORGANISM-5.0.262.tab3.zip",
                     "ensembl/Mus_musculus.GRCm38.pep.all.fa.gz"),
    "regulation": ("trrust/trrust_rawdata.mouse.tsv", "jaspar/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt",
                   "ensembl/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz"),
    "evolution": ("uniprot/mouse.tsv", "oma/oma-groups.txt.gz"),
    "protein": ("mgi/MGI_InterProDomains.rpt", "ensembl/Mus_musculus.GRCm38.pep.all.fa.gz", "uniprot/mouse.tsv"),
    "genomic": ("ensembl/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz",),
    "literature": ("uniprot/mouse.tsv",),
    "phenotypes": ("mgi/MGI_GenePheno.rpt",),
    "perturbation": ("biogrid/BIOGRID-ORGANISM-5.0.262.tab3.zip",),
}


def prepare_all(output_config=DEFAULT_OUTPUT_CONFIG, resource_dir=None,
                families=None, download_models=True, go_config_path=None):
    """Download real inputs, normalize provider tables and write a runnable config.

    Defaults retrieve every family and both pretrained models. Repeated calls
    reuse verified downloads and pinned model commits. A family subset updates
    that subset in an existing generated config, preserving the other families.
    No feature construction or neural inference is performed here.
    """
    output = Path(output_config).expanduser().resolve()
    requested = list(dict.fromkeys(families or FAMILIES))
    if set(requested) - set(FAMILIES):
        raise ValueError("Unsupported prior family")
    previous = json.loads(output.read_text()) if output.is_file() else {}
    if previous and previous.get("prepared_by") != "preprocessing.biological_priors.prepare_all":
        raise ValueError("Output config already exists and was not generated by prepare_all; choose a different --output-config")
    store = ResourceStore(resource_dir or output.parent / "biological_prior_resources/all")
    go_path = Path(go_config_path or output.parent / "priors.go.json").expanduser().resolve()
    if go_path == output:
        raise ValueError("The all-family output must differ from the GO source config")
    if not go_path.is_file():
        from .prepare_go import prepare_go
        prepare_go(go_path, store.root.parent / "go")
    go_config = json.loads(go_path.read_text())
    # Resolve relative paths in an existing GO configuration before reusing it.
    from .resources import prepare as resolve_source, materialize
    go_cache = materialize(go_config.get("cache_dir", ".prior_cache"), go_path.parent, go_path.parent)
    go_cache.mkdir(parents=True, exist_ok=True)
    mapping = materialize(go_config["identifiers"]["path"], go_path.parent, go_cache)
    if not mapping.is_file():
        raise FileNotFoundError(f"GO identifier mapping missing: run prepare_go for {go_path}")
    files = sorted({key for family in requested for key in FAMILY_FILES[family]})
    download_errors = {}
    available = {}
    # Independent raw downloads share a persistent, locked receipt manifest.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(store.fetch, key, FILES[key]): key for key in files}
        for future in as_completed(pending):
            key = pending[future]
            try:
                available[key] = future.result()
            except Exception as error:
                download_errors[key] = f"{type(error).__name__}: {error}"
                print(f"Download failed: {key}: {error}", flush=True)
    base_mapping = previous.get("identifiers", {}).get("path")
    if base_mapping and Path(base_mapping).is_file():
        mapping = Path(base_mapping)
    enriched = enrich_identifiers(store, mapping, available.get("uniprot/mouse.tsv"),
                                  available.get("ensembl/Mus_musculus.GRCm38.pep.all.fa.gz"))
    ids = Identifiers(enriched, [])
    recipes = MousePreparation(store, ids, go_config, download_models)
    sources = [s for s in previous.get("sources", []) if s.get("family") not in requested]
    statuses = {}
    report_path = store.root / "preparation-report.json"
    config = dict(organism="10090", prepared_by="preprocessing.biological_priors.prepare_all",
                  cache_dir=str(store.root / "cache"),
                  identifiers=dict(go_config["identifiers"], path=str(enriched),
                                   version="MGI mapping plus pinned UniProt/Ensembl protein cross-reference snapshots"),
                  sources=sources,
                  resource_snapshot=dict(directory=str(store.root), downloads_manifest=str(store.manifest),
                                         reference_assembly="GRCm38; Ensembl release 102; not inferred from sample"))

    def checkpoint():
        config["sources"] = sources
        config["resource_snapshot"]["downloads_manifest_sha256"] = digest(store.manifest)
        atomic_json(output, config)
        atomic_json(report_path, dict(config=str(output), requested_families=requested, families=statuses,
                                      downloads=download_errors, model_downloads_enabled=download_models,
                                      complete=False))

    for family in requested:
        print(f"Preparing family: {family}", flush=True)
        try:
            failures = {key: download_errors[key] for key in FAMILY_FILES[family] if key in download_errors}
            if failures:
                raise RuntimeError("Required downloads failed: " + json.dumps(failures))
            if family == "go":
                result = [resolve_source(spec, go_path.parent, go_cache)[0] for spec in go_config["sources"]]
            else:
                result = getattr(recipes, family)()
            if not result:
                raise ValueError("Recipe produced no sources")
            sources.extend(result)
            statuses[family] = dict(status="prepared", sources=[s["name"] for s in result])
        except Exception as error:
            statuses[family] = dict(status="failed", reason=f"{type(error).__name__}: {error}")
            print(f"{family}: {statuses[family]['reason']}", flush=True)
        checkpoint()
    # Check availability of generated inputs/model files, not feature matrices.
    from .prepare_resources import prepare_resources
    _, inventory = prepare_resources(output, families=requested)
    atomic_json(store.root / "input-inventory.json", inventory)
    report = dict(config=str(output), requested_families=requested, families=statuses,
                  download_errors=download_errors, model_downloads_enabled=download_models,
                  complete=inventory["ready"] and all(s["status"] == "prepared" for s in statuses.values()),
                  inventory=str(store.root / "input-inventory.json"), downloads_manifest=str(store.manifest))
    atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--resource-dir", type=Path, help="Persistent raw/prepared/model cache; defaults beneath preprocessing/results")
    parser.add_argument("--go-config", type=Path, help="Reuse an existing GO/mouse identifier config; create the default if absent")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, help="Retrieve only these families; default: all ten")
    parser.add_argument("--skip-models", action="store_true", help="Fetch data only; missing model assets will remain explicit in the inventory")
    parser.add_argument("--probe", action="store_true", help="Check public endpoints without full dataset downloads")
    args = parser.parse_args()
    if args.probe:
        probe()
        return
    try:
        report = prepare_all(args.output_config, args.resource_dir, args.families,
                             not args.skip_models, args.go_config)
    except Exception as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    for family, status in report["families"].items():
        print(f"{family}: {status['status']}" + (f" — {status['reason']}" if "reason" in status else ""))
    print(f"Configuration: {report['config']}")
    print(f"Input inventory: {report['inventory']}")
    print("All requested inputs ready" if report["complete"] else "Preparation incomplete; inspect preparation-report.json and input-inventory.json")
    raise SystemExit(0 if report["complete"] else 2)


if __name__ == "__main__":
    main()
