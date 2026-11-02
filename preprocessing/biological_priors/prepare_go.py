"""Download/cache official GO and MGI inputs and build an audited mouse config.

python -m preprocessing.biological_priors.prepare_go --output-config preprocessing/results/priors.go.json
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re

import pooch

from .core import digest, write_json, write_table

DEFAULT_OUTPUT_CONFIG = Path(__file__).resolve().parents[1] / "results/priors.go.json"

GO_RELEASE = "2026-08-05"
FILES = {
    "MRK_ENSEMBL.rpt": "https://www.informatics.jax.org/downloads/reports/MRK_ENSEMBL.rpt",
    "MGI_EntrezGene.rpt": "https://www.informatics.jax.org/downloads/reports/MGI_EntrezGene.rpt",
    "Mus_musculus.GRCm38.102.gtf.gz": "https://ftp.ensembl.org/pub/release-102/gtf/mus_musculus/Mus_musculus.GRCm38.102.gtf.gz",
    f"MOUSE-mod.{GO_RELEASE}.gaf.gz": f"https://release.geneontology.org/{GO_RELEASE}/annotations/gaf/MOUSE-mod.gaf.gz",
    f"go-basic.{GO_RELEASE}.obo": f"https://release.geneontology.org/{GO_RELEASE}/ontology/go-basic.obo",
}


def prepare_go(output_config=DEFAULT_OUTPUT_CONFIG, resource_dir=None):
    output_config = Path(output_config).expanduser().resolve()
    folder = Path(resource_dir or output_config.parent / "biological_prior_resources/go").resolve()
    folder.mkdir(parents=True, exist_ok=True)
    previous_manifest = folder / "resource-manifest.json"
    previous = json.loads(previous_manifest.read_text()).get("files", {}) if previous_manifest.is_file() else {}
    artifacts = {}
    for name, url in FILES.items():
        path = folder / name
        expected = previous.get(name, {}).get("sha256")
        if path.exists() and expected and digest(path) != expected:
            raise ValueError(f"Cached GO resource differs from its recorded snapshot: {path}")
        if not path.exists():
            path = Path(pooch.retrieve(url, known_hash="sha256:" + expected if expected else None,
                                       fname=name, path=folder,
                                       downloader=pooch.HTTPDownloader(timeout=180)))
        if path.stat().st_size < 1000:
            raise ValueError(f"Empty/invalid resource: {path}")
        artifacts[name] = dict(url=url, path=str(path), sha256=digest(path), size_bytes=path.stat().st_size)
    gaf_path = folder / f"MOUSE-mod.{GO_RELEASE}.gaf.gz"
    with gzip.open(gaf_path, "rt") as handle:
        gaf_header = []
        for line in handle:
            if not line.startswith("!"):
                break
            gaf_header.append(line.strip())
    ontology_path = folder / f"go-basic.{GO_RELEASE}.obo"
    ontology_version = None
    with ontology_path.open() as handle:
        for line in handle:
            if line.startswith("data-version:"):
                ontology_version = line.split(":", 1)[1].strip()
            if line.startswith("[Term]"):
                break
    if not ontology_version or not any(line.startswith("!gaf-version:") for line in gaf_header):
        raise ValueError("Downloaded GO resources lack required format/version headers")
    # These reports contain native mouse identifiers/biotypes only. No expression
    # tables, values, tissue-specific annotations or human orthologs are used.
    extras = defaultdict(lambda: {"aliases": set(), "entrez_id": set(), "symbol": set()})
    with (folder / "MGI_EntrezGene.rpt").open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 15:
                raise ValueError("Unexpected MGI_EntrezGene report schema")
            extras[row[0]]["aliases"].update(x for x in row[9].split("|") if x)
            extras[row[0]]["entrez_id"].update(x for x in row[8].split("|") if x)
            if row[2] == "O":
                extras[row[0]]["symbol"].add(row[1])
    genes = defaultdict(lambda: defaultdict(set))
    with (folder / "MRK_ENSEMBL.rpt").open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 13:
                raise ValueError("Unexpected MRK_ENSEMBL report schema")
            for gid in re.findall(r"ENSMUSG\d+", row[5]):
                rec = genes[gid]
                rec["symbol"].add(row[1])
                rec["mgi_id"].add(row[0])
                rec["aliases"].update(extras[row[0]]["aliases"])
                rec["entrez_id"].update(extras[row[0]]["entrez_id"])
                rec["protein_id"].update(row[7].split())
                rec["biotype"].update(row[12].split("|"))
    # Historical native mouse names are cross-references, not an assumption
    # about the assay's genome build. gffutils parses GTF attributes.
    from gffutils.feature import feature_from_line
    symbols, synonyms, current = defaultdict(set), defaultdict(set), defaultdict(set)
    for mgi, values in extras.items():
        for name in values["symbol"]:
            symbols[name].add(mgi)
        for name in values["aliases"]:
            synonyms[name].add(mgi)
    for gid, values in genes.items():
        values["mapping_source"].add("MGI current Ensembl report")
        for mgi in values["mgi_id"]:
            current[mgi].add(gid)
    with gzip.open(folder / "Mus_musculus.GRCm38.102.gtf.gz", "rt") as handle:
        for line in handle:
            if line.startswith("#") or len(line.split("\t", 3)) < 4 or line.split("\t", 3)[2] != "gene":
                continue
            feature = feature_from_line(line)
            gid = feature.attributes["gene_id"][0]
            names = feature.attributes.get("gene_name", [])
            mgi_matches = set().union(*(symbols.get(name, set()) for name in names))
            if not mgi_matches:
                mgi_matches = set().union(*(synonyms.get(name, set()) for name in names))
            targets = {gid} if gid in genes else set().union(*(current.get(mgi, set()) for mgi in mgi_matches))
            if not targets:
                targets = {gid}
            for target in targets:
                rec = genes[target]
                rec["historical_symbols"].update(names)
                rec["ensembl_aliases"].add(gid)
                rec["mapping_source"].add("Ensembl GRCm38 release 102 identifier annotations")
                if target == gid and gid not in {g for mgi in mgi_matches for g in current.get(mgi, set())} and not rec["symbol"]:
                    # A reference-only stable ID is valid but is not declared a
                    # current MGI symbol/model. Uncertain MGI links stay absent.
                    rec["biotype"].update(feature.attributes.get("gene_biotype", ["unknown"]))
                    if len(mgi_matches) == 1 and not current.get(next(iter(mgi_matches))):
                        mgi = next(iter(mgi_matches))
                        rec["mgi_id"].add(mgi)
                        rec["symbol"].update(extras[mgi]["symbol"])
                        rec["aliases"].update(extras[mgi]["aliases"])
                        rec["entrez_id"].update(extras[mgi]["entrez_id"])
    rows = []
    for gid, values in sorted(genes.items()):
        types = values["biotype"] - {""}
        coding = types & {"protein_coding", "protein-coding"}
        biotype = "protein_coding" if coding and types <= {"protein_coding", "protein-coding"} else "|".join(sorted(types)) or "unknown"
        rows.append(dict(ensembl_gene_id=gid, symbol="|".join(sorted(values["symbol"])), taxon_id="10090", biotype=biotype,
                         mapping_status="current_mgi_model" if "MGI current Ensembl report" in values["mapping_source"] else
                                        "historical_model_with_mgi_link" if values["mgi_id"] else "historical_reference_only",
                         **{key: "|".join(sorted(values[key] - {""})) for key in ("mgi_id", "aliases", "entrez_id", "protein_id", "historical_symbols", "ensembl_aliases", "mapping_source")}))
    mapping = folder / "mouse_identifiers.tsv"
    write_table(mapping, rows, ("ensembl_gene_id", "symbol", "taxon_id", "biotype", "mgi_id", "aliases", "entrez_id", "protein_id", "historical_symbols", "ensembl_aliases", "mapping_source", "mapping_status"))
    snapshot = datetime.now(timezone.utc).isoformat()
    audit = dict(expression_independent=True,
        upstream_components=[f"Official mouse MGI GAF and go-basic from GO release bundle {GO_RELEASE}",
                             "MGI MRK_ENSEMBL and MGI_EntrezGene native mouse nomenclature/cross-reference reports",
                             "Ensembl GRCm38 release 102 gene IDs/names only; not an assertion of sample genome assembly"],
        evidence_filters=["Primary taxon 10090 only; IDA/IPI/IMP/IGI/TAS/IC only",
                          "Exclude IEP/HEP, IEA, similarity-transfer and all other unapproved codes; exclude NOT memberships",
                          "GODag propagation only is_a and part_of; omit roots, obsolete and unknown terms",
                          "No expression matrices, expression-derived weights, pretrained gene models or human substitutions"],
        justification="Curated functional annotations are permitted under the stated restrictions; numerical features use permitted annotation memberships/frequencies only. Nomenclature reports supply identifiers only.",
        reviewed_documentation=["https://geneontology.org/docs/guide-go-evidence-codes/",
            "https://geneontology.org/docs/download-ontology/", "https://www.informatics.jax.org/downloads/reports/index.html"],
        release_note="Ontology and GAF are from one official bundle; their internal ontology/GO-plus version IRIs are recorded separately. Unknown terms are excluded explicitly.")
    config = dict(organism="10090", cache_dir=str(folder / "cache"),
        identifiers=dict(path=str(mapping), version=f"MGI snapshot retrieved {snapshot}",
                         url="https://www.informatics.jax.org/downloads/reports/index.html",
                         license="CC BY 4.0; https://www.informatics.jax.org/mgihome/other/copyright.shtml"),
        sources=[dict(name="mouse_go", family="go", kind="go_gaf", path=str(gaf_path), ontology_path=str(ontology_path),
                      version=f"GO release bundle {GO_RELEASE}; ontology {ontology_version}",
                      url=f"https://release.geneontology.org/{GO_RELEASE}/", species="10090",
                      license="CC BY 4.0; https://geneontology.org/docs/go-citation-policy/",
                      evidence_basis="curated_annotation", allowed_evidence=["IDA", "IPI", "IMP", "IGI", "TAS", "IC"], audit=audit)],
        resource_snapshot=dict(retrieved_utc=snapshot, files=artifacts, gaf_header=gaf_header,
                               ontology_data_version=ontology_version, mapping_rows=len(rows), mapping_sha256=digest(mapping)))
    output_config.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_config, config)
    write_json(folder / "resource-manifest.json", config["resource_snapshot"])
    print(f"Prepared {len(rows):,} native mouse identifier records: {output_config}", flush=True)
    return output_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-config", type=Path, default=DEFAULT_OUTPUT_CONFIG)
    parser.add_argument("--resource-dir", type=Path)
    args = parser.parse_args()
    prepare_go(args.output_config, args.resource_dir)
