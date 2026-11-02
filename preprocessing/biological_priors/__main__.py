"""python -m preprocessing.biological_priors --genes genes.txt --config config.json --output results"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import scipy
from scipy import sparse

from .core import FAMILIES, Identifiers, audit, digest, write_json, write_table
from . import sources, sequence
from .resources import materialize, prepare


def save_block(block, ids, output):
    for expected, actual in zip(ids.rows, block.coverage):
        if any(expected[key] != actual[key] for key in ("row_id", "input_id", "canonical_id")):
            raise ValueError(f"{block.name}: gene order differs from shared input order")
    kernel, smallest = block.validate(len(ids.genes))
    folder = output / block.name
    folder.mkdir()
    sparse.save_npz(folder / "features.npz", block.matrix)
    sparse.save_npz(folder / "similarity.npz", kernel)
    write_table(folder / "rows.tsv", ids.rows, ("row_id", "input_id", "canonical_id"))
    write_table(folder / "features.tsv", ({"column_index": i, "feature": feature} for i, feature in enumerate(block.features)),
                ("column_index", "feature"))
    columns = ("row_id", "input_id", "canonical_id", "status", "observed", "usable", "nonzero_features")
    write_table(folder / "coverage.tsv", block.coverage, columns)
    provenance = dict(block.provenance, validation={"smallest_kernel_eigenvalue": smallest, "psd_tolerance": 1e-7,
                                                    "n_rows": len(ids.genes), "n_features": len(block.features),
                                                    "finite_symmetric": True, "same_input_order": True})
    inputs = block.provenance.get("artifacts", {}).get("inputs_path")
    if inputs:
        suffix = "".join(Path(inputs["path"]).suffixes)
        exact_path = folder / ("exact_inputs" + suffix)
        shutil.copyfile(inputs["path"], exact_path)
        provenance["exact_inputs_file"] = exact_path.name
    write_json(folder / "provenance.json", provenance)
    files = {key: f"{block.name}/{key}" for key in ("features.npz", "similarity.npz", "rows.tsv", "features.tsv", "coverage.tsv", "provenance.json")}
    if inputs:
        files[exact_path.name] = f"{block.name}/{exact_path.name}"
    return {"name": block.name, "family": block.provenance["source"]["family"],
            "shape": list(block.matrix.shape), "usable_rows": sum(row["usable"] for row in block.coverage),
            "status": "populated" if block.matrix.nnz else "no_usable_features", "files": files,
            "sha256": {key: digest(output / value) for key, value in files.items()}}


def run(genes_path, config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    config = json.loads(config_path.read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty, to avoid overwriting/mixing runs")
    if str(config.get("organism", "10090")) != "10090":
        raise ValueError("This implementation resolves native mouse identifiers only")
    base = config_path.parent
    cache = materialize(config.get("cache_dir", ".prior_cache"), base, base)
    cache.mkdir(parents=True, exist_ok=True)
    mapping_spec = config["identifiers"]
    for key in ("path", "version", "url", "license"):
        if not mapping_spec.get(key):
            raise ValueError(f"Identifier mapping requires {key}")
    mapping_path = materialize(mapping_spec["path"], base, cache)
    with Path(genes_path).open() as handle:
        inputs = [line.rstrip("\r\n") for line in handle]
    if not inputs or any(not gene or gene != gene.strip() for gene in inputs):
        raise ValueError("Gene list must contain one nonempty identifier per line, without surrounding whitespace")
    ids = Identifiers(mapping_path, inputs)
    specs = config.get("sources", [])
    names = [spec.get("name", "") for spec in specs]
    if len(names) != len(set(names)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError("Source names must be unique simple filenames: letters, numbers, _ or -")
    loaders = dict(sources.LOADERS, **sequence.LOADERS)
    blocks, statuses, failures = [], [], []
    for spec in specs:
        status = {"name": spec["name"], "family": spec.get("family"), "kind": spec.get("kind")}
        try:
            audit(spec)
            if spec["kind"] not in loaders or spec["family"] not in sources.KIND_FAMILY[spec["kind"]]:
                raise ValueError("Source kind is unsupported or does not match the declared family")
            if spec["evidence_basis"] not in sources.KIND_BASES[spec["kind"]]:
                raise ValueError("Evidence basis does not match source adapter")
            if spec["family"] == "phenotypes" and spec["evidence_basis"] != "genetic_phenotype":
                raise ValueError("Phenotype memberships must have curated genetic phenotype evidence")
            if spec["kind"] in {"go_gaf", "reactome", "string", "genome", "motifs"} and str(spec["species"]) != "10090":
                raise ValueError("This native-mouse adapter does not accept orthology-transferred input")
            resolved, artifacts = prepare(spec, base, cache)
            mapper = sources.SourceMapper(resolved, ids)
            provenance = {"source": spec, "artifacts": artifacts,
                          "identifier_mapping_sha256": digest(mapping_path)}
            result = loaders[spec["kind"]](resolved, ids, provenance, mapper)
            for block in result:
                if not re.fullmatch(r"[A-Za-z0-9_. -]+", block.name) or ".." in block.name or "/" in block.name:
                    raise ValueError("Unsafe block/channel name")
                block.provenance["unmapped_source_identifiers"] = sorted(mapper.unmapped)
                block.validate_features(len(inputs))
            blocks.extend(result)
            status.update(status="constructed" if result else "no_eligible_records", blocks=[block.name for block in result],
                          unmapped_identifiers=len(mapper.unmapped))
        except (FileNotFoundError, ImportError) as error:
            status.update(status="unavailable", reason=str(error))
        except Exception as error:
            status.update(status="failed_validation", reason=f"{type(error).__name__}: {error}")
            failures.append(spec["name"])
        statuses.append(status)
    if len({block.name for block in blocks}) != len(blocks):
        raise ValueError("Block names collide")
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "genes.tsv", ids.rows, ("row_id", "input_id", "canonical_id"))
    write_table(output / "identifier_mapping.tsv", ids.rows, tuple(ids.rows[0]))
    dependency_versions = {}
    for package in ("numpy", "scipy", "goatools", "gffutils", "biopython", "intervaltree", "pooch", "scikit-learn", "pyfaidx", "transformers", "torch", "sentence-transformers", "huggingface-hub"):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = "not installed"
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "organism": "Mus musculus", "taxon_id": 10090,
                "software": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                             "dependencies": dependency_versions,
                             "implementation_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}},
                "configuration": config, "config_sha256": digest(config_path),
                "gene_list_sha256": digest(genes_path), "identifiers": dict(mapping_spec, sha256=digest(mapping_path)),
                "ordered_genes": "genes.tsv", "identifier_mapping": "identifier_mapping.tsv",
                "identifier_resolution_policy": ["exact native identifier", "exact current mouse symbol", "exact synonym/historical alias",
                                                 "unique case-insensitive native symbol", "unique case-insensitive native alias",
                                                 "preserve genuine ambiguity; no ortholog or fuzzy-name substitution"],
                "sources": statuses, "blocks": [], "aggregation": "unimplemented; no combined outputs"}
    for block in blocks:
        manifest["blocks"].append(save_block(block, ids, output))
    manifest["families"] = {family: {
        "supported": True,
        "configured_sources": [status["name"] for status in statuses if status["family"] == family],
        "populated_blocks": [entry["name"] for entry in manifest["blocks"] if entry["family"] == family and entry["status"] == "populated"],
        "incomplete_coverage": (not any(entry["family"] == family for entry in manifest["blocks"]) or
                                any(entry["family"] == family and entry["usable_rows"] < len(inputs) for entry in manifest["blocks"]) or
                                any(status["family"] == family and status["status"] != "constructed" for status in statuses)),
    } for family in FAMILIES}
    manifest["failed_sources"] = failures
    write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# Separate biological priors\n\n"
        "See manifest.json for successfully populated blocks, unavailable sources, audits and exact file hashes.\n"
        "features.npz and similarity.npz are scipy.sparse CSR matrices. rows.tsv labels both axes of each similarity; "
        "features.tsv labels feature columns. Row IDs distinguish duplicate input identifiers.\n\n"
        "Cosine Gram kernels are positive semidefinite. Missing/zero vectors have zero rows and diagonals, "
        "not fabricated self-similarity. Consult coverage.tsv to distinguish absent records from explicit known absence. "
        "Signed profiles can yield negative similarity. These matrices are not expression covariance. "
        "No combined feature or similarity matrix is produced.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genes", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true", help="Exit nonzero if any family is unpopulated or a configured source is unavailable")
    args = parser.parse_args()
    try:
        manifest = run(args.genes, args.config, args.output)
    except Exception as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    for family, status in manifest["families"].items():
        print(f"{family}: {len(status['populated_blocks'])} populated blocks; incomplete coverage={status['incomplete_coverage']}")
    for source in manifest["sources"]:
        print(f"{source['name']}: {source['status']}" + (f" ({source['reason']})" if "reason" in source else ""))
    print(f"Manifest: {args.output / 'manifest.json'}")
    if manifest["failed_sources"] or (args.require_complete and (
        any(not status["populated_blocks"] for status in manifest["families"].values()) or
        any(source["status"] == "unavailable" for source in manifest["sources"]))):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
