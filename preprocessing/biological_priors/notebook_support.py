"""Sample identifiers, notebook setup, synthetic fixtures, and block inspection.

This module wraps the existing pipeline. It never aggregates representations or
uses expression measurements. Demonstration vectors are hand-authored fixtures,
not inferred protein or text embeddings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import sparse

from .__main__ import run
from .core import FAMILIES, digest, open_text, write_json, write_table


INFERENCE_KINDS = {"protein_esm", "descriptions"}


def sample_gene_list(sample_path, output_folder, gene_id_column=None):
    """Read only an H5AD /var identifier field, never X, layers, raw or obs.

    The sample defines the supplied gene universe/order only. No expression
    filtering, ranking, numerical metadata or gene-name conversion is performed.
    """
    import h5py

    sample = Path(sample_path).expanduser().resolve()
    with h5py.File(sample, "r") as handle:
        var = handle["var"]
        key = gene_id_column if gene_id_column is not None else var.attrs.get("_index", "_index")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if not isinstance(key, str) or "/" in key or key in {".", ".."}:
            raise ValueError("Gene identifier field must name a direct child of /var")
        if key not in var:
            raise ValueError(f"Gene identifier field /var/{key} is unavailable in {sample.name}")
        field = var[key]
        if isinstance(field, h5py.Group):
            if field.attrs.get("encoding-type") not in {"categorical", b"categorical"}:
                raise ValueError("Gene identifiers must be a string array or string categorical")
            categories = field["categories"]
            if h5py.check_string_dtype(categories.dtype) is None:
                raise ValueError("Gene identifier categories must be strings")
            labels = categories.asstr()[:]
            codes = field["codes"][:]
            if (codes < 0).any() or (codes >= len(labels)).any():
                raise ValueError("Gene identifier categorical has missing/invalid codes")
            genes = labels[codes].tolist()
        else:
            if field.ndim != 1 or h5py.check_string_dtype(field.dtype) is None:
                raise ValueError("Gene identifiers must be a one-dimensional string array")
            genes = field.asstr()[:].tolist()
    if not genes or any(not gene or gene != gene.strip() or any(c in gene for c in "\r\n\t") for gene in genes):
        raise ValueError("Gene identifiers must be nonempty strings without surrounding whitespace or tab/newline characters")
    content = "\n".join(genes) + "\n"
    fingerprint = hashlib.sha256(content.encode("utf-8")).hexdigest()
    folder = Path(output_folder).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{sample.stem}.{fingerprint[:16]}.genes.txt"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError("Cached sample gene list differs from the sample's ordered identifiers")
    else:
        destination.write_text(content, encoding="utf-8")
    stat = sample.stat()
    provenance = dict(sample_path=str(sample), sample_filename=sample.name,
                      sample_size_bytes=stat.st_size, sample_mtime_ns=stat.st_mtime_ns,
                      identifier_field=f"/var/{key}", gene_count=len(genes),
                      unique_gene_count=len(set(genes)), duplicate_rows=len(genes) - len(set(genes)),
                      ordered_gene_list_sha256=fingerprint, gene_list_path=str(destination),
                      accessed_data="Only the selected /var identifier field (and its categories/codes if categorical)",
                      expression_values_used=False, filtering="none; preserve all original identifiers and their order")
    write_json(destination.with_suffix(".source.json"), provenance)
    return destination, provenance


def synthetic_config(family, folder):
    """Small runnable fixtures. No fabricated entry is attributed to a database."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def text(name, value):
        path = folder / name
        path.write_text(value)
        return str(path)

    def tsv(name, columns, rows):
        path = folder / name
        write_table(path, [dict(zip(columns, row)) for row in rows], columns)
        return str(path)

    genes = text("genes.txt", "ToyB\nToyA\nToyNc\nShared\nMissing\nToyA\nENSG00000000001\n")
    gene_ids = ["ENSMUSG" + str(i).zfill(11) for i in (90001, 90002, 90003)]
    identifiers = tsv("identifiers.tsv",
        ("ensembl_gene_id", "symbol", "taxon_id", "biotype", "aliases", "string_id", "entrez_id"),
        [(gene_ids[i], symbol, "10090", biotype, "Shared" if i < 2 else "", f"10090.toy{i}", str(i + 1))
         for i, (symbol, biotype) in enumerate((("ToyA", "protein_coding"), ("ToyB", "protein_coding"), ("ToyNc", "lncRNA")))])
    sources = []

    def source(name, kind, path, basis, **kwargs):
        item = dict(name="synthetic_" + name, family=family, kind=kind, path=path,
                    version="SYNTHETIC-DEMONSTRATION-v1", species="10090",
                    url="https://example.org/synthetic-not-biological-data", license="Hand-authored synthetic fixture",
                    evidence_basis=basis, audit={
                        "expression_independent": True,
                        "upstream_components": ["Hand-authored toy records; not a real biological resource"],
                        "evidence_filters": ["Only the synthetic records explicitly listed in this fixture"],
                        "justification": "Demonstrate software behavior only; establishes no real biological associations",
                    }, **kwargs)
        sources.append(item)
        return item

    def membership(name, features, basis):
        path = tsv(name + ".tsv", ("gene_id", "feature", "evidence", "association"),
                   [(gene, feature, "synthetic", "positive") for gene, feature in features])
        source(name, "membership", path, basis, allowed_evidence=["synthetic"])

    def genome_files():
        fasta = text("genome.fa", ">toy_chr\n" + "ACGT" * 100 + "\n")
        gtf = text("genes.gtf", "\n".join(
            f'toy_chr\tsynthetic\texon\t{start}\t{end}\t.\t{strand}\t.\tgene_id "{gene_ids[i]}"; transcript_id "toy_tx{i}";'
            for i, start, end, strand in ((0, 31, 60, "+"), (0, 71, 90, "+"), (1, 121, 160, "-"), (2, 211, 250, "+"))) + "\n")
        return dict(gtf_path=gtf, fasta_path=fasta, assembly="SYNTHETIC-ASSEMBLY",
                    gtf_assembly="SYNTHETIC-ASSEMBLY", fasta_assembly="SYNTHETIC-ASSEMBLY",
                    transcript_rule="longest_exonic_then_id", promoter_window=[20, 10])

    if family == "go":
        terms = [("GO:0008150", "biological_process", None), ("GO:0003674", "molecular_function", None),
                 ("GO:0005575", "cellular_component", None), ("GO:9000001", "biological_process", "GO:0008150"),
                 ("GO:9000002", "biological_process", "GO:9000001"), ("GO:9000003", "molecular_function", "GO:0003674"),
                 ("GO:9000004", "cellular_component", "GO:0005575")]
        ontology = text("synthetic.obo", "format-version: 1.2\n\n" + "\n".join(
            "[Term]\nid: " + term + "\nname: SYNTHETIC toy term\nnamespace: " + aspect + "\n" +
            ("is_a: " + parent + " ! toy parent\n" if parent else "") for term, aspect, parent in terms))
        records = []
        for gene, term, evidence in (("ToyA", "GO:9000002", "IDA"), ("ToyB", "GO:9000001", "IMP"),
                                    ("ToyA", "GO:9000003", "IPI"), ("ToyB", "GO:9000004", "TAS"),
                                    ("ToyNc", "GO:9000002", "IEP")):
            aspect, qualifier = (("F", "enables") if term == "GO:9000003" else
                                 ("C", "located_in") if term == "GO:9000004" else ("P", "involved_in"))
            records.append("\t".join(["SYNTHETIC", gene, gene, qualifier, term, "SYNTHETIC:reference", evidence,
                                       "", aspect, "", "", "gene", "taxon:10090", "20261030", "SYNTHETIC", "", ""]))
        source("go", "go_gaf", text("synthetic.gaf", "!gaf-version: 2.2\n!generated-by: SYNTHETIC\n!date-generated: 2026-10-30T00:00:00Z\n" + "\n".join(records) + "\n"),
               "curated_annotation", ontology_path=ontology)
    elif family == "pathways":
        raw = "\n".join(f"{gene}\tSYNTHETIC-PATHWAY\thttps://example.org\tSynthetic pathway\tTAS\tMus musculus" for gene in gene_ids[:2]) + "\n"
        source("pathways", "reactome", text("reactome-format.tsv", raw), "curated_pathway")
        membership("complexes", [("ToyA", "SYNTHETIC-COMPLEX")], "curated_pathway")
    elif family == "interactions":
        raw = ("protein1 protein2 neighborhood fusion cooccurence coexpression coexpression_transferred combined_score\n"
               "10090.toy0 10090.external 900 700 0 999 999 999\n"
               "10090.external 10090.toy1 800 600 0 999 999 999\n")
        source("string", "string", text("string-format.txt", raw), "sequence", channels=["neighborhood", "fusion"])
        path = tsv("biogrid-format.tsv", ("Entrez Gene Interactor A", "Entrez Gene Interactor B", "Organism ID Interactor A",
                   "Organism ID Interactor B", "Experimental System Type", "Experimental System", "Pubmed ID"),
                   [("1", "2", "10090", "10090", "physical", "Affinity Capture-MS", "SYNTHETIC")])
        source("physical", "biogrid", path, "physical_interaction", allowed_experimental_systems=["Affinity Capture-MS"])
    elif family == "regulation":
        source("regulation", "trrust", text("trrust-format.tsv", "ToyA\tToyB\tActivation\tSYNTHETIC\nToyB\tToyA\tRepression\tSYNTHETIC\nToyNc\tToyB\tUnknown\tSYNTHETIC\n"), "curated_regulation")
        options = genome_files()
        path = folder / "motifs.json"
        write_json(path, {"SYNTHETIC-MOTIF": [[10, 0, 0, 0], [0, 10, 0, 0]]})
        source("motifs", "motifs", str(path), "sequence", **options, motif_resource="SYNTHETIC-PFM",
               motif_version="toy-v1", motif_url="https://example.org", score_threshold=1.)
    elif family == "evolution":
        for name in ("families", "orthogroups", "paralogs"):
            membership(name, [("ToyA", "SYNTHETIC-" + name), ("ToyB", "SYNTHETIC-" + name)], "sequence")
        path = tsv("conservation.tsv", ("gene_id", "synthetic_sequence_statistic"), [("ToyA", .8), ("ToyB", .6)])
        source("conservation", "numeric", path, "sequence")
    elif family in {"protein", "literature"}:
        # Explicit synthetic vectors exercise precomputed-file handling only.
        # They are not attributed to ESM or Sentence Transformers inference.
        path = tsv("synthetic-vectors.tsv", ("gene_id", "synthetic_dim_0", "synthetic_dim_1", "synthetic_dim_2"),
                   [("ToyA", 1, -.5, .2), ("ToyB", -.2, 1, .1)])
        if family == "protein":
            membership("domains", [("ToyA", "SYNTHETIC-DOMAIN-A"), ("ToyB", "SYNTHETIC-DOMAIN-B")], "sequence")
            inputs = text("exact-proteins.fa", ">ToyA\nACDEF\n>ToyB\nACDEFG\n")
            source("protein_vectors", "protein_embeddings", path, "sequence_only_model",
                   model="facebook/esm2_t6_8M_UR50D", model_revision="SYNTHETIC-NOT-A-MODEL-REVISION",
                   model_audit="Hand-authored SYNTHETIC vectors; ESM-2 was NOT executed. Model ID demonstrates adapter configuration only.",
                   inputs_path=inputs, pooling="SYNTHETIC-vectors-no-inference", isoform_rule="SYNTHETIC-one-sequence-per-gene")
        else:
            inputs = tsv("exact-descriptions.tsv", ("gene_id", "description", "reference"),
                         [("ToyA", "Synthetic description A for software demonstration only.", "SYNTHETIC"),
                          ("ToyB", "Synthetic description B for software demonstration only.", "SYNTHETIC")])
            source("text_vectors", "text_embeddings", path, "curated_description",
                   model="SYNTHETIC-no-text-model", model_revision="SYNTHETIC-v1",
                   model_audit="Hand-authored SYNTHETIC vectors; no embedding model was executed.",
                   inputs_path=inputs, pooling="SYNTHETIC-vectors-no-inference")
    elif family == "genomic":
        options = genome_files()
        source("genome", "genome", options["gtf_path"], "sequence", **options)
    elif family == "phenotypes":
        for name, context in (("knockout", "SYNTHETIC homozygous knockout"), ("perturbation", "SYNTHETIC genetic perturbation")):
            path = tsv(name + ".tsv", ("gene_id", "feature", "evidence", "association", "allele", "genotype", "context", "reference"),
                       [("ToyA", "SYNTHETIC-MP-1", "synthetic", "positive", "toy_allele", "toy_genotype", context, "SYNTHETIC"),
                        ("ToyB", "SYNTHETIC-MP-2", "synthetic", "positive", "toy_allele", "toy_genotype", context, "SYNTHETIC")])
            source(name, "membership", path, "genetic_phenotype", allowed_evidence=["synthetic"], context=context)
    elif family == "perturbation":
        path = tsv("fitness.tsv", ("gene_id", "synthetic_assay_1", "synthetic_assay_2", "synthetic_assay_3"),
                   [("ToyA", -1, .5, -.2), ("ToyB", 1, -.5, .2)])
        source("fitness", "numeric", path, "fitness", readout="fitness", assay="SYNTHETIC viability",
               context="SYNTHETIC mouse context; no embryonic claim", transform="center_resource")
        path = tsv("genetic.tsv", ("Entrez Gene Interactor A", "Entrez Gene Interactor B", "Organism ID Interactor A",
                   "Organism ID Interactor B", "Experimental System Type", "Experimental System", "Pubmed ID"),
                   [("1", "2", "10090", "10090", "genetic", "Synthetic Lethality", "SYNTHETIC")])
        source("genetic", "biogrid", path, "genetic_interaction", allowed_experimental_systems=["Synthetic Lethality"],
               assay="SYNTHETIC double perturbation", context="SYNTHETIC mouse context")
    else:
        raise ValueError(f"Unsupported family: {family}")
    config = dict(organism="10090", data_kind="synthetic_demonstration", cache_dir=str(folder / "cache"),
                  identifiers=dict(path=identifiers, version="SYNTHETIC-v1", url="https://example.org", license="Synthetic fixture"), sources=sources)
    return Path(genes), config


def _paths_absolute(value, base, allow_downloads):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"path", "cache_dir"} or key.endswith("_path"):
                if isinstance(item, dict):
                    if not allow_downloads:
                        raise ValueError("Remote download descriptor encountered; set ALLOW_DOWNLOADS=True explicitly")
                    result[key] = item
                else:
                    result[key] = str((base / Path(item).expanduser()).resolve())
            else:
                result[key] = _paths_absolute(item, base, allow_downloads)
        return result
    if isinstance(value, list):
        return [_paths_absolute(item, base, allow_downloads) for item in value]
    return value


def prepare_notebook(family, gene_list_path=None, config_path=None, output_root=None,
                     allow_synthetic=True, enable_inference=False, allow_downloads=False,
                     sample_path=None, gene_id_column=None):
    if family not in FAMILIES:
        raise ValueError(f"Unsupported family: {family}")
    root = Path(output_root or "prior_notebook_runs").expanduser().resolve()
    sample_provenance = None
    if sample_path is not None:
        if gene_list_path is not None or config_path is None:
            raise ValueError("Supply SAMPLE_PATH and CONFIG_PATH together, without a separate gene-list override")
        gene_list_path, sample_provenance = sample_gene_list(sample_path, root / "sample_inputs", gene_id_column)
    if bool(gene_list_path) != bool(config_path):
        raise ValueError("Supply both gene-list and config paths, or both sample and config paths; synthetic demo requires neither")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    session = root / family / stamp
    demo = gene_list_path is None
    if demo:
        if not allow_synthetic:
            raise ValueError("No real inputs supplied and synthetic demonstration is disabled")
        gene_list, config = synthetic_config(family, root / "synthetic_inputs" / family)
    else:
        gene_list = Path(gene_list_path).expanduser().resolve()
        original_path = Path(config_path).expanduser().resolve()
        # Explicit real inputs never silently fall back to a synthetic panel.
        config = json.loads(original_path.read_text())
    selected = [source for source in config.get("sources", []) if source.get("family") == family]
    disabled = [source["name"] for source in selected if source.get("kind") in INFERENCE_KINDS and not enable_inference]
    config["sources"] = [source for source in selected if source["name"] not in disabled]
    if not demo:
        # Unrelated families and disabled inference must not trigger downloads
        # or make their local resources prerequisites for this notebook.
        config = _paths_absolute(config, original_path.parent, allow_downloads)
        config.setdefault("cache_dir", str(original_path.parent / ".prior_cache"))
    if sample_provenance:
        if config.get("data_kind") == "synthetic_demonstration":
            raise ValueError("Do not apply synthetic prior fixtures to a real sample's gene list")
        config["input_gene_source"] = sample_provenance
    session.mkdir(parents=True, exist_ok=False)
    configuration = session / "family-config.json"
    write_json(configuration, config)
    synthetic = demo or config.get("data_kind") == "synthetic_demonstration"
    return dict(family=family, mode="SYNTHETIC DEMONSTRATION" if synthetic else "REAL CONFIGURED RESOURCES",
                genes=gene_list, config=config, config_path=configuration, session=session,
                output=session / "artifacts", disabled_sources=disabled, manifest=None, dependency_error=None,
                sample_provenance=sample_provenance)


def source_overview(context):
    columns = ("name", "kind", "species", "version", "url", "license", "evidence_basis")
    return pd.DataFrame([{key: spec.get(key, "") for key in columns} for spec in context["config"]["sources"]], columns=columns)


def input_preview(context, lines=8):
    """Bounded raw source previews retain evidence columns without loading resources."""
    result = {}
    for spec in context["config"]["sources"]:
        path = spec.get("path")
        if isinstance(path, str) and Path(path).is_file():
            with open_text(path) as handle:
                result[spec["name"]] = []
                for _, line in zip(range(lines), handle):
                    result[spec["name"]].append(line.rstrip("\n")[:1500])
        else:
            result[spec["name"]] = ["Unavailable locally or remote descriptor; construction will report access status."]
    return result


def execute_notebook(context):
    try:
        context["manifest"] = run(context["genes"], context["config_path"], context["output"])
    except (FileNotFoundError, ImportError) as error:
        context["dependency_error"] = str(error)
    manifest = context.get("manifest") or {}
    write_json(context["session"] / "notebook-report.json", {
        "mode": context["mode"], "family": context["family"], "disabled_inference_sources": context["disabled_sources"],
        "input_gene_source": context.get("sample_provenance"),
        "dependency_error": context["dependency_error"], "sources": manifest.get("sources", []),
        "populated_blocks": [entry["name"] for entry in manifest.get("blocks", []) if entry["status"] == "populated"],
        "real_biological_coverage_assessed": context["mode"] != "SYNTHETIC DEMONSTRATION" and bool(manifest.get("blocks")),
    })
    return context


def source_report(context):
    manifest = context.get("manifest") or {}
    rows = list(manifest.get("sources", []))
    rows.extend(dict(name=name, status="disabled_inference", reason="ENABLE_INFERENCE=False") for name in context["disabled_sources"])
    if context["dependency_error"]:
        rows.append(dict(name="required_input", status="unavailable", reason=context["dependency_error"]))
    if not rows:
        rows.append(dict(name=context["family"], status="not_configured", reason="No enabled sources for this family"))
    return pd.DataFrame(rows)


def read_block(context, entry):
    folder = context["output"] / entry["name"]
    return dict(entry=entry, folder=folder,
                features=sparse.load_npz(folder / "features.npz"), similarity=sparse.load_npz(folder / "similarity.npz"),
                rows=pd.read_csv(folder / "rows.tsv", sep="\t", keep_default_na=False),
                labels=pd.read_csv(folder / "features.tsv", sep="\t", keep_default_na=False),
                coverage=pd.read_csv(folder / "coverage.tsv", sep="\t", keep_default_na=False),
                provenance=json.loads((folder / "provenance.json").read_text()))


def block_summary(context):
    return pd.DataFrame([dict(name=entry["name"], genes=entry["shape"][0], dimensions=entry["shape"][1],
                              usable_genes=entry["usable_rows"], status=entry["status"])
                         for entry in (context.get("manifest") or {}).get("blocks", [])])


def validate_exports(context):
    manifest = context.get("manifest")
    if not manifest:
        return pd.DataFrame([dict(status="not_run", reason=context["dependency_error"] or "No manifest")])
    shared = pd.read_csv(context["output"] / "genes.tsv", sep="\t", keep_default_na=False)
    results = []
    for entry in manifest["blocks"]:
        block = read_block(context, entry)
        x, k, coverage = block["features"], block["similarity"], block["coverage"]
        assert block["rows"].equals(shared), f"{entry['name']}: row order mismatch"
        assert coverage["row_id"].tolist() == shared["row_id"].tolist()
        assert x.shape == (len(shared), len(block["labels"])) and k.shape == (len(shared), len(shared))
        assert block["labels"]["feature"].is_unique
        assert np.isfinite(x.data).all() and np.isfinite(k.data).all()
        delta = k - k.T
        assert not delta.nnz or np.max(np.abs(delta.data)) <= 1e-10
        missing = ~coverage["usable"].astype(str).str.lower().eq("true").to_numpy()
        assert not x[missing].nnz and not k[missing].nnz and not k[:, missing].nnz
        for filename, expected in entry["sha256"].items():
            assert digest(context["output"] / entry["files"][filename]) == expected, "Artifact checksum mismatch"
        validation = block["provenance"]["validation"]
        minimum = validation["smallest_kernel_eigenvalue"]
        if minimum is not None:
            assert minimum >= -1e-7 * max(1., float(k.diagonal().sum()))
        else:
            assert block["provenance"]["kernel_validation"]["psd_by_gram_construction"]
        results.append(dict(block=entry["name"], order_dimensions_finite_symmetry_missingness_hashes="pass",
                            minimum_eigenvalue=minimum, psd_check=block["provenance"].get("kernel_validation", {}).get("method", "full eigenspectrum")))
    return pd.DataFrame(results)


def covered_genes(block):
    """Usable feature rows, retaining the original input order and positional IDs."""
    coverage = block["coverage"]
    mask = coverage["usable"].astype(str).str.lower().eq("true")
    result = coverage.loc[mask, ["row_id", "input_id", "canonical_id", "nonzero_features"]].copy()
    result.insert(0, "input_row", np.flatnonzero(mask.to_numpy()))
    return result.reset_index(drop=True)


def similar_covered_genes(block, query=None, n_queries=3, neighbors=5):
    """Inspect profile similarity among covered genes; not a selection algorithm."""
    columns = ["query_row_id", "query_gene", "neighbor_row_id", "neighbor_gene", "similarity"]
    covered = covered_genes(block)
    selected = covered if query is None else covered.loc[
        covered["input_id"].eq(query) | covered["canonical_id"].eq(query) | covered["row_id"].eq(query)]
    results = []
    for _, row in selected.iloc[:n_queries].iterrows():
        candidates = covered.loc[~covered["canonical_id"].eq(row["canonical_id"])]
        positions = candidates["input_row"].to_numpy()
        scores = block["similarity"][int(row["input_row"]), positions].toarray().ravel()
        order = np.lexsort((positions, -scores))[:neighbors]
        for i in order:
            other = candidates.iloc[i]
            results.append(dict(query_row_id=row["row_id"], query_gene=row["input_id"],
                                neighbor_row_id=other["row_id"], neighbor_gene=other["input_id"],
                                similarity=float(scores[i])))
    return pd.DataFrame(results, columns=columns)


@lru_cache(maxsize=2)
def _go_names(ontology_path, cache_dir, mtime_ns):
    from goatools.obo_parser import GODag
    from .sources import _goatools_input
    dag = GODag(_goatools_input(ontology_path, cache_dir, ".obo"), prt=None)
    return {key: node.name for key, node in dag.items()}


def covered_go_annotations(block, query=None, n_genes=8, max_terms=12):
    """Explain nonzero GO features of covered sample genes using GOATOOLS names."""
    spec = block["provenance"]["source"]
    ontology = block["provenance"].get("artifacts", {}).get("ontology_path", {}).get("path", spec["ontology_path"])
    names = _go_names(ontology, str(block["folder"].parent.parent / "inspection_cache"), Path(ontology).stat().st_mtime_ns)
    covered = covered_genes(block)
    if query is not None:
        covered = covered.loc[covered["input_id"].eq(query) | covered["canonical_id"].eq(query) | covered["row_id"].eq(query)]
    records = []
    for _, gene in covered.iloc[:n_genes].iterrows():
        vector = block["features"].getrow(int(gene["input_row"]))
        for column, value in sorted(zip(vector.indices, vector.data))[:max_terms]:
            term = block["labels"]["feature"].iloc[column]
            records.append(dict(row_id=gene["row_id"], gene=gene["input_id"], canonical_id=gene["canonical_id"],
                                term=term, term_name=names.get(term, ""), feature_value=float(value)))
    return pd.DataFrame(records, columns=["row_id", "gene", "canonical_id", "term", "term_name", "feature_value"])


def feature_preview(block, n_rows=8, n_features=10, covered_only=True):
    # Densify only a bounded preview; retain full sparse matrices on disk.
    positions = covered_genes(block)["input_row"].to_numpy()[:n_rows] if covered_only else np.arange(min(n_rows, len(block["rows"])))
    rows = block["rows"].iloc[positions]
    selected = block["features"][positions]
    # Show actual nonzero features of these genes, rather than arbitrary columns of zeros.
    columns = np.flatnonzero(np.asarray(selected.getnnz(axis=0)).ravel())[:n_features]
    labels = block["labels"]["feature"].iloc[columns].tolist()
    index = rows["row_id"] + ": " + rows["input_id"]
    return pd.DataFrame(selected[:, columns].toarray(), index=index, columns=labels)


def plot_block(block, max_genes=20, covered_only=True):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    covered = covered_genes(block)
    if covered.empty:
        for axis in axes:
            axis.text(.5, .5, "No usable genes in this block", ha="center", va="center", transform=axis.transAxes)
        plt.show(); plt.close(fig)
        return
    axes[0].hist(covered["nonzero_features"], bins=min(20, max(1, covered["nonzero_features"].nunique())))
    axes[0].set(title=f"Features of {len(covered):,} covered input genes", xlabel="Nonzero features per gene", ylabel="Input rows")
    positions = covered["input_row"].to_numpy()[:max_genes] if covered_only else np.arange(min(max_genes, len(block["rows"])))
    n = len(positions)
    labels = (block["rows"]["row_id"] + ": " + block["rows"]["input_id"]).iloc[positions]
    artist = axes[1].imshow(block["similarity"][positions][:, positions].toarray(), vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(range(n), labels, rotation=90)
    axes[1].set_yticks(range(n), labels)
    axes[1].set_title(f"Separate kernel (first {n} {'covered ' if covered_only else ''}input rows)")
    fig.colorbar(artist, ax=axes[1], label="Biological similarity")
    fig.suptitle(block["entry"]["name"])
    plt.show()
    plt.close(fig)


def plot_go_clustermap(blocks, representation="binary", source=None, max_genes=None, output_dir=None):
    """One inspection map of all three GO aspects, using package clustering.

    Concatenate term features of one representation/source, then compute cosine
    similarity. Binary and IC are alternatives, never counted together. The
    same average-linkage tree from 1-cosine orders both axes. Exported pipeline
    blocks are not changed; rows lacking every GO feature are excluded.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics.pairwise import cosine_similarity

    if representation not in {"binary", "ic"}:
        raise ValueError("GO representation must be binary or ic")
    if max_genes is not None and (isinstance(max_genes, bool) or not isinstance(max_genes, int) or max_genes < 2):
        raise ValueError("max_genes must be None or an integer >= 2")
    candidates = [b for b in blocks if b["provenance"]["source"]["family"] == "go"
                  and b["entry"]["name"].endswith("." + representation)]
    sources = sorted({b["provenance"]["source"]["name"] for b in candidates})
    if not sources:
        print("No GO blocks available for a clustermap.")
        return None
    if source is None:
        if len(sources) != 1:
            raise ValueError(f"Choose one GO source for the clustermap: {sources}")
        source = sources[0]
    selected = [b for b in candidates if b["provenance"]["source"]["name"] == source]
    aspects = ("biological_process", "molecular_function", "cellular_component")
    expected = [f"{source}.{aspect}.{representation}" for aspect in aspects]
    by_name = {b["entry"]["name"]: b for b in selected}
    if set(by_name) != set(expected) or len(selected) != len(expected):
        raise ValueError("The GO clustermap needs BP, MF and CC blocks from the selected source")
    selected = [by_name[name] for name in expected]
    rows = selected[0]["rows"]
    if any(not b["rows"].equals(rows) for b in selected[1:]):
        raise ValueError("GO blocks have different gene rows/order")
    features = sparse.hstack([b["features"] for b in selected], format="csr")
    positions = np.flatnonzero(np.asarray(features.getnnz(axis=1)).ravel())
    total = len(positions)
    if max_genes is not None:
        positions = positions[:max_genes]
    n = len(positions)
    print(f"GO {representation}: plotting {n:,} of {total:,} annotated input rows; "
          f"{len(rows) - total:,} rows without usable GO features excluded.")
    if n < 2:
        print("At least two annotated input rows are needed for a clustermap.")
        return None
    # Densify the gene-gene matrix only; term features remain sparse.
    similarity = cosine_similarity(features[positions])
    np.clip(similarity, 0., 1., out=similarity)
    np.fill_diagonal(similarity, 1.)
    distances = squareform(similarity, checks=False)
    np.subtract(1., distances, out=distances)
    tree = linkage(distances, method="average", optimal_ordering=False)
    del distances
    genes = rows.iloc[positions].copy()
    genes.insert(0, "input_row", positions)
    labels = (genes["row_id"] + ": " + genes["input_id"]).tolist()
    grid = sns.clustermap(pd.DataFrame(similarity, index=labels, columns=labels),
                          row_linkage=tree, col_linkage=tree, cmap="viridis", vmin=0, vmax=1,
                          figsize=(14, 14), xticklabels=n <= 60, yticklabels=n <= 60,
                          cbar_kws={"label": "GO cosine similarity"}, rasterized=True)
    grid.fig.suptitle(f"Gene Ontology gene–gene relationships ({representation}; BP + MF + CC; n={n:,})", y=1.02)
    grid.ax_heatmap.set(xlabel="Genes", ylabel="Genes")
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        prefix = folder / f"go.{representation}.clustermap"
        grid.savefig(str(prefix) + ".png", dpi=200)
        ordered = genes.iloc[grid.dendrogram_row.reordered_ind].copy()
        ordered.insert(0, "cluster_order", np.arange(n))
        ordered.to_csv(str(prefix) + ".genes.tsv", sep="\t", index=False)
        np.save(str(prefix) + ".linkage.npy", tree, allow_pickle=False)
        write_json(str(prefix) + ".json", dict(source=source, blocks=expected, representation=representation,
                   construction="concatenate GO term features across BP/MF/CC; cosine similarity",
                   clustering="scipy average linkage on 1 - cosine; identical order on both axes",
                   annotated_input_rows=total, plotted_input_rows=n, max_genes=max_genes,
                   excluded_unannotated_rows=len(rows) - total,
                   row_policy="retain duplicate input rows; optional limit takes first annotated input rows"))
        print("Clustermap and clustered gene order:", prefix)
    plt.show()
    plt.close(grid.fig)
    return grid
