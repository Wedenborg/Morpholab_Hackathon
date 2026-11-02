"""Audited file adapters. Construct representations on the full supplied resource."""

from __future__ import annotations

import csv
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import shutil
import tempfile

import numpy as np
from scipy import sparse

from .core import digest, make_block, open_text, table


KIND_FAMILY = {
    "go_gaf": {"go"}, "reactome": {"pathways"}, "membership": {"pathways", "evolution", "protein", "phenotypes"},
    "string": {"interactions"}, "biogrid": {"interactions", "perturbation"},
    "regulatory_tsv": {"regulation"}, "trrust": {"regulation"},
    "numeric": {"evolution", "perturbation"}, "protein_embeddings": {"protein"},
    "text_embeddings": {"literature"}, "protein_esm": {"protein"},
    "descriptions": {"literature"}, "genome": {"genomic"}, "motifs": {"regulation"},
}

KIND_BASES = {
    "go_gaf": {"curated_annotation"}, "reactome": {"curated_pathway"},
    "membership": {"curated_pathway", "curated_annotation", "sequence", "genetic_phenotype"},
    "string": {"sequence", "physical_interaction", "curated_annotation"},
    "biogrid": {"physical_interaction", "genetic_interaction"},
    "regulatory_tsv": {"curated_regulation"}, "trrust": {"curated_regulation"},
    "numeric": {"sequence", "fitness", "viability", "genetic_interaction"},
    "protein_embeddings": {"sequence_only_model"}, "protein_esm": {"sequence_only_model"},
    "text_embeddings": {"curated_description"}, "descriptions": {"curated_description"},
    "genome": {"sequence"}, "motifs": {"sequence"},
}


class SourceMapper:
    def __init__(self, spec, ids):
        self.ids = ids
        self.transfers = None
        self.unmapped = set()
        if str(spec["species"]) != "10090":
            self.transfers = defaultdict(set)
            for row in table(spec["transfer"]["mapping_path"], ("source_id", "mouse_id", "confidence", "orthology_type")):
                if not row["confidence"] or not row["orthology_type"]:
                    raise ValueError("Orthology mapping needs confidence and relationship type for every row")
                gid = ids.resolve(row["mouse_id"])
                if gid:
                    self.transfers[row["source_id"]].add(gid)

    def __call__(self, identifier):
        if self.transfers is not None:
            result = sorted(self.transfers.get(identifier, ()))
        else:
            gene = self.ids.resolve(identifier)
            result = [gene] if gene else []
        if not result:
            self.unmapped.add(identifier)
        return result


def membership_block(spec, ids, provenance, records, suffix="memberships"):
    """records: (gene, feature, value); every gene is canonical already."""
    labels = sorted({feature for _, feature, _ in records})
    index = {label: i for i, label in enumerate(labels)}
    entries = defaultdict(dict)
    for gene, feature, value in records:
        # Memberships are idempotent: repeated evidence must not inflate values.
        old = entries[gene].get(index[feature])
        if old is not None and old != value:
            raise ValueError(f"Conflicting membership values: {gene}, {feature}")
        entries[gene][index[feature]] = value
    values = {gene: sparse.csr_matrix((list(items.values()), ([0] * len(items), list(items))), shape=(1, len(labels)))
              for gene, items in entries.items()}
    return make_block(f"{spec['name']}.{suffix}", values, labels, ids,
                      dict(provenance, construction="binary memberships; no cross-block normalization"))


def memberships(spec, ids, provenance, mapper):
    rows = table(spec["path"], ("gene_id", "feature", "evidence", "association"))
    records, known_zero = [], set()
    admitted = set(spec.get("allowed_evidence", []))
    if not admitted:
        raise ValueError("membership requires an explicit allowed_evidence list")
    if spec["family"] == "protein" and spec["evidence_basis"] != "sequence":
        raise ValueError("Domain annotations must be sequence-derived")
    for row in rows:
        if row["evidence"] not in admitted:
            continue
        if row["association"] not in {"positive", "known_none"}:
            raise ValueError("association must be positive or known_none; blank is not evidence of absence")
        for gene in mapper(row["gene_id"]):
            if spec["family"] == "protein" and ids.records[gene]["biotype"] != "protein_coding":
                raise ValueError("Protein domains supplied for a noncoding gene")
            if row["association"] == "known_none":
                known_zero.add(gene)
            elif row["feature"]:
                records.append((gene, row["feature"], 1.))
            else:
                raise ValueError("Positive membership needs a feature")
    block = membership_block(spec, ids, provenance, records)
    if known_zero & {gene for gene, _, _ in records}:
        raise ValueError("Gene has both positive memberships and a resource-wide known_none assertion")
    for row in block.coverage:
        if row["canonical_id"] in known_zero:
            row.update(status="known_no_association", observed=True)
    return [block]


def _goatools_input(path, cache_dir, suffix):
    """GOATOOLS readers expect plain files; reuse cached decompression of .gz inputs."""
    if not str(path).endswith(".gz"):
        return str(path)
    cache = Path(cache_dir) / "goatools"
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (digest(path) + suffix)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with open_text(path) as handle:
                    shutil.copyfileobj(handle, temporary)
                temporary.close()
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
    return str(destination)


def go(spec, ids, provenance, mapper):
    from goatools.anno.gaf_reader import GafReader
    from goatools.anno.init.reader_gaf import GafHdr
    from goatools.godag.go_tasks import get_go2ancestors
    from goatools.obo_parser import GODag

    roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
    allowed = set(spec.get("allowed_evidence", ["IDA", "IPI", "IMP", "IGI", "TAS", "IC"]))
    if allowed & {"IEP", "HEP"}:
        raise ValueError("GO expression-pattern evidence IEP/HEP is prohibited")
    if allowed - {"IDA", "IPI", "IMP", "IGI", "TAS", "IC"} and not spec.get("additional_evidence_audit"):
        raise ValueError("Additional GO evidence requires additional_evidence_audit of upstream derivation")
    ontology = _goatools_input(spec["ontology_path"], spec["cache_dir"], ".obo")
    gaf_path = _goatools_input(spec["path"], spec["cache_dir"], ".gaf")
    dag = GODag(ontology, optional_attrs={"relationship"}, load_obsolete=True, prt=None)
    terms = {term.id: term for term in dag.values() if not term.is_obsolete}
    # GOATOOLS includes is_a automatically and only the selected optional relationships.
    ancestor_map = get_go2ancestors(terms.values(), relationships={"part_of"})
    gaf_header, data_rows = [], 0
    with open_text(gaf_path) as handle:
        for line in handle:
            if line.startswith("!"):
                gaf_header.append(line.rstrip("\n"))
            elif line.strip():
                data_rows += 1
    version_headers = [line for line in gaf_header if line.startswith("!gaf-version:")]
    if len(version_headers) != 1:
        raise ValueError("GAF must declare exactly one !gaf-version header")
    header = GafHdr(version=version_headers[0].split(":", 1)[1].strip())
    for line in gaf_header:
        header.chkaddhdr(line)
    if not header.validate():
        raise ValueError("GOATOOLS GAF header validation failed; GAF 2.2 requires generated-by and date-generated")
    try:
        reader = GafReader(gaf_path, prt=None)
    except SystemExit as error:
        raise ValueError("GOATOOLS rejected the GAF; inspect its version, fields and formatting") from error
    if len(reader.associations) != data_rows:
        raise ValueError("GOATOOLS omitted GAF records; fix malformed rows instead of silently dropping evidence")

    annotations = defaultdict(set)
    evidence, excluded = [], []
    observed = defaultdict(set)
    ns_names = {"BP": "biological_process", "MF": "molecular_function", "CC": "cellular_component"}
    for row in reader.associations:
        if not row.Taxon or str(row.Taxon[0]) != "10090":
            continue
        node = dag.get(row.GO_ID)  # GODag resolves alternative IDs to canonical records.
        aspect = node.namespace if node is not None else ns_names.get(row.NS)
        if node is not None and ns_names.get(row.NS) != node.namespace:
            raise ValueError(f"GAF aspect disagrees with ontology namespace: {row.GO_ID}")
        for gene in mapper(row.DB_ID):
            record = {"gene": gene, "term": row.GO_ID, "canonical_term": node.id if node else None,
                      "qualifier": "|".join(sorted(row.Qualifier)), "evidence": row.Evidence_Code,
                      "reference": "|".join(sorted(row.DB_Reference)), "with_from": "|".join(sorted(row.With_From)),
                      "date": row.Date.strftime("%Y%m%d"), "assigned_by": row.Assigned_By}
            if aspect:
                observed[aspect].add(gene)
            if "NOT" in row.Qualifier or row.Evidence_Code not in allowed:
                excluded.append(dict(record, reason="negated_or_excluded_evidence"))
                continue
            if node is None or node.is_obsolete:
                excluded.append(dict(record, reason="unknown_or_obsolete_term"))
                continue
            propagated = ({node.id} | ancestor_map.get(node.id, set())) & terms.keys()
            annotations[gene] |= propagated - roots
            evidence.append(record)
    blocks = []
    for aspect in ("biological_process", "molecular_function", "cellular_component"):
        records = [(gene, term, 1.) for gene, terms_for_gene in annotations.items() for term in terms_for_gene
                   if terms[term].namespace == aspect]
        p = dict(provenance, admitted_evidence=evidence, excluded_annotations=excluded, allowed_evidence=sorted(allowed),
                 ontology_backend={"package": "goatools", "version": version("goatools"),
                                   "ontology_parser": "goatools.obo_parser.GODag",
                                   "annotation_parser": "goatools.anno.gaf_reader.GafReader",
                                   "ancestor_method": "goatools.godag.go_tasks.get_go2ancestors",
                                   "optional_relationships": ["part_of"], "ontology_data_version": dag.data_version},
                 gaf_header=gaf_header, parsed_gaf_records=len(reader.associations),
                 parsed_inputs={"ontology_sha256": digest(ontology), "gaf_sha256": digest(gaf_path)},
                 propagation="is_a and part_of ancestors; exclude three GO roots; obsolete terms omitted",
                 annotation_universe="all mapped mouse genes in the supplied full GAF, not input panel")
        block = membership_block(spec, ids, p, records, aspect + ".binary")
        for row in block.coverage:
            if row["canonical_id"] in observed[aspect] and not row["usable"]:
                row.update(observed=True, status="observed_no_usable_annotations")
        blocks.append(block)
        # IC uses annotation frequencies over the resource universe after filtering.
        genes = sorted(annotations)
        counts = defaultdict(set)
        for gene, term, _ in records:
            counts[term].add(gene)
        weights = {term: -np.log((len(counts[term]) + 1) / (len(genes) + 1)) for term in block.features}
        weighted_records = [(gene, term, float(np.sqrt(weights[term]))) for gene, term, _ in records]
        weighted = membership_block(spec, ids, dict(p, weighting="sqrt(-log((annotated_genes+1)/(universe_genes+1)))",
                                                   information_content=weights), weighted_records, aspect + ".ic")
        # Membership construction's default text is replaced with the precise transform.
        weighted.provenance["construction"] = "propagated memberships times square-root annotation IC; cosine Gram"
        for row in weighted.coverage:
            if row["canonical_id"] in observed[aspect] and not row["usable"]:
                row.update(observed=True, status="observed_no_usable_annotations_or_zero_ic")
        blocks.append(weighted)
    return blocks


def reactome(spec, ids, provenance, mapper):
    records = []
    with open_text(spec["path"]) as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) != 6:
                raise ValueError("Reactome Ensembl mapping must have six columns and no header")
            if row[5] != "Mus musculus":
                continue
            for gene in mapper(row[0]):
                records.append((gene, row[1], 1.))
    return [membership_block(spec, ids, dict(provenance,
            limitation="Mouse Reactome pathways may be orthology-inferred; not all mouse memberships are experimentally curated",
            membership_policy="mouse rows only; raw six-column mapping retains pathway names, evidence, URL and species"), records)]


def graph_blocks(spec, ids, provenance, edges):
    """Two-step neighborhoods on the complete external graph; one block per channel."""
    from sklearn.preprocessing import normalize
    alpha = float(spec.get("two_hop_weight", .5))
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("two_hop_weight must be finite and nonnegative")
    blocks = []
    for channel, channel_edges in sorted(edges.items()):
        nodes = sorted({node for a, b, _ in channel_edges for node in (a, b)})
        ix = {node: i for i, node in enumerate(nodes)}
        weights = {}
        for a, b, value in channel_edges:
            weights[a, b] = max(value, weights.get((a, b), 0.))
            weights[b, a] = max(value, weights.get((b, a), 0.))
        rr, cc, vv = [], [], []
        for (a, b), value in weights.items():
            rr.append(ix[a]); cc.append(ix[b]); vv.append(value)
        adjacency = sparse.csr_matrix((vv, (rr, cc)), shape=(len(nodes), len(nodes)))
        transition = normalize(adjacency, norm="l1", axis=1) if nodes else adjacency
        # Subset rows first, but multiply against the full graph to retain external paths.
        selected = sorted(set(ids.genes) & set(nodes))
        first = transition[[ix[gene] for gene in selected]]
        profiles = (first + alpha * (first @ transition)).tocsr()
        values = {gene: profiles[i] for i, gene in enumerate(selected)}
        blocks.append(make_block(f"{spec['name']}.{channel}", values, [f"neighbor:{node}" for node in nodes], ids,
                                 dict(provenance, channel=channel, directed=False,
                                      construction="P + alpha P^2, row stochastic full external graph; then input rows",
                                      two_hop_weight=float(spec.get("two_hop_weight", .5)),
                                      interpretation="neighborhood-profile similarity; neither causal edges nor expression correlation")))
    return blocks


def string(spec, ids, provenance, mapper):
    # Only sequence/genome-derived channels are enabled without a channel-specific upstream review.
    safe = {"neighborhood", "neighborhood_transferred", "fusion", "cooccurence", "homology"}
    reviewable = {"experiments", "experiments_transferred", "database", "database_transferred", "textmining", "textmining_transferred"}
    channels = spec.get("channels", ["neighborhood", "fusion", "cooccurence"])
    reviews = spec.get("channel_audits", {})
    if not channels or len(channels) != len(set(channels)):
        raise ValueError("STRING requires a unique nonempty channel list")
    for channel in channels:
        if channel not in safe | reviewable:
            raise ValueError(f"Prohibited/unreviewed STRING channel: {channel}; combined and coexpression scores are forbidden")
        if channel in reviewable and not reviews.get(channel):
            raise ValueError(f"STRING {channel} requires review of contributing databases and measurement types")
    edges = {channel: [] for channel in channels}
    threshold = float(spec.get("min_score", 1))
    if not 0 < threshold <= 1000:
        raise ValueError("STRING min_score must be in (0,1000]")
    with open_text(spec["path"]) as handle:
        header = next(handle).split()
        if not {"protein1", "protein2", *channels} <= set(header):
            raise ValueError("Require STRING full links download with requested individual channels")
        for line in handle:
            fields = line.split()
            if len(fields) != len(header):
                raise ValueError("Malformed STRING full-links row")
            row = dict(zip(header, fields))
            if not all(row[key].startswith("10090.") for key in ("protein1", "protein2")):
                continue
            # Retain unmapped external mouse proteins as graph nodes, so two-hop
            # neighborhoods do not collapse to the panel or identifier-table subset.
            left = mapper(row["protein1"]) or ["STRING:" + row["protein1"]]
            right = mapper(row["protein2"]) or ["STRING:" + row["protein2"]]
            for channel in channels:
                score = float(row[channel])
                if not np.isfinite(score) or not 0 <= score <= 1000:
                    raise ValueError("Invalid STRING channel score")
                if score >= threshold:
                    edges[channel].extend((a, b, score / 1000) for a in left for b in right)
    return graph_blocks(spec, ids, dict(provenance, excluded_channels=["coexpression", "coexpression_transferred", "combined_score"]), edges)


def biogrid(spec, ids, provenance, mapper):
    required = ("Entrez Gene Interactor A", "Entrez Gene Interactor B", "Organism ID Interactor A", "Organism ID Interactor B",
                "Experimental System Type", "Experimental System", "Pubmed ID")
    rows = table(spec["path"], required)
    allowed = set(spec.get("allowed_experimental_systems", []))
    if not allowed:
        raise ValueError("BioGRID requires allowed_experimental_systems after readout review")
    prohibited = {"Co-expression", "Coexpression", "Expression", "Phenotypic Enhancement (Expression)", "Phenotypic Suppression (Expression)"}
    if any(assay in prohibited or "expression" in assay.lower() for assay in allowed):
        raise ValueError("Expression-derived BioGRID experimental systems are prohibited")
    edges, evidence = defaultdict(list), []
    expected = str(spec["species"])
    for row in rows:
        if row["Organism ID Interactor A"] != expected or row["Organism ID Interactor B"] != expected:
            continue
        category = row["Experimental System Type"].lower()
        if category not in {"physical", "genetic"} or row["Experimental System"] not in allowed:
            continue
        expected_category = "physical" if spec["evidence_basis"] == "physical_interaction" else "genetic"
        if category != expected_category:
            continue
        if spec["family"] == "perturbation" and category != "genetic":
            continue
        if spec["family"] == "interactions" and category == "genetic" and spec["evidence_basis"] != "genetic_interaction":
            # A separate source declaration prevents physical/genetic evidence conflation.
            continue
        left = mapper(row["Entrez Gene Interactor A"])
        right = mapper(row["Entrez Gene Interactor B"])
        if str(spec["species"]) == "10090":
            left = left or ["NCBI:" + row["Entrez Gene Interactor A"]]
            right = right or ["NCBI:" + row["Entrez Gene Interactor B"]]
        for a in left:
            for b in right:
                edges[category + "." + row["Experimental System"]].append((a, b, 1.))
                evidence.append({"source": a, "target": b, "type": category, "assay": row["Experimental System"], "reference": row["Pubmed ID"]})
    return graph_blocks(spec, ids, dict(provenance, admitted_evidence=evidence), edges)


def regulation(spec, ids, provenance, mapper):
    if spec["kind"] == "trrust":
        rows = []
        with open_text(spec["path"]) as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) != 4:
                    raise ValueError("TRRUST must have four columns, without a header")
                rows.append(dict(zip(("regulator", "target", "sign", "reference"), row), evidence="literature_curated"))
    else:
        rows = table(spec["path"], ("regulator", "target", "sign", "evidence", "reference"))
    allowed = set(spec.get("allowed_evidence", ["literature_curated"]))
    records, evidence = defaultdict(list), []
    for row in rows:
        if row["evidence"] not in allowed:
            continue
        sign = {"Activation": "activation", "Repression": "repression", "Unknown": "unknown",
                "activation": "activation", "repression": "repression", "unknown": "unknown", "": "unknown"}.get(row["sign"])
        if sign is None:
            raise ValueError("Regulatory sign must be activation, repression, or unknown")
        for regulator in mapper(row["regulator"]):
            for target in mapper(row["target"]):
                # Separate sign/evidence blocks keep unknown signs distinct from zero/activation.
                channel = sign + "." + row["evidence"]
                records["incoming." + channel].append((target, regulator, 1.))
                records["outgoing." + channel].append((regulator, target, 1.))
                evidence.append(dict(row, mouse_regulator=regulator, mouse_target=target, normalized_sign=sign))
    return [membership_block(spec, ids, dict(provenance, admitted_evidence=evidence,
             direction=channel.split(".")[0], sign=channel.split(".")[1],
             interpretation="symmetric similarity of directed role profiles; not the original directed regulatory network"), items, channel)
            for channel, items in sorted(records.items())]


def numeric(spec, ids, provenance, mapper):
    rows = table(spec["path"], ("gene_id",))
    if not rows:
        return []
    labels = [label for label in rows[0] if label != "gene_id"]
    if not labels:
        raise ValueError("Numeric input requires at least one labeled dimension")
    if spec["family"] == "perturbation":
        for key in ("assay", "context", "readout"):
            if not spec.get(key):
                raise ValueError(f"Perturbation input needs {key}")
        if spec["readout"] not in {"fitness", "viability", "genetic_interaction"}:
            raise ValueError("Only non-expression perturbation readouts are allowed")
        if spec["readout"] != spec["evidence_basis"]:
            raise ValueError("Perturbation readout and evidence_basis must agree")
    if spec["kind"] in {"protein_embeddings", "text_embeddings"}:
        for key in ("model", "model_revision", "pooling", "model_audit", "inputs_path"):
            if not spec.get(key):
                raise ValueError(f"Precomputed embeddings need {key}")
        if spec["kind"] == "protein_embeddings" and spec["evidence_basis"] != "sequence_only_model":
            raise ValueError("Protein model must be pretrained solely on sequence")
        if spec["kind"] == "protein_embeddings" and not spec.get("isoform_rule"):
            raise ValueError("Protein embeddings need isoform_rule")
        if spec["model_revision"] in {"main", "latest", "master"}:
            raise ValueError("Precomputed embeddings require an immutable model revision")
        if spec["kind"] == "protein_embeddings" and not spec["model"].startswith("facebook/esm2_"):
            raise ValueError("Precomputed protein adapter supports audited official ESM-2 embeddings only")
    values, rejected = {}, []
    for row in rows:
        # Strict complete-case rows: partial missingness is reported, never imputed as biology.
        if any(row[label] in {"", "NA", "NaN", "nan"} for label in labels):
            rejected.append(row["gene_id"])
            continue
        value = np.array([float(row[label]) for label in labels])
        if not np.isfinite(value).all():
            raise ValueError("Numeric values must be finite")
        for gene in mapper(row["gene_id"]):
            if spec["kind"] == "protein_embeddings" and ids.records[gene]["biotype"] != "protein_coding":
                raise ValueError("Protein embedding supplied for noncoding gene")
            if gene in values and not np.array_equal(values[gene], value):
                raise ValueError(f"Multiple conflicting numeric profiles for {gene}; explicit upstream pooling required")
            values[gene] = value
    transform = spec.get("transform", "none")
    if transform == "center_resource" and values:
        mean = np.mean(list(values.values()), axis=0)
        values = {gene: vector - mean for gene, vector in values.items()}
        provenance = dict(provenance, centering_mean=mean.tolist())
    elif transform != "none":
        raise ValueError("Numeric transform must be none or center_resource")
    return [make_block(spec["name"] + ".features", values, labels, ids,
                       dict(provenance, construction=transform + "; cosine Gram",
                            incomplete_rows_omitted=rejected))]


LOADERS = {"membership": memberships, "go_gaf": go, "reactome": reactome,
           "string": string, "biogrid": biogrid, "trrust": regulation,
           "regulatory_tsv": regulation, "numeric": numeric,
           "protein_embeddings": numeric, "text_embeddings": numeric}
