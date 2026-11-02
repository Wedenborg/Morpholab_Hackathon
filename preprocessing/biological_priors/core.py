"""Identifier resolution, provenance, sparse feature blocks, and cosine kernels."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse


FAMILIES = (
    "go", "pathways", "interactions", "regulation", "evolution", "protein",
    "genomic", "literature", "phenotypes", "perturbation",
)

# This is an allowlist of readouts, not a claim that an entire database is safe.
BASES = {
    "curated_annotation", "curated_pathway", "physical_interaction",
    "curated_regulation", "sequence", "sequence_only_model", "curated_description",
    "genetic_phenotype", "fitness", "viability", "genetic_interaction",
}

FAMILY_BASES = {
    "go": {"curated_annotation"},
    "pathways": {"curated_pathway", "curated_annotation"},
    "interactions": {"sequence", "curated_annotation", "physical_interaction", "genetic_interaction"},
    "regulation": {"sequence", "curated_regulation"},
    "evolution": {"sequence", "curated_annotation"},
    "protein": {"sequence", "sequence_only_model"},
    "genomic": {"sequence"}, "literature": {"curated_description"},
    "phenotypes": {"genetic_phenotype"},
    "perturbation": {"fitness", "viability", "genetic_interaction"},
}


def aggregate_features(feature_blocks):
    pass


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else Path(path).open(encoding="utf-8")


def table(path, required=()):
    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not set(required) <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: required TSV columns: {sorted(required)}")
        fields = reader.fieldnames or []
        if not fields or any(not field for field in fields) or len(fields) != len(set(fields)):
            raise ValueError(f"{path}: TSV headers must be unique and nonempty")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError(f"{path}: TSV row width does not match its header")
        return rows


def write_table(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def audit(spec):
    """Require a review of upstream components, not just a database name."""
    for key in ("name", "family", "kind", "version", "url", "species", "license", "evidence_basis", "audit"):
        if key not in spec or not spec[key]:
            raise ValueError(f"Source requires nonempty {key!r}")
    if spec["family"] not in FAMILIES:
        raise ValueError("Unsupported biological-prior family")
    if spec["evidence_basis"] not in BASES:
        raise ValueError("Evidence basis is not permitted")
    if spec["evidence_basis"] not in FAMILY_BASES[spec["family"]]:
        raise ValueError("Evidence basis is incompatible with the biological-prior family")
    review = spec["audit"]
    if review.get("expression_independent") is not True:
        raise ValueError("Source must have an explicit expression-independence review")
    for key in ("upstream_components", "evidence_filters", "justification"):
        if not review.get(key):
            raise ValueError(f"Source audit requires {key}")
    if review.get("expression_supervised", False):
        raise ValueError("Expression-supervised models are prohibited")
    model = str(spec.get("model", "")).lower()
    prohibited_models = ("gene2vec", "sivae", "cofast", "hotspot", "geneformer", "scgpt",
                         "genecompass", "scprint", "genie3", "scenic", "enformer")
    if any(name in model for name in prohibited_models):
        raise ValueError("Prohibited expression-derived model or inference method")
    if model and not spec.get("model_audit"):
        raise ValueError("Every pretrained model requires model_audit of training inputs/objective")
    if str(spec["species"]) != "10090":
        transfer = spec.get("transfer", {})
        for key in ("mapping_path", "orthology_confidence", "context", "one_to_many_rule"):
            if not transfer.get(key):
                raise ValueError(f"Non-mouse source requires explicit transfer.{key}")
        if transfer["one_to_many_rule"] != "retain_all":
            raise ValueError("Only explicit retain_all orthology transfer is supported")


class Identifiers:
    """Native mouse IDs/current symbols precede synonyms; record every fallback."""

    def __init__(self, path, inputs):
        records = table(path, ("ensembl_gene_id", "symbol", "taxon_id", "biotype"))
        self.records = {}
        self.aliases = {}
        self.symbols = {}
        self.identifiers = {}
        self.folded_symbols = {}
        self.folded_aliases = {}
        for record in records:
            if record["taxon_id"] != "10090":
                continue
            gid = self.key(record["ensembl_gene_id"])
            if not re.fullmatch(r"ENSMUSG\d+", gid):
                raise ValueError(f"Invalid mouse Ensembl ID in mapping: {gid}")
            if gid in self.records and record != self.records[gid]:
                raise ValueError(f"Conflicting mapping records for {gid}; put multivalued IDs in | lists")
            self.records[gid] = record
            for column, value in record.items():
                if column in {"taxon_id", "biotype", "mapping_source", "mapping_status"}:
                    continue
                for alias in value.split("|"):
                    if alias:
                        self.aliases.setdefault(self.key(alias), set()).add(gid)
                        self.folded_aliases.setdefault(self.key(alias).casefold(), set()).add(gid)
                        if column == "symbol":
                            self.symbols.setdefault(self.key(alias), set()).add(gid)
                            self.folded_symbols.setdefault(self.key(alias).casefold(), set()).add(gid)
                        elif column in {"ensembl_gene_id", "mgi_id", "entrez_id", "protein_id", "uniprot_id", "string_id"}:
                            self.identifiers.setdefault(self.key(alias), set()).add(gid)
        self.rows = []
        self.genes = []
        seen_input, seen_gene = {}, {}
        for i, original in enumerate(inputs):
            key = self.key(original)
            candidates, match_type, alternatives = self.match(original)
            ensembl_input = original.split(":")[-1]
            unsupported = (match_type != "exact_official_symbol" and ensembl_input.upper().startswith("ENS")
                           and bool(re.search(r"\d", ensembl_input))
                           and not re.fullmatch(r"ENSMUSG\d+(?:\.\d+)?", ensembl_input, re.I))
            status = "unsupported" if unsupported else "resolved" if len(candidates) == 1 else "ambiguous" if candidates else "unresolved"
            gid = candidates[0] if status == "resolved" else ""
            row = {"row_id": f"g{i:06d}", "input_id": original, "canonical_id": gid,
                   "status": status, "candidates": "|".join(candidates),
                   "lookup_key": key, "match_type": match_type if not unsupported else "unsupported",
                   "shadowed_candidates": "|".join(sorted(alternatives - set(candidates))),
                   "duplicate_input_of": seen_input.get(original, ""),
                   "duplicate_gene_of": seen_gene.get(gid, "") if gid else "",
                   "mapping_record": json.dumps(self.records.get(gid, {}), sort_keys=True)}
            self.rows.append(row)
            self.genes.append(gid)
            seen_input.setdefault(original, row["row_id"])
            if gid:
                seen_gene.setdefault(gid, row["row_id"])

    @staticmethod
    def key(identifier):
        gene = re.fullmatch(r"(?:ensembl:)?(ENSMUSG\d+)(?:\.\d+)?", identifier, re.I)
        if gene:
            return gene.group(1).upper()
        mgi = re.fullmatch(r"(?:MGI:)+(\d+)", identifier, re.I)
        if mgi:
            return "MGI:" + mgi.group(1)
        entrez = re.fullmatch(r"(?:NCBIGene|Entrez|GeneID):(\d+)", identifier, re.I)
        return entrez.group(1) if entrez else identifier

    def match(self, identifier):
        key = self.key(identifier)
        alternatives = self.aliases.get(key, set()) | self.folded_aliases.get(key.casefold(), set())
        for kind, index, lookup in (
            ("exact_identifier", self.identifiers, key), ("exact_official_symbol", self.symbols, key),
            ("exact_alias", self.aliases, key), ("casefold_official_symbol", self.folded_symbols, key.casefold()),
            ("casefold_alias", self.folded_aliases, key.casefold()),
        ):
            if index.get(lookup):
                return sorted(index[lookup]), kind, alternatives
        return [], "unresolved", alternatives

    def resolve(self, identifier):
        candidates, _, _ = self.match(identifier)
        return candidates[0] if len(candidates) == 1 else None


@dataclass
class FeatureBlock:
    name: str
    matrix: sparse.csr_matrix
    features: list[str]
    coverage: list[dict]
    provenance: dict

    def kernel(self):
        """Zero-fill unavailable rows before normalization: zero rows/diagonals."""
        from sklearn.preprocessing import normalize
        x = self.matrix.tocsr().astype(np.float64)
        # sklearn rejects zero-feature inputs; their Gram matrix is all zero.
        if 0 in x.shape:
            return sparse.csr_matrix((x.shape[0], x.shape[0]), dtype=np.float64)
        z = normalize(x, norm="l2", axis=1)
        return (z @ z.T).tocsr()

    def validate_features(self, n):
        x = self.matrix
        if x.shape != (n, len(self.features)) or len(self.coverage) != n:
            raise ValueError(f"{self.name}: dimensions/labels disagree")
        if len(set(self.features)) != len(self.features) or not np.isfinite(x.data).all():
            raise ValueError(f"{self.name}: invalid features")
        if not self.provenance:
            raise ValueError("Missing provenance")
        unavailable = [i for i, row in enumerate(self.coverage) if not row["usable"]]
        if unavailable and x[unavailable].nnz:
            raise ValueError("Missing-information feature rows must be zero")

    def validate(self, n):
        self.validate_features(n)
        x = self.matrix
        k = self.kernel()
        delta = k - k.T
        if not np.isfinite(k.data).all() or (delta.nnz and abs(delta.data).max() > 1e-10):
            raise ValueError("Nonfinite or asymmetric similarity")
        unavailable = [i for i, row in enumerate(self.coverage) if not row["usable"]]
        if unavailable and (x[unavailable].nnz or k[unavailable].nnz):
            raise ValueError("Missing-information rows must be zero")
        # PSD follows from the explicit Gram construction. Large full-spectrum
        # eigensolves of rank-deficient kernels are prohibitively slow; report
        # bounded numerical checks honestly rather than an invented eigenvalue.
        if n <= 512:
            smallest = float(np.linalg.eigvalsh(k.toarray()).min()) if n else 0.
            checked = smallest
            self.provenance["kernel_validation"] = {"method": "full dense eigenspectrum", "psd_by_gram_construction": True}
        else:
            rng = np.random.default_rng(0)
            indices = np.sort(rng.choice(n, min(256, n), replace=False))
            checked = float(np.linalg.eigvalsh(k[indices][:, indices].toarray()).min())
            vectors = rng.normal(size=(n, 8)) / np.sqrt(n)
            quadratic_forms = np.sum(vectors * (k @ vectors), axis=0)
            if quadratic_forms.min() < -1e-7 * max(1., float(k.diagonal().sum())):
                raise ValueError("Kernel has a negative numerical quadratic form")
            smallest = None
            self.provenance["kernel_validation"] = {
                "method": "Gram PSD guarantee plus sampled principal eigenspectrum and full-kernel random quadratic forms",
                "psd_by_gram_construction": True, "full_eigenspectrum_computed": False,
                "principal_indices": indices.tolist(), "smallest_principal_eigenvalue": checked,
                "quadratic_forms": quadratic_forms.tolist(), "random_seed": 0,
            }
        if checked < -1e-7 * max(1., float(k.diagonal().sum())):
            raise ValueError(f"Kernel failed numerical PSD check: {checked}")
        return k, smallest


def make_block(name, values, features, ids, provenance, observed=None, known_zero=None):
    """values maps canonical genes to vectors; missingness never becomes a feature."""
    observed = set(values) if observed is None else set(observed)
    known_zero = set() if known_zero is None else set(known_zero)
    n_features = len(features)
    parts, coverage = [], []
    for mapping, gene in zip(ids.rows, ids.genes):
        vector = values.get(gene)
        row = sparse.csr_matrix((1, n_features)) if vector is None else sparse.csr_matrix(vector).reshape((1, n_features))
        row.eliminate_zeros()
        usable = bool(row.nnz)
        status = ("identifier_" + mapping["status"] if not gene else
                  "usable" if usable else "known_no_association" if gene in known_zero else
                  "observed_zero_vector" if gene in observed else "absent_from_resource")
        coverage.append({"row_id": mapping["row_id"], "input_id": mapping["input_id"],
                         "canonical_id": gene, "status": status, "observed": gene in observed,
                         "usable": usable, "nonzero_features": row.nnz})
        parts.append(row)
    matrix = sparse.vstack(parts, format="csr") if parts else sparse.csr_matrix((0, n_features))
    provenance = dict(provenance, similarity={
        "method": "cosine Gram: row-L2-normalized X times its transpose", "range": [-1, 1],
        "positive_semidefinite": True,
        "missing_information": "zero feature row, zero similarity row/column including diagonal",
        "interpretation": "biological similarity; not calibrated expression covariance",
        "negative_values": "opposing signed feature profiles; retained without clipping",
    })
    return FeatureBlock(name, matrix, features, coverage, provenance)
