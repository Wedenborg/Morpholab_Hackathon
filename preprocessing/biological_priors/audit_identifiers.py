"""Trace native mouse naming matches and compare them with an earlier run."""

from collections import Counter
from pathlib import Path

from .core import Identifiers, digest, table, write_json, write_table


def audit_identifiers(genes_path, mapping_path, output_dir, previous_mapping=None):
    genes = Path(genes_path).read_text().splitlines()
    ids = Identifiers(mapping_path, genes)
    previous = table(previous_mapping) if previous_mapping else []
    if previous and [row["input_id"] for row in previous] != genes:
        raise ValueError("Previous mapping does not preserve the same ordered input gene list")
    rows = []
    for i, current in enumerate(ids.rows):
        row = dict(current)
        old = previous[i] if previous else None
        row["previous_status"] = old["status"] if old else "not_compared"
        row["previous_canonical_id"] = old["canonical_id"] if old else ""
        row["change"] = ("newly_resolved" if old and old["status"] != "resolved" and row["status"] == "resolved" else
                         "assignment_changed" if old and old["canonical_id"] != row["canonical_id"] else
                         "unchanged" if old else "not_compared")
        rows.append(row)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_table(output / "identifier_audit.tsv", rows, tuple(rows[0]))
    summary = {"input_rows": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
               "match_type_counts": dict(Counter(row["match_type"] for row in rows)),
               "change_counts": dict(Counter(row["change"] for row in rows)),
               "newly_resolved_genes": [row["input_id"] for row in rows if row["change"] == "newly_resolved"],
               "resolved_with_shadowed_aliases": [row["input_id"] for row in rows if row["status"] == "resolved" and row["shadowed_candidates"]],
               "gene_list_sha256": digest(genes_path), "mapping_sha256": digest(mapping_path),
               "previous_mapping_sha256": digest(previous_mapping) if previous_mapping else None,
               "scope": "Identifier naming audit only; resolving a name does not imply permitted GO annotation coverage",
               "no_fuzzy_matching_or_ortholog_substitution": True,
               "historical_reference_note": "Ensembl GRCm38 release 102 is a native identifier cross-reference; sample assembly is not inferred"}
    write_json(output / "identifier_audit.json", summary)
    return summary
