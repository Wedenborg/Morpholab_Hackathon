from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .models import DatasetSummary, MappingRequest, ValidationIssue

MAX_UPLOAD_MB = int(os.getenv("GENE_PANEL_MAX_UPLOAD_MB", "250"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
PREVIEW_LIMIT = 1200
VALIDATION_CHUNK_ELEMENTS = 2_000_000


def open_annotated(path: Path) -> ad.AnnData:
    return ad.read_h5ad(path, backed="r")


def transpose_h5ad(source: Path, destination: Path) -> None:
    adata = ad.read_h5ad(source)
    adata.transpose().write_h5ad(destination)


def close_annotated(adata: ad.AnnData) -> None:
    adata.file.close()


def expression_matrix(adata: ad.AnnData, layer: str) -> Any:
    if layer == "X":
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Expression layer {layer!r} is not present.")
    return adata.layers[layer]


def _identifiers(adata: ad.AnnData, field: str) -> list[str]:
    if field == "var_names":
        values = adata.var_names.tolist()
    else:
        if field not in adata.var.columns:
            raise KeyError(f"Gene identifier field {field!r} is not present in var.")
        values = adata.var[field].tolist()
    return ["" if _missing(value) else str(value) for value in values]


def _display_names(adata: ad.AnnData, field: str) -> list[str]:
    if field == "var_names":
        values = adata.var_names.tolist()
    else:
        if field not in adata.var.columns:
            raise KeyError(f"Gene display field {field!r} is not present in var.")
        values = adata.var[field].tolist()
    return ["" if _missing(value) else str(value) for value in values]


def _missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _duplicate_examples(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
        if len(duplicates) == 5:
            break
    return duplicates


def _matrix_issues(matrix: Any, n_obs: int, n_vars: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if matrix is None:
        return [
            ValidationIssue(
                severity="error",
                code="missing_expression",
                what="The selected expression matrix is missing.",
                why="Selection and expression previews require values for observations and genes.",
                fix="Select X or a layer that contains expression values.",
            )
        ]
    shape = getattr(matrix, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != n_obs or shape[1] != n_vars:
        return [
            ValidationIssue(
                severity="error",
                code="matrix_shape",
                what=f"Expression matrix dimensions {shape} do not match {n_obs} observations × {n_vars} genes.",
                why="The interface treats rows as cells/spots and columns as genes.",
                fix="Select a matrix aligned to this AnnData object's obs and var metadata.",
            )
        ]
    if shape[0] == 0 or shape[1] == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                code="empty_expression",
                what="The expression matrix has no observations or genes.",
                why="A nonempty matrix is required for a panel.",
                fix="Choose a nonempty dataset or expression layer.",
            )
        )
    dtype = getattr(matrix, "dtype", None)
    if dtype is None or not np.issubdtype(dtype, np.number):
        issues.append(
            ValidationIssue(
                severity="error",
                code="non_numeric_expression",
                what=f"The selected matrix has non-numeric dtype {dtype}.",
                why="Expression previews require numeric values.",
                fix="Choose X or a numeric expression layer. Normalized and transformed numeric values are accepted.",
            )
        )
        return issues
    if shape[0] and shape[1]:
        rows_per_chunk = max(1, min(1024, VALIDATION_CHUNK_ELEMENTS // max(shape[1], 1)))
        for start in range(0, shape[0], rows_per_chunk):
            block = matrix[start : start + rows_per_chunk]
            values = block.data if sparse.issparse(block) else np.asarray(block)
            if not np.isfinite(values).all():
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="non_finite_expression",
                        what="Expression values contain NaN or infinity.",
                        why="Non-finite values cannot be used reliably in previews or scoring.",
                        fix="Clean the selected matrix or choose a layer with finite values.",
                    )
                )
                break
    return issues


def _coordinate_data(
    adata: ad.AnnData, mapping: MappingRequest
) -> tuple[np.ndarray | None, list[ValidationIssue]]:
    if mapping.coordinate_source == "obsm":
        if mapping.coordinate_field not in adata.obsm:
            return None, [
                ValidationIssue(
                    severity="error",
                    code="missing_coordinates",
                    what=f"obsm[{mapping.coordinate_field!r}] is not present.",
                    why="Spatial preview needs coordinates for each cell or spot.",
                    fix=f"Choose one of the available obsm entries: {', '.join(adata.obsm.keys()) or 'none'}; or map coordinate columns from obs.",
                )
            ]
        raw = adata.obsm[mapping.coordinate_field]
    else:
        columns = mapping.coordinate_columns
        if len(columns) not in (2, 3) or any(col not in adata.obs for col in columns):
            return None, [
                ValidationIssue(
                    severity="error",
                    code="coordinate_columns",
                    what="Two or three valid observation columns were not selected for coordinates.",
                    why="Each cell or spot must have one coordinate per plotted dimension.",
                    fix="Select two or three numeric columns from obs.",
                )
            ]
        raw = adata.obs[columns].to_numpy()
    try:
        coordinates = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        coordinates = np.empty((0, 0), dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != adata.n_obs or coordinates.shape[1] not in (2, 3):
        return None, [
            ValidationIssue(
                severity="error",
                code="coordinate_shape",
                what=f"Coordinates have shape {coordinates.shape}; expected ({adata.n_obs}, 2) or ({adata.n_obs}, 3).",
                why="The preview supports two-dimensional coordinates and three-dimensional coordinates shown as a labelled projection.",
                fix="Choose a compatible obsm entry or map two/three numeric obs columns.",
            )
        ]
    if not np.isfinite(coordinates).all():
        return None, [
            ValidationIssue(
                severity="error",
                code="coordinate_non_finite",
                what="Coordinates contain non-numeric, NaN, or infinite values.",
                why="Every plotted observation needs finite numeric coordinates.",
                fix="Choose or create coordinate columns containing finite numeric values.",
            )
        ]
    return coordinates, []


def validate_dataset(
    path: Path,
    kind: Literal["spatial", "reference"],
    filename: str,
    size_bytes: int,
    mapping: MappingRequest,
) -> tuple[DatasetSummary, list[str], list[str], np.ndarray | None]:
    adata = open_annotated(path)
    try:
        issues: list[ValidationIssue] = []
        obs_names = ["" if _missing(value) else str(value) for value in adata.obs_names.tolist()]
        gene_ids: list[str] = []
        display_names: list[str] = []
        try:
            gene_ids = _identifiers(adata, mapping.gene_field)
        except KeyError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_gene_field",
                    what=str(exc),
                    why="The selected identifier field provides the stable IDs used for matching and exports.",
                    fix=f"Choose var_names or a field from var: {', '.join(adata.var.columns) or 'none'}.",
                    technical=str(exc),
                )
            )
        try:
            display_names = _display_names(adata, mapping.display_field)
        except KeyError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_display_field",
                    what=str(exc),
                    why="The display field provides readable gene names in results and previews.",
                    fix=f"Choose var_names or a field from var: {', '.join(adata.var.columns) or 'none'}.",
                    technical=str(exc),
                )
            )
        for label, values, code in (
            ("Observation", obs_names, "observation_identifiers"),
            ("Gene", gene_ids, "gene_identifiers"),
        ):
            if not values:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code=code,
                        what=f"{label} identifiers are missing.",
                        why="Identifiers must identify observations and genes unambiguously.",
                        fix="Provide nonempty observation names and gene identifiers in obs_names, var_names, or the selected var field.",
                    )
                )
                continue
            if any(not value for value in values):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code=code,
                        what=f"Some {label.lower()} identifiers are empty.",
                        why="Empty IDs cannot be matched or exported reliably.",
                        fix="Fill the missing identifiers or choose another identifier field.",
                    )
                )
            duplicates = _duplicate_examples(values)
            if duplicates:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code=f"duplicate_{label.lower()}_identifiers",
                        what=f"{label} identifiers are not unique; examples: {', '.join(duplicates)}.",
                        why="Duplicate IDs make gene matching and row-level previews ambiguous.",
                        fix="Make the identifiers unique or select a unique identifier field.",
                    )
                )
        if mapping.cell_type_field and mapping.cell_type_field not in adata.obs:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="missing_cell_type_field",
                    what=f"Cell-type field {mapping.cell_type_field!r} was not found in obs.",
                    why="Cell-type labels are optional and only affect preview coloring.",
                    fix="Choose an available obs column or clear the optional mapping.",
                )
            )
        matrix: Any = None
        try:
            matrix = expression_matrix(adata, mapping.matrix_layer)
            issues.extend(_matrix_issues(matrix, adata.n_obs, adata.n_vars))
        except KeyError as exc:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="missing_expression_layer",
                    what=str(exc),
                    why="Selection and previews require an existing expression matrix.",
                    fix=f"Choose X or a layer from: {', '.join(adata.layers.keys()) or 'none'}.",
                    technical=str(exc),
                )
            )
        coordinate_dimensions: int | None = None
        if kind == "spatial":
            coordinates, coordinate_issues = _coordinate_data(adata, mapping)
            issues.extend(coordinate_issues)
            if coordinates is not None:
                coordinate_dimensions = coordinates.shape[1]
        else:
            coordinates = None
        if adata.n_obs < adata.n_vars:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="orientation_review",
                    what="This file has fewer observations than genes.",
                    why="The expected AnnData layout is observations × genes; dimensions alone cannot determine whether a file is transposed.",
                    fix="Review the interpreted axes and example identifiers below. Select the reviewed orientation correction mapping to create a temporary transposed copy, or correct the AnnData obs/var orientation yourself.",
                )
            )
        summary = DatasetSummary(
            kind=kind,
            filename=filename,
            size_bytes=size_bytes,
            n_observations=adata.n_obs,
            n_genes=adata.n_vars,
            matrix="X" if mapping.matrix_layer == "X" else f"layers[{mapping.matrix_layer!r}]",
            matrix_dtype=str(getattr(matrix, "dtype", "unknown")),
            observation_examples=obs_names[:5],
            gene_examples=gene_ids[:5],
            gene_display_examples=display_names[:5],
            gene_field=mapping.gene_field,
            display_field=mapping.display_field,
            matrix_layer=mapping.matrix_layer,
            observation_columns=[str(key) for key in adata.obs.columns],
            gene_columns=[str(key) for key in adata.var.columns],
            layer_names=[str(key) for key in adata.layers.keys()],
            obsm_keys=[str(key) for key in adata.obsm.keys()],
            coordinate_source=mapping.coordinate_source if kind == "spatial" else None,
            coordinate_field=(
                mapping.coordinate_field if mapping.coordinate_source == "obsm" and kind == "spatial"
                else ", ".join(mapping.coordinate_columns) if kind == "spatial"
                else None
            ),
            coordinate_dimensions=coordinate_dimensions,
            cell_type_field=mapping.cell_type_field,
            valid=not any(issue.severity == "error" for issue in issues),
            issues=issues,
        )
        return summary, gene_ids, display_names, coordinates
    finally:
        close_annotated(adata)


def mapping_options(path: Path) -> dict[str, list[str]]:
    adata = open_annotated(path)
    try:
        return {
            "gene_fields": ["var_names", *[str(col) for col in adata.var.columns]],
            "layers": ["X", *[str(key) for key in adata.layers.keys()]],
            "obsm": [str(key) for key in adata.obsm.keys()],
            "obs_columns": [str(col) for col in adata.obs.columns],
        }
    finally:
        close_annotated(adata)


def read_preview_values(
    path: Path, mapping: MappingRequest, kind: str, selected_gene: str | None, color_by: str | None
) -> tuple[np.ndarray | None, list[Any], str]:
    adata = open_annotated(path)
    try:
        coordinates: np.ndarray | None = None
        if kind == "spatial":
            coordinates, issues = _coordinate_data(adata, mapping)
            if issues:
                return None, [], "Coordinates unavailable"
        color_values: list[Any] = []
        color_label = "Uncolored"
        if color_by == "none":
            pass
        elif color_by:
            if color_by.startswith("gene:"):
                gene = color_by.removeprefix("gene:")
                identifiers = _identifiers(adata, mapping.gene_field)
                try:
                    col_index = identifiers.index(gene)
                except ValueError:
                    return coordinates, [], "Gene expression unavailable"
                values = expression_matrix(adata, mapping.matrix_layer)[:, col_index]
                color_values = np.asarray(values.toarray() if sparse.issparse(values) else values).reshape(-1).tolist()
                color_label = f"Expression: {gene}"
            elif color_by in adata.obs:
                color_values = [None if _missing(value) else str(value) for value in adata.obs[color_by].tolist()]
                color_label = color_by
        elif selected_gene:
            identifiers = _identifiers(adata, mapping.gene_field)
            try:
                col_index = identifiers.index(selected_gene)
            except ValueError:
                return coordinates, [], "Gene expression unavailable"
            values = expression_matrix(adata, mapping.matrix_layer)[:, col_index]
            color_values = np.asarray(values.toarray() if sparse.issparse(values) else values).reshape(-1).tolist()
            color_label = f"Expression: {selected_gene}"
        return coordinates, color_values, color_label
    finally:
        close_annotated(adata)


def preview_limit() -> int:
    return PREVIEW_LIMIT
