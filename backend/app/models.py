from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


IssueSeverity = Literal["error", "warning"]


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    code: str
    what: str
    why: str
    fix: str
    technical: str | None = None


class DatasetSummary(BaseModel):
    kind: Literal["spatial", "reference"]
    filename: str
    size_bytes: int
    n_observations: int
    n_genes: int
    matrix: str
    matrix_dtype: str
    observation_examples: list[str]
    gene_examples: list[str]
    gene_display_examples: list[str]
    gene_field: str
    display_field: str
    matrix_layer: str
    observation_columns: list[str]
    gene_columns: list[str]
    layer_names: list[str]
    obsm_keys: list[str]
    coordinate_source: str | None = None
    coordinate_field: str | None = None
    coordinate_dimensions: int | None = None
    cell_type_field: str | None = None
    transpose_applied: bool = False
    valid: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    session_id: str
    dataset: DatasetSummary
    mapping_options: dict[str, list[str]]


class MappingRequest(BaseModel):
    gene_field: str = "var_names"
    display_field: str = "var_names"
    matrix_layer: str = "X"
    coordinate_source: Literal["obsm", "obs"] = "obsm"
    coordinate_field: str = "spatial"
    coordinate_columns: list[str] = Field(default_factory=list)
    cell_type_field: str | None = None
    transpose: bool = False


class SelectionRequest(BaseModel):
    session_id: str
    previous_genes: list[str] = Field(default_factory=list)
    method: Literal["random", "dummy_markers", "dummy_variance", "dummy_random", "manual"] = "random"
    chosen_genes: list[str] = Field(default_factory=list)
    ng: int = Field(strict=True, gt=0)
    seed: int | None = None


class CandidateRequest(BaseModel):
    session_id: str
    previous_genes: list[str] = Field(default_factory=list)


class CandidateSummary(BaseModel):
    candidate_count: int
    previous_count: int
    matched_previous_count: int
    unmatched_previous_genes: list[str]
    eligible_count: int
    reference_overlap_count: int


class SelectedGene(BaseModel):
    gene_id: str
    display_name: str
    rank: int
    score: float | None = None
    score_provenance: str


class SelectionResponse(BaseModel):
    method: str
    method_label: str
    seed: int
    selected_genes: list[SelectedGene]
    candidate_count: int
    previous_count: int
    matched_previous_count: int
    unmatched_previous_genes: list[str]
    eligible_count: int
    requested_count: int
    reference_overlap_count: int
    evaluation: dict[str, Any]
    run_summary: dict[str, Any]


class PreviewResponse(BaseModel):
    kind: Literal["spatial", "reference"]
    available: bool
    message: str | None = None
    points: list[dict[str, Any]] = Field(default_factory=list)
    coordinate_dimensions: int | None = None
    sampled: bool = False
    total_observations: int = 0
    color_label: str = "Uncolored"
    color_values: list[Any] = Field(default_factory=list)
