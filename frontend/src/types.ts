export type DatasetKind = "spatial" | "reference";

export interface ValidationIssue {
  severity: "error" | "warning";
  code: string;
  what: string;
  why: string;
  fix: string;
  technical?: string | null;
}

export interface DatasetSummary {
  kind: DatasetKind;
  filename: string;
  size_bytes: number;
  n_observations: number;
  n_genes: number;
  matrix: string;
  matrix_dtype: string;
  observation_examples: string[];
  gene_examples: string[];
  gene_display_examples: string[];
  gene_field: string;
  display_field: string;
  matrix_layer: string;
  observation_columns: string[];
  gene_columns: string[];
  layer_names: string[];
  obsm_keys: string[];
  coordinate_source?: string | null;
  coordinate_field?: string | null;
  coordinate_dimensions?: number | null;
  cell_type_field?: string | null;
  transpose_applied: boolean;
  valid: boolean;
  issues: ValidationIssue[];
}

export interface MappingOptions {
  gene_fields: string[];
  layers: string[];
  obsm: string[];
  obs_columns: string[];
}

export interface DatasetUploadResponse {
  session_id: string;
  dataset: DatasetSummary;
  mapping_options: MappingOptions;
}

export interface MappingForm {
  gene_field: string;
  display_field: string;
  matrix_layer: string;
  coordinate_source: "obsm" | "obs";
  coordinate_field: string;
  coordinate_columns: string[];
  cell_type_field: string;
  transpose: boolean;
}

export interface SelectedGene {
  gene_id: string;
  display_name: string;
  rank: number;
  score: number;
  score_provenance: string;
}

export type MethodId =
  | "random-demo-v1"
  | "rotating-markers-spatial"
  | "rotating-markers-reference"
  | "gaussian-information-spatial"
  | "gaussian-information-reference";

export interface EvaluationStep {
  round: number;
  genes: string[];
  n_measured: number;
  expected_share: number | null;
  heldout_corr: number | null;
}

export interface SelectionResponse {
  method: MethodId;
  method_label: string;
  score_label: string;
  seed: number;
  selected_genes: SelectedGene[];
  candidate_count: number;
  previous_count: number;
  matched_previous_count: number;
  unmatched_previous_genes: string[];
  eligible_count: number;
  requested_count: number;
  reference_overlap_count: number;
  evaluation: {
    label: string;
    synthetic: boolean;
    notice: string;
    metric: string;
    points: { round: number; value: number }[];
    steps?: EvaluationStep[];
  };
  run_summary: Record<string, unknown>;
}

export interface CandidateSummary {
  candidate_count: number;
  previous_count: number;
  matched_previous_count: number;
  unmatched_previous_genes: string[];
  eligible_count: number;
  reference_overlap_count: number;
}

export interface PreviewResponse {
  kind: DatasetKind;
  available: boolean;
  message?: string | null;
  points: { x: number; y: number }[];
  coordinate_dimensions?: number | null;
  sampled: boolean;
  total_observations: number;
  color_label: string;
  color_values: (string | number | null)[];
}
