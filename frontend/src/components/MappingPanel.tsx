import { useEffect, useState } from "react";
import type { DatasetKind, DatasetSummary, MappingForm, MappingOptions } from "../types";

interface Props {
  kind: DatasetKind;
  dataset: DatasetSummary;
  options: MappingOptions;
  initial: MappingForm;
  onApply: (mapping: MappingForm) => Promise<void>;
  onChange: () => void;
  saving: boolean;
}

export default function MappingPanel({ kind, dataset, options, initial, onApply, onChange, saving }: Props) {
  const [form, setForm] = useState(initial);
  useEffect(() => setForm(initial), [initial]);
  const geneFields = options.gene_fields.includes(form.gene_field)
    ? options.gene_fields
    : [form.gene_field, ...options.gene_fields];
  const displayFields = options.gene_fields.includes(form.display_field)
    ? options.gene_fields
    : [form.display_field, ...options.gene_fields];
  const coordinateFields = options.obsm.includes(form.coordinate_field)
    ? options.obsm
    : [form.coordinate_field, ...options.obsm];
  function update(patch: Partial<MappingForm>) {
    setForm((current) => ({ ...current, ...patch }));
    onChange();
  }

  return (
    <details className="mapping-panel">
      <summary>Metadata and expression mapping</summary>
      <div className="mapping-grid">
        <label>
          Gene identifiers
          <select value={form.gene_field} onChange={(event) => update({ gene_field: event.target.value })}>
            {geneFields.map((field) => <option key={field} value={field}>{field}{options.gene_fields.includes(field) ? "" : " (not found)"}</option>)}
          </select>
        </label>
        <label>
          Gene display names
          <select value={form.display_field} onChange={(event) => update({ display_field: event.target.value })}>
            {displayFields.map((field) => <option key={field} value={field}>{field}{options.gene_fields.includes(field) ? "" : " (not found)"}</option>)}
          </select>
        </label>
        {(dataset.issues.some((issue) => issue.code === "orientation_review") || dataset.transpose_applied) && (
          <label className="transpose-option">
            <span>Reviewed orientation correction</span>
            <span>
              <input
                type="checkbox"
                checked={form.transpose}
                onChange={(event) => update({ transpose: event.target.checked })}
              />
              Transpose rows and columns before validation
            </span>
          </label>
        )}
        <label>
          Expression matrix
          <select value={form.matrix_layer} onChange={(event) => update({ matrix_layer: event.target.value })}>
            {options.layers.map((layer) => <option key={layer}>{layer}</option>)}
          </select>
        </label>
        {kind === "spatial" && (
          <>
            <label>
              Coordinate source
              <select
                value={form.coordinate_source}
                onChange={(event) => update({ coordinate_source: event.target.value as "obsm" | "obs" })}
              >
                <option value="obsm">obsm entry</option>
                <option value="obs">obs columns</option>
              </select>
            </label>
            {form.coordinate_source === "obsm" ? (
              <label>
                Coordinate entry
                <select value={form.coordinate_field} onChange={(event) => update({ coordinate_field: event.target.value })}>
                  {coordinateFields.map((field) => <option key={field} value={field}>{field}{options.obsm.includes(field) ? "" : " (not found)"}</option>)}
                </select>
              </label>
            ) : (
              <label>
                Coordinate columns (choose 2 or 3)
                <select
                  multiple
                  value={form.coordinate_columns}
                  onChange={(event) => update({ coordinate_columns: Array.from(event.target.selectedOptions, (option) => option.value) })}
                  size={Math.min(4, Math.max(2, options.obs_columns.length))}
                >
                  {options.obs_columns.map((field) => <option key={field}>{field}</option>)}
                </select>
              </label>
            )}
          </>
        )}
        <label>
          Cell-type annotation <span className="optional">(optional)</span>
          <select value={form.cell_type_field} onChange={(event) => update({ cell_type_field: event.target.value })}>
            <option value="">None selected</option>
            {form.cell_type_field && !options.obs_columns.includes(form.cell_type_field) && <option value={form.cell_type_field}>{form.cell_type_field} (not found)</option>}
            {options.obs_columns.map((field) => <option key={field}>{field}</option>)}
          </select>
        </label>
      </div>
      <button className="button button-small" disabled={saving} onClick={() => void onApply(form)}>
        {saving ? "Checking mapping…" : "Apply mapping"}
      </button>
      <p className="mapping-note">
        Interpreted as {dataset.n_observations.toLocaleString()} observations × {dataset.n_genes.toLocaleString()} genes.
        Dimensions alone cannot prove orientation. Review example IDs first; this explicit option makes a temporary corrected copy and can be unchecked to return to the original.
      </p>
    </details>
  );
}
