import { useEffect, useMemo, useState } from "react";
import { api, errorTechnical } from "./api";
import MappingPanel from "./components/MappingPanel";
import ScatterPreview from "./components/ScatterPreview";
import type {
  DatasetKind,
  DatasetSummary,
  CandidateSummary,
  MappingForm,
  MappingOptions,
  PreviewResponse,
  SelectedGene,
  SelectionResponse,
} from "./types";

type DatasetMap<T> = Record<DatasetKind, T>;

const initialMapping = (kind: DatasetKind): MappingForm => ({
  gene_field: "var_names",
  display_field: "var_names",
  matrix_layer: "X",
  coordinate_source: "obsm",
  coordinate_field: "spatial",
  coordinate_columns: [],
  cell_type_field: "",
  transpose: false,
});

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function splitGenes(text: string) {
  return [...new Set(text.split(/[\n,;\t]+/).map((value) => value.trim()).filter(Boolean))];
}

function fileToMappingOptions(summary: DatasetSummary): MappingOptions {
  return {
    gene_fields: ["var_names", ...summary.gene_columns],
    layers: ["X", ...summary.layer_names],
    obsm: summary.obsm_keys,
    obs_columns: summary.observation_columns,
  };
}

function download(filename: string, contents: string, type: string) {
  const blob = new Blob([contents], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function csvEscape(value: string | number) {
  return `"${String(value).replaceAll('"', '""')}"`;
}

function UploadCard({
  kind,
  summary,
  options,
  mapping,
  busy,
  onUpload,
  onMapping,
  onMappingChange,
  onChange,
}: {
  kind: DatasetKind;
  summary: DatasetSummary | null;
  options: MappingOptions | null;
  mapping: MappingForm;
  busy: boolean;
  onUpload: (file: File) => void;
  onMapping: (mapping: MappingForm) => Promise<void>;
  onMappingChange: () => void;
  onChange: () => void;
}) {
  const [dragging, setDragging] = useState(false);
  const title = kind === "spatial" ? "Spatial dataset" : "Single-cell reference";
  function choose(file?: File) {
    if (!file) return;
    onChange();
    onUpload(file);
  }
  return (
    <article className="dataset-card">
      <div className="dataset-card-head">
        <div>
          <div className="eyebrow">{kind === "spatial" ? "Required · panel candidates" : "Required · comparison only"}</div>
          <h3>{title}</h3>
        </div>
        {summary && (
          <span className={`status-pill ${summary.valid ? "status-good" : "status-bad"}`}>
            <span className="status-dot" /> {summary.valid ? "Checked" : "Needs attention"}
          </span>
        )}
      </div>
      <label
        className={`drop-zone ${dragging ? "drag-active" : ""} ${busy ? "drop-busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          choose(event.dataTransfer.files[0]);
        }}
      >
        <input
          type="file"
          accept=".h5ad,application/x-hdf5"
          disabled={busy}
          aria-label={`Choose ${title} .h5ad file`}
          onChange={(event) => choose(event.target.files?.[0])}
        />
        <span className="upload-symbol" aria-hidden="true">↑</span>
        <span className="drop-copy">
          <strong>{busy ? "Uploading and validating…" : "Drop a .h5ad file here"}</strong>
          <span>or <span className="file-picker-text">browse files</span></span>
        </span>
      </label>
      {summary ? (
        <div className="file-summary">
          <div className="file-line"><span className="file-mark">H5</span><strong title={summary.filename}>{summary.filename}</strong><span>{formatBytes(summary.size_bytes)}</span></div>
          <div className="summary-facts">
            <span><strong>{summary.n_observations.toLocaleString()}</strong> cells / spots</span>
            <span><strong>{summary.n_genes.toLocaleString()}</strong> genes</span>
            <span><strong>{summary.matrix}</strong> · {summary.matrix_dtype}</span>
          </div>
          <div className="identifier-preview">
            <div><span>Observation IDs</span><code>{summary.observation_examples.slice(0, 3).join(", ") || "none found"}</code></div>
            <div><span>Gene IDs · {summary.gene_field}</span><code>{summary.gene_examples.slice(0, 3).join(", ") || "none found"}</code></div>
            <div><span>Display names · {summary.display_field}</span><code>{summary.gene_display_examples.slice(0, 3).join(", ") || "none found"}</code></div>
          </div>
          {summary.issues.length > 0 && (
            <div className="issues" aria-live="polite">
              {summary.issues.map((issue, index) => (
                <div className={`issue issue-${issue.severity}`} key={`${issue.code}-${index}`}>
                  <strong>{issue.severity === "error" ? "Needs fixing" : "Review"} · {issue.what}</strong>
                  <span>{issue.why}</span>
                  <span className="issue-fix">Next: {issue.fix}</span>
                  {issue.technical && <details><summary>Technical details</summary><code>{issue.technical}</code></details>}
                </div>
              ))}
            </div>
          )}
          {options && (
            <MappingPanel
              kind={kind}
              dataset={summary}
              options={options}
              initial={mapping}
              onApply={onMapping}
              onChange={onMappingChange}
              saving={busy}
            />
          )}
        </div>
      ) : (
        <p className="upload-hint">
          {kind === "spatial"
            ? "Rows are cells or spots; columns are genes. Spatial coordinates are needed for the preview."
            : "The reference is used to report exact gene overlap; the demo selector does not use it to filter candidates."}
        </p>
      )}
    </article>
  );
}

function ResultsTable({
  genes,
  search,
  setSearch,
  selected,
  onSelect,
}: {
  genes: SelectedGene[];
  search: string;
  setSearch: (value: string) => void;
  selected: SelectedGene | null;
  onSelect: (gene: SelectedGene) => void;
}) {
  const filtered = genes.filter((gene) =>
    `${gene.gene_id} ${gene.display_name}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <div className="table-section">
      <div className="table-topline">
        <div><strong>Selected genes</strong><span>{filtered.length} shown · {genes.length} selected</span></div>
        <label className="search-field"><span className="sr-only">Search selected genes</span><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search genes" /></label>
      </div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Rank</th><th>Gene</th><th>Stable identifier</th><th>Demo score</th></tr></thead>
          <tbody>
            {filtered.map((gene) => (
              <tr
                key={gene.gene_id}
                className={selected?.gene_id === gene.gene_id ? "row-selected" : ""}
                onClick={() => onSelect(gene)}
                tabIndex={0}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") onSelect(gene);
                }}
                aria-label={`View details for ${gene.display_name}`}
              >
                <td>{String(gene.rank).padStart(2, "0")}</td>
                <td className="gene-name">{gene.display_name}</td>
                <td><code>{gene.gene_id}</code></td>
                <td><span className="score-value">{gene.score.toFixed(3)}</span></td>
              </tr>
            ))}
            {!filtered.length && <tr><td colSpan={4} className="empty-table">No selected genes match this search.</td></tr>}
          </tbody>
        </table>
      </div>
      <p className="table-footnote">Scores are synthetic demo scores, not estimated information gain, probabilities, or evidence of biological usefulness.</p>
    </div>
  );
}

function ExpressionPreview({ label, preview }: { label: string; preview: PreviewResponse | null }) {
  if (!preview) return <div className="expression-preview"><span>{label}</span><small>Waiting for gene selection…</small></div>;
  if (!preview.available) {
    return <div className="expression-preview"><span>{label}</span><small>{preview.message ?? "Expression unavailable."}</small></div>;
  }
  const values = preview.color_values.filter((value): value is number => typeof value === "number");
  if (!values.length) {
    return <div className="expression-preview"><span>{label}</span><small>No numeric expression values were returned.</small></div>;
  }
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const bins = Array.from({ length: 8 }, () => 0);
  for (const value of values) {
    const index = Math.min(7, Math.floor(((value - minimum) / Math.max(maximum - minimum, 1e-12)) * bins.length));
    bins[index] += 1;
  }
  const peak = Math.max(...bins, 1);
  return (
    <div className="expression-preview">
      <div className="expression-heading"><span>{label}</span><small>{values.length.toLocaleString()} sampled cells</small></div>
      <div className="expression-bars" role="img" aria-label={`${label} expression distribution, sampled values`}>
        {bins.map((count, index) => <i key={index} style={{ height: `${Math.max(3, (count / peak) * 100)}%` }} />)}
      </div>
      <small>Sample range {minimum.toPrecision(3)}–{maximum.toPrecision(3)} · mean {mean.toPrecision(3)}</small>
    </div>
  );
}

function EvaluationPanel({ result }: { result: SelectionResponse }) {
  const points = result.evaluation.points;
  const width = 520;
  const height = 150;
  const min = Math.min(...points.map((point) => point.value));
  const max = Math.max(...points.map((point) => point.value));
  const y = (value: number) => height - 20 - ((value - min) / Math.max(max - min, 0.001)) * (height - 46);
  const path = points.map((point, index) => `${index ? "L" : "M"} ${36 + index * 108} ${y(point.value)}`).join(" ");
  return (
    <section className="evaluation-panel">
      <div className="evaluation-head">
        <div><span className="demo-label"><span /> Dedicated demo evaluation</span><h3>{result.evaluation.label}</h3></div>
        <span className="synthetic-tag">SYNTHETIC</span>
      </div>
      <p>{result.evaluation.notice}</p>
      <div className="evaluation-chart">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${result.evaluation.metric} over demo rounds`}>
          {[0, 1, 2, 3].map((line) => <line key={line} x1="30" x2="500" y1={20 + line * 34} y2={20 + line * 34} className="chart-grid" />)}
          <path d={path} className="chart-path" />
          {points.map((point, index) => <g key={point.round}><circle cx={36 + index * 108} cy={y(point.value)} r="4" className="chart-point" /><text x={36 + index * 108} y="145" textAnchor="middle">Round {point.round}</text></g>)}
        </svg>
        <span className="chart-axis-label">{result.evaluation.metric}</span>
      </div>
      <div className="evaluation-values">{points.map((point) => <span key={point.round}>R{point.round} <strong>{point.value.toFixed(3)}</strong></span>)}</div>
    </section>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState("");
  const [datasets, setDatasets] = useState<DatasetMap<DatasetSummary | null>>({ spatial: null, reference: null });
  const [options, setOptions] = useState<DatasetMap<MappingOptions | null>>({ spatial: null, reference: null });
  const [mappings, setMappings] = useState<DatasetMap<MappingForm>>({
    spatial: initialMapping("spatial"),
    reference: initialMapping("reference"),
  });
  const [busy, setBusy] = useState<DatasetMap<boolean>>({ spatial: false, reference: false });
  const [mappingDirty, setMappingDirty] = useState<DatasetMap<boolean>>({ spatial: false, reference: false });
  const [previousText, setPreviousText] = useState("");
  const [panelSize, setPanelSize] = useState("5");
  const [seedText, setSeedText] = useState("");
  const [result, setResult] = useState<SelectionResponse | null>(null);
  const [selectedGene, setSelectedGene] = useState<SelectedGene | null>(null);
  const [search, setSearch] = useState("");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [referencePreview, setReferencePreview] = useState<PreviewResponse | null>(null);
  const [spatialGenePreview, setSpatialGenePreview] = useState<PreviewResponse | null>(null);
  const [previewColor, setPreviewColor] = useState("auto");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [candidateSummary, setCandidateSummary] = useState<CandidateSummary | null>(null);
  const [error, setError] = useState("");
  const [technicalError, setTechnicalError] = useState("");
  const [selectionBusy, setSelectionBusy] = useState(false);
  const [exampleBusy, setExampleBusy] = useState(false);

  useEffect(() => {
    let active = true;
    void api.createSession().then(({ session_id }) => {
      if (active) setSessionId(session_id);
      else void api.deleteSession(session_id);
    }).catch((failure: unknown) => {
      if (active) setError(failure instanceof Error ? failure.message : "Could not start a session.");
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    return () => { void api.deleteSession(sessionId); };
  }, [sessionId]);

  const previous = useMemo(() => splitGenes(previousText), [previousText]);
  const eligibleCount = candidateSummary?.eligible_count ?? 0;
  const allReady = Boolean(datasets.spatial?.valid && datasets.reference?.valid);
  const parsedNg = Number(panelSize);
  const ngValid = Number.isInteger(parsedNg) && parsedNg > 0 && parsedNg <= eligibleCount;
  const parsedSeed = Number(seedText);
  const seedValid = !seedText.trim() || Number.isSafeInteger(parsedSeed);
  const validForSelection = allReady && ngValid && seedValid && Boolean(sessionId)
    && !mappingDirty.spatial && !mappingDirty.reference && !busy.spatial && !busy.reference && !exampleBusy;

  useEffect(() => {
    if (!sessionId || !datasets.spatial?.valid || !datasets.reference?.valid) {
      setCandidateSummary(null);
      return;
    }
    setCandidateSummary(null);
    let active = true;
    void api.candidates(sessionId, previous).then((value) => {
      if (active) setCandidateSummary(value);
    }).catch(() => {
      if (active) setCandidateSummary(null);
    });
    return () => { active = false; };
  }, [sessionId, datasets.spatial, datasets.reference, previous]);

  function clearResults() {
    setResult(null);
    setSelectedGene(null);
    setReferencePreview(null);
    setSpatialGenePreview(null);
  }

  function setErrorState(failure: unknown) {
    setError(failure instanceof Error ? failure.message : "Something went wrong. Please try again.");
    setTechnicalError(errorTechnical(failure) ?? "");
  }

  async function upload(kind: DatasetKind, file: File) {
    if (!sessionId) {
      setError("Your session is still starting. Please try again in a moment.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".h5ad")) {
      setError("Choose a file with the .h5ad extension.");
      return;
    }
    setError("");
    setTechnicalError("");
    clearResults();
    if (kind === "spatial") {
      setPreview(null);
      setPreviewLoading(false);
    }
    setBusy((current) => ({ ...current, [kind]: true }));
    try {
      const currentMapping = mappings[kind];
      const response = await api.upload(sessionId, kind, file, currentMapping);
      setDatasets((current) => ({ ...current, [kind]: response.dataset }));
      setOptions((current) => ({ ...current, [kind]: response.mapping_options }));
      setMappings((current) => ({ ...current, [kind]: currentMapping }));
      setMappingDirty((current) => ({ ...current, [kind]: false }));
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setBusy((current) => ({ ...current, [kind]: false }));
    }
  }

  async function applyMapping(kind: DatasetKind, mapping: MappingForm) {
    if (!sessionId) return;
    setBusy((current) => ({ ...current, [kind]: true }));
    setError("");
    clearResults();
    if (kind === "spatial") {
      setPreview(null);
      setPreviewLoading(false);
    }
    try {
      const response = await api.updateMapping(sessionId, kind, mapping);
      setDatasets((current) => ({ ...current, [kind]: response.dataset }));
      setOptions((current) => ({ ...current, [kind]: response.mapping_options }));
      setMappings((current) => ({ ...current, [kind]: mapping }));
      setMappingDirty((current) => ({ ...current, [kind]: false }));
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setBusy((current) => ({ ...current, [kind]: false }));
    }
  }

  async function tryExample() {
    setExampleBusy(true);
    setError("");
    setTechnicalError("");
    clearResults();
    setPreview(null);
    setPreviewLoading(false);
    try {
      const response = await api.example();
      const sessionResponse = await api.getSession(response.session_id);
      const spatial = sessionResponse.datasets.spatial;
      const reference = sessionResponse.datasets.reference;
      if (!spatial || !reference) throw new Error("Example datasets did not pass validation.");
      const spatialOptions = fileToMappingOptions(spatial);
      const referenceOptions = fileToMappingOptions(reference);
      setSessionId(response.session_id);
      setDatasets({ spatial, reference });
      setOptions({ spatial: spatialOptions, reference: referenceOptions });
      setMappings({
        spatial: { ...initialMapping("spatial"), cell_type_field: "cell_type" },
        reference: { ...initialMapping("reference"), cell_type_field: "cell_type" },
      });
      setMappingDirty({ spatial: false, reference: false });
      setPreviousText(response.previous_genes);
      setPanelSize("5");
      setSeedText("42");
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setExampleBusy(false);
    }
  }

  async function importPreviousFile(file?: File) {
    if (!file) return;
    const lines = (await file.text()).replace(/^\uFEFF/, "").split(/\r?\n/);
    const firstCell = lines[0]?.split(",")[0]?.trim().replace(/^"|"$/g, "").toLowerCase();
    const hasHeader = ["gene", "gene_id", "gene id", "gene_identifier", "identifier"].includes(firstCell ?? "");
    const text = (hasHeader ? lines.slice(1) : lines).join("\n");
    setPreviousText(text);
    clearResults();
  }

  async function runSelection() {
    if (!validForSelection) return;
    setSelectionBusy(true);
    setError("");
    setTechnicalError("");
    clearResults();
    try {
      const response = await api.select({
        session_id: sessionId,
        previous_genes: previous,
        ng: parsedNg,
        seed: seedText.trim() ? parsedSeed : null,
      });
      setResult(response);
      setSelectedGene(response.selected_genes[0] ?? null);
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setSelectionBusy(false);
    }
  }

  useEffect(() => {
    if (!sessionId || !datasets.spatial?.valid) return;
    let active = true;
    setPreviewLoading(true);
    void api.preview(
      sessionId,
      "spatial",
      selectedGene?.gene_id,
      previewColor === "auto"
        ? selectedGene
          ? undefined
          : mappings.spatial.cell_type_field || undefined
        : previewColor === "gene" && selectedGene
          ? `gene:${selectedGene.gene_id}`
          : previewColor === "none"
            ? "none"
            : previewColor,
    ).then((value) => {
      if (active) setPreview(value);
    }).catch(() => {
      if (active) setPreview(null);
    }).finally(() => {
      if (active) setPreviewLoading(false);
    });
    return () => { active = false; };
  }, [sessionId, datasets.spatial, datasets.reference, selectedGene, mappings.spatial.cell_type_field, previewColor]);

  useEffect(() => {
    if (!sessionId || !selectedGene) {
      setReferencePreview(null);
      setSpatialGenePreview(null);
      return;
    }
    let active = true;
    void api.preview(sessionId, "spatial", selectedGene.gene_id, `gene:${selectedGene.gene_id}`).then((value) => {
      if (active) setSpatialGenePreview(value);
    }).catch(() => {
      if (active) setSpatialGenePreview(null);
    });
    void api.preview(sessionId, "reference", selectedGene.gene_id).then((value) => {
      if (active) setReferencePreview(value);
    }).catch(() => {
      if (active) setReferencePreview(null);
    });
    return () => { active = false; };
  }, [sessionId, datasets.spatial, datasets.reference, selectedGene]);

  function exportGenes(format: "csv" | "txt") {
    if (!result) return;
    if (format === "txt") {
      download("selected-genes.txt", `${result.selected_genes.map((gene) => gene.gene_id).join("\n")}\n`, "text/plain");
      return;
    }
    const rows = [
      ["gene_id", "display_name", "rank", "demo_score", "score_provenance"],
      ...result.selected_genes.map((gene) => [gene.gene_id, gene.display_name, gene.rank, gene.score, gene.score_provenance]),
    ];
    download("selected-genes.csv", `${rows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`, "text/csv");
  }

  function exportSummary() {
    if (!result) return;
    download("gene-panel-run-summary.json", `${JSON.stringify(result.run_summary, null, 2)}\n`, "application/json");
  }

  const filteredChartGenes = result?.selected_genes.slice(0, 8) ?? [];
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Fieldnote home">
          <span className="brand-icon"><i /><i /><i /><i /></span>
          <span>fieldnote<span className="brand-dot">.</span></span>
        </a>
        <div className="topbar-right">
          <span className="prototype-chip">RESEARCH PROTOTYPE</span>
          <a className="help-link" href="#about">About this demo <span aria-hidden="true">↗</span></a>
        </div>
      </header>

      <main>
        <section className="intro">
          <div className="intro-copy">
            <div className="eyebrow">SPATIAL TRANSCRIPTOMICS · PANEL DESIGN</div>
            <h1>Plan your next<br /><em>measurement round.</em></h1>
            <p>Prepare a gene panel from your spatial data, check dataset overlap, and explore a transparent demo selection.</p>
          </div>
          <button className="button button-example" onClick={() => void tryExample()} disabled={exampleBusy}>
            <span aria-hidden="true">✳</span>{exampleBusy ? "Loading example…" : "Try example data"}
          </button>
        </section>

        <nav className="workflow" aria-label="Workflow stages">
          <a href="#data" className={`workflow-step ${datasets.spatial ? "step-done" : "step-current"}`}><span>01</span> Upload & check data</a>
          <a href="#configure" className={`workflow-step ${allReady ? "step-current" : ""}`}><span>02</span> Configure gene panel</a>
          <a href="#results" className={`workflow-step ${result ? "step-done" : ""}`}><span>03</span> Explore & export</a>
        </nav>

        {error && (
          <div className="global-error" role="alert">
            <div><strong>We couldn’t complete that step.</strong><span>{error}</span></div>
            {technicalError && <details><summary>Technical details</summary><code>{technicalError}</code></details>}
            <button onClick={() => { setError(""); setTechnicalError(""); }} aria-label="Dismiss error">×</button>
          </div>
        )}

        <section className="workflow-section" id="data">
          <div className="section-heading">
            <div className="stage-number">01</div>
            <div><div className="eyebrow">UPLOAD AND CHECK DATA</div><h2>Start with your datasets</h2><p>Both files are checked before any genes can be selected. Data stays in this app’s temporary session storage.</p></div>
          </div>
          <div className="dataset-grid">
            <UploadCard
              kind="spatial"
              summary={datasets.spatial}
              options={options.spatial}
              mapping={mappings.spatial}
              busy={busy.spatial}
              onUpload={(file) => void upload("spatial", file)}
              onMapping={(mapping) => applyMapping("spatial", mapping)}
              onMappingChange={() => { setMappingDirty((current) => ({ ...current, spatial: true })); clearResults(); setPreview(null); }}
              onChange={() => { clearResults(); setPreview(null); }}
            />
            <UploadCard
              kind="reference"
              summary={datasets.reference}
              options={options.reference}
              mapping={mappings.reference}
              busy={busy.reference}
              onUpload={(file) => void upload("reference", file)}
              onMapping={(mapping) => applyMapping("reference", mapping)}
              onMappingChange={() => { setMappingDirty((current) => ({ ...current, reference: true })); clearResults(); }}
              onChange={clearResults}
            />
          </div>
          {datasets.spatial?.valid && datasets.reference?.valid && (
            <div className="overlap-banner">
              <span className="overlap-icon">↔</span>
              <div><strong>{candidateSummary ? `${candidateSummary.reference_overlap_count.toLocaleString()} matching genes` : "Checking exact gene overlap…"}</strong>
                <span>Reference overlap is informational; this demo samples from spatial genes only and never adds reference-only genes.</span>
              </div>
              <span className="overlap-note">exact IDs only</span>
            </div>
          )}
          <div className="coordinate-preview">
            <div className="preview-heading">
              <div><div className="eyebrow">SPATIAL OVERVIEW</div><h3>Cells and spots</h3></div>
              <div className="preview-controls">
                {datasets.spatial && (
                  <label>Color by
                    <select aria-label="Color spatial preview by" value={previewColor} onChange={(event) => setPreviewColor(event.target.value)}>
                      <option value="auto">Automatic</option>
                      <option value="none">No color</option>
                      <option value="gene" disabled={!selectedGene}>Selected gene expression</option>
                      {datasets.spatial.observation_columns.map((column) => <option key={column} value={column}>{column}</option>)}
                    </select>
                  </label>
                )}
                {preview?.coordinate_dimensions === 3 && <span className="projection-tag">3D data · X–Y projection</span>}
              </div>
            </div>
            <ScatterPreview preview={preview} loading={previewLoading} />
          </div>
        </section>

        <section className="workflow-section configure-section" id="configure">
          <div className="section-heading">
            <div className="stage-number">02</div>
            <div><div className="eyebrow">CONFIGURE THE NEXT PANEL</div><h2>Choose what to measure</h2><p>Previous measurements are entered explicitly; uploading a matrix does not mark its genes as already measured.</p></div>
          </div>
          <div className="configure-grid">
            <div className="previous-entry">
              <div className="field-heading"><label htmlFor="previous-genes">Previously measured genes</label><span className="optional">Optional</span></div>
              <textarea
                id="previous-genes"
                value={previousText}
                onChange={(event) => { setPreviousText(event.target.value); clearResults(); }}
                placeholder={"Paste one gene identifier per line\nor separate with commas"}
                rows={5}
              />
              <div className="field-foot">
                <span>{previous.length} unique identifier{previous.length === 1 ? "" : "s"}</span>
                <label className="file-link">Upload CSV / TXT<input type="file" accept=".csv,.txt,text/plain,text/csv" onChange={(event) => void importPreviousFile(event.target.files?.[0])} /></label>
              </div>
              <p className="helper-text">Exact identifiers are matched as written. Single-column CSV/TXT lists are supported; common gene-ID headers are skipped. Matched genes are excluded and unmatched entries are reported.</p>
              {candidateSummary && candidateSummary.unmatched_previous_genes.length > 0 && (
                <p className="previous-warning">
                  {candidateSummary.matched_previous_count} matched and excluded · {candidateSummary.unmatched_previous_genes.length} not found:
                  {" "}{candidateSummary.unmatched_previous_genes.join(", ")}
                </p>
              )}
            </div>
            <div className="panel-settings">
              <div className="field-heading"><label htmlFor="ng-input">Additional genes to select</label><span className="required">Required</span></div>
              <input
                id="ng-input"
                type="number"
                min="1"
                max={eligibleCount || undefined}
                step="1"
                value={panelSize}
                onChange={(event) => { setPanelSize(event.target.value); clearResults(); }}
                aria-invalid={Boolean(datasets.spatial && !ngValid)}
              />
              {datasets.spatial && !ngValid && <p className="input-error" role="alert">Enter a whole number from 1 to {eligibleCount} eligible genes.</p>}
              <div className="pool-summary">
                <div><span>Candidate genes</span><strong>{candidateSummary?.candidate_count.toLocaleString() ?? "—"}</strong></div>
                <div><span>Previous IDs supplied</span><strong>{candidateSummary?.previous_count.toLocaleString() ?? "—"}</strong></div>
                <div><span>Eligible remaining</span><strong>{datasets.spatial ? eligibleCount.toLocaleString() : "—"}</strong></div>
              </div>
              <details className="advanced-settings">
                <summary>Advanced settings</summary>
                <label htmlFor="seed-input">Random seed <span className="optional">Optional · fixed seed repeats this selection</span></label>
                <input id="seed-input" type="number" step="1" value={seedText} placeholder="Generated automatically" onChange={(event) => { setSeedText(event.target.value); clearResults(); }} />
                {!seedValid && <p className="input-error" role="alert">Use a whole-number seed within the safe integer range.</p>}
              </details>
              <div className="method-note">
                <span className="method-mark">↗</span>
                <span><strong>Demo method</strong> · reproducible random sampling<br />The reference dataset is validated but is not used to filter or rank candidates.</span>
              </div>
              <button className="button button-primary run-button" disabled={!validForSelection || selectionBusy} onClick={() => void runSelection()}>
                {selectionBusy ? <><span className="spinner" /> Selecting genes…</> : "Select next-round genes"}
                {!selectionBusy && <span aria-hidden="true">→</span>}
              </button>
              {!allReady && <p className="helper-text centered">Upload and validate both datasets to continue.</p>}
            </div>
          </div>
        </section>

        <section className="workflow-section results-section" id="results">
          <div className="section-heading">
            <div className="stage-number">03</div>
            <div><div className="eyebrow">EXPLORE AND EXPORT RESULTS</div><h2>Your next-round panel</h2><p>Results are tied to the current data, mappings, exclusions, panel size, and seed.</p></div>
          </div>
          {!result ? (
            <div className="empty-results">
              <div className="empty-art" aria-hidden="true"><span>G</span><i /><i /><i /></div>
              <strong>Your selected genes will appear here</strong>
              <span>Complete the dataset checks and select a panel to view the result.</span>
            </div>
          ) : (
            <div className="results-content">
              <div className="result-banner">
                <div><span className="demo-label"><span /> Demo method</span><h3>{result.selected_genes.length} genes selected for the next measurement round</h3>
                  <span>Seed {result.seed} · {result.eligible_count.toLocaleString()} eligible from {result.candidate_count.toLocaleString()} spatial genes</span>
                </div>
                <div className="export-buttons">
                  <button className="button button-outline" onClick={() => exportGenes("csv")}>↓ CSV</button>
                  <button className="button button-outline" onClick={() => exportGenes("txt")}>↓ Gene list</button>
                  <button className="button button-outline" onClick={exportSummary}>↓ Run JSON</button>
                </div>
              </div>
              <div className="results-grid">
                <section className="result-panel chart-panel">
                  <div className="panel-title"><div><div className="eyebrow">ORDERED BY DEMO SCORE</div><h3>Selection scores</h3></div><span className="demo-score-tag">DEMO SCORE</span></div>
                  <div className="bar-chart">
                    {filteredChartGenes.map((gene) => (
                      <button key={gene.gene_id} className={`bar-row ${selectedGene?.gene_id === gene.gene_id ? "bar-selected" : ""}`} onClick={() => setSelectedGene(gene)}>
                        <span className="bar-label">{gene.display_name}</span>
                        <span className="bar-track"><i style={{ width: `${gene.score * 100}%` }} /></span>
                        <span className="bar-number">{gene.score.toFixed(3)}</span>
                      </button>
                    ))}
                  </div>
                  <p className="helper-text">Synthetic deterministic scores; future expected information gain is not implemented.</p>
                </section>
                <section className="result-panel detail-panel">
                  <div className="panel-title"><div><div className="eyebrow">GENE DETAIL</div><h3>{selectedGene?.display_name ?? "Select a gene"}</h3></div></div>
                  {selectedGene ? (
                    <>
                      <div className="gene-meta"><span>Stable ID</span><code>{selectedGene.gene_id}</code><span>Rank</span><strong>#{selectedGene.rank}</strong></div>
                      <p className="detail-score"><span>Demo score</span><strong>{selectedGene.score.toFixed(4)}</strong></p>
                      <ExpressionPreview label="Spatial dataset" preview={spatialGenePreview} />
                      <ExpressionPreview label="Reference dataset" preview={referencePreview} />
                      <p className="detail-notice">Expression values are real sampled values from the uploaded matrix where available. Scores are synthetic.</p>
                    </>
                  ) : <p className="helper-text">Choose a gene above to inspect its identifier and data availability.</p>}
                </section>
              </div>
              <ResultsTable genes={result.selected_genes} search={search} setSearch={setSearch} selected={selectedGene} onSelect={setSelectedGene} />
              {result.unmatched_previous_genes.length > 0 && (
                <div className="unmatched-note"><strong>{result.unmatched_previous_genes.length} previous identifier(s) were not found in spatial genes.</strong><span>{result.unmatched_previous_genes.join(", ")}</span></div>
              )}
              <div className="result-facts">
                <span><strong>{result.candidate_count}</strong> spatial candidate genes</span>
                <span><strong>{result.matched_previous_count}</strong> previous genes excluded</span>
                <span><strong>{result.reference_overlap_count}</strong> exact reference matches</span>
              </div>
              <EvaluationPanel result={result} />
            </div>
          )}
        </section>
        <section className="about-demo" id="about">
          <span className="about-mark">i</span>
          <div><strong>About this prototype</strong><p>“Next round” means the next measurement round, not a prediction of biological change over time. The candidate universe is the spatial dataset’s selected gene identifiers; reference-only genes are not added. The random selector and demo evaluation are placeholders.</p></div>
        </section>
      </main>
      <footer><span>FIELDNOTE · GENE PANEL PLANNER</span><span>Prototype for research workflows</span></footer>
    </div>
  );
}
