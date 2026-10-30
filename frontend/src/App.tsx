import { useEffect, useMemo, useState } from "react";
import { api, errorTechnical } from "./api";
import MappingPanel from "./components/MappingPanel";
import ScatterPreview from "./components/ScatterPreview";
import type {
  DatasetKind,
  DatasetSummary,
  CandidateSummary,
  EvaluationStep,
  MappingForm,
  MappingOptions,
  PreviewResponse,
  PresetSessionResponse,
  SelectionMethod,
  SelectedGene,
  SelectionResponse,
} from "./types";

type DatasetMap<T> = Record<DatasetKind, T>;

const DESIGN_METHODS: Partial<Record<SelectionMethod, { label: string; note: string }>> = {
  gaussian_visible: {
    label: "Gaussian information gain · visible embryo",
    note: "Factor model fitted on the reference (visible) embryo's neighbourhoods; genes chosen greedily by expected information gain about the target's factors, given the genes already measured.",
  },
  gaussian_mdcla: {
    label: "Gaussian information gain · mdCLA",
    note: "As above, with factor directions and gene weights from mdCLA E7.5 single cells, rescaled per gene to the visible embryo.",
  },
  rotating_visible: {
    label: "Rotating markers · visible embryo",
    note: "Each cell type's next-best marker in turn, from the reference (visible) embryo's cell types.",
  },
  rotating_mdcla: {
    label: "Rotating markers · mdCLA",
    note: "Each cell subtype's next-best marker in turn, from mdCLA E7.5 single cells.",
  },
};

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

function selectionMethodFromValue(value: string): SelectionMethod {
  switch (value) {
    case "dummy_markers":
    case "dummy_variance":
    case "dummy_random":
    case "manual":
    case "gaussian_visible":
    case "gaussian_mdcla":
    case "rotating_visible":
    case "rotating_mdcla":
      return value;
    default:
      return "random";
  }
}

function fileToMappingOptions(summary: DatasetSummary): MappingOptions {
  return {
    gene_fields: ["var_names", ...summary.gene_columns],
    layers: ["X", ...summary.layer_names],
    obsm: summary.obsm_keys,
    obs_columns: summary.observation_columns,
  };
}

function mappingForSummary(summary: DatasetSummary): MappingForm {
  const coordinateSource = summary.coordinate_source === "obs" ? "obs" : "obsm";
  const coordinateField = summary.coordinate_field ?? "spatial";
  return {
    ...initialMapping(summary.kind),
    gene_field: summary.gene_field,
    display_field: summary.display_field,
    matrix_layer: summary.matrix_layer,
    coordinate_source: coordinateSource,
    coordinate_field: coordinateSource === "obsm" ? coordinateField : "spatial",
    coordinate_columns: coordinateSource === "obs"
      ? coordinateField.split(",").map((column) => column.trim()).filter(Boolean)
      : [],
    cell_type_field: summary.cell_type_field ?? "",
    transpose: summary.transpose_applied,
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

function csvEscape(value: string | number | null) {
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
  scoreLabel,
  synthetic,
  search,
  setSearch,
  selected,
  onSelect,
}: {
  genes: SelectedGene[];
  scoreLabel: string;
  synthetic: boolean;
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
          <thead><tr><th>Rank</th><th>Gene</th><th>Stable identifier</th><th>{scoreLabel}</th></tr></thead>
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
                <td><span className="score-value">{gene.score === null ? "—" : gene.score.toFixed(3)}</span></td>
              </tr>
            ))}
            {!filtered.length && <tr><td colSpan={4} className="empty-table">No selected genes match this search.</td></tr>}
          </tbody>
        </table>
      </div>
      {synthetic && <p className="table-footnote">Scores are synthetic demo scores, not estimated information gain, probabilities, or evidence of biological usefulness.</p>}
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

const EXPECTED_SHARE_TIP = "The Gaussian model's own expectation of how much of the unmeasured genes' neighbourhood-level variance it predicts, given the genes sequenced so far. Fitted on the visible embryo; it is not compared with this embryo.";

type MetricInfo = { id: string; label: string; explanation?: string };

function StepsPanel({ steps, metrics }: { steps: EvaluationStep[]; metrics: MetricInfo[] }) {
  const width = 520;
  const height = 170;
  const left = 36;
  const right = 500;
  const x = (index: number) => left + (steps.length > 1 ? (index / (steps.length - 1)) * (right - left) : (right - left) / 2);
  const shown = metrics.filter((metric) => steps.some((step) => step.benchmark?.[metric.id] !== undefined));
  const series = [
    ...(shown.some((metric) => metric.id === "heldout_corr")
      ? [{ label: "Accuracy on unmeasured genes (eval_team)", value: (step: EvaluationStep) => step.benchmark?.heldout_corr ?? null, alt: false }]
      : []),
    { label: "Expected share predicted (model)", value: (step: EvaluationStep) => step.expected_share, alt: true },
  ];
  const values = steps.flatMap((step) => series.map((item) => item.value(step))).filter((value): value is number => value !== null);
  const max = Math.max(0.05, ...values);
  const y = (value: number) => height - 24 - (value / max) * (height - 44);
  const labelEvery = Math.ceil(steps.length / 10);
  return (
    <div className="steps-panel">
      <h4>Scores per round</h4>
      <div className="evaluation-chart">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Scores per round">
          {[0, 0.25, 0.5, 0.75, 1].map((fraction) => (
            <g key={fraction}>
              <line x1="30" x2="505" y1={y(fraction * max)} y2={y(fraction * max)} className="chart-grid" />
              <text x="26" y={y(fraction * max) + 3} textAnchor="end">{(fraction * max).toFixed(2)}</text>
            </g>
          ))}
          {series.map((item) => {
            const pts = steps
              .map((step, index) => ({ index, value: item.value(step) }))
              .filter((point): point is { index: number; value: number } => point.value !== null);
            const d = pts.map((point, i) => `${i ? "L" : "M"} ${x(point.index)} ${y(point.value)}`).join(" ");
            return (
              <g key={item.label}>
                <path d={d} className={item.alt ? "chart-path chart-path-alt" : "chart-path"} />
                {pts.map((point) => (
                  <circle key={point.index} cx={x(point.index)} cy={y(point.value)} r="3.5" className={item.alt ? "chart-point chart-point-alt" : "chart-point"} />
                ))}
              </g>
            );
          })}
          {steps.map((step, index) => index % labelEvery === 0 && (
            <text key={step.round} x={x(index)} y={height - 6} textAnchor="middle">{step.n_measured} genes</text>
          ))}
        </svg>
        <div className="chart-legend">
          {series.map((item) => <span key={item.label} className={item.alt ? "legend-alt" : "legend-main"}><i />{item.label}</span>)}
        </div>
      </div>
      <div className="table-scroll">
        <table className="steps-table">
          <thead>
            <tr>
              <th>Round</th>
              <th>Genes sequenced</th>
              {shown.map((metric) => <th key={metric.id} className="has-tip" data-tip={metric.explanation} tabIndex={0}>{metric.label}</th>)}
              <th className="has-tip" data-tip={EXPECTED_SHARE_TIP} tabIndex={0}>Expected share (model)</th>
              <th>Genes added</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((step) => (
              <tr key={step.round}>
                <td>{step.round === 0 ? "0 (previous)" : step.round}</td>
                <td>{step.n_measured}</td>
                {shown.map((metric) => <td key={metric.id}>{step.benchmark?.[metric.id]?.toFixed(3) ?? "—"}</td>)}
                <td>{step.expected_share === null ? "—" : step.expected_share.toFixed(3)}</td>
                <td className="steps-genes">{step.genes.join(", ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EvaluationPanel({ result }: { result: SelectionResponse }) {
  return (
    <section className="evaluation-panel">
      <div className="evaluation-head">
        <div><span className="demo-label"><span /> Repository benchmark</span><h3>{result.evaluation.label}</h3></div>
        <span className="benchmark-tag">{result.evaluation.available ? "EVAL_TEAM" : "NOT RUN"}</span>
      </div>
      <p>{result.evaluation.notice}</p>
      {result.evaluation.available ? (
        <>
          <div className="evaluation-metrics">
            {result.evaluation.metrics.map((metric) => (
              <article className="evaluation-metric has-tip" key={metric.id} data-tip={metric.explanation} tabIndex={0}>
                <span>{metric.label}{metric.explanation && <i className="tip-mark" aria-hidden="true">i</i>}</span>
                <strong>{metric.value.toFixed(3)}</strong>
                <small>{metric.description}</small>
                {metric.true_data !== null && (
                  <small>True-data reference: {metric.true_data.toFixed(3)}</small>
                )}
              </article>
            ))}
          </div>
          <div className="evaluation-meta">
            {result.evaluation.measured_gene_count !== undefined && (
              <span>{result.evaluation.measured_gene_count} measured genes</span>
            )}
            {result.evaluation.experiment_count !== undefined && (
              <span>{result.evaluation.experiment_count.toLocaleString()} logged section experiments</span>
            )}
            {result.evaluation.external_references?.length ? (
              <span>External references: {result.evaluation.external_references.join(", ")}</span>
            ) : null}
          </div>
          {result.evaluation.summary && (
            <details className="evaluation-details">
              <summary>Full eval_team report</summary>
              <pre>{JSON.stringify(result.evaluation.summary, null, 2)}</pre>
            </details>
          )}
        </>
      ) : (
        <p className="evaluation-unavailable">This run has no benchmark scores.</p>
      )}
      {result.evaluation.steps?.length ? <StepsPanel steps={result.evaluation.steps} metrics={result.evaluation.metrics} /> : null}
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
  const [selectionMethod, setSelectionMethod] = useState<SelectionMethod>("random");
  const [e75Preset, setE75Preset] = useState(false);
  const [manualMode, setManualMode] = useState(false);
  const [chosenGenesText, setChosenGenesText] = useState("");
  const [seedText, setSeedText] = useState("");
  const [roundsText, setRoundsText] = useState("5");
  const [result, setResult] = useState<SelectionResponse | null>(null);
  const [addedToPrevious, setAddedToPrevious] = useState(0);
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
  const [mouseBusy, setMouseBusy] = useState(false);

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
  const isEvalTeamMethod = selectionMethod.startsWith("dummy_");
  const designMethod = DESIGN_METHODS[selectionMethod];
  const evalTeamMethodReady = e75Preset
    && mappings.spatial.gene_field === "var_names"
    && mappings.spatial.matrix_layer === "X"
    && !mappings.spatial.transpose;
  const parsedNg = Number(panelSize);
  const parsedRounds = Number(roundsText);
  const roundsValid = Number.isInteger(parsedRounds) && parsedRounds >= 1 && parsedRounds <= 50;
  const ngValid = Number.isInteger(parsedNg) && parsedNg > 0 && parsedNg <= eligibleCount
    && (!isEvalTeamMethod || (evalTeamMethodReady && parsedNg % 4 === 0));
  const parsedSeed = Number(seedText);
  const seedValid = !seedText.trim() || Number.isSafeInteger(parsedSeed);
  const validForSelection = allReady && ngValid && seedValid && (!designMethod || roundsValid) && Boolean(sessionId)
    && !mappingDirty.spatial && !mappingDirty.reference && !busy.spatial && !busy.reference
    && !exampleBusy && !mouseBusy && (!isEvalTeamMethod || evalTeamMethodReady);

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
    setAddedToPrevious(0);
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
      setE75Preset(false);
      setSelectionMethod("random");
      setManualMode(false);
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

  async function installPreset(
    response: PresetSessionResponse,
    defaultPanelSize: string,
    defaultSeed: string,
    isMouseEmbryo = false,
  ) {
    const sessionResponse = await api.getSession(response.session_id);
    const spatial = sessionResponse.datasets.spatial;
    const reference = sessionResponse.datasets.reference;
    if (!spatial || !reference) throw new Error("The selected datasets did not pass validation.");
    setSessionId(response.session_id);
    setDatasets({ spatial, reference });
    setOptions({
      spatial: fileToMappingOptions(spatial),
      reference: fileToMappingOptions(reference),
    });
    setMappings({
      spatial: mappingForSummary(spatial),
      reference: mappingForSummary(reference),
    });
    setMappingDirty({ spatial: false, reference: false });
    setPreviousText(response.previous_genes);
    setPanelSize(String(response.panel_size ?? defaultPanelSize));
    setSeedText(String(response.seed ?? defaultSeed));
    setE75Preset(isMouseEmbryo);
    setSelectionMethod("random");
    setManualMode(false);
  }

  function preparePreset() {
    setError("");
    setTechnicalError("");
    clearResults();
    setPreview(null);
    setPreviewLoading(false);
    setE75Preset(false);
    setSelectionMethod("random");
    setManualMode(false);
  }

  async function tryExample() {
    setExampleBusy(true);
    preparePreset();
    try {
      await installPreset(await api.example(), "5", "42");
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setExampleBusy(false);
    }
  }

  async function tryMouseEmbryo() {
    setMouseBusy(true);
    preparePreset();
    try {
      await installPreset(await api.mouseEmbryo(), "10", "42", true);
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setMouseBusy(false);
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

  async function submitSelection(method: SelectionMethod, ng: number, chosenGenes: string[] = []) {
    setSelectionBusy(true);
    setError("");
    setTechnicalError("");
    clearResults();
    try {
      const response = await api.select({
        session_id: sessionId,
        previous_genes: previous,
        method,
        chosen_genes: chosenGenes,
        ng,
        seed: seedText.trim() ? parsedSeed : null,
        rounds: parsedRounds,
      });
      setResult(response);
      setSelectedGene(response.selected_genes[0] ?? null);
      // Treat this round's genes as measured so the next round builds on them.
      const known = new Set(previous);
      const added = response.selected_genes.map((gene) => gene.gene_id).filter((id) => !known.has(id));
      if (added.length) setPreviousText([...previous, ...added].join("\n"));
      setAddedToPrevious(added.length);
      if (method === "manual") setChosenGenesText("");
    } catch (failure) {
      setErrorState(failure);
    } finally {
      setSelectionBusy(false);
    }
  }

  function runSelection() {
    if (!validForSelection) return;
    void submitSelection(selectionMethod, parsedNg);
  }

  function runChosenSelection() {
    const chosenGenes = splitGenes(chosenGenesText);
    if (!sessionId || !allReady || !chosenGenes.length) return;
    void submitSelection("manual", chosenGenes.length, chosenGenes);
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
      ["gene_id", "display_name", "rank", "score", "score_provenance"],
      ...result.selected_genes.map((gene) => [gene.gene_id, gene.display_name, gene.rank, gene.score, gene.score_provenance]),
    ];
    download("selected-genes.csv", `${rows.map((row) => row.map(csvEscape).join(",")).join("\n")}\n`, "text/csv");
  }

  function exportSummary() {
    if (!result) return;
    download("gene-panel-run-summary.json", `${JSON.stringify(result.run_summary, null, 2)}\n`, "application/json");
  }

  const maxScore = Math.max(1e-12, ...(result?.selected_genes.map((gene) => gene.score ?? 0) ?? [1]));
  const filteredChartGenes = result?.selected_genes.filter((gene) => gene.score !== null).slice(0, 8) ?? [];
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
            <p>Start with the locally available E7.5 mouse embryo datasets or upload your own, then plan a reproducible random gene panel.</p>
          </div>
          <div className="intro-actions">
            <button className="button button-example" onClick={() => void tryMouseEmbryo()} disabled={exampleBusy || mouseBusy}>
              <span aria-hidden="true">M</span>{mouseBusy ? "Loading E7.5 datasets…" : "Use E7.5 mouse embryo"}
            </button>
            <button className="button button-example button-example-secondary" onClick={() => void tryExample()} disabled={exampleBusy || mouseBusy}>
              {exampleBusy ? "Loading synthetic example…" : "Try synthetic example"}
            </button>
          </div>
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
            <div><div className="eyebrow">UPLOAD AND CHECK DATA</div><h2>Start with your datasets</h2><p>Both files are checked before any genes can be selected. Uploads use temporary session storage; the local preset references source files in place.</p></div>
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
              <label className="method-select-label" htmlFor="selection-method">Selection method</label>
              <select
                id="selection-method"
                value={selectionMethod}
                onChange={(event) => {
                  const nextMethod = selectionMethodFromValue(event.target.value);
                  setSelectionMethod(nextMethod);
                  if (nextMethod.startsWith("dummy_") && parsedNg % 4 !== 0) {
                    setPanelSize(String(Math.max(4, Math.floor(parsedNg / 4) * 4)));
                  }
                  clearResults();
                }}
              >
                <option value="random">Web random panel</option>
                {Object.entries(DESIGN_METHODS).map(([id, info]) => <option key={id} value={id}>{info?.label}</option>)}
                <option value="dummy_markers" disabled={!evalTeamMethodReady}>eval_team dummy · cell-type markers</option>
                <option value="dummy_variance" disabled={!evalTeamMethodReady}>eval_team dummy · variable genes</option>
                <option value="dummy_random" disabled={!evalTeamMethodReady}>eval_team dummy · random expressed genes</option>
              </select>
              <p className="helper-text">
                {designMethod
                  ? designMethod.note
                  : isEvalTeamMethod
                  ? "Uses the repository dummy reconstruction method. These methods measure genes in groups of four."
                  : "The web random selector samples spatial genes without replacement. Select the local E7.5 preset to enable eval_team dummy methods."}
              </p>
              <div className="field-heading">
                <label htmlFor="ng-input">{isEvalTeamMethod ? "Genes to measure (sets of four)" : designMethod ? "Genes sequenced per round" : "Additional genes to select"}</label>
                <span className="required">Required</span>
              </div>
              <input
                id="ng-input"
                type="number"
                min={isEvalTeamMethod ? "4" : "1"}
                max={isEvalTeamMethod ? Math.floor(eligibleCount / 4) * 4 || undefined : eligibleCount || undefined}
                step={isEvalTeamMethod ? "4" : "1"}
                value={panelSize}
                onChange={(event) => { setPanelSize(event.target.value); clearResults(); }}
                aria-invalid={Boolean(datasets.spatial && !ngValid)}
              />
              {datasets.spatial && !ngValid && (
                <p className="input-error" role="alert">
                  {isEvalTeamMethod
                    ? `Enter a multiple of 4 from 4 to ${Math.floor(eligibleCount / 4) * 4} genes.`
                    : `Enter a whole number from 1 to ${eligibleCount} eligible genes.`}
                </p>
              )}
              {designMethod && (
                <>
                  <div className="field-heading">
                    <label htmlFor="rounds-input">Rounds (experiment budget)</label>
                    <span className="optional">{roundsValid && ngValid ? `${parsedRounds} × ${parsedNg} = ${parsedRounds * parsedNg} genes sequenced` : "1–50"}</span>
                  </div>
                  <input
                    id="rounds-input"
                    type="number"
                    min="1"
                    max="50"
                    step="1"
                    value={roundsText}
                    onChange={(event) => { setRoundsText(event.target.value); clearResults(); }}
                    aria-invalid={!roundsValid}
                  />
                  {!roundsValid && <p className="input-error" role="alert">Enter a whole number of rounds from 1 to 50.</p>}
                  <p className="helper-text">Round 1 is the panel returned; later rounds are simulated with the same method and scored, so the chart shows how the scores grow with the number of genes sequenced.</p>
                </>
              )}
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
                <span>
                  <strong>{designMethod ? designMethod.label : isEvalTeamMethod ? "eval_team dummy method" : "Web random method"}</strong>
                  {designMethod
                    ? " · the spatial dataset contributes positions only; scores per round use eval_team on the E7.5 preset."
                    : isEvalTeamMethod
                    ? " · reconstructs from the reference embryo with nearest-neighbour matching."
                    : " · reproducible random sampling; the reference does not filter candidates."}
                </span>
              </div>
              <button className="button button-primary run-button" disabled={!validForSelection || selectionBusy} onClick={() => void runSelection()}>
                {selectionBusy ? <><span className="spinner" /> Running method and benchmark…</> : "Select next-round genes"}
                {!selectionBusy && <span aria-hidden="true">→</span>}
              </button>
              <button
                className="button button-outline other-genes-toggle"
                type="button"
                onClick={() => { setManualMode((current) => !current); clearResults(); }}
              >
                {manualMode ? "Cancel other genes" : "I chose other genes"}
              </button>
              {manualMode && (
                <div className="chosen-genes-form">
                  <label htmlFor="chosen-genes">Genes you chose</label>
                  <textarea
                    id="chosen-genes"
                    rows={4}
                    value={chosenGenesText}
                    onChange={(event) => { setChosenGenesText(event.target.value); clearResults(); }}
                    placeholder={"Enter gene IDs, separated by commas or new lines"}
                  />
                  <p className="helper-text">The exact genes you enter will be evaluated as your chosen panel. IDs must match the spatial dataset.</p>
                  <button
                    className="button button-primary"
                    type="button"
                    disabled={!allReady || !sessionId || !splitGenes(chosenGenesText).length || selectionBusy}
                    onClick={runChosenSelection}
                  >
                    {selectionBusy ? "Evaluating your genes…" : "Evaluate my chosen genes"}
                  </button>
                </div>
              )}
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
                <div><span className="demo-label"><span /> {result.method_label}</span><h3>{result.selected_genes.length} genes selected for the next measurement round</h3>
                  <span>Seed {result.seed} · {result.eligible_count.toLocaleString()} eligible from {result.candidate_count.toLocaleString()} spatial genes</span>
                </div>
                <div className="export-buttons">
                  <button className="button button-outline" onClick={() => exportGenes("csv")}>↓ CSV</button>
                  <button className="button button-outline" onClick={() => exportGenes("txt")}>↓ Gene list</button>
                  <button className="button button-outline" onClick={exportSummary}>↓ Run JSON</button>
                </div>
              </div>
              {addedToPrevious > 0 && (
                <div className="unmatched-note"><strong>{addedToPrevious} gene(s) added to “Previously measured genes”.</strong><span>The next selection will exclude them and use them as already measured.</span></div>
              )}
              <div className="results-grid">
                <section className="result-panel chart-panel">
                  <div className="panel-title"><div><div className="eyebrow">IN SELECTION ORDER</div><h3>Selection scores</h3></div><span className="demo-score-tag">{result.score_label.toUpperCase()}</span></div>
                  <div className="bar-chart">
                    {filteredChartGenes.map((gene) => (
                      <button key={gene.gene_id} className={`bar-row ${selectedGene?.gene_id === gene.gene_id ? "bar-selected" : ""}`} onClick={() => setSelectedGene(gene)}>
                        <span className="bar-label">{gene.display_name}</span>
                        <span className="bar-track"><i style={{ width: `${((gene.score ?? 0) / maxScore) * 100}%` }} /></span>
                        <span className="bar-number">{gene.score?.toFixed(3)}</span>
                      </button>
                    ))}
                    {!filteredChartGenes.length && <p className="helper-text">This method does not assign per-gene scores.</p>}
                  </div>
                  {filteredChartGenes.length > 0 && <p className="helper-text">{result.method === "random-demo-v1" ? "Synthetic deterministic ranking scores; not biological utility or information gain." : result.selected_genes[0]?.score_provenance}</p>}
                </section>
                <section className="result-panel detail-panel">
                  <div className="panel-title"><div><div className="eyebrow">GENE DETAIL</div><h3>{selectedGene?.display_name ?? "Select a gene"}</h3></div></div>
                  {selectedGene ? (
                    <>
                      <div className="gene-meta"><span>Stable ID</span><code>{selectedGene.gene_id}</code><span>Rank</span><strong>#{selectedGene.rank}</strong></div>
                      <p className="detail-score"><span>{result.score_label}</span><strong>{selectedGene.score?.toFixed(4) ?? "Not assigned"}</strong></p>
                      <ExpressionPreview label="Spatial dataset" preview={spatialGenePreview} />
                      <ExpressionPreview label="Reference dataset" preview={referencePreview} />
                      <p className="detail-notice">{result.method === "random-demo-v1" ? "Expression values are sampled from the datasets where available. Any displayed random-panel ranking score is synthetic." : selectedGene.score_provenance}</p>
                    </>
                  ) : <p className="helper-text">Choose a gene above to inspect its identifier and data availability.</p>}
                </section>
              </div>
              <ResultsTable genes={result.selected_genes} scoreLabel={result.score_label} synthetic={result.method === "random-demo-v1"} search={search} setSearch={setSearch} selected={selectedGene} onSelect={setSelectedGene} />
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
          <div><strong>About this prototype</strong><p>“Next round” means the next measurement round, not a prediction of biological change over time. The candidate universe is the spatial dataset’s selected gene identifiers; reference-only genes are not added. Gene-row scores are synthetic for the web random panel; the Gaussian methods report information gain and the rotating methods marker strength. For the local E7.5 preset, reconstruction metrics come from the repository’s eval_team benchmark; other datasets are not given fabricated evaluation scores.</p></div>
        </section>
      </main>
      <footer><span>FIELDNOTE · GENE PANEL PLANNER</span><span>Prototype for research workflows</span></footer>
    </div>
  );
}
