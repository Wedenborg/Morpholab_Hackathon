import { useEffect, useMemo, useState } from "react";

interface PrerunMetric { id: string; label: string; true_data: number | null; higher_is_better: boolean }
interface PrerunRound { round: number; n_measured: number; genes: string[]; metrics: Record<string, number> }
interface PrerunRun { method: string; label: string; ng: number; rounds: PrerunRound[] }
interface Prerun {
  dataset: string;
  previous_genes: string[];
  notice: string;
  metrics: PrerunMetric[];
  runs: PrerunRun[];
}

const COLORS: Record<string, string> = {
  gaussian_visible: "#13766f",
  rotating_visible: "#c98d4f",
  dummy_markers: "#ae4f48",
  dummy_variance: "#5b6fa8",
  dummy_random: "#8b8f8c",
};
const W = 640;
const H = 300;
const PAD = { left: 52, right: 16, top: 14, bottom: 36 };

function ticks(max: number): number[] {
  const step = max <= 10 ? 2 : max <= 60 ? 10 : max <= 120 ? 20 : max <= 500 ? 100 : 200;
  const out: number[] = [];
  for (let value = 0; value <= max; value += step) out.push(value);
  return out;
}

export default function PrerunExplorer() {
  const [data, setData] = useState<Prerun | null>(null);
  const [failed, setFailed] = useState(false);
  const [metricId, setMetricId] = useState("");
  const [ng, setNg] = useState(0);
  const [xAxis, setXAxis] = useState<"round" | "genes">("round");
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  useEffect(() => {
    let active = true;
    void fetch(`${import.meta.env.BASE_URL}prerun/e75_prerun.json`)
      .then((response) => (response.ok ? (response.json() as Promise<Prerun>) : Promise.reject(new Error("missing"))))
      .then((value) => {
        if (!active) return;
        setData(value);
        setMetricId(value.metrics[0]?.id ?? "");
        setNg(Math.min(...value.runs.map((run) => run.ng)));
      })
      .catch(() => { if (active) setFailed(true); });
    return () => { active = false; };
  }, []);

  const sizes = useMemo(() => [...new Set(data?.runs.map((run) => run.ng) ?? [])].sort((a, b) => a - b), [data]);
  const metric = data?.metrics.find((item) => item.id === metricId);
  const runs = useMemo(() => (data?.runs ?? []).filter((run) => run.ng === ng), [data, ng]);

  if (failed) return <p className="helper-text">Pre-run results are not available in this build.</p>;
  if (!data || !metric) return <p className="helper-text">Loading pre-run results…</p>;

  const series = runs.map((run) => ({
    run,
    points: run.rounds
      .filter((point) => point.metrics[metric.id] !== undefined)
      .map((point) => ({ x: xAxis === "round" ? point.round : point.n_measured, y: point.metrics[metric.id] })),
  }));
  const xs = series.flatMap((item) => item.points.map((point) => point.x));
  const ys = [...series.flatMap((item) => item.points.map((point) => point.y)), ...(metric.true_data !== null ? [metric.true_data] : [])];
  const xMax = Math.max(1, ...xs);
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(0.01, ...ys) * 1.08;
  const sx = (x: number) => PAD.left + (x / xMax) * (W - PAD.left - PAD.right);
  const sy = (y: number) => H - PAD.bottom - ((y - yMin) / (yMax - yMin)) * (H - PAD.top - PAD.bottom);
  const yTicks = Array.from({ length: 5 }, (_, i) => yMin + ((yMax - yMin) * i) / 4);

  const finals = series
    .filter((item) => item.points.length)
    .map((item) => ({ item, last: item.points[item.points.length - 1] }))
    .sort((a, b) => (metric.higher_is_better ? b.last.y - a.last.y : a.last.y - b.last.y));

  return (
    <div className="prerun">
      <div className="prerun-controls">
        <div className="prerun-group" role="group" aria-label="Genes per round">
          <span className="eyebrow">Genes per round</span>
          {sizes.map((size) => (
            <button key={size} className={`chip ${size === ng ? "chip-on" : ""}`} onClick={() => setNg(size)}>{size}</button>
          ))}
        </div>
        <div className="prerun-group" role="group" aria-label="Score">
          <span className="eyebrow">Score</span>
          {data.metrics.map((item) => (
            <button key={item.id} className={`chip ${item.id === metricId ? "chip-on" : ""}`} onClick={() => setMetricId(item.id)}>{item.label}</button>
          ))}
        </div>
        <div className="prerun-group" role="group" aria-label="X axis">
          <span className="eyebrow">X axis</span>
          <button className={`chip ${xAxis === "round" ? "chip-on" : ""}`} onClick={() => setXAxis("round")}>Rounds</button>
          <button className={`chip ${xAxis === "genes" ? "chip-on" : ""}`} onClick={() => setXAxis("genes")}>Genes measured</button>
        </div>
      </div>

      <section className="result-panel prerun-chart">
        <div className="panel-title">
          <div><div className="eyebrow">{metric.higher_is_better ? "HIGHER IS BETTER" : "LOWER IS BETTER"}</div><h3>{metric.label}</h3></div>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${metric.label} by ${xAxis} for each method`}>
          {yTicks.map((value) => (
            <g key={value}>
              <line x1={PAD.left} x2={W - PAD.right} y1={sy(value)} y2={sy(value)} stroke="#e2e7e1" />
              <text x={PAD.left - 8} y={sy(value) + 3} textAnchor="end" fontSize="10" fill="#71817c">{value.toFixed(2)}</text>
            </g>
          ))}
          {ticks(xMax).map((value) => (
            <text key={value} x={sx(value)} y={H - PAD.bottom + 16} textAnchor="middle" fontSize="10" fill="#71817c">{value}</text>
          ))}
          <text x={(PAD.left + W - PAD.right) / 2} y={H - 4} textAnchor="middle" fontSize="10" fill="#71817c">
            {xAxis === "round" ? `Round (each adds ${ng} genes)` : "Genes measured (incl. 4 previous)"}
          </text>
          {metric.true_data !== null && (
            <g>
              <line x1={PAD.left} x2={W - PAD.right} y1={sy(metric.true_data)} y2={sy(metric.true_data)} stroke="#203b39" strokeDasharray="5 4" />
              <text x={W - PAD.right} y={sy(metric.true_data) - 4} textAnchor="end" fontSize="9" fill="#203b39">True-data reference {metric.true_data.toFixed(3)}</text>
            </g>
          )}
          {series.filter((item) => !hidden.has(item.run.method)).map(({ run, points }) => (
            <g key={run.method}>
              <polyline fill="none" stroke={COLORS[run.method]} strokeWidth="2" points={points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
              {points.map((p) => (
                <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r="3" fill={COLORS[run.method]}>
                  <title>{`${run.label}: ${p.y.toFixed(3)} at ${xAxis === "round" ? "round" : "genes"} ${p.x}`}</title>
                </circle>
              ))}
            </g>
          ))}
        </svg>
        <div className="prerun-legend">
          {finals.map(({ item, last }) => (
            <button
              key={item.run.method}
              className={`legend-item ${hidden.has(item.run.method) ? "legend-off" : ""}`}
              onClick={() => setHidden((current) => {
                const next = new Set(current);
                if (next.has(item.run.method)) next.delete(item.run.method); else next.add(item.run.method);
                return next;
              })}
            >
              <i style={{ background: COLORS[item.run.method] }} />
              <span>{item.run.label}</span>
              <strong>{last.y.toFixed(3)}</strong>
            </button>
          ))}
        </div>
      </section>
      <p className="helper-text">{data.notice} Previous genes: {data.previous_genes.join(", ")}. Scores are computed at selected rounds only (rounds 0, 1, 2, 5, 10, 25, 50, 100, up to about 420 measured genes). Click a legend entry to hide a method.</p>
    </div>
  );
}
