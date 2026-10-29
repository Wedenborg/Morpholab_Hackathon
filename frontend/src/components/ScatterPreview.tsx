import type { PreviewResponse } from "../types";

interface Props {
  preview: PreviewResponse | null;
  loading: boolean;
}

function colorFor(value: string | number | null, index: number, values: (string | number | null)[]) {
  if (typeof value === "number") {
    const numbers = values.filter((item): item is number => typeof item === "number");
    const low = Math.min(...numbers);
    const high = Math.max(...numbers);
    const t = high === low ? 0.5 : (value - low) / (high - low);
    return `hsl(${188 - t * 155} 48% ${45 - t * 9}%)`;
  }
  if (typeof value === "string") {
    const palette = ["#187c77", "#d19150", "#7c75a7", "#799c68", "#d17871", "#5687a3"];
    const categories = [...new Set(values.filter((item): item is string => typeof item === "string"))];
    return palette[categories.indexOf(value) % palette.length];
  }
  return index % 2 === 0 ? "#19827b" : "#c2ccc6";
}

export default function ScatterPreview({ preview, loading }: Props) {
  if (loading) return <div className="plot-message">Preparing a sampled preview…</div>;
  if (!preview?.available) {
    return <div className="plot-message">{preview?.message ?? "Spatial preview will appear after upload."}</div>;
  }
  if (!preview.points.length) {
    return <div className="plot-message">This dataset has no coordinates to plot. Expression values are still available.</div>;
  }
  const xs = preview.points.map((point) => point.x);
  const ys = preview.points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 1e-9);
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  const scale = 176 / span;
  return (
    <div className="scatter-wrap">
      <svg className="scatter" viewBox="0 0 300 230" role="img" aria-label="Spatial coordinate preview with equal axis scaling">
        <line x1="25" y1="205" x2="280" y2="205" className="axis" />
        <line x1="25" y1="18" x2="25" y2="205" className="axis" />
        {preview.points.map((point, index) => (
          <circle
            key={`${point.x}-${point.y}-${index}`}
            cx={150 + (point.x - midX) * scale}
            cy={112 - (point.y - midY) * scale}
            r="2.7"
            fill={colorFor(preview.color_values[index] ?? null, index, preview.color_values)}
            fillOpacity="0.76"
          />
        ))}
      </svg>
      <div className="plot-caption">
        <span>{preview.color_label}</span>
        <span>
          {preview.total_observations.toLocaleString()} {preview.kind === "spatial" ? "cells / spots" : "cells"}
          {preview.sampled ? " · sampled for display" : ""}
        </span>
      </div>
    </div>
  );
}
