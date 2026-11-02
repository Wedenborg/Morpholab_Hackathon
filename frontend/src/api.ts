import type {
  DatasetKind,
  DatasetUploadResponse,
  CandidateSummary,
  MappingForm,
  PreviewResponse,
  PresetSessionResponse,
  SelectionMethod,
  SelectionResponse,
} from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, init);
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string | { message?: string; technical?: string } }
      | null;
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ?? `Request failed (${response.status}).`;
    const error = new Error(message);
    if (typeof detail === "object" && detail.technical) {
      Object.assign(error, { technical: detail.technical });
    }
    throw error;
  }
  return (await response.json()) as T;
}

export const api = {
  createSession: () => request<{ session_id: string }>("/api/sessions", { method: "POST" }),
  deleteSession: (id: string) =>
    request<{ deleted: boolean }>(`/api/sessions/${id}`, { method: "DELETE", keepalive: true }),
  upload: async (
    sessionId: string,
    kind: DatasetKind,
    file: File,
    mapping: MappingForm,
  ): Promise<DatasetUploadResponse> => {
    const body = new FormData();
    body.append("session_id", sessionId);
    body.append("file", file);
    body.append("gene_field", mapping.gene_field);
    body.append("display_field", mapping.display_field);
    body.append("matrix_layer", mapping.matrix_layer);
    body.append("coordinate_source", mapping.coordinate_source);
    body.append("coordinate_field", mapping.coordinate_field);
    body.append("coordinate_columns", mapping.coordinate_columns.join(","));
    body.append("cell_type_field", mapping.cell_type_field);
    body.append("transpose", String(mapping.transpose));
    return request(`/api/datasets/${kind}`, { method: "POST", body });
  },
  updateMapping: (sessionId: string, kind: DatasetKind, mapping: MappingForm) =>
    request<DatasetUploadResponse>(`/api/datasets/${kind}/mapping?session_id=${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(mapping),
    }),
  example: () =>
    request<PresetSessionResponse>("/api/example", { method: "POST" }),
  mouseEmbryo: () =>
    request<PresetSessionResponse>("/api/mouse-embryo", { method: "POST" }),
  getSession: (id: string) =>
    request<{ datasets: Record<string, import("./types").DatasetSummary> }>(`/api/sessions/${id}`),
  select: (payload: {
    session_id: string;
    previous_genes: string[];
    method: SelectionMethod;
    chosen_genes: string[];
    ng: number;
    seed: number | null;
    rounds: number;
  }) =>
    request<SelectionResponse>("/api/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  candidates: (sessionId: string, previousGenes: string[]) =>
    request<CandidateSummary>("/api/candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, previous_genes: previousGenes }),
    }),
  preview: (sessionId: string, kind: DatasetKind, geneId?: string, colorBy?: string) => {
    const query = new URLSearchParams({ session_id: sessionId });
    if (geneId) query.set("gene_id", geneId);
    if (colorBy) query.set("color_by", colorBy);
    return request<PreviewResponse>(`/api/preview/${kind}?${query}`);
  },
};

export function errorTechnical(error: unknown): string | undefined {
  return error instanceof Error && "technical" in error
    ? String((error as Error & { technical?: string }).technical)
    : undefined;
}
