const API_BASE = import.meta.env.VITE_API_URL || "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export type Status = {
  baseline_ready: boolean;
  demo_mode: boolean;
  sample_interval_minutes: number;
  llm_model: string;
  llm_base_url: string;
  total_samples: number;
  baseline_samples: number;
  live_samples: number;
  latest_overall_score: number | null;
  drifted: boolean | null;
};

export type DriftScore = {
  id: number;
  category: string;
  mmd_score: number;
  kl_score: number;
  cosine_score: number;
  combined_score: number;
  threshold: number;
  is_alert: boolean;
  sample_count: number;
  created_at: string;
};

export type Probe = {
  id: string;
  category: string;
  prompt: string;
  description: string;
};

export type Compare = {
  probe_id: string;
  category: string;
  prompt: string;
  baseline_responses: string[];
  current_response: string | null;
  cosine_similarities: number[];
};

export type HeatmapCell = {
  category: string;
  probe_id: string;
  score: number;
};

export type Explainability = {
  available: boolean;
  category?: string;
  components: Array<{
    component: number;
    baseline_mean: number;
    live_mean: number;
    delta: number;
    explained_variance_ratio: number;
  }>;
  total_explained?: number;
};

export type Alert = {
  id: number;
  category: string;
  score: number;
  threshold: number;
  message: string;
  delivered: boolean;
  created_at: string;
};

export const api = {
  status: () => request<Status>("/api/status"),
  probes: () => request<Probe[]>("/api/probes"),
  drift: (category?: string) =>
    request<DriftScore[]>(
      `/api/drift?limit=120${category ? `&category=${category}` : ""}`
    ),
  latest: () => request<Record<string, DriftScore>>("/api/drift/latest"),
  heatmap: () => request<HeatmapCell[]>("/api/heatmap"),
  compare: (probeId: string) => request<Compare>(`/api/compare/${probeId}`),
  explain: (category?: string) =>
    request<Explainability>(
      `/api/explainability${category ? `?category=${category}` : ""}`
    ),
  alerts: () => request<Alert[]>("/api/alerts"),
  sample: () => request("/api/sample", { method: "POST", body: "{}" }),
  baseline: () => request("/api/baseline", { method: "POST", body: "{}" }),
  setDrift: (enable: boolean) =>
    request("/api/demo/drift", {
      method: "POST",
      body: JSON.stringify({ enable }),
    }),
};
