const API_BASE = import.meta.env.VITE_API_URL || "";
const TOKEN_KEY = "tone_token";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string | null) {
  if (token) localStorage.setItem(TOKEN_KEY, token);
  else localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(init?.headers as Record<string, string> | undefined),
  };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (res.status === 401) {
    setToken(null);
  }
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  return res.json();
}

export type User = {
  id: number;
  email: string;
  name: string;
  avatar_url?: string | null;
  auth_provider?: string;
};

export type Job = {
  id: string;
  kind: "baseline" | "sample" | string;
  status: "queued" | "running" | "succeeded" | "failed" | string;
  message: string;
  error?: string | null;
  result?: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
};

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
  system_prompt_poisoned: boolean;
  llm_connected: boolean;
  llm_connection_name: string | null;
  llm_provider: string | null;
  active_job?: Job | null;
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

export type ChatReviewPoint = {
  turn_index: number;
  window_start: number;
  window_end: number;
  label: string;
  drift_score: number;
  similarity: number;
  is_alert: boolean;
  excerpt: string;
};

export type ChatReviewResult = {
  goal: string;
  message_count: number;
  assistant_turns: number;
  user_turns: number;
  threshold: number;
  overall_drift: number;
  is_alert: boolean;
  peak: ChatReviewPoint | null;
  first_alert: ChatReviewPoint | null;
  timeline: ChatReviewPoint[];
  tips: string[];
  tips_source: string;
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

export type LLMConnection = {
  id?: number | null;
  name: string;
  provider: string;
  base_url: string;
  model: string;
  system_prompt: string;
  api_key_masked: string;
  has_api_key: boolean;
  is_active: boolean;
  connected: boolean;
  last_tested_at?: string | null;
  last_test_ok?: boolean | null;
  last_test_message?: string | null;
};

export type LLMConnectionPayload = {
  name: string;
  provider: string;
  base_url: string;
  api_key: string;
  model: string;
  system_prompt: string;
  keep_existing_key: boolean;
};

export const api = {
  authProviders: () => request<{ password: boolean; google: boolean }>("/api/auth/providers"),
  register: async (email: string, password: string, name: string) => {
    const res = await request<{ access_token: string }>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, name }),
    });
    setToken(res.access_token);
    return res;
  },
  login: async (email: string, password: string) => {
    const res = await request<{ access_token: string }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    setToken(res.access_token);
    return res;
  },
  me: () => request<User>("/api/auth/me"),
  logout: () => setToken(null),
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
  sample: () =>
    request<Job & { job_id: string }>("/api/sample", { method: "POST", body: "{}" }),
  baseline: () =>
    request<Job & { job_id: string }>("/api/baseline", { method: "POST", body: "{}" }),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${jobId}`),
  setDrift: (enable: boolean) =>
    request("/api/demo/drift", {
      method: "POST",
      body: JSON.stringify({ enable }),
    }),
  poison: (enable: boolean) =>
    request(`/api/llm/poison?enable=${enable}`, { method: "POST" }),
  getConnection: () => request<LLMConnection>("/api/connection"),
  saveConnection: (body: LLMConnectionPayload) =>
    request<LLMConnection>("/api/connection", {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  testConnection: (body: {
    base_url: string;
    api_key: string;
    model: string;
    keep_existing_key: boolean;
  }) =>
    request<{ ok: boolean; message: string; latency_ms: number }>(
      "/api/connection/test",
      { method: "POST", body: JSON.stringify(body) }
    ),
  chatReview: (body: {
    transcript: string;
    goal?: string;
    threshold?: number;
    use_llm_tips?: boolean;
  }) =>
    request<ChatReviewResult>("/api/chat-review", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  generateLlmTest: (body: {
    goal: string;
    overall_drift: number;
    tips: string[];
    peak?: ChatReviewPoint | null;
    first_alert?: ChatReviewPoint | null;
    transcript?: string;
    timeline?: ChatReviewPoint[];
  }) =>
    request<{ code: string; filename_hint: string }>("/api/chat-review/generate-test", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
