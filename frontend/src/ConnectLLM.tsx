import { useEffect, useState } from "react";
import { api, LLMConnection } from "./api";

/**
 * Shortcuts only — Tone talks to any OpenAI-compatible /v1/chat/completions API.
 * Self-hosted / custom models use the "custom" option (or edit Base URL + Model freely).
 */
const PRESETS: Record<
  string,
  { label: string; base_url: string; model: string; api_key_hint: string }
> = {
  custom: {
    label: "Any OpenAI-compatible (self-hosted / custom)",
    base_url: "",
    model: "",
    api_key_hint: "API key (or any placeholder if none required)",
  },
  openrouter: {
    label: "OpenRouter",
    base_url: "https://openrouter.ai/api/v1",
    model: "openrouter/free",
    api_key_hint: "sk-or-…",
  },
  openai: {
    label: "OpenAI",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    api_key_hint: "sk-…",
  },
  groq: {
    label: "Groq",
    base_url: "https://api.groq.com/openai/v1",
    model: "llama-3.3-70b-versatile",
    api_key_hint: "gsk_…",
  },
  together: {
    label: "Together AI",
    base_url: "https://api.together.xyz/v1",
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    api_key_hint: "API key",
  },
  deepseek: {
    label: "DeepSeek",
    base_url: "https://api.deepseek.com/v1",
    model: "deepseek-chat",
    api_key_hint: "sk-…",
  },
  fireworks: {
    label: "Fireworks",
    base_url: "https://api.fireworks.ai/inference/v1",
    model: "accounts/fireworks/models/llama-v3p1-8b-instruct",
    api_key_hint: "fw_…",
  },
  ollama: {
    label: "Ollama (local only)",
    base_url: "http://localhost:11434/v1",
    model: "llama3.2",
    api_key_hint: "ollama",
  },
};

type Props = {
  onSaved: () => Promise<void> | void;
};

export default function ConnectLLM({ onSaved }: Props) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<LLMConnection | null>(null);

  const [name, setName] = useState("Production LLM");
  const [provider, setProvider] = useState("custom");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");

  useEffect(() => {
    api
      .getConnection()
      .then((c) => {
        setSaved(c);
        setName(c.name || "Production LLM");
        // Unknown providers still work — treat as custom shortcut
        const known = c.provider && PRESETS[c.provider] ? c.provider : "custom";
        setProvider(known);
        setBaseUrl(c.base_url || "");
        setModel(c.model || "");
        setSystemPrompt(c.system_prompt || "");
        if (!c.connected) setOpen(true);
      })
      .catch((err) => {
        setError(
          err instanceof Error
            ? `Could not load saved connection (${err.message}). You are still logged in — try Save again if needed.`
            : "Could not load saved connection"
        );
        setOpen(true);
      });
  }, []);

  function applyPreset(key: string) {
    setProvider(key);
    const p = PRESETS[key];
    if (!p) return;
    // Custom = keep whatever the user already typed (for their own API)
    if (key === "custom") return;
    if (p.base_url) setBaseUrl(p.base_url);
    if (p.model) setModel(p.model);
    if (key === "ollama" && !apiKey) setApiKey("ollama");
  }

  async function handleTest() {
    setTesting(true);
    setError(null);
    setMessage(null);
    try {
      const res = await api.testConnection({
        base_url: baseUrl,
        api_key: apiKey,
        model,
        keep_existing_key: !apiKey && Boolean(saved?.has_api_key),
      });
      if (res.ok) setMessage(`${res.message} (${Math.round(res.latency_ms)} ms)`);
      else setError(res.message);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Test failed");
    } finally {
      setTesting(false);
    }
  }

  async function handleSave() {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const c = await api.saveConnection({
        name,
        provider: provider || "custom",
        base_url: baseUrl,
        api_key: apiKey,
        model,
        system_prompt: systemPrompt,
        keep_existing_key: !apiKey && Boolean(saved?.has_api_key),
      });
      setSaved(c);
      setApiKey("");
      setMessage("Connected. Establish a baseline next.");
      setOpen(false);
      await onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="panel connect-panel">
      <div className="connect-head">
        <div>
          <h2>Connect your LLM</h2>
          <p className="muted" style={{ margin: "0.25rem 0 0" }}>
            Works with <strong>any</strong> OpenAI-compatible API — your own model server,
            vLLM, LiteLLM, Hugging Face, cloud providers, etc. Paste base URL + model + key.
          </p>
        </div>
        <button className="btn" onClick={() => setOpen((v) => !v)}>
          {open ? "Hide" : saved?.connected ? "Edit connection" : "Set up"}
        </button>
      </div>

      {saved?.connected && !open && (
        <p className="connect-summary">
          <strong>{saved.name}</strong> · {saved.provider} · {saved.model}
          <span className="muted"> · {saved.base_url}</span>
          {saved.api_key_masked ? (
            <span className="muted"> · key {saved.api_key_masked}</span>
          ) : null}
        </p>
      )}

      {open && (
        <div className="connect-form">
          <label>
            Shortcut (optional)
            <select value={provider} onChange={(e) => applyPreset(e.target.value)}>
              {Object.entries(PRESETS).map(([key, p]) => (
                <option key={key} value={key}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>
          <p className="muted" style={{ margin: "-0.35rem 0 0.5rem", fontSize: "0.85rem" }}>
            Shortcuts only fill the fields below. For a model you built, pick{" "}
            <strong>Any OpenAI-compatible</strong> and enter your URL (must expose{" "}
            <code>/v1/chat/completions</code>).
          </p>

          <label>
            Display name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Support bot" />
          </label>

          <label>
            Base URL
            <input
              value={baseUrl}
              onChange={(e) => {
                setBaseUrl(e.target.value);
                setProvider("custom");
              }}
              placeholder="https://your-api.example.com/v1"
            />
          </label>

          <label>
            Model
            <input
              value={model}
              onChange={(e) => {
                setModel(e.target.value);
                setProvider("custom");
              }}
              placeholder="your-model-id"
            />
          </label>

          <label>
            API key {saved?.has_api_key ? `(saved ${saved.api_key_masked})` : ""}
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder={
                saved?.has_api_key
                  ? "Leave blank to keep existing key"
                  : PRESETS[provider]?.api_key_hint || "API key"
              }
              autoComplete="off"
            />
          </label>

          <label>
            Default system prompt (optional)
            <textarea
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              placeholder="You are a helpful support assistant for Acme…"
              rows={3}
            />
          </label>

          <div className="connect-actions">
            <button className="btn" disabled={testing || loading} onClick={handleTest}>
              {testing ? "Testing…" : "Test connection"}
            </button>
            <button className="btn primary" disabled={loading || testing} onClick={handleSave}>
              {loading ? "Saving…" : "Save & connect"}
            </button>
          </div>

          {message && <p className="ok">{message}</p>}
          {error && <p className="error">{error}</p>}
        </div>
      )}
    </section>
  );
}
