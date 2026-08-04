import { useEffect, useState } from "react";
import { api, LLMConnection } from "./api";

const PRESETS: Record<
  string,
  { label: string; base_url: string; model: string; api_key_hint: string }
> = {
  openai: {
    label: "OpenAI",
    base_url: "https://api.openai.com/v1",
    model: "gpt-4o-mini",
    api_key_hint: "sk-…",
  },
  ollama: {
    label: "Ollama (local)",
    base_url: "http://localhost:11434/v1",
    model: "llama3.2",
    api_key_hint: "ollama",
  },
  openrouter: {
    label: "OpenRouter (Claude/GPT)",
    base_url: "https://openrouter.ai/api/v1",
    model: "anthropic/claude-3.5-sonnet",
    api_key_hint: "sk-or-…",
  },
  custom: {
    label: "Custom OpenAI-compatible",
    base_url: "",
    model: "",
    api_key_hint: "API key",
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
  const [provider, setProvider] = useState("openai");
  const [baseUrl, setBaseUrl] = useState(PRESETS.openai.base_url);
  const [model, setModel] = useState(PRESETS.openai.model);
  const [apiKey, setApiKey] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("");

  useEffect(() => {
    api
      .getConnection()
      .then((c) => {
        setSaved(c);
        setName(c.name || "Production LLM");
        setProvider(c.provider || "custom");
        setBaseUrl(c.base_url || "");
        setModel(c.model || "");
        setSystemPrompt(c.system_prompt || "");
        if (!c.connected) setOpen(true);
      })
      .catch(() => setOpen(true));
  }, []);

  function applyPreset(key: string) {
    setProvider(key);
    const p = PRESETS[key];
    if (!p) return;
    if (p.base_url) setBaseUrl(p.base_url);
    if (p.model) setModel(p.model);
    // Ollama accepts any non-empty key; prefill so Test/Save never send blank.
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
        provider,
        base_url: baseUrl,
        api_key: apiKey,
        model,
        system_prompt: systemPrompt,
        keep_existing_key: !apiKey && Boolean(saved?.has_api_key),
      });
      setSaved(c);
      setApiKey("");
      setMessage("Connected. Establish a baseline next.");
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
            Companies paste an OpenAI-compatible endpoint here — no server files needed.
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
            Provider preset
            <select
              value={provider}
              onChange={(e) => applyPreset(e.target.value)}
            >
              {Object.entries(PRESETS).map(([key, p]) => (
                <option key={key} value={key}>
                  {p.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Display name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Support bot" />
          </label>

          <label>
            Base URL
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://api.openai.com/v1"
            />
          </label>

          <label>
            Model
            <input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              placeholder="gpt-4o-mini"
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
