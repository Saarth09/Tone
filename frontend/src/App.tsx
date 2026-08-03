import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  api,
  Alert,
  Compare,
  DriftScore,
  Explainability,
  HeatmapCell,
  Probe,
  Status,
} from "./api";

function fmtTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function heatColor(score: number) {
  const t = Math.max(0, Math.min(1, score));
  const r = Math.round(40 + t * 180);
  const g = Math.round(160 - t * 110);
  const b = Math.round(100 - t * 40);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [driftHistory, setDriftHistory] = useState<DriftScore[]>([]);
  const [latest, setLatest] = useState<Record<string, DriftScore>>({});
  const [heatmap, setHeatmap] = useState<HeatmapCell[]>([]);
  const [probes, setProbes] = useState<Probe[]>([]);
  const [selectedProbe, setSelectedProbe] = useState<string>("tone_quantum");
  const [compare, setCompare] = useState<Compare | null>(null);
  const [explain, setExplain] = useState<Explainability | null>(null);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [s, d, l, h, p, e, a] = await Promise.all([
        api.status(),
        api.drift("overall"),
        api.latest(),
        api.heatmap(),
        api.probes(),
        api.explain(),
        api.alerts(),
      ]);
      setStatus(s);
      setDriftHistory([...d].reverse());
      setLatest(l);
      setHeatmap(h);
      setProbes(p);
      setExplain(e);
      setAlerts(a);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 8000);
    return () => clearInterval(id);
  }, [refresh]);

  useEffect(() => {
    if (!selectedProbe) return;
    api.compare(selectedProbe).then(setCompare).catch(() => setCompare(null));
  }, [selectedProbe, driftHistory.length]);

  const chartData = useMemo(
    () =>
      driftHistory.map((row) => ({
        time: fmtTime(row.created_at),
        combined: Number(row.combined_score.toFixed(3)),
        mmd: Number(row.mmd_score.toFixed(3)),
        kl: Number(row.kl_score.toFixed(3)),
        cosine: Number(row.cosine_score.toFixed(3)),
        threshold: row.threshold,
      })),
    [driftHistory]
  );

  const overall = latest.overall;
  const alertActive = Boolean(overall?.is_alert);

  async function runAction(fn: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await fn();
      await refresh();
      if (selectedProbe) {
        setCompare(await api.compare(selectedProbe));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusy(false);
    }
  }

  const groupedHeat = useMemo(() => {
    const cats = ["tone", "fact", "persona"] as const;
    return cats.map((category) => ({
      category,
      cells: heatmap.filter((c) => c.category === category),
    }));
  }, [heatmap]);

  return (
    <div className="app">
      <header className="hero">
        <h1 className="brand">
          Tone<span>.</span>
        </h1>
        <p className="tagline">
          Behavioral drift detector for LLM outputs — continuous probe sampling,
          embedding-space MMD, token KL divergence, and persona cosine checks.
        </p>
        <div className="toolbar">
          <span className={`pill ${alertActive ? "danger" : status?.drifted ? "warn" : ""}`}>
            <span className="dot" />
            {alertActive
              ? "Drift alert"
              : status?.drifted
                ? "Injected persona active"
                : "Monitoring"}
          </span>
          <span className="pill">
            {status?.demo_mode ? "Demo mode" : status?.llm_model || "…"}
          </span>
          <span className="pill">
            every {status?.sample_interval_minutes ?? "—"} min
          </span>
          <button
            className="btn primary"
            disabled={busy}
            onClick={() => runAction(() => api.sample())}
          >
            Run sample cycle
          </button>
          {!status?.baseline_ready && (
            <button
              className="btn"
              disabled={busy}
              onClick={() => runAction(() => api.baseline())}
            >
              Establish baseline
            </button>
          )}
          {status?.demo_mode && (
            <>
              <button
                className="btn danger"
                disabled={busy || Boolean(status.drifted)}
                onClick={() =>
                  runAction(async () => {
                    await api.setDrift(true);
                    await api.sample();
                  })
                }
              >
                Inject drift
              </button>
              <button
                className="btn"
                disabled={busy || !status.drifted}
                onClick={() => runAction(() => api.setDrift(false))}
              >
                Restore baseline persona
              </button>
            </>
          )}
        </div>
        {error && <p className="error">{error}</p>}
      </header>

      <section className="grid stats">
        <div className={`stat ${alertActive ? "alert" : ""}`}>
          <label>Overall drift</label>
          <strong>
            {overall ? overall.combined_score.toFixed(3) : "—"}
          </strong>
        </div>
        <div className="stat">
          <label>MMD / KL / Cosine</label>
          <strong style={{ fontSize: "1.1rem" }}>
            {overall
              ? `${overall.mmd_score.toFixed(2)} / ${overall.kl_score.toFixed(2)} / ${overall.cosine_score.toFixed(2)}`
              : "—"}
          </strong>
        </div>
        <div className="stat">
          <label>Baseline samples</label>
          <strong>{status?.baseline_samples ?? "—"}</strong>
        </div>
        <div className="stat">
          <label>Live samples</label>
          <strong>{status?.live_samples ?? "—"}</strong>
        </div>
      </section>

      <section className="grid two-col" style={{ marginTop: "1rem" }}>
        <div className="panel">
          <h2>Drift score over time</h2>
          <div className="chart-wrap">
            {chartData.length === 0 ? (
              <p className="muted">No drift scores yet — run a sample cycle.</p>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid stroke="rgba(168,196,176,0.12)" />
                  <XAxis dataKey="time" stroke="#8fa398" tick={{ fontSize: 11 }} />
                  <YAxis domain={[0, 1]} stroke="#8fa398" tick={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#121a17",
                      border: "1px solid rgba(168,196,176,0.2)",
                      borderRadius: 8,
                    }}
                  />
                  <Legend />
                  <ReferenceLine
                    y={overall?.threshold ?? 0.55}
                    stroke="#e0a45a"
                    strokeDasharray="4 4"
                    label={{ value: "threshold", fill: "#e0a45a", fontSize: 11 }}
                  />
                  <Line type="monotone" dataKey="combined" stroke="#3dbe84" strokeWidth={2.4} dot={false} />
                  <Line type="monotone" dataKey="mmd" stroke="#5ec4a0" strokeWidth={1.2} dot={false} />
                  <Line type="monotone" dataKey="kl" stroke="#6aa8e0" strokeWidth={1.2} dot={false} />
                  <Line type="monotone" dataKey="cosine" stroke="#c4a05e" strokeWidth={1.2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        <div className="panel">
          <h2>Probe category heatmap</h2>
          <div className="heatmap">
            {groupedHeat.map((row) => (
              <div className="heat-row" key={row.category}>
                <span>{row.category}</span>
                <div className="heat-cells">
                  {row.cells.map((cell) => (
                    <button
                      key={cell.probe_id}
                      className={`heat-cell ${selectedProbe === cell.probe_id ? "active" : ""}`}
                      title={`${cell.probe_id}: ${cell.score.toFixed(3)}`}
                      style={{ background: heatColor(cell.score) }}
                      onClick={() => setSelectedProbe(cell.probe_id)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
          <div className="grid three-col" style={{ marginTop: "1rem" }}>
            {(["tone", "fact", "persona"] as const).map((cat) => (
              <div className="stat" key={cat} style={{ padding: "0.7rem" }}>
                <label>{cat}</label>
                <strong style={{ fontSize: "1.2rem" }}>
                  {latest[cat] ? latest[cat].combined_score.toFixed(3) : "—"}
                </strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="grid two-col" style={{ marginTop: "1rem" }}>
        <div className="panel">
          <h2>Baseline vs current</h2>
          <select
            className="btn"
            value={selectedProbe}
            onChange={(e) => setSelectedProbe(e.target.value)}
            style={{ marginBottom: "0.75rem" }}
          >
            {probes.map((p) => (
              <option key={p.id} value={p.id}>
                [{p.category}] {p.id}
              </option>
            ))}
          </select>
          {compare && (
            <>
              <p className="prompt">{compare.prompt}</p>
              <div className="compare">
                <article>
                  <h3>Baseline</h3>
                  <p>
                    {compare.baseline_responses[0] ||
                      "No baseline response stored yet."}
                  </p>
                </article>
                <article>
                  <h3>Current</h3>
                  <p>{compare.current_response || "No live sample yet."}</p>
                </article>
              </div>
              {compare.cosine_similarities.length > 0 && (
                <p className="muted" style={{ marginTop: "0.75rem" }}>
                  Cosine to nearest baselines:{" "}
                  {compare.cosine_similarities.map((s) => s.toFixed(3)).join(", ")}
                </p>
              )}
            </>
          )}
        </div>

        <div className="panel">
          <h2>Explainability (PCA deltas)</h2>
          {!explain?.available ? (
            <p className="muted">
              Need more baseline and live embeddings before PCA explanation is available.
            </p>
          ) : (
            <div className="pca-list">
              {explain.components.map((c) => (
                <div className="pca-item" key={c.component}>
                  <span>PC{c.component}</span>
                  <div className="bar">
                    <i
                      style={{
                        width: `${Math.min(100, Math.abs(c.delta) * 120)}%`,
                        background:
                          c.delta >= 0
                            ? "linear-gradient(90deg,#2a8f62,#3dbe84)"
                            : "linear-gradient(90deg,#a85a5a,#e07070)",
                      }}
                    />
                  </div>
                  <span className="muted">Δ {c.delta.toFixed(3)}</span>
                </div>
              ))}
              <p className="muted" style={{ marginTop: "0.4rem" }}>
                Explained variance:{" "}
                {((explain.total_explained || 0) * 100).toFixed(1)}%
              </p>
            </div>
          )}

          <h2 style={{ marginTop: "1.25rem" }}>Alerts</h2>
          <div className="alerts">
            {alerts.length === 0 ? (
              <p className="muted">No alerts yet.</p>
            ) : (
              alerts.slice(0, 8).map((a) => (
                <div className="alert-item" key={a.id}>
                  {a.message}
                </div>
              ))
            )}
          </div>
        </div>
      </section>
    </div>
  );
}
