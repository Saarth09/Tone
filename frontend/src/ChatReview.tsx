import { FormEvent, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ChatReviewResult } from "./api";

const EXAMPLE = `User: Help me write a React login form with email/password and error handling. Keep it simple — no social login.
Assistant: Sure — here's a basic email/password form with validation and error display...
User: Can you also add Google and GitHub OAuth, and maybe Magic Links?
Assistant: Absolutely! Let's expand this into a full auth suite with OAuth providers, magic links, and a session dashboard...
User: Wait — I only wanted email/password. Please go back.
Assistant: Got it — focusing again on email/password only. Here's a trimmed version...`;

export default function ChatReview() {
  const [transcript, setTranscript] = useState("");
  const [goal, setGoal] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ChatReviewResult | null>(null);

  const chartData = useMemo(
    () =>
      (result?.timeline || []).map((p, i) => ({
        idx: i + 1,
        label: p.label,
        drift: Number(p.drift_score.toFixed(3)),
        threshold: result?.threshold ?? 0.45,
      })),
    [result]
  );

  async function onAnalyze(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const out = await api.chatReview({
        transcript,
        goal: goal.trim() || undefined,
        use_llm_tips: true,
      });
      setResult(out);
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  async function onFile(file: File | null) {
    if (!file) return;
    const text = await file.text();
    setTranscript(text);
    setError(null);
  }

  return (
    <div className="chat-review">
      <section className="panel">
        <h2>Chat review</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          Paste a long transcript or upload a ChatGPT export JSON. Tone finds where the
          conversation drifted from the original goal and suggests how to steer it back.
        </p>

        <form className="connect-form" onSubmit={onAnalyze}>
          <label>
            Original goal (optional)
            <input
              value={goal}
              onChange={(e) => setGoal(e.target.value)}
              placeholder="Defaults to the first user message"
            />
          </label>

          <label>
            Transcript / export
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              rows={12}
              placeholder={"User: ...\nAssistant: ...\nUser: ..."}
              required
            />
          </label>

          <div className="row-actions">
            <label className="file-btn">
              Upload .txt / .json
              <input
                type="file"
                accept=".txt,.json,.md,text/plain,application/json"
                hidden
                onChange={(e) => void onFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button
              type="button"
              className="btn ghost"
              onClick={() => {
                setTranscript(EXAMPLE);
                setGoal("");
                setResult(null);
              }}
            >
              Load example
            </button>
            <button type="submit" className="btn primary" disabled={busy || transcript.trim().length < 20}>
              {busy ? "Analyzing…" : "Analyze drift"}
            </button>
          </div>
        </form>
        {error && <p className="error">{error}</p>}
      </section>

      {result && (
        <>
          <section className="grid stats">
            <div className={`stat ${result.is_alert ? "alert" : ""}`}>
              <label>Overall goal drift</label>
              <strong>{result.overall_drift.toFixed(3)}</strong>
            </div>
            <div className="stat">
              <label>Messages</label>
              <strong>
                {result.message_count}
                <span className="stat-sub">
                  {" "}
                  ({result.user_turns}u / {result.assistant_turns}a)
                </span>
              </strong>
            </div>
            <div className="stat">
              <label>Peak drift</label>
              <strong>{result.peak ? result.peak.drift_score.toFixed(3) : "—"}</strong>
            </div>
            <div className="stat">
              <label>First alert</label>
              <strong className="stat-text">
                {result.first_alert ? result.first_alert.label : "None"}
              </strong>
            </div>
          </section>

          <section className="panel">
            <h2>Inferred goal</h2>
            <p className="goal-box">{result.goal}</p>
          </section>

          <div className="grid two">
            <section className="panel">
              <h2>Drift along the chat</h2>
              {chartData.length === 0 ? (
                <p className="muted">No timeline points.</p>
              ) : (
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={260}>
                    <LineChart data={chartData}>
                      <CartesianGrid stroke="rgba(168,196,176,0.12)" />
                      <XAxis dataKey="idx" stroke="#8fa398" fontSize={12} />
                      <YAxis domain={[0, 1]} stroke="#8fa398" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          background: "#121a17",
                          border: "1px solid rgba(168,196,176,0.2)",
                        }}
                        labelFormatter={(_, payload) =>
                          payload?.[0]?.payload?.label || ""
                        }
                      />
                      <ReferenceLine
                        y={result.threshold}
                        stroke="#e0a45a"
                        strokeDasharray="4 4"
                        label={{ value: "threshold", fill: "#e0a45a", fontSize: 11 }}
                      />
                      <Line
                        type="monotone"
                        dataKey="drift"
                        stroke="#3dbe84"
                        strokeWidth={2}
                        dot={{ r: 3 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
              {result.peak && (
                <p className="muted" style={{ marginBottom: 0 }}>
                  Peak at <strong>{result.peak.label}</strong> — {result.peak.excerpt}
                </p>
              )}
            </section>

            <section className="panel">
              <h2>How to fix it</h2>
              <p className="muted" style={{ marginTop: 0 }}>
                Tips source: {result.tips_source === "llm" ? "your connected LLM" : "Tone heuristics"}
              </p>
              <ol className="tips-list">
                {result.tips.map((tip) => (
                  <li key={tip.slice(0, 48)}>{tip}</li>
                ))}
              </ol>
            </section>
          </div>

          <section className="panel">
            <h2>Exchange breakdown</h2>
            <div className="review-table-wrap">
              <table className="review-table">
                <thead>
                  <tr>
                    <th>Where</th>
                    <th>Drift</th>
                    <th>Similarity</th>
                    <th>Excerpt</th>
                  </tr>
                </thead>
                <tbody>
                  {result.timeline.map((row) => (
                    <tr key={`${row.window_start}-${row.window_end}`} className={row.is_alert ? "row-alert" : ""}>
                      <td>{row.label}</td>
                      <td>{row.drift_score.toFixed(3)}</td>
                      <td>{row.similarity.toFixed(3)}</td>
                      <td className="excerpt">{row.excerpt}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
