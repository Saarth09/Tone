import { FormEvent, useEffect, useState } from "react";
import { api, apiUrl } from "./api";

type Props = {
  onAuthed: () => void;
};

export default function AuthScreen({ onAuthed }: Props) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [googleEnabled, setGoogleEnabled] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const authError = params.get("auth_error");
    if (authError) {
      setError(`Google sign-in failed (${authError}).`);
      window.history.replaceState({}, "", window.location.pathname);
    }
    api
      .authProviders()
      .then((p) => setGoogleEnabled(Boolean(p.google)))
      .catch(() => setGoogleEnabled(false));
  }, []);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") {
        await api.login(email, password);
      } else {
        await api.register(email, password, name);
      }
      onAuthed();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Auth failed");
    } finally {
      setBusy(false);
    }
  }

  function loginWithGoogle() {
    // Must hit the API host (VITE_API_URL on Railway; same-origin via Vite proxy locally)
    window.location.href = apiUrl("/api/auth/google");
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <h1 className="brand">
          Tone<span>.</span>
        </h1>
        <h2 className="auth-title">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h2>
        <p className="tagline">
          {mode === "login"
            ? "Enter your email below to sign in to your account"
            : "Monitor your LLM’s tone, facts, and persona — private to your account."}
        </p>

        <form className="connect-form" onSubmit={submit}>
          {mode === "register" && (
            <label>
              Name
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Alex" />
            </label>
          )}
          <label>
            Email
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
              autoComplete="email"
            />
          </label>
          <label>
            Password
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </label>
          <button className="btn primary auth-submit" disabled={busy} type="submit">
            {busy ? "Please wait…" : mode === "login" ? "Sign In" : "Create account"}
          </button>
          {error && <p className="error">{error}</p>}
        </form>

        <div className="auth-divider">
          <span>Or</span>
        </div>

        <button
          type="button"
          className="btn google-btn"
          onClick={loginWithGoogle}
          disabled={!googleEnabled}
          title={
            googleEnabled
              ? "Continue with Google"
              : "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in backend/.env"
          }
        >
          <GoogleMark />
          Login with Google
        </button>
        {!googleEnabled && (
          <p className="muted auth-hint">
            Google login needs <code>GOOGLE_CLIENT_ID</code> + <code>GOOGLE_CLIENT_SECRET</code> in
            the server env.
          </p>
        )}

        <p className="auth-switch">
          {mode === "login" ? (
            <>
              Don&apos;t have an account?{" "}
              <button type="button" className="linkish" onClick={() => setMode("register")}>
                Sign up
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button type="button" className="linkish" onClick={() => setMode("login")}>
                Sign in
              </button>
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function GoogleMark() {
  return (
    <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#FFC107"
        d="M43.6 20.5H42V20H24v8h11.3C33.7 32.7 29.3 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3 0 5.8 1.1 7.9 3l5.7-5.7C34 5.1 29.3 3 24 3 12.3 3 3 12.3 3 24s9.3 21 21 21 21-9.3 21-21c0-1.2-.1-2.3-.4-3.5z"
      />
      <path
        fill="#FF3D00"
        d="M6.3 14.7l6.6 4.8C14.7 16 19 13 24 13c3 0 5.8 1.1 7.9 3l5.7-5.7C34 5.1 29.3 3 24 3 16.3 3 9.6 7.3 6.3 14.7z"
      />
      <path
        fill="#4CAF50"
        d="M24 45c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.3 36.7 26.8 37.5 24 37.5c-5.3 0-9.7-3.3-11.3-8l-6.5 5C9.5 40.6 16.2 45 24 45z"
      />
      <path
        fill="#1976D2"
        d="M43.6 20.5H42V20H24v8h11.3c-1.1 3.1-3.5 5.5-6.5 6.9l.1.1 6.2 5.2C36.8 41.4 45 35 45 24c0-1.2-.1-2.3-.4-3.5z"
      />
    </svg>
  );
}
