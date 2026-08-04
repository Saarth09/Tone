# Tone — Behavioral Drift Detector for LLM Outputs

Continuous monitoring for deployed LLMs. Tone samples probe prompts on a schedule, embeds responses, and detects personality / tone / factual drift with three parallel statistical signals.

## Architecture

| Layer | Role |
|---|---|
| **Sampler** | APScheduler + probe library (tone / fact / persona) hits any OpenAI-compatible endpoint |
| **Embedder** | `all-MiniLM-L6-v2` → ChromaDB (baseline + live vectors) |
| **Drift detector** | MMD (RBF kernel), token KL divergence, cosine sliding window → weighted score |
| **Dashboard + alerts** | React + Recharts UI, FastAPI API, optional Slack webhook |

```
Deployed LLM → Sampler → Embedder → ChromaDB
                              ↘
                         Drift Detector (MMD / KL / Cosine)
                              ↓
                         Dashboard → Slack / webhook
```

## Quick start (demo mode)

Demo mode uses a synthetic LLM so you can inject persona corruption without Ollama/OpenAI.

### Docker Compose

```bash
docker compose up --build
```

- Dashboard: http://localhost:3000
- API: http://localhost:8000/docs

On first boot the API establishes a baseline, runs a live sample cycle, then you can click **Inject drift** in the UI.

### Local (Windows / macOS / Linux)

**Backend**

```bash
cd backend
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env   # or: cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Demo walkthrough

1. Wait for status to show baseline samples > 0 (auto in `DEMO_MODE`).
2. Click **Run sample cycle** — scores should stay low.
3. Click **Inject drift** — responses become curt / factually wrong / persona-hijacked.
4. Watch overall drift, category heatmap, and baseline-vs-current panel spike.
5. Optional: set `SLACK_WEBHOOK_URL` to receive alerts when scores cross threshold.

## Database (PostgreSQL)

Tone stores per-user accounts, LLM connections (encrypted API keys), probe samples, drift scores, and alerts in a SQL database. **PostgreSQL** is the production target; SQLite still works for quick local demos.

### What was set up on this machine
- PostgreSQL 17 is installed and the Windows service `postgresql-x64-17` is running
- App code + Docker Compose are wired for Postgres
- Tables are created automatically on API startup (`init_db`) or via `python -m scripts.init_db`

### Finish creating the Tone database

During install, PostgreSQL asked for a **postgres** superuser password. Use that once:

```powershell
cd c:\projects\Tone\backend
$env:PGPASSWORD = "YOUR_POSTGRES_PASSWORD_HERE"
.\scripts\setup_postgres.ps1
```

Then set in `backend/.env`:

```env
DATABASE_URL=postgresql+asyncpg://tone:tone@127.0.0.1:5432/tone
```

Restart the API. Confirm with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

You should see `"dialect": "postgresql"`.

### Docker alternative (when Docker Desktop is installed)

```bash
docker compose up -d db
```

Default URL: `postgresql+asyncpg://tone:tone@127.0.0.1:5432/tone`

## Auth + per-user isolation

Tone requires an account. Each user’s baselines, samples, drift scores, alerts, and LLM API keys are scoped by `user_id`.

- `POST /api/auth/register` · `POST /api/auth/login` · `GET /api/auth/me`
- API keys encrypted at rest (Fernet derived from `SECRET_KEY`)
- JWT bearer tokens (set `SECRET_KEY` in production)
- Optional **Login with Google** via OAuth (`GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`, `FRONTEND_URL`)

### Enable Google login

1. Create an OAuth client in [Google Cloud Console](https://console.cloud.google.com/apis/credentials) (Web application)
2. Authorized redirect URI: `http://localhost:8000/api/auth/google/callback`
3. Put the client ID/secret in `backend/.env`
4. Restart the API — **Login with Google** becomes active on the auth screen

## How companies use it (no backend access)

When Tone is deployed as a site, customers never touch `.env` or server files:

1. Open the dashboard  
2. **Connect your LLM** — pick OpenAI / Ollama / OpenRouter / custom, paste base URL + API key + model  
3. **Test connection** → **Save & connect**  
4. **Establish baseline** while the model is known-good  
5. Leave monitoring on — Tone probes on a schedule and alerts on drift  

API for the same flow: `GET/PUT /api/connection`, `POST /api/connection/test`.

## Point at a real LLM (UI or env)

**Preferred:** use **Connect your LLM** in the dashboard.

Or set defaults in `backend/.env` / Compose (used until a UI connection is saved):

```env
DEMO_MODE=false
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=llama3.2
```

Works with Ollama, vLLM, OpenAI, or any OpenAI-compatible chat completions API.

Then:

```bash
curl -X POST http://localhost:8000/api/baseline
curl -X POST http://localhost:8000/api/sample
```

## Probe categories

- **Tone** — warmth / style consistency (“Explain quantum computing…”)
- **Fact** — checkable answers with keyword validation
- **Persona** — identity + prompt-injection resistance

## Drift score

```
score = 0.4·MMD + 0.3·KL + 0.3·CosineDrift
```

Weights and per-category thresholds are configurable via env (`MMD_WEIGHT`, `TONE_THRESHOLD`, …).

## API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/status` | Health + counters |
| POST | `/api/baseline` | Establish baseline distribution |
| POST | `/api/sample` | Fire probes + compute drift |
| GET | `/api/drift` | Historical scores |
| GET | `/api/compare/{probe_id}` | Baseline vs current text |
| GET | `/api/heatmap` | Per-probe drift cells |
| GET | `/api/explainability` | PCA dimension deltas |
| POST | `/api/demo/drift` | Toggle synthetic corruption |

## Tests

```bash
cd backend
pip install pytest
pytest -q
```

## Stack

Python · FastAPI · APScheduler · sentence-transformers · ChromaDB · scikit-learn · SQLite · React · Recharts · Docker Compose
