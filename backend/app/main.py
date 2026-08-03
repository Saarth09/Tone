from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alerting import Alerter
from app.config import get_settings
from app.db import SessionLocal, get_session, init_db
from app.db.models import AlertEvent, DriftScore, Sample, SystemState
from app.detector import DriftDetector
from app.embedder import Embedder
from app.probes import PROBES, get_probe
from app.sampler import Sampler
from app.sampler.llm_client import DemoLLMClient
from app.scheduler import JobRunner
from app.schemas import (
    AlertOut,
    BaselineRequest,
    CompareOut,
    DriftScoreOut,
    DriftSimulateRequest,
    HeatmapCell,
    ProbeOut,
    SampleOut,
    SampleRequest,
    StatusOut,
)

logger = logging.getLogger(__name__)
settings = get_settings()

embedder: Optional[Embedder] = None
sampler: Optional[Sampler] = None
detector: Optional[DriftDetector] = None
alerter: Optional[Alerter] = None
runner: Optional[JobRunner] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, sampler, detector, alerter, runner
    logging.basicConfig(level=logging.INFO)
    await init_db()

    embedder = Embedder()
    sampler = Sampler()
    detector = DriftDetector(embedder)
    alerter = Alerter()
    runner = JobRunner(sampler, embedder, detector, alerter)
    runner.start()

    if settings.demo_mode:
        async with SessionLocal() as session:
            ready = await sampler.is_baseline_ready(session)
            if not ready:
                logger.info("Demo mode: establishing baseline automatically")
                await runner.run_baseline(session, runs=max(15, len(PROBES) * 2))
                # Collect a few live samples at baseline behavior
                await runner.run_sample_and_detect(session)

    yield
    if runner:
        runner.shutdown()


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"ok": True, "service": "tone"}


@app.get("/api/status", response_model=StatusOut)
async def status(session: AsyncSession = Depends(get_session)):
    assert sampler is not None
    total = await session.scalar(select(func.count()).select_from(Sample)) or 0
    baseline = await session.scalar(
        select(func.count()).select_from(Sample).where(Sample.is_baseline.is_(True))
    ) or 0
    live = await session.scalar(
        select(func.count()).select_from(Sample).where(Sample.is_baseline.is_(False))
    ) or 0
    latest = await session.scalar(
        select(DriftScore)
        .where(DriftScore.category == "overall")
        .order_by(DriftScore.created_at.desc())
        .limit(1)
    )
    drifted = None
    if isinstance(sampler.llm, DemoLLMClient):
        drifted = sampler.llm.drifted
    return StatusOut(
        baseline_ready=await sampler.is_baseline_ready(session),
        demo_mode=settings.demo_mode,
        sample_interval_minutes=settings.sample_interval_minutes,
        llm_model=settings.llm_model,
        llm_base_url=settings.llm_base_url,
        total_samples=total,
        baseline_samples=baseline,
        live_samples=live,
        latest_overall_score=latest.combined_score if latest else None,
        drifted=drifted,
    )


@app.get("/api/probes", response_model=list[ProbeOut])
async def list_probes():
    return [
        ProbeOut(
            id=p.id,
            category=p.category.value,
            prompt=p.prompt,
            description=p.description,
        )
        for p in PROBES
    ]


@app.post("/api/baseline")
async def create_baseline(
    body: BaselineRequest,
    session: AsyncSession = Depends(get_session),
):
    assert runner is not None
    count = await runner.run_baseline(session, runs=body.runs)
    return {"baseline_samples": count, "ready": True}


@app.post("/api/sample")
async def trigger_sample(
    body: SampleRequest,
    session: AsyncSession = Depends(get_session),
):
    assert runner is not None and sampler is not None
    if not await sampler.is_baseline_ready(session):
        raise HTTPException(400, "Baseline not established. Call POST /api/baseline first.")
    results = await runner.run_sample_and_detect(session, probe_ids=body.probe_ids)
    return {
        "samples": len(results["samples"]),
        "drift": [
            {
                "category": r.category,
                "combined_score": r.combined_score,
                "is_alert": r.is_alert,
                "mmd_score": r.mmd_score,
                "kl_score": r.kl_score,
                "cosine_score": r.cosine_score,
            }
            for r in results["drift"]
        ],
    }


@app.get("/api/samples", response_model=list[SampleOut])
async def list_samples(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=500),
    category: Optional[str] = None,
    baseline: Optional[bool] = None,
    probe_id: Optional[str] = None,
):
    q = select(Sample).order_by(Sample.created_at.desc()).limit(limit)
    if category:
        q = q.where(Sample.category == category)
    if baseline is not None:
        q = q.where(Sample.is_baseline.is_(baseline))
    if probe_id:
        q = q.where(Sample.probe_id == probe_id)
    result = await session.execute(q)
    return list(result.scalars().all())


@app.get("/api/drift", response_model=list[DriftScoreOut])
async def list_drift(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
):
    q = select(DriftScore).order_by(DriftScore.created_at.desc()).limit(limit)
    if category:
        q = q.where(DriftScore.category == category)
    result = await session.execute(q)
    return list(result.scalars().all())


@app.get("/api/drift/latest")
async def latest_drift(session: AsyncSession = Depends(get_session)):
    categories = ["overall", "tone", "fact", "persona"]
    out = {}
    for cat in categories:
        row = await session.scalar(
            select(DriftScore)
            .where(DriftScore.category == cat)
            .order_by(DriftScore.created_at.desc())
            .limit(1)
        )
        if row:
            out[cat] = DriftScoreOut.model_validate(row)
    return out


@app.get("/api/compare/{probe_id}", response_model=CompareOut)
async def compare_probe(probe_id: str, session: AsyncSession = Depends(get_session)):
    assert embedder is not None
    probe = get_probe(probe_id)
    if not probe:
        raise HTTPException(404, "Unknown probe")

    baselines = embedder.all_baseline_docs(probe_id, limit=3)
    current = await session.scalar(
        select(Sample)
        .where(Sample.probe_id == probe_id, Sample.is_baseline.is_(False))
        .order_by(Sample.created_at.desc())
        .limit(1)
    )
    sims: list[float] = []
    if current:
        vec = embedder.embed([current.response])[0]
        sims, _ = embedder.nearest_baseline(vec, probe_id, k=5)

    return CompareOut(
        probe_id=probe_id,
        category=probe.category.value,
        prompt=probe.prompt,
        baseline_responses=baselines,
        current_response=current.response if current else None,
        cosine_similarities=sims,
    )


@app.get("/api/heatmap", response_model=list[HeatmapCell])
async def heatmap(session: AsyncSession = Depends(get_session)):
    assert detector is not None
    # Use latest overall evaluation details if present; else compute lightly from cosine
    cells: list[HeatmapCell] = []
    for probe in PROBES:
        latest_live = await session.scalar(
            select(Sample)
            .where(Sample.probe_id == probe.id, Sample.is_baseline.is_(False))
            .order_by(Sample.created_at.desc())
            .limit(1)
        )
        score = 0.0
        if latest_live and embedder is not None:
            vec = embedder.embed([latest_live.response])[0]
            sims, _ = embedder.nearest_baseline(vec, probe.id, k=5)
            if sims:
                score = 1.0 - sum(sims) / len(sims)
            if probe.category.value == "fact" and latest_live.fact_ok is False:
                score = max(score, 0.9)
        cells.append(HeatmapCell(category=probe.category.value, probe_id=probe.id, score=score))
    return cells


@app.get("/api/explainability")
async def explainability(category: Optional[str] = None):
    assert detector is not None
    return detector.pca_explanation(category=category)


@app.get("/api/alerts", response_model=list[AlertOut])
async def list_alerts(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(50, ge=1, le=200),
):
    result = await session.execute(
        select(AlertEvent).order_by(AlertEvent.created_at.desc()).limit(limit)
    )
    return list(result.scalars().all())


@app.post("/api/demo/drift")
async def simulate_drift(body: DriftSimulateRequest):
    """Toggle synthetic persona corruption in demo mode."""
    assert sampler is not None
    if not isinstance(sampler.llm, DemoLLMClient):
        raise HTTPException(400, "Drift simulation only available in DEMO_MODE")
    sampler.llm.set_drifted(body.enable)
    return {"drifted": sampler.llm.drifted}
