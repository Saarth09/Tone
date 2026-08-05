from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from app.alerting import Alerter
from app.config import get_settings
from app.connections import ConnectionStore
from app.db import SessionLocal
from app.db.models import LLMConnection, Sample
from app.detector import DriftDetector
from app.embedder import Embedder
from app.sampler import Sampler

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(
        self,
        sampler: Sampler,
        embedder: Embedder,
        detector: DriftDetector,
        alerter: Alerter,
        connections: ConnectionStore,
    ) -> None:
        self.settings = get_settings()
        self.sampler = sampler
        self.embedder = embedder
        self.detector = detector
        self.alerter = alerter
        self.connections = connections
        self.scheduler = AsyncIOScheduler()
        self._lock = asyncio.Lock()

    def start(self) -> None:
        minutes = self.settings.sample_interval_minutes
        self.scheduler.add_job(
            self.scheduled_cycle,
            "interval",
            minutes=minutes,
            id="sample_cycle",
            replace_existing=True,
            next_run_time=None,
        )
        self.scheduler.start()
        logger.info("Scheduler started — sampling every %s minutes", minutes)

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)

    async def scheduled_cycle(self) -> None:
        async with SessionLocal() as session:
            result = await session.execute(
                select(LLMConnection.user_id).where(LLMConnection.is_active.is_(True)).distinct()
            )
            user_ids = [row[0] for row in result.all()]
            for user_id in user_ids:
                try:
                    if not await self.sampler.is_baseline_ready(session, user_id):
                        continue
                    await self.connections.load_user(session, user_id)
                    await self.run_sample_and_detect(session, user_id=user_id)
                except Exception:
                    logger.exception("Scheduled cycle failed for user %s", user_id)

    async def embed_samples(self, samples: list[Sample]) -> None:
        if not samples:
            return

        # sentence-transformers encode is CPU-blocking — never run on the event loop
        # or the whole API freezes (browser shows "Failed to fetch").
        def _run() -> None:
            for sample in samples:
                created = sample.created_at or datetime.now(timezone.utc)
                self.embedder.store(
                    sample.id,
                    sample.response,
                    user_id=sample.user_id,
                    probe_id=sample.probe_id,
                    category=sample.category,
                    is_baseline=sample.is_baseline,
                    created_at=created.isoformat(),
                )

        await asyncio.to_thread(_run)

    async def run_baseline(
        self, session, *, user_id: int, runs: Optional[int] = None
    ) -> int:
        async with self._lock:
            await self.connections.load_user(session, user_id)
            samples = await self.sampler.establish_baseline(
                session, user_id=user_id, runs=runs
            )
            await self.embed_samples(samples)
            return len(samples)

    async def run_sample_and_detect(
        self,
        session,
        *,
        user_id: int,
        probe_ids: Optional[list[str]] = None,
    ) -> dict:
        async with self._lock:
            await self.connections.load_user(session, user_id)
            samples = await self.sampler.run_cycle(
                session, user_id=user_id, is_baseline=False, probe_ids=probe_ids
            )
            await self.embed_samples(samples)
            drift_results = await self.detector.evaluate_all(session, user_id=user_id)
            for r in drift_results:
                if r.is_alert:
                    await self.alerter.maybe_alert(session, user_id=user_id, result=r)
            return {"samples": samples, "drift": drift_results}
