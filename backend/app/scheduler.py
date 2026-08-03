from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.config import get_settings
from app.db import SessionLocal
from app.alerting import Alerter
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
    ) -> None:
        self.settings = get_settings()
        self.sampler = sampler
        self.embedder = embedder
        self.detector = detector
        self.alerter = alerter
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
            next_run_time=None,  # wait for first interval unless triggered
        )
        self.scheduler.start()
        logger.info("Scheduler started — sampling every %s minutes", minutes)

    def shutdown(self) -> None:
        self.scheduler.shutdown(wait=False)

    async def scheduled_cycle(self) -> None:
        async with SessionLocal() as session:
            if not await self.sampler.is_baseline_ready(session):
                logger.info("Skipping scheduled cycle — baseline not ready")
                return
            await self.run_sample_and_detect(session)

    async def embed_samples(self, samples) -> None:
        for sample in samples:
            created = sample.created_at or datetime.now(timezone.utc)
            self.embedder.store(
                sample.id,
                sample.response,
                probe_id=sample.probe_id,
                category=sample.category,
                is_baseline=sample.is_baseline,
                created_at=created.isoformat(),
            )

    async def run_baseline(self, session, runs: Optional[int] = None) -> int:
        async with self._lock:
            samples_total = await self.sampler.establish_baseline(session, runs=runs)
            # Re-fetch baseline samples to embed (establish_baseline already committed)
            from sqlalchemy import select
            from app.db.models import Sample

            result = await session.execute(
                select(Sample).where(Sample.is_baseline.is_(True))
            )
            samples = list(result.scalars().all())
            await self.embed_samples(samples)
            return samples_total

    async def run_sample_and_detect(self, session, probe_ids: Optional[list[str]] = None) -> dict:
        async with self._lock:
            samples = await self.sampler.run_cycle(session, is_baseline=False, probe_ids=probe_ids)
            await self.embed_samples(samples)
            drift_results = await self.detector.evaluate_all(session)
            for r in drift_results:
                if r.is_alert:
                    await self.alerter.maybe_alert(session, r)
            return {"samples": samples, "drift": drift_results}
