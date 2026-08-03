from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Sample, SystemState
from app.probes import PROBES, Probe, get_probe
from app.sampler.llm_client import DemoLLMClient, LLMClient, check_fact

logger = logging.getLogger(__name__)


class Sampler:
    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self.settings = get_settings()
        if llm is not None:
            self.llm = llm
        elif self.settings.demo_mode:
            self.llm = DemoLLMClient()
        else:
            self.llm = LLMClient()

    async def sample_probe(
        self, session: AsyncSession, probe: Probe, *, is_baseline: bool = False
    ) -> Sample:
        response, latency_ms = await self.llm.complete(probe.prompt)
        sample = Sample(
            probe_id=probe.id,
            category=probe.category.value,
            prompt=probe.prompt,
            response=response,
            is_baseline=is_baseline,
            latency_ms=latency_ms,
            model=self.settings.llm_model if not self.settings.demo_mode else "demo",
            token_count=len(response.split()),
            fact_ok=check_fact(probe, response),
        )
        session.add(sample)
        await session.flush()
        await session.refresh(sample)
        return sample

    async def run_cycle(
        self, session: AsyncSession, *, is_baseline: bool = False, probe_ids: Optional[list[str]] = None
    ) -> list[Sample]:
        probes = list(PROBES)
        if probe_ids:
            probes = [p for p in probes if p.id in probe_ids]

        samples: list[Sample] = []
        for probe in probes:
            try:
                sample = await self.sample_probe(session, probe, is_baseline=is_baseline)
                samples.append(sample)
            except Exception:
                logger.exception("Failed sampling probe %s", probe.id)
        await session.commit()
        return samples

    async def establish_baseline(self, session: AsyncSession, runs: Optional[int] = None) -> int:
        runs = runs or self.settings.baseline_runs
        # Cap per-probe baseline samples for practical demo speed
        per_probe = max(3, runs // max(1, len(PROBES)))
        total = 0
        for _ in range(per_probe):
            batch = await self.run_cycle(session, is_baseline=True)
            total += len(batch)

        state = await session.get(SystemState, "baseline_ready")
        if state is None:
            session.add(SystemState(key="baseline_ready", value="true"))
        else:
            state.value = "true"
        await session.commit()
        return total

    async def is_baseline_ready(self, session: AsyncSession) -> bool:
        state = await session.get(SystemState, "baseline_ready")
        return bool(state and state.value == "true")

    async def get_recent_samples(
        self, session: AsyncSession, *, limit: int = 50, category: Optional[str] = None
    ) -> list[Sample]:
        q = select(Sample).order_by(Sample.created_at.desc()).limit(limit)
        if category:
            q = q.where(Sample.category == category)
        result = await session.execute(q)
        return list(result.scalars().all())
