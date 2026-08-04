from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Sample, SystemState
from app.probes import PROBES, Probe
from app.sampler.llm_client import DemoLLMClient, LLMClient, check_fact

if TYPE_CHECKING:
    from app.connections import ConnectionStore

logger = logging.getLogger(__name__)


def baseline_key(user_id: int) -> str:
    return f"baseline_ready:{user_id}"


class Sampler:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        connections: Optional["ConnectionStore"] = None,
    ) -> None:
        self.settings = get_settings()
        self.connections = connections
        if llm is not None:
            self.llm = llm
        elif self.settings.demo_mode:
            self.llm = DemoLLMClient()
        else:
            self.llm = LLMClient(connections=connections)

    def _active_model_label(self, user_id: int) -> str:
        if self.settings.demo_mode:
            return "demo"
        if self.connections is not None:
            return self.connections.live_for(user_id).model
        return self.settings.llm_model

    def _bind_user_llm(self, user_id: int) -> None:
        """Point the shared client at this user's connection for the duration of a cycle."""
        if self.connections is None or isinstance(self.llm, DemoLLMClient):
            return
        cfg = self.connections.live_for(user_id)
        # LLMClient reads connections.live_for via a request-scoped helper
        self.llm.active_user_id = user_id  # type: ignore[attr-defined]

    async def sample_probe(
        self,
        session: AsyncSession,
        probe: Probe,
        *,
        user_id: int,
        is_baseline: bool = False,
    ) -> Sample:
        self._bind_user_llm(user_id)
        response, latency_ms = await self.llm.complete(probe.prompt)
        sample = Sample(
            user_id=user_id,
            probe_id=probe.id,
            category=probe.category.value,
            prompt=probe.prompt,
            response=response,
            is_baseline=is_baseline,
            latency_ms=latency_ms,
            model=self._active_model_label(user_id),
            token_count=len(response.split()),
            fact_ok=check_fact(probe, response),
        )
        session.add(sample)
        await session.flush()
        await session.refresh(sample)
        return sample

    async def run_cycle(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        is_baseline: bool = False,
        probe_ids: Optional[list[str]] = None,
    ) -> list[Sample]:
        probes = list(PROBES)
        if probe_ids:
            probes = [p for p in probes if p.id in probe_ids]

        samples: list[Sample] = []
        for probe in probes:
            try:
                sample = await self.sample_probe(
                    session, probe, user_id=user_id, is_baseline=is_baseline
                )
                samples.append(sample)
            except Exception:
                logger.exception("Failed sampling probe %s for user %s", probe.id, user_id)
        await session.commit()
        return samples

    async def establish_baseline(
        self, session: AsyncSession, *, user_id: int, runs: Optional[int] = None
    ) -> int:
        runs = runs or self.settings.baseline_runs
        per_probe = max(1, runs // max(1, len(PROBES)))
        total = 0
        for _ in range(per_probe):
            batch = await self.run_cycle(session, user_id=user_id, is_baseline=True)
            total += len(batch)

        key = baseline_key(user_id)
        if total <= 0:
            # Do not mark ready when every probe call failed (e.g. bad LLM config).
            state = await session.get(SystemState, key)
            if state is not None:
                state.value = "false"
            await session.commit()
            return total

        state = await session.get(SystemState, key)
        if state is None:
            session.add(SystemState(key=key, value="true"))
        else:
            state.value = "true"
        await session.commit()
        return total

    async def is_baseline_ready(self, session: AsyncSession, user_id: int) -> bool:
        state = await session.get(SystemState, baseline_key(user_id))
        if not (state and state.value == "true"):
            return False
        count = await session.scalar(
            select(func.count())
            .select_from(Sample)
            .where(Sample.user_id == user_id, Sample.is_baseline.is_(True))
        )
        return int(count or 0) > 0

    async def get_recent_samples(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        limit: int = 50,
        category: Optional[str] = None,
    ) -> list[Sample]:
        q = (
            select(Sample)
            .where(Sample.user_id == user_id)
            .order_by(Sample.created_at.desc())
            .limit(limit)
        )
        if category:
            q = q.where(Sample.category == category)
        result = await session.execute(q)
        return list(result.scalars().all())
