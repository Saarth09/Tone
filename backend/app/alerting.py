from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import AlertEvent
from app.detector import CategoryDriftResult

logger = logging.getLogger(__name__)


class Alerter:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def _recently_alerted(self, session: AsyncSession, category: str) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.settings.alert_cooldown_minutes)
        q = (
            select(AlertEvent)
            .where(AlertEvent.category == category, AlertEvent.created_at >= cutoff)
            .limit(1)
        )
        result = await session.execute(q)
        return result.scalar_one_or_none() is not None

    async def maybe_alert(self, session: AsyncSession, result: CategoryDriftResult) -> AlertEvent | None:
        if not result.is_alert:
            return None
        if await self._recently_alerted(session, result.category):
            return None

        message = (
            f"[Tone] Behavioral drift detected in *{result.category}* — "
            f"score {result.combined_score:.3f} (threshold {result.threshold:.3f}). "
            f"MMD={result.mmd_score:.3f} KL={result.kl_score:.3f} Cosine={result.cosine_score:.3f}"
        )
        delivered = await self._send_slack(message)
        event = AlertEvent(
            category=result.category,
            score=result.combined_score,
            threshold=result.threshold,
            message=message,
            delivered=delivered,
        )
        session.add(event)
        await session.commit()
        return event

    async def _send_slack(self, message: str) -> bool:
        url = self.settings.slack_webhook_url
        if not url:
            logger.info("Alert (no webhook configured): %s", message)
            return False
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(url, json={"text": message})
                resp.raise_for_status()
            return True
        except Exception:
            logger.exception("Failed to deliver Slack alert")
            return False
