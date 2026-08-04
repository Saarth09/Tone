from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import LLMConnection
from app.security import decrypt_secret, encrypt_secret


@dataclass
class LiveLLMConfig:
    base_url: str
    api_key: str
    model: str
    system_prompt: str = ""
    name: str = "Production LLM"
    provider: str = "custom"
    connected: bool = False
    user_id: Optional[int] = None


def mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "••••••••"
    return f"{key[:4]}…{key[-4:]}"


class ConnectionStore:
    """Per-user LLM connection cache used by the sampler."""

    def __init__(self) -> None:
        self._by_user: dict[int, LiveLLMConfig] = {}
        settings = get_settings()
        self._fallback = LiveLLMConfig(
            base_url=settings.llm_base_url.rstrip("/"),
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            system_prompt=settings.llm_system_prompt,
            connected=False,
        )

    def live_for(self, user_id: int) -> LiveLLMConfig:
        return self._by_user.get(user_id, self._fallback)

    # Back-compat for code paths that still read .live during a bound request
    @property
    def live(self) -> LiveLLMConfig:
        if len(self._by_user) == 1:
            return next(iter(self._by_user.values()))
        return self._fallback

    def apply_row(self, row: LLMConnection) -> LiveLLMConfig:
        cfg = LiveLLMConfig(
            base_url=row.base_url.rstrip("/"),
            api_key=decrypt_secret(row.api_key),
            model=row.model,
            system_prompt=row.system_prompt or "",
            name=row.name,
            provider=row.provider,
            connected=True,
            user_id=row.user_id,
        )
        self._by_user[row.user_id] = cfg
        return cfg

    async def load_user(self, session: AsyncSession, user_id: int) -> LiveLLMConfig:
        row = await self.get_active_row(session, user_id)
        if row:
            return self.apply_row(row)
        cfg = LiveLLMConfig(
            base_url=self._fallback.base_url,
            api_key=self._fallback.api_key,
            model=self._fallback.model,
            system_prompt=self._fallback.system_prompt,
            connected=False,
            user_id=user_id,
        )
        self._by_user[user_id] = cfg
        return cfg

    async def load_all_active(self, session: AsyncSession) -> list[LiveLLMConfig]:
        result = await session.execute(
            select(LLMConnection).where(LLMConnection.is_active.is_(True))
        )
        rows = list(result.scalars().all())
        return [self.apply_row(row) for row in rows]

    async def get_active_row(
        self, session: AsyncSession, user_id: int
    ) -> Optional[LLMConnection]:
        return await session.scalar(
            select(LLMConnection)
            .where(LLMConnection.user_id == user_id, LLMConnection.is_active.is_(True))
            .order_by(LLMConnection.updated_at.desc())
            .limit(1)
        )

    async def upsert(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        name: str,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        system_prompt: str = "",
        keep_existing_key: bool = False,
    ) -> LLMConnection:
        # Local Ollama does not require a real key, but httpx needs a non-empty
        # Authorization value — default so blank saves still work.
        effective_key = api_key.strip() if api_key else ""
        if not effective_key and not keep_existing_key:
            if provider == "ollama" or "11434" in base_url:
                effective_key = "ollama"

        row = await self.get_active_row(session, user_id)
        if row is None:
            row = LLMConnection(
                user_id=user_id,
                name=name,
                provider=provider,
                base_url=base_url.rstrip("/"),
                api_key=encrypt_secret(effective_key) if effective_key else "",
                model=model,
                system_prompt=system_prompt,
                is_active=True,
            )
            session.add(row)
        else:
            row.name = name
            row.provider = provider
            row.base_url = base_url.rstrip("/")
            row.model = model
            row.system_prompt = system_prompt
            row.is_active = True
            if effective_key:
                row.api_key = encrypt_secret(effective_key)
            elif not keep_existing_key:
                row.api_key = ""

        await session.commit()
        await session.refresh(row)
        self.apply_row(row)
        return row


async def test_llm_connection(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = 45.0,
) -> tuple[bool, str, float]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key or 'none'}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "temperature": 0,
        "max_tokens": 16,
    }
    start = datetime.now(timezone.utc)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}: {resp.text[:300]}", 0.0
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
        latency = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        return True, f"OK — model responded: {content[:80]}", latency
    except Exception as exc:
        return False, str(exc), 0.0
