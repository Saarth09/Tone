from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    user_id: int
    kind: str  # "baseline" | "sample"
    status: str = "queued"  # queued | running | succeeded | failed
    message: str = ""
    error: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self, *, status: Optional[str] = None, message: Optional[str] = None) -> None:
        if status is not None:
            self.status = status
        if message is not None:
            self.message = message
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class JobStore:
    """In-memory async jobs so long LLM runs don't hold open HTTP requests."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._active_by_user: dict[int, str] = {}
        self._lock = asyncio.Lock()

    async def start(
        self,
        *,
        user_id: int,
        kind: str,
        coro_factory,
        message: str,
    ) -> Job:
        async with self._lock:
            existing_id = self._active_by_user.get(user_id)
            if existing_id and existing_id in self._jobs:
                existing = self._jobs[existing_id]
                if existing.status in {"queued", "running"}:
                    raise RuntimeError(
                        f"A {existing.kind} job is already running. Wait for it to finish."
                    )

            job_id = secrets.token_urlsafe(12)
            job = Job(id=job_id, user_id=user_id, kind=kind, message=message)
            self._jobs[job_id] = job
            self._active_by_user[user_id] = job_id

        async def _runner() -> None:
            job.touch(status="running", message=message)
            try:
                result = await coro_factory(job)
                job.result = result
                job.touch(status="succeeded", message="Done")
            except Exception as exc:
                logger.exception("Job %s (%s) failed", job.id, job.kind)
                job.error = str(exc)
                job.touch(status="failed", message="Failed")
            finally:
                async with self._lock:
                    if self._active_by_user.get(user_id) == job_id:
                        self._active_by_user.pop(user_id, None)

        asyncio.create_task(_runner())
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def active_for_user(self, user_id: int) -> Optional[Job]:
        job_id = self._active_by_user.get(user_id)
        if not job_id:
            return None
        job = self._jobs.get(job_id)
        if job and job.status in {"queued", "running"}:
            return job
        return None


jobs = JobStore()
