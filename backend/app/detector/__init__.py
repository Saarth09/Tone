from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import DriftScore, Sample
from app.detector.cosine import drift_from_similarity, mean_top_k_similarity
from app.detector.kl import token_kl_between_corpora
from app.detector.mmd import mmd2, normalize_mmd
from app.embedder import Embedder
from app.probes import ProbeCategory

logger = logging.getLogger(__name__)


@dataclass
class CategoryDriftResult:
    category: str
    mmd_score: float
    kl_score: float
    cosine_score: float
    combined_score: float
    threshold: float
    is_alert: bool
    sample_count: int
    details: dict


class DriftDetector:
    def __init__(self, embedder: Embedder) -> None:
        self.settings = get_settings()
        self.embedder = embedder

    def _threshold_for(self, category: str) -> float:
        mapping = {
            "tone": self.settings.tone_threshold,
            "fact": self.settings.fact_threshold,
            "persona": self.settings.persona_threshold,
            "overall": self.settings.drift_threshold,
        }
        return mapping.get(category, self.settings.drift_threshold)

    def _combine(self, mmd: float, kl: float, cosine: float) -> float:
        w_m = self.settings.mmd_weight
        w_k = self.settings.kl_weight
        w_c = self.settings.cosine_weight
        total = w_m + w_k + w_c
        return float((w_m * mmd + w_k * kl + w_c * cosine) / total)

    async def _texts(
        self,
        session: AsyncSession,
        *,
        user_id: int,
        category: str,
        baseline: bool,
        limit: int,
    ) -> list[str]:
        q = (
            select(Sample)
            .where(
                Sample.user_id == user_id,
                Sample.category == category,
                Sample.is_baseline == baseline,
            )
            .order_by(Sample.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(q)
        return [s.response for s in result.scalars().all()]

    async def _cosine_drift(
        self, session: AsyncSession, *, user_id: int, category: str
    ) -> tuple[float, dict]:
        q = (
            select(Sample)
            .where(
                Sample.user_id == user_id,
                Sample.category == category,
                Sample.is_baseline.is_(False),
            )
            .order_by(Sample.created_at.desc())
            .limit(max(5, self.settings.recent_window_size // 3))
        )
        result = await session.execute(q)
        live_samples = list(result.scalars().all())
        if not live_samples:
            return 0.0, {"per_probe": {}}

        per_probe: dict[str, float] = {}
        drifts: list[float] = []
        for sample in live_samples:
            vector = self.embedder.embed([sample.response])[0]
            sims, _ = self.embedder.nearest_baseline(
                user_id, vector, sample.probe_id, k=self.settings.cosine_k
            )
            sim = mean_top_k_similarity(sims)
            d = drift_from_similarity(sim)
            drifts.append(d)
            per_probe[sample.probe_id] = max(per_probe.get(sample.probe_id, 0.0), d)

        return float(np.mean(drifts)) if drifts else 0.0, {"per_probe": per_probe}

    async def evaluate_category(
        self, session: AsyncSession, *, user_id: int, category: str
    ) -> CategoryDriftResult:
        baseline_vecs = self.embedder.get_baseline_vectors(user_id, category)
        per_category_window = max(5, self.settings.recent_window_size // 3)
        live_vecs = self.embedder.get_recent_live_vectors(
            user_id, category, limit=per_category_window
        )

        if len(baseline_vecs) and len(live_vecs):
            raw_mmd = mmd2(
                baseline_vecs,
                live_vecs,
                gamma=self.settings.mmd_gamma,
                biased=True,
            )
        else:
            raw_mmd = 0.0
        mmd_score = normalize_mmd(raw_mmd)

        baseline_texts = await self._texts(
            session, user_id=user_id, category=category, baseline=True, limit=200
        )
        live_texts = await self._texts(
            session,
            user_id=user_id,
            category=category,
            baseline=False,
            limit=per_category_window,
        )
        kl_score = token_kl_between_corpora(baseline_texts, live_texts)
        cosine_score, cosine_details = await self._cosine_drift(
            session, user_id=user_id, category=category
        )

        if category == "fact":
            fact_rate = await self._fact_failure_rate(
                session, user_id=user_id, limit=per_category_window
            )
            cosine_score = max(cosine_score, fact_rate)
            cosine_details["fact_failure_rate"] = fact_rate

        combined = self._combine(mmd_score, kl_score, cosine_score)
        threshold = self._threshold_for(category)
        details = {
            "raw_mmd2": raw_mmd,
            "baseline_n": int(len(baseline_vecs)),
            "live_n": int(len(live_vecs)),
            **cosine_details,
        }

        return CategoryDriftResult(
            category=category,
            mmd_score=mmd_score,
            kl_score=kl_score,
            cosine_score=cosine_score,
            combined_score=combined,
            threshold=threshold,
            is_alert=combined >= threshold and len(live_vecs) > 0,
            sample_count=len(live_vecs),
            details=details,
        )

    async def _fact_failure_rate(
        self, session: AsyncSession, *, user_id: int, limit: int = 10
    ) -> float:
        q = (
            select(Sample)
            .where(
                Sample.user_id == user_id,
                Sample.category == "fact",
                Sample.is_baseline.is_(False),
            )
            .order_by(Sample.created_at.desc())
            .limit(limit)
        )
        result = await session.execute(q)
        rows = list(result.scalars().all())
        checked = [r for r in rows if r.fact_ok is not None]
        if not checked:
            return 0.0
        return float(sum(1 for r in checked if r.fact_ok is False) / len(checked))

    async def evaluate_all(
        self, session: AsyncSession, *, user_id: int
    ) -> list[CategoryDriftResult]:
        results: list[CategoryDriftResult] = []
        for cat in ProbeCategory:
            results.append(
                await self.evaluate_category(session, user_id=user_id, category=cat.value)
            )

        if results:
            overall_mmd = float(np.mean([r.mmd_score for r in results]))
            overall_kl = float(np.mean([r.kl_score for r in results]))
            overall_cos = float(np.mean([r.cosine_score for r in results]))
            combined = self._combine(overall_mmd, overall_kl, overall_cos)
            threshold = self._threshold_for("overall")
            overall = CategoryDriftResult(
                category="overall",
                mmd_score=overall_mmd,
                kl_score=overall_kl,
                cosine_score=overall_cos,
                combined_score=combined,
                threshold=threshold,
                is_alert=combined >= threshold,
                sample_count=sum(r.sample_count for r in results),
                details={"categories": [r.category for r in results]},
            )
            results.append(overall)

        for r in results:
            session.add(
                DriftScore(
                    user_id=user_id,
                    category=r.category,
                    mmd_score=r.mmd_score,
                    kl_score=r.kl_score,
                    cosine_score=r.cosine_score,
                    combined_score=r.combined_score,
                    threshold=r.threshold,
                    is_alert=r.is_alert,
                    sample_count=r.sample_count,
                    details_json=json.dumps(r.details),
                )
            )
        await session.commit()
        return results

    def pca_explanation(
        self, user_id: int, category: Optional[str] = None, n_components: int = 3
    ) -> dict:
        baseline = self.embedder.get_baseline_vectors(user_id, category)
        live = self.embedder.get_recent_live_vectors(user_id, category, limit=100)
        if len(baseline) < 3 or len(live) < 2:
            return {"available": False, "components": []}

        pca = PCA(n_components=min(n_components, baseline.shape[1], len(baseline)))
        pca.fit(baseline)
        b_proj = pca.transform(baseline)
        l_proj = pca.transform(live)
        b_mean = b_proj.mean(axis=0)
        l_mean = l_proj.mean(axis=0)
        delta = l_mean - b_mean

        components = []
        for i in range(len(delta)):
            components.append(
                {
                    "component": i + 1,
                    "baseline_mean": float(b_mean[i]),
                    "live_mean": float(l_mean[i]),
                    "delta": float(delta[i]),
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[i]),
                }
            )
        return {
            "available": True,
            "category": category or "all",
            "components": components,
            "total_explained": float(sum(pca.explained_variance_ratio_)),
        }
