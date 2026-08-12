from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from llmtest.embedder import Embedder
from llmtest.similarity import cosine_similarity


@dataclass
class AssertionResult:
    name: str
    passed: bool
    score: float
    threshold: float
    message: str
    kind: str


class SemanticAssertionError(AssertionError):
    def __init__(self, result: AssertionResult) -> None:
        super().__init__(result.message)
        self.result = result


_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def set_embedder(embedder: Embedder) -> None:
    global _embedder
    _embedder = embedder


def assert_semantically_equals(
    response: str,
    baseline: str,
    *,
    threshold: float = 0.82,
    name: str = "semantically_equals",
) -> AssertionResult:
    emb = get_embedder()
    vecs = emb.embed([response, baseline])
    score = cosine_similarity(vecs[0], vecs[1])
    passed = score >= threshold
    message = (
        f"{name}: similarity={score:.3f} (need >= {threshold:.3f})"
        if passed
        else f"{name} FAILED: similarity={score:.3f} < {threshold:.3f}"
    )
    result = AssertionResult(
        name=name,
        passed=passed,
        score=score,
        threshold=threshold,
        message=message,
        kind="equals",
    )
    if not passed:
        raise SemanticAssertionError(result)
    return result


def assert_tone_matches(
    response: str,
    persona: str,
    *,
    threshold: float = 0.45,
    name: str = "tone_matches",
) -> AssertionResult:
    """
    Score the reply against concrete exemplar utterances (not bare adjectives).
    """
    emb = get_embedder()
    exemplars = [
        "I'm sorry you're dealing with this. I understand how frustrating it is, and I'm here to help.",
        "I hear your frustration. Let's work through this calmly and professionally.",
        "Thank you for telling me — I want to help carefully and respectfully.",
        f"I'll respond in a way that is {persona}.",
    ]
    vectors = emb.embed([response] + exemplars)
    live = vectors[0]
    score = max(cosine_similarity(live, vectors[i]) for i in range(1, len(vectors)))
    passed = score >= threshold
    message = (
        f"{name}: tone_similarity={score:.3f} (need >= {threshold:.3f})"
        if passed
        else f"{name} FAILED: tone_similarity={score:.3f} < {threshold:.3f} for persona '{persona}'"
    )
    result = AssertionResult(
        name=name,
        passed=passed,
        score=score,
        threshold=threshold,
        message=message,
        kind="tone",
    )
    if not passed:
        raise SemanticAssertionError(result)
    return result


_DENIAL_RE = re.compile(
    r"\b(do not|don't|does not|doesn't|not currently|unable to|cannot|can't|no longer)\b"
    r".{0,40}\b(support|offer|provide|integrate|available)\b"
    r"|\b(not supported|unsupported|unavailable)\b",
    re.IGNORECASE | re.DOTALL,
)


def assert_semantically_excludes(
    response: str,
    concept: str,
    *,
    max_similarity: float = 0.55,
    name: str = "semantically_excludes",
) -> AssertionResult:
    """
    Prefer clear refusals over raw cosine-to-forbidden-phrase, which false-fires when
    the model mentions a topic only to deny it.
    """
    emb = get_embedder()
    topic = re.sub(r"^(yes[,.]?\s*|no[,.]?\s*)", "", concept, flags=re.I).strip()
    affirmation = f"Yes — we fully support this and can enable it today: {topic}"
    refusal = f"No — we do not support this and cannot enable it: {topic}"
    vectors = emb.embed([response, affirmation, refusal])
    sim_yes = cosine_similarity(vectors[0], vectors[1])
    sim_no = cosine_similarity(vectors[0], vectors[2])
    lexical_denial = bool(_DENIAL_RE.search(response))

    passed = lexical_denial or sim_no >= sim_yes or sim_yes <= max_similarity
    score = sim_yes
    message = (
        f"{name}: yes_sim={sim_yes:.3f} no_sim={sim_no:.3f} denial={lexical_denial}"
        if passed
        else (
            f"{name} FAILED: reply aligns with forbidden concept "
            f"(yes_sim={sim_yes:.3f} > no_sim={sim_no:.3f}, denial={lexical_denial})"
        )
    )
    result = AssertionResult(
        name=name,
        passed=passed,
        score=score,
        threshold=max_similarity,
        message=message,
        kind="excludes",
    )
    if not passed:
        raise SemanticAssertionError(result)
    return result


def similarity_to_text(a: str, b: str) -> float:
    emb = get_embedder()
    vecs = emb.embed([a, b])
    return cosine_similarity(vecs[0], vecs[1])


def similarity_to_vector(text: str, vector: list[float] | np.ndarray) -> float:
    emb = get_embedder()
    live = emb.embed_one(text)
    return cosine_similarity(live, np.asarray(vector, dtype=np.float32))
