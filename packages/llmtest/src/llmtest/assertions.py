from __future__ import annotations

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
    threshold: float = 0.82,
    name: str = "tone_matches",
) -> AssertionResult:
    emb = get_embedder()
    # Phrase the persona as an exemplar utterance for better embedding alignment
    persona_text = f"A response that is {persona}."
    vecs = emb.embed([response, persona_text])
    score = cosine_similarity(vecs[0], vecs[1])
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


def assert_semantically_excludes(
    response: str,
    concept: str,
    *,
    max_similarity: float = 0.55,
    name: str = "semantically_excludes",
) -> AssertionResult:
    emb = get_embedder()
    vecs = emb.embed([response, concept])
    score = cosine_similarity(vecs[0], vecs[1])
    passed = score <= max_similarity
    message = (
        f"{name}: concept_similarity={score:.3f} (need <= {max_similarity:.3f})"
        if passed
        else f"{name} FAILED: concept_similarity={score:.3f} > {max_similarity:.3f} for '{concept}'"
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
