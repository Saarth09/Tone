from __future__ import annotations

import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_distribution(texts: list[str], vocab: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Build a smoothed unigram distribution over a shared vocabulary."""
    counters = [Counter(tokenize(t)) for t in texts]
    if vocab is None:
        merged: Counter[str] = Counter()
        for c in counters:
            merged.update(c)
        # Keep top tokens for stability
        vocab = [w for w, _ in merged.most_common(2000)]
        if not vocab:
            vocab = ["<empty>"]

    total_counts = np.zeros(len(vocab), dtype=np.float64)
    index = {w: i for i, w in enumerate(vocab)}
    for c in counters:
        for w, n in c.items():
            if w in index:
                total_counts[index[w]] += n

    # Additive smoothing
    total_counts += 1e-6
    total_counts /= total_counts.sum()
    return total_counts, vocab


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with numerical guards."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def normalize_kl(raw: float, scale: float = 0.5) -> float:
    """Map KL into roughly [0, 1]."""
    return float(1.0 - np.exp(-raw * scale))


def token_kl_between_corpora(baseline_texts: list[str], live_texts: list[str]) -> float:
    if not baseline_texts or not live_texts:
        return 0.0
    p, vocab = build_distribution(baseline_texts)
    q, _ = build_distribution(live_texts, vocab=vocab)
    return normalize_kl(kl_divergence(q, p))
