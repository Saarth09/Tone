"""Unit tests for MMD and KL helpers (no model download required)."""

import numpy as np

from app.detector.kl import kl_divergence, tokenize, token_kl_between_corpora
from app.detector.mmd import mmd2, normalize_mmd


def test_mmd_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8))
    score = mmd2(x, x.copy(), gamma=1.0)
    assert score < 1e-6


def test_mmd_shifted_distributions_higher():
    rng = np.random.default_rng(1)
    x = rng.normal(loc=0.0, size=(50, 8))
    y = rng.normal(loc=2.5, size=(50, 8))
    same = mmd2(x, x.copy(), gamma=1.0)
    diff = mmd2(x, y, gamma=1.0)
    assert diff > same
    assert normalize_mmd(diff) > normalize_mmd(same)


def test_kl_and_tokenize():
    assert "hello" in tokenize("Hello, world!")
    p = np.array([0.7, 0.3])
    q = np.array([0.7, 0.3])
    assert kl_divergence(p, q) < 1e-9

    baseline = ["The capital of France is Paris."] * 5
    live = ["The capital of France is Berlin."] * 5
    drifted = token_kl_between_corpora(baseline, live)
    stable = token_kl_between_corpora(baseline, baseline)
    assert drifted > stable
