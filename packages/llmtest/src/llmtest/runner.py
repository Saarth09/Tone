from __future__ import annotations

import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from llmtest.assertions import similarity_to_text, similarity_to_vector
from llmtest.baseline import load_baseline, save_baseline
from llmtest.embedder import get_embedder
from llmtest.suite import LLMTestSuite, TestCase


@dataclass
class TestReport:
    suite: str
    name: str
    passed: bool
    score: Optional[float]
    threshold: Optional[float]
    message: str
    prompt: Optional[str] = None
    baseline_response: Optional[str] = None
    current_response: Optional[str] = None


@dataclass
class RunReport:
    results: list[TestReport] = field(default_factory=list)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def to_markdown(self) -> str:
        lines = [
            "| Test | Score | Result | Notes |",
            "|---|---:|:---:|---|",
        ]
        for r in self.results:
            score = "—" if r.score is None else f"{r.score:.3f}"
            status = "pass" if r.passed else "FAIL"
            note = r.message.replace("|", "\\|")[:120]
            lines.append(f"| `{r.suite}.{r.name}` | {score} | {status} | {note} |")
        if self.failed:
            lines.append("")
            lines.append("### Failures")
            for r in self.results:
                if r.passed:
                    continue
                lines.append(f"**`{r.suite}.{r.name}`** — {r.message}")
                if r.baseline_response or r.current_response:
                    lines.append("")
                    lines.append("| | |")
                    lines.append("|---|---|")
                    lines.append(f"| Baseline | {(r.baseline_response or '')[:400]} |")
                    lines.append(f"| Current | {(r.current_response or '')[:400]} |")
                    lines.append("")
        return "\n".join(lines)


def discover_suites(test_dir: Path) -> list[LLMTestSuite]:
    suites: list[LLMTestSuite] = []
    paths = sorted(test_dir.rglob("test_*.py"))
    for path in paths:
        mod_name = f"llmtest_user_{path.stem}_{abs(hash(path))}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            traceback.print_exc()
            continue
        for value in vars(module).values():
            if isinstance(value, LLMTestSuite):
                if not value.name or value.name == "default":
                    value.name = path.stem
                suites.append(value)
    return suites


def _best_score(results: list[Any]) -> Optional[float]:
    if not results:
        return None
    # Prefer equals/tone scores (higher is better); for excludes, invert display
    for r in results:
        if r.kind in {"equals", "tone"}:
            return r.score
    return results[0].score


def run_suite(
    suite: LLMTestSuite,
    *,
    mode: str = "run",
    cwd: Optional[Path] = None,
    default_threshold: Optional[float] = None,
) -> list[TestReport]:
    """
    mode: run | baseline | update-baseline
    """
    reports: list[TestReport] = []
    threshold = default_threshold if default_threshold is not None else suite.threshold
    emb = get_embedder()

    for case in suite.tests:
        if mode in {"baseline", "update-baseline"}:
            passed, assertion_results, err = suite.run_test(case)
            if case.last_response is None:
                reports.append(
                    TestReport(
                        suite=suite.name or "default",
                        name=case.name,
                        passed=False,
                        score=None,
                        threshold=threshold,
                        message=err or "Test did not call suite.query()",
                    )
                )
                continue
            vector = emb.embed_one(case.last_response)
            save_baseline(
                suite.name or "default",
                case.name,
                {
                    "prompt": case.last_prompt,
                    "response": case.last_response,
                    "embedding": vector.tolist(),
                    "meta": {
                        "model": suite.model,
                        "threshold": threshold,
                    },
                },
                cwd=cwd,
            )
            reports.append(
                TestReport(
                    suite=suite.name or "default",
                    name=case.name,
                    passed=True,
                    score=1.0,
                    threshold=threshold,
                    message=f"Baseline saved ({mode})",
                    prompt=case.last_prompt,
                    current_response=case.last_response,
                    baseline_response=case.last_response,
                )
            )
            continue

        # Normal run: execute test asserts + baseline semantic compare if present
        passed, assertion_results, err = suite.run_test(case)
        baseline = load_baseline(suite.name or "default", case.name, cwd=cwd)
        score = _best_score(assertion_results)
        message = err or (
            assertion_results[-1].message if assertion_results else "ok"
        )
        baseline_response = None
        if baseline and case.last_response:
            baseline_response = baseline.get("response")
            if baseline.get("embedding"):
                base_score = similarity_to_vector(case.last_response, baseline["embedding"])
            else:
                base_score = similarity_to_text(case.last_response, baseline_response or "")
            score = base_score if score is None else min(score, base_score)
            if base_score < threshold:
                passed = False
                message = (
                    f"baseline regression: similarity={base_score:.3f} < {threshold:.3f}"
                )
        elif baseline is None and mode == "run":
            # Soft warning — assertions may still pass without snapshot
            if passed and not assertion_results:
                message = "no baseline (run `llmtest --baseline`) and no assertions recorded"

        reports.append(
            TestReport(
                suite=suite.name or "default",
                name=case.name,
                passed=passed and err is None,
                score=score,
                threshold=threshold,
                message=message,
                prompt=case.last_prompt,
                baseline_response=baseline_response,
                current_response=case.last_response,
            )
        )
    return reports


def run_directory(
    test_dir: Path,
    *,
    mode: str = "run",
    cwd: Optional[Path] = None,
    default_threshold: Optional[float] = None,
) -> RunReport:
    suites = discover_suites(test_dir)
    report = RunReport()
    if not suites:
        report.results.append(
            TestReport(
                suite="-",
                name="discover",
                passed=False,
                score=None,
                threshold=None,
                message=f"No LLMTestSuite found under {test_dir}",
            )
        )
        return report
    for suite in suites:
        report.results.extend(
            run_suite(
                suite,
                mode=mode,
                cwd=cwd,
                default_threshold=default_threshold,
            )
        )
    return report
