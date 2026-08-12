from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

from llmtest.assertions import AssertionResult, SemanticAssertionError
from llmtest.embedder import get_embedder


TestFn = Callable[..., Any]


@dataclass
class TestCase:
    name: str
    fn: TestFn
    last_prompt: Optional[str] = None
    last_response: Optional[str] = None
    assertion_results: list[AssertionResult] = field(default_factory=list)


@dataclass
class LLMTestSuite:
    model: str = "gpt-4o-mini"
    system_prompt: str = ""
    system_prompt_path: Optional[str] = None
    threshold: float = 0.82
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    name: Optional[str] = None
    _tests: list[TestCase] = field(default_factory=list, init=False, repr=False)
    _current: Optional[TestCase] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.system_prompt_path:
            path = Path(self.system_prompt_path)
            if path.exists():
                self.system_prompt = path.read_text(encoding="utf-8").strip()
        if not self.name:
            self.name = "default"

    def test(self, fn: TestFn) -> TestFn:
        self._tests.append(TestCase(name=fn.__name__, fn=fn))
        return fn

    @property
    def tests(self) -> list[TestCase]:
        return list(self._tests)

    def _endpoint(self) -> tuple[str, str, str]:
        base = (self.base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        key = self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
        model = self.model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return base, key, model

    def query(self, prompt: str, *, temperature: float = 0.2) -> str:
        base, key, model = self._endpoint()
        if not key and "localhost" not in base and "127.0.0.1" not in base:
            raise RuntimeError(
                "No API key set. Export OPENAI_API_KEY (or LLM_API_KEY), "
                "or pass api_key= to LLMTestSuite."
            )
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        headers = {
            "Authorization": f"Bearer {key.strip() or 'none'}",
            "Content-Type": "application/json",
        }
        payload = {"model": model, "messages": messages, "temperature": temperature}
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(f"{base}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        if self._current is not None:
            self._current.last_prompt = prompt
            self._current.last_response = content
        return content

    def run_test(self, case: TestCase) -> tuple[bool, list[AssertionResult], Optional[str]]:
        self._current = case
        case.assertion_results = []
        case.last_prompt = None
        case.last_response = None
        err: Optional[str] = None
        try:
            params = inspect.signature(case.fn).parameters
            if len(params) >= 1:
                case.fn(self)
            else:
                case.fn()
        except SemanticAssertionError as exc:
            case.assertion_results.append(exc.result)
            err = str(exc)
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
        finally:
            self._current = None
        passed = err is None
        return passed, case.assertion_results, err
