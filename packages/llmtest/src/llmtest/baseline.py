from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def baseline_root(cwd: Optional[Path] = None) -> Path:
    root = Path(cwd or Path.cwd()) / ".llmtest" / "baseline"
    return root


def baseline_path(suite_name: str, test_name: str, cwd: Optional[Path] = None) -> Path:
    safe_suite = "".join(c if c.isalnum() or c in "-_" else "_" for c in suite_name)
    safe_test = "".join(c if c.isalnum() or c in "-_" else "_" for c in test_name)
    return baseline_root(cwd) / safe_suite / f"{safe_test}.json"


def load_baseline(suite_name: str, test_name: str, cwd: Optional[Path] = None) -> Optional[dict[str, Any]]:
    path = baseline_path(suite_name, test_name, cwd)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(
    suite_name: str,
    test_name: str,
    payload: dict[str, Any],
    cwd: Optional[Path] = None,
) -> Path:
    path = baseline_path(suite_name, test_name, cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
