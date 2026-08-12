from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llmtest.runner import run_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmtest",
        description="Semantic regression tests for LLMs (pytest for AI behavior).",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=["run", "baseline", "update-baseline"],
        help="run (default), baseline (capture), or update-baseline",
    )
    # Also support flags matching the pitch: llmtest --baseline
    parser.add_argument("--baseline", action="store_true", help="Capture baselines")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baselines with current responses",
    )
    parser.add_argument(
        "--test-dir",
        default="llmtests",
        help="Directory containing test_*.py suites (default: llmtests)",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        default=True,
        help="Exit non-zero when any test fails (default: true)",
    )
    parser.add_argument(
        "--no-fail-on-drift",
        action="store_true",
        help="Always exit 0 (report only)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=None,
        help="Override suite cosine similarity floor for baseline compares",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print a markdown summary table",
    )
    args = parser.parse_args(argv)

    mode = args.command
    if args.baseline:
        mode = "baseline"
    if args.update_baseline:
        mode = "update-baseline"

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}", file=sys.stderr)
        return 2

    report = run_directory(
        test_dir,
        mode=mode,
        cwd=Path.cwd(),
        default_threshold=args.drift_threshold,
    )

    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        score = "—" if r.score is None else f"{r.score:.3f}"
        print(f"[{mark}] {r.suite}.{r.name}  score={score}  {r.message}")

    if args.markdown or mode == "run":
        print()
        print(report.to_markdown())

    print()
    print(f"{report.passed} passed, {report.failed} failed")

    fail = not args.no_fail_on_drift
    if fail and report.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
