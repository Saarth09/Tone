# llmtest

**pytest for AI behavior** — semantic regression tests for LLM apps.

Part of [Tone](https://github.com/Saarth09/Tone): monitoring in production, prevention in CI.

## Install

```bash
pip install -e packages/llmtest
# needs sentence-transformers / torch (CPU is fine)
```

## Write tests

```python
# llmtests/test_customer_support.py
from llmtest import LLMTestSuite, assert_semantically_equals, assert_tone_matches

suite = LLMTestSuite(
    model="gpt-4o-mini",
    system_prompt_path="prompts/customer_support.txt",
    threshold=0.82,
)

@suite.test
def test_refund_policy_explanation():
    response = suite.query("How do I get a refund?")
    assert_semantically_equals(
        response,
        baseline="You can request a refund within 30 days by contacting support.",
        threshold=0.85,
    )
```

## Run

```bash
export OPENAI_API_KEY=sk-...
# optional: LLM_BASE_URL=https://api.openai.com/v1

llmtest --baseline --test-dir packages/llmtest/examples   # capture snapshots
llmtest run --test-dir packages/llmtest/examples          # compare / assert
llmtest --update-baseline --test-dir packages/llmtest/examples
```

Baselines land in `.llmtest/baseline/` (commit them like Jest snapshots).

## Assertions

| Helper | Passes when |
|--------|-------------|
| `assert_semantically_equals` | cosine(response, baseline) ≥ threshold |
| `assert_tone_matches` | response aligns with a persona phrase |
| `assert_semantically_excludes` | cosine(response, forbidden concept) ≤ ceiling |

## GitHub Action

See [`.github/actions/llmtest`](../../.github/actions/llmtest) and the root README.
