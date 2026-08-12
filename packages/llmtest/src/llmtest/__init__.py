"""llmtest — semantic regression testing for LLMs."""

from llmtest.assertions import (
    AssertionResult,
    SemanticAssertionError,
    assert_semantically_equals,
    assert_semantically_excludes,
    assert_tone_matches,
)
from llmtest.suite import LLMTestSuite

__all__ = [
    "LLMTestSuite",
    "AssertionResult",
    "SemanticAssertionError",
    "assert_semantically_equals",
    "assert_semantically_excludes",
    "assert_tone_matches",
]

__version__ = "0.1.0"
