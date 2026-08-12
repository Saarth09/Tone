# Example behavioral suite for llmtest

from llmtest import (
    LLMTestSuite,
    assert_semantically_equals,
    assert_semantically_excludes,
    assert_tone_matches,
)

suite = LLMTestSuite(
    model="gpt-4o-mini",
    system_prompt_path="packages/llmtest/examples/prompts/customer_support.txt",
    threshold=0.75,
    name="customer_support",
)


@suite.test
def test_refund_policy_explanation():
    response = suite.query("How do I get a refund?")
    assert_semantically_equals(
        response,
        baseline=(
            "You can request a refund within 30 days by contacting our support team. "
            "They will help you complete the refund process."
        ),
        threshold=0.75,
    )


@suite.test
def test_tone_stays_friendly():
    response = suite.query("This product is broken and I'm furious.")
    assert_tone_matches(response, persona="empathetic, professional, calm", threshold=0.40)


@suite.test
def test_never_promises_unsupported_features():
    response = suite.query("Can you integrate with Salesforce?")
    assert_semantically_excludes(
        response,
        concept="Yes, we fully support Salesforce integration and can enable it for you today",
        max_similarity=0.60,
    )
