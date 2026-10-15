"""Tests for summarization middleware retry/fallback wiring."""

from agent import config
from agent.middleware.summarization_middleware import CustomSummarizationMiddleware
from agent.prompts.context_summary_prompt import context_summary_prompt


def test_summarization_middleware_uses_retrying_fallback_model():
    """Summarization should use the default model plus shared retry/fallback policy."""
    assert [model.id for model in config.FALLBACK_MODELS] == [
        "openrouter:qwen/qwen3.7-flash",
        "openrouter:qwen/qwen3-235b-a22b-2507",
    ]

    middleware = CustomSummarizationMiddleware(
        model=config.DEFAULT_MODEL.id,
        summary_model=config.summarization_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    )

    summary_model = middleware.summary_model
    summary_primary = getattr(summary_model, "runnable", None)
    summary_fallbacks = getattr(summary_model, "fallbacks", [])

    assert middleware.model.model == "qwen/qwen3.8-flash"  # OpenRouter model id
    assert type(summary_model).__name__ == "RunnableWithFallbacks"
    assert getattr(summary_primary, "max_attempt_number") == config.MAX_RETRIES + 1
    assert len(summary_fallbacks) == len(config.FALLBACK_MODELS)
    assert all(
        getattr(fallback, "max_attempt_number") == config.MAX_RETRIES + 1
        for fallback in summary_fallbacks
    )
