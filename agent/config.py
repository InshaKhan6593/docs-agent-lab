"""Shared configuration for all agents."""

import logging
import os
from dataclasses import dataclass

import dotenv
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from agent.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
)
from agent.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a supported chat model."""

    id: str  # e.g., "openrouter:qwen/qwen3.8-flash"
    name: str  # Display name, e.g., "Gemini 3.5 Flash Lite"
    provider: str  # e.g., "openrouter"
    api_key_env: str  # Environment variable for API key
    description: str | None = None


# Backend-supported models: cheap Qwen models via OpenRouter (one key, one bill), in the same roles
# chat-langchain uses (main / guardrails / fallbacks). `init_chat_model("openrouter:...")` uses
# langchain-openrouter's ChatOpenRouter. Prices are OpenRouter's per 1M tokens (in / out), Oct 2026.
# chat-langchain's originals: Gemini 3.5 Flash Lite (main), GPT-5.4 Nano, Claude Haiku 4.5.
MODELS: dict[str, ModelConfig] = {
    "qwen3.8-flash": ModelConfig(
        id="openrouter:qwen/qwen3.8-flash",
        name="Qwen3.8 Flash",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        description="Main agent: newest Qwen Flash, 1M context, tool calling ($0.15 / $0.47)",
    ),
    "qwen3.7-flash": ModelConfig(
        id="openrouter:qwen/qwen3.7-flash",
        name="Qwen3.7 Flash",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        description="Cheapest Qwen with tools + structured output, 1M context ($0.03 / $0.13)",
    ),
    "qwen3-235b-a22b-2507": ModelConfig(
        id="openrouter:qwen/qwen3-235b-a22b-2507",
        name="Qwen3 235B A22B (2507)",
        provider="openrouter",
        api_key_env="OPENROUTER_API_KEY",
        description="Larger open-weight Qwen, a different line from Flash, 262k context ($0.09 / $0.55)",
    ),
}

# Default models for different use cases
DEFAULT_MODEL = MODELS["qwen3.8-flash"]
GUARDRAILS_MODEL = MODELS["qwen3.7-flash"]

# Fallback chain (in order of preference)
FALLBACK_MODELS = [
    MODELS["qwen3.7-flash"],
    MODELS["qwen3-235b-a22b-2507"],
]

# Pylon is LangChain's private support knowledge base. Its tools (and the exact chat-langchain prompt,
# which relies on them) are used only when both keys are set; otherwise the docs-only prompt is used.
PYLON_ENABLED = bool(os.getenv("PYLON_API_KEY") and os.getenv("PYLON_KB_ID"))

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENROUTER_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))

# Primary model. Public callers cannot switch this at runtime.
default_model = init_chat_model(model=DEFAULT_MODEL.id)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str) -> Runnable:
    return (
        init_chat_model(model=model)
        | RunnableLambda(_raise_for_retryable_finish_reason)
    ).with_retry(stop_after_attempt=MAX_RETRIES + 1)


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [
        _init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS
    ]
    return primary_model.with_fallbacks(fallback_models)


summarization_model = init_retry_fallback_model(DEFAULT_MODEL.id)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)

model_fallback_middleware = ModelFallbackMiddleware(*[m.id for m in FALLBACK_MODELS])
logger.info(f"Fallback chain: {' -> '.join(m.name for m in FALLBACK_MODELS)}")

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MODELS",
    "DEFAULT_MODEL",
    "GUARDRAILS_MODEL",
    "FALLBACK_MODELS",
    "ModelConfig",
    # Models
    "default_model",
    "init_retry_fallback_model",
    "summarization_model",
    # Middleware
    "model_retry_middleware",
    "tool_retry_middleware",
    "model_fallback_middleware",
    # Config
    "MAX_RETRIES",
    "PYLON_ENABLED",
    "logger",
]
