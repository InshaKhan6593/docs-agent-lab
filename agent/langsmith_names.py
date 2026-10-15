"""Every LangSmith resource this agent reads or writes, in one place.

chat-langchain points at LangChain's own workspace (`public-chat-langchain-test` prompts, the
`Chat-LangChain-Guardrails-Samples` dataset). These are the same resources, in your workspace.
"""

import os

# LANGSMITH_ENV=dev -> use the :staging tag of each Hub prompt instead of :production
STAGING = os.getenv("LANGSMITH_ENV") == "dev"
TAG = "staging" if STAGING else "production"

DOCS_AGENT_PROMPT = os.getenv("DOCS_AGENT_PROMPT", "docs-agent-lab")
GUARDRAILS_PROMPT = os.getenv("GUARDRAILS_PROMPT", "docs-agent-lab-guardrails")
GUARDRAILS_DATASET = os.getenv("GUARDRAILS_DATASET", "docs-agent-lab-guardrails-samples")
EVAL_DATASET = os.getenv("EVAL_DATASET", "docs-agent-lab-eval")

# chat-langchain copies 100% of blocked and 1% of allowed questions into a dataset, so real
# traffic becomes eval data. That writes to your workspace, so it is opt-in here.
SAMPLE_TO_DATASET = os.getenv("GUARDRAILS_SAMPLE_TO_DATASET", "false").lower() in {"1", "true", "yes"}

# Root-run metadata tag (chat-langchain uses "Chat-LangChain").
SOURCE_TYPE = os.getenv("SOURCE_TYPE", "docs-agent-lab")


def hub_ref(name: str) -> str:
    return f"{name}:{TAG}"
