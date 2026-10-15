"""Test defaults: unit tests never call a model or LangSmith, but the model clients need a key to construct."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("OPENROUTER_API_KEY", "placeholder-unit-tests-make-no-model-calls")
os.environ.setdefault("USE_LOCAL_PROMPTS", "true")
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_TRACING"] = "false"
