"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

from agent import langsmith_names as names
from agent.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"
_AGENT_DIR = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def _local_code_version() -> str:
    """Fingerprint of the agent's code and prompts, for runs with no deploy revision.

    chat-langchain gets LANGSMITH_AGENT_VERSION from its deployment revision. Running
    locally there is none, so hash agent/*.py + agent/*.md instead: any code or prompt
    change gives a new version, and traces can be filtered by it.
    """
    digest = hashlib.sha256()
    for path in sorted(_AGENT_DIR.rglob("*")):
        if path.suffix in {".py", ".md"} and "__pycache__" not in path.parts:
            digest.update(path.relative_to(_AGENT_DIR).as_posix().encode())
            digest.update(path.read_bytes())
    return f"local-{digest.hexdigest()[:12]}"


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
) -> dict[str, str]:
    """Return metadata that should land on the root LangSmith run."""
    metadata: dict[str, str] = {
        "source_type": names.SOURCE_TYPE,
        # chat-langchain's frontend sends `graph` with every run; set it server-side so any UI gets it.
        "graph": graph_id,
        **get_prompt_provenance(graph_id),
    }
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    metadata["LANGSMITH_AGENT_VERSION"] = revision or _local_code_version()
    return metadata


__all__ = ["build_docs_agent_trace_metadata"]
