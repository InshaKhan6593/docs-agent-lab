"""The docs agent's system prompt (Hub push and eval mirror of instructions.md)."""

from pathlib import Path

from agent.config import PYLON_ENABLED

_AGENT_DIR = Path(__file__).resolve().parent.parent
# chat-langchain's exact prompt needs the Pylon support KB; without it, use the docs-only derivative.
PROMPT_FILE = _AGENT_DIR / ("instructions.chat-langchain.md" if PYLON_ENABLED else "instructions.md")
docs_agent_prompt = PROMPT_FILE.read_text(encoding="utf-8")
