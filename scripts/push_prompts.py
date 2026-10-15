"""Push the docs-agent and guardrails prompts to LangSmith Prompt Hub (private, in your workspace).

Same split as chat-langchain:
* guardrails prompt: pulled live from `<name>:production` at startup (falls back to the local file), so a
  change ships by moving a tag, no redeploy.
* docs-agent prompt: the agent runs from agent/instructions*.md; the Hub copy is pulled only to stamp
  `prompt_source` / `prompt_commit` on every root trace, so you can tell which prompt version answered.

    uv run python scripts/push_prompts.py --tag staging       # try a new prompt version (LANGSMITH_ENV=dev reads it)
    uv run python scripts/push_prompts.py --tag production    # promote it
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from langchain_core.messages import SystemMessage  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langsmith import Client  # noqa: E402

from agent import langsmith_names as names  # noqa: E402
from agent.prompts.docs_agent_prompt import PROMPT_FILE, docs_agent_prompt  # noqa: E402
from agent.prompts.guardrails_prompts import guardrails_system_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", choices=["staging", "production"], required=True)
    args = parser.parse_args()

    client = Client()
    prompts = [
        (names.DOCS_AGENT_PROMPT, docs_agent_prompt, f"Docs agent system prompt ({PROMPT_FILE.name})"),
        (names.GUARDRAILS_PROMPT, guardrails_system_prompt, "Guardrails classifier system prompt"),
    ]
    for name, text, description in prompts:
        # A SystemMessage is stored literally (not as a template), so braces in the prompt are safe.
        prompt =ChatPromptTemplate.from_messages([SystemMessage(content=text)])
        url = client.push_prompt(name, object=prompt, is_public=False, description=description, commit_tags=[args.tag])
        print(f"{name}:{args.tag} -> {url}")


if __name__ == "__main__":
    main()
