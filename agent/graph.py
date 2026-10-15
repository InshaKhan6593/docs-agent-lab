"""The docs agent: chat-langchain's agent, rebuilt from its open-source parts.

chat-langchain declares its agent with `managed_deepagents.define_deep_agent` (LangChain's hosted runtime).
That runtime compiles it into the open-source `deepagents.create_deep_agent`, appends the MCP connector's
tools, uses instructions.md as the system prompt and applies `metadata` with `.with_config` so it lands on
the LangSmith root run. This module does exactly that, without the hosted parts:

    hosted (MDA)                          here
    define_deep_agent(...)          ->    create_deep_agent(...)
    connectors/mcp.py               ->    MultiServerMCPClient (DOCS_MCP_URL)
    instructions.md from Context Hub ->   agent/instructions*.md
    platform checkpointer/store      ->   LangGraph server (`langgraph dev`) provides them
"""

import asyncio
import os

from deepagents import create_deep_agent
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import CallToolResult

from agent import langsmith_names as names
from agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    PYLON_ENABLED,
    model_fallback_middleware,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from agent.middleware.guardrails_middleware import GuardrailsMiddleware
from agent.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from agent.middleware.summarization_middleware import CustomSummarizationMiddleware
from agent.prompts.context_summary_prompt import context_summary_prompt
from agent.prompts.docs_agent_prompt import docs_agent_prompt
from agent.tools.link_check_tools import check_links
from agent.tools.pricing_tools import fetch_langchain_pricing
from agent.utils.trace_root_metadata import build_docs_agent_trace_metadata

# Our local ChromaFs server by default; set https://docs.langchain.com/mcp to run against Mintlify's.
DOCS_MCP_URL = os.getenv("DOCS_MCP_URL", "http://localhost:3333/mcp")


def authored_tools() -> list:
    """chat-langchain's own tools, in its order. Pylon (support KB) only when its keys are set."""
    tools: list = []
    if PYLON_ENABLED:
        from agent.tools.pylon_tools import get_support_article_content, search_support_articles

        tools += [search_support_articles, get_support_article_content]
    return tools + [fetch_langchain_pricing, check_links]


def docs_agent_middleware() -> list:
    """Identical to chat-langchain's agent.py."""
    return [
        # Cap oversized user input (was auth.py). Trace metadata is applied via
        # .with_config(metadata=...) so it lands on the LangSmith root run.
        IngressGuardsMiddleware(),
        GuardrailsMiddleware(
            model=GUARDRAILS_MODEL.id,
            fallback_model=DEFAULT_MODEL.id,
            block_off_topic=True,
        ),
        CustomSummarizationMiddleware(
            model=DEFAULT_MODEL.id,
            summary_model=summarization_model,
            trigger=("tokens", 130_000),
            keep=("tokens", 30_000),
            summary_prompt=context_summary_prompt,
            trim_tokens_to_summarize=None,
        ),
        tool_retry_middleware,
        model_retry_middleware,
        model_fallback_middleware,
    ]


async def text_results(request, handler):
    """Return pure-text MCP results as one plain string instead of a list of text blocks.

    Same text for the model, but chat UIs that only render string tool output (Agent Chat UI shows
    "[object Object]" for block lists) can display it. Non-text or error results pass through unchanged.
    """
    result = await handler(request)
    call_id = getattr(request.runtime, "tool_call_id", None)
    if (
        call_id
        and isinstance(result, CallToolResult)
        and not result.isError
        and result.content
        and all(block.type == "text" for block in result.content)
    ):
        text = "\n\n".join(block.text for block in result.content)
        return ToolMessage(content=text, tool_call_id=call_id, name=request.name)
    return result


async def load_docs_tools() -> list:
    """The docs MCP tools: search_docs_by_lang_chain, query_docs_filesystem_docs_by_lang_chain (+ submit_feedback on Mintlify)."""
    client = MultiServerMCPClient(
        {"langchain-docs": {"transport": "streamable_http", "url": DOCS_MCP_URL}},
        tool_interceptors=[text_results],
    )
    # Read-only agent: never let it file docs feedback on someone's site.
    return [t for t in await client.get_tools() if t.name != "submit_feedback"]


def _build(tools: list):
    agent = create_deep_agent(
        model=DEFAULT_MODEL.id,
        tools=tools,
        middleware=docs_agent_middleware(),
        system_prompt=docs_agent_prompt,
        name="docs_agent",
    )
    # chat-langchain's frontend tags every run [source_type, graph]; Agent Chat UI sends no tags, so the
    # graph adds them itself. Metadata goes on the root run, exactly as their managed runtime does it.
    return agent.with_config(
        {"metadata": build_docs_agent_trace_metadata(), "tags": [names.SOURCE_TYPE, "docs_agent"]}
    )


_graph = None
_graph_lock = asyncio.Lock()


async def make_graph(config=None):
    """Graph factory for `langgraph dev` / LangSmith Deployments (see langgraph.json).

    The server calls this for every run, so the agent is built once and reused. The build does
    synchronous work (model clients, Prompt Hub lookups, deepagents' own file checks), so it runs in
    a worker thread: `langgraph dev` rejects blocking calls on its event loop.
    """
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                tools = authored_tools() + await load_docs_tools()
                _graph = await asyncio.to_thread(_build, tools)
    return _graph
