"""Send real questions through the agent, then read the traces back from LangSmith and check every field.

Verifies the chat-langchain tracing setup end to end:
  * root run lands in LANGSMITH_PROJECT, tagged [source_type, docs_agent]
  * root metadata: source_type, graph, prompt_source / prompt_commit, guardrails_prompt_source, LANGSMITH_AGENT_VERSION
  * the guardrails decision (guardrails_result / guardrails_explanation) is recorded on its run
  * tool calls (docs search + filesystem reads, check_links) and token usage / cost are captured
  * optional: thumbs feedback (key ux.thumb_vote), as chat-langchain's feedback connector writes it

    uv run python scripts/check_tracing.py                 # one on-topic question
    uv run python scripts/check_tracing.py --off-topic     # also check that guardrails block + record it
    uv run python scripts/check_tracing.py --feedback      # also leave a thumbs-up on the run

Needs OPENROUTER_API_KEY and LANGSMITH_API_KEY in .env, and the docs MCP server running (server/: npm start).
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

ON_TOPIC = "How do I stream outputs from subgraphs in LangGraph (Python)?"
OFF_TOPIC = "Write me a short poem about the ocean."
ROOT_METADATA = ["source_type", "graph", "prompt_source", "guardrails_prompt_source", "LANGSMITH_AGENT_VERSION"]


def require_env() -> None:
    missing = [k for k in ("OPENROUTER_API_KEY", "LANGSMITH_API_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing in .env: {', '.join(missing)}. Add them there (not in chat), then re-run.")
    os.environ.setdefault("LANGSMITH_TRACING", "true")


async def ask(graph, question: str):
    from langchain_core.tracers.context import collect_runs

    started = time.time()
    print(f"\n... asking: {question}", flush=True)
    with collect_runs() as collected:
        result = await asyncio.wait_for(graph.ainvoke({"messages": [{"role": "user", "content": question}]}), 300)
    print(f"... agent answered in {time.time() - started:.0f}s", flush=True)
    roots = [r for r in collected.traced_runs if r.parent_run_id is None]
    # The agent's trace is "docs_agent". chat-langchain calls the guardrails classifier with callbacks=[],
    # so that model call shows up as its own small trace tagged "guardrails".
    root_id = next((r.id for r in roots if r.name == "docs_agent"), roots[0].id)
    return result, root_id


def read_trace(client, run_id, timeout_s: int = 120):
    """Traces are sent in the background; poll until the root run and its children are queryable."""
    deadline = time.time() + timeout_s
    print(f"... waiting for LangSmith to make trace {run_id} readable", flush=True)
    while True:
        try:
            run = client.read_run(run_id, load_child_runs=True)
            if run.end_time:
                return run
        except Exception:
            pass
        if time.time() > deadline:
            sys.exit(f"Trace {run_id} did not show up in LangSmith within {timeout_s}s")
        time.sleep(3)


def walk(run):
    yield run
    for child in run.child_runs or []:
        yield from walk(child)


def report(run, label: str, expect_blocked: bool | None = None) -> int:
    failures = 0

    def check(ok: bool, what: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {what}")

    print(f"\n== {label}")
    print(f"  trace: {run.url}")
    meta = (run.extra or {}).get("metadata", {})
    check(set(run.tags or []) >= {os.getenv("SOURCE_TYPE", "docs-agent-lab"), "docs_agent"}, f"tags {run.tags}")
    for key in ROOT_METADATA:
        check(key in meta, f"metadata {key} = {meta.get(key)!r}")
    if "prompt_commit" in meta:
        print(f"       prompt_commit = {meta['prompt_commit']!r} (Prompt Hub version)")

    runs = list(walk(run))
    guard = next((r for r in runs if "guardrails_result" in ((r.extra or {}).get("metadata") or {})), None)
    check(guard is not None, "guardrails decision recorded")
    if guard:
        gm = guard.extra["metadata"]
        print(f"       guardrails_result = {gm['guardrails_result']} ({gm.get('guardrails_explanation', '')[:90]})")
        if expect_blocked is not None:
            check((gm["guardrails_result"] == "BLOCKED") == expect_blocked, f"guardrails {'blocked' if expect_blocked else 'allowed'} it")

    tools = [r.name for r in runs if r.run_type == "tool"]
    llms = [r for r in runs if r.run_type == "llm"]
    print(f"       tool calls: {tools or 'none'}")
    print(f"       model calls: {len(llms)} ({', '.join(sorted({(r.extra or {}).get('metadata', {}).get('ls_model_name', '?') for r in llms}))})")
    if expect_blocked is False:
        check(any(t.startswith("search_docs") for t in tools), "searched the docs")
        check(any(t.startswith("query_docs_filesystem") for t in tools), "read docs pages")
    check(bool(run.total_tokens), f"tokens recorded: {run.total_tokens} (cost: {run.total_cost})")
    return failures


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--off-topic", action="store_true")
    parser.add_argument("--feedback", action="store_true")
    args = parser.parse_args()
    require_env()

    from langchain_core.tracers.langchain import wait_for_all_tracers
    from langsmith import Client

    from agent.graph import make_graph

    print(f"project: {os.getenv('LANGSMITH_PROJECT', 'default')}  |  docs MCP: {os.getenv('DOCS_MCP_URL', 'http://localhost:3333/mcp')}")
    graph = await make_graph()
    client = Client()
    failures = 0

    result, run_id = await ask(graph, ON_TOPIC)
    print(f"\nQ: {ON_TOPIC}\nA: {result['messages'][-1].content[:400]}...")
    wait_for_all_tracers()
    run = read_trace(client, run_id)
    failures += report(run, "on-topic question", expect_blocked=False)

    if args.feedback:
        fb = client.create_feedback(run_id=run_id, key="ux.thumb_vote", score=1, comment="check_tracing.py")
        print(f"  ok   feedback ux.thumb_vote=1 left on the run (id {fb.id})")

    if args.off_topic:
        result, run_id = await ask(graph, OFF_TOPIC)
        print(f"\nQ: {OFF_TOPIC}\nA: {result['messages'][-1].content[:300]}")
        wait_for_all_tracers()
        failures += report(read_trace(client, run_id), "off-topic question", expect_blocked=True)

    print("\nall tracing checks passed" if not failures else f"\n{failures} tracing check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
