"""Run the same tool calls against Mintlify's live docs MCP and our local ChromaFs MCP, side by side.

    uv run python scripts/compare_with_mintlify.py
"""

import json
import sys
import time
import urllib.request

SERVERS = {"mintlify (live)": "https://docs.langchain.com/mcp", "ours (local)": "http://localhost:3333/mcp"}
FS = "query_docs_filesystem_docs_by_lang_chain"
SEARCH = "search_docs_by_lang_chain"

CASES = [
    (FS, {"command": "tree / -L 1"}),
    (FS, {"command": "ls /oss/python/langgraph | head -8"}),
    (FS, {"command": "head -12 /oss/python/langgraph/streaming.mdx"}),
    (FS, {"command": 'rg -l "interrupt_before" /oss/python/langgraph'}),
    (FS, {"command": "rg -c 'checkpointer' /oss/python/langgraph/persistence.mdx"}),
    (FS, {"command": "ls /openapi/langsmith"}),
    (FS, {"command": "echo hi > /x.mdx"}),
    (SEARCH, {"query": "stream subgraph outputs"}),
]


def call(url: str, tool: str, args: dict) -> tuple[str, float]:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}}).encode()
    req = urllib.request.Request(url, body, {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    started = time.perf_counter()
    raw = urllib.request.urlopen(req, timeout=90).read().decode("utf-8")
    ms = (time.perf_counter() - started) * 1000
    data = json.loads(next(line[6:] for line in raw.splitlines() if line.startswith("data: ")))
    content = data["result"]["content"]
    if tool == SEARCH:
        lines = [c["text"].split("\n")[0].removeprefix("Title: ") + "  <" + c["text"].split("\n")[1].removeprefix("Link: ") + ">" for c in content]
        return "\n".join(lines[:6]), ms
    return "".join(c.get("text", "") for c in content), ms


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for tool, args in CASES:
        print("=" * 100)
        print(f"{tool.split('_docs')[0]}  {json.dumps(args)}")
        for name, url in SERVERS.items():
            try:
                text, ms = call(url, tool, args)
            except Exception as err:  # keep comparing even if one side fails
                text, ms = f"ERROR: {err}", 0.0
            print(f"--- {name}  ({ms:.0f} ms)")
            print(text.strip()[:900])


if __name__ == "__main__":
    main()
