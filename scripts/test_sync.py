"""End-to-end test of incremental sync against the running MCP server.

Adds, edits and deletes a throwaway page in ingest/extra/, runs sync.py after each step, and checks
through the MCP tools that the server serves the new state (filesystem, grep and search).

    uv run python scripts/test_sync.py          # needs Chroma (:8000) and the MCP server (:3333) running
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANARY = ROOT / "ingest" / "extra" / "sync-test" / "canary.mdx"
MCP = "http://localhost:3333/mcp"
FS = "query_docs_filesystem_docs_by_lang_chain"
SEARCH = "search_docs_by_lang_chain"


def mcp(tool: str, args: dict) -> str:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": args}}).encode()
    req = urllib.request.Request(MCP, body, {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
    raw = urllib.request.urlopen(req, timeout=120).read().decode("utf-8")
    data = json.loads(next(line[6:] for line in raw.splitlines() if line.startswith("data: ")))
    return "\n".join(c.get("text", "") for c in data["result"]["content"])


def run_sync() -> dict:
    out = subprocess.run(
        ["uv", "run", "python", "ingest/sync.py", "--no-pull", "--json"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def write_canary(token: str, heading: str) -> None:
    CANARY.parent.mkdir(parents=True, exist_ok=True)
    CANARY.write_text(
        f"---\ntitle: Sync canary\ndescription: Throwaway page for the sync test.\n---\n\n"
        f"Intro line.\n\n## {heading}\n\nThe canary token is {token}.\n",
        encoding="utf-8",
    )


failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"\n       {detail[:300]}" if not ok and detail else ""))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    path = "/sync-test/canary.mdx"
    token_a, token_b = f"CANARY-{secrets.token_hex(4)}", f"CANARY-{secrets.token_hex(4)}"
    try:
        print("1. add a page")
        write_canary(token_a, "First version")
        r = run_sync()
        check("sync reports it as added", r["added"] == [path], json.dumps(r))
        check("only that page was embedded", r["chunks_written"] == 3, f"chunks_written={r['chunks_written']}")
        check("server reloaded", r["server_reloaded"] is True)
        out = mcp(FS, {"command": f"rg -l {token_a} /"})
        check("rg finds it", path in out, out)
        out = mcp(FS, {"command": "ls /sync-test"})
        check("it is in the tree", "canary.mdx" in out, out)
        out = mcp(SEARCH, {"query": f"sync canary {token_a}"})
        check("search finds it", f"Link: https://docs.langchain.com/sync-test/canary" in out, out[:300])

        print("2. edit it")
        write_canary(token_b, "Second version")
        r = run_sync()
        check("sync reports it as changed", r["changed"] == [path] and not r["added"], json.dumps(r))
        out = mcp(FS, {"command": f"cat {path}"})
        check("cat shows the new text (cache invalidated)", token_b in out and token_a not in out, out)
        out = mcp(FS, {"command": f"rg -l {token_a} /"})
        check("old text is gone from grep", path not in out, out)
        out = mcp(SEARCH, {"query": "Second version sync canary"})
        check("search sees the new heading", "Title: Second version" in out, out[:300])

        print("3. no-op sync")
        r = run_sync()
        check("nothing to do", not (r["added"] or r["changed"] or r["removed"]) and not r["tree_written"], json.dumps(r))

        print("4. delete it")
        CANARY.unlink()
        r = run_sync()
        check("sync reports it as removed", r["removed"] == [path], json.dumps(r))
        out = mcp(FS, {"command": "ls /sync-test"})
        check("gone from the tree", "No such file" in out or "exit: 2" in out or "exit: 1" in out, out)
        out = mcp(FS, {"command": f"rg -l {token_b} /"})
        check("gone from grep", path not in out, out)
    finally:
        if CANARY.exists():
            CANARY.unlink()
        if CANARY.parent.exists() and not any(CANARY.parent.iterdir()):
            CANARY.parent.rmdir()
    print("\nall sync checks passed" if not failures else f"\n{failures} sync check(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
