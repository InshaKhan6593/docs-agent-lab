"""Incremental sync: bring Chroma in line with the docs source, touching only pages that changed.

Mintlify doesn't describe this part, so this is our design:

1. Pull + build (optional): fetch the newest docs commit and run LangChain's docs build.
2. Diff: render + chunk every page (remark output is cached by content, so this is fast), fingerprint
   it, and compare with what is *actually stored in Chroma* -> added / changed / removed pages.
3. Apply without a gap for readers:
     a. upsert chunks of added + changed pages (only these get embedded)
     b. delete leftover chunks when a changed page now has fewer chunks
     c. write the new __path_tree__ (new revision -> running servers reload)
     d. delete chunks of removed pages (the new tree no longer lists them)
4. Tell the local server to reload now (it also polls the tree revision on its own).

    uv run python ingest/sync.py                 # pull, build, sync
    uv run python ingest/sync.py --no-pull       # sync the current build (e.g. after editing ingest/extra/)
    uv run python ingest/sync.py --dry-run       # show the diff only
    uv run python ingest/sync.py --every 600     # keep syncing every 10 minutes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ingest as I  # noqa: E402

BUILD_CMD = ["uvx", "uv@0.9.30", "run", "pipeline", "build"]  # the docs repo pins uv>=0.9.26


@dataclass
class SyncResult:
    commit_before: str = ""
    commit_after: str = ""
    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    chunks_written: int = 0
    chunks_deleted: int = 0
    tree_written: bool = False
    revision: str = ""
    server_reloaded: bool | None = None
    seconds: float = 0.0


def git(*args: str) -> str:
    out = subprocess.run(["git", "-C", str(I.DOCS_REPO), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def pull_and_build(force_build: bool) -> tuple[str, str]:
    """Move the docs checkout to the newest upstream commit, then rebuild if anything changed.

    Never overwrites your work: if you have edited files in vendor/langchain-docs, it stops and tells you
    instead of updating. `git reset --keep` also refuses rather than clobber a locally modified file.
    """
    edits = git("status", "--porcelain", "--untracked-files=no")
    if edits:
        raise SystemExit(
            "vendor/langchain-docs has local edits, so sync won't pull over them:\n"
            f"{edits}\n"
            "Commit or stash them there, or run with --no-pull to sync your edited build as-is."
        )
    before = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch == "HEAD":  # detached: fall back to the remote default
        branch = "main"
    print(f"fetching {branch} ...", flush=True)
    git("fetch", "--depth", "1", "origin", branch)
    git("reset", "--keep", "FETCH_HEAD")
    after = git("rev-parse", "HEAD")
    if after == before and (I.BUILD_DIR / "docs.json").exists() and not force_build:
        print(f"docs already at {after[:10]}, build is current", flush=True)
        return before, after
    print(f"docs {before[:10]} -> {after[:10]}; running the docs build ...", flush=True)
    env = {**os.environ, "PYTHONPATH": str(I.DOCS_REPO)}
    subprocess.run(BUILD_CMD, cwd=I.DOCS_REPO, env=env, check=True, capture_output=True)
    return before, after


def stored_pages(col) -> dict[str, dict]:
    """Fingerprint of every page as it is stored in Chroma right now (not as the tree claims)."""
    pages: dict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    limit = 2000
    offset = 0
    while True:
        res = col.get(where={"kind": "chunk"}, include=["documents", "metadatas"], limit=limit, offset=offset)
        for doc, meta in zip(res["documents"], res["metadatas"]):
            pages[meta["page"]].append((meta["chunk_index"], doc, meta))
        if len(res["ids"]) < limit:
            break
        offset += limit
    state = {}
    for path, chunks in pages.items():
        chunks.sort(key=lambda c: c[0])
        state[path] = {
            "hash": I.page_fingerprint([c[1] for c in chunks], [c[2] for c in chunks]),
            "chunks": len(chunks),
        }
    return state


def stored_tree_revision(col) -> str | None:
    res = col.get(ids=[I.PATH_TREE_ID], include=["metadatas"])
    return (res["metadatas"][0] or {}).get("revision") if res["ids"] else None


def notify_server(url: str) -> bool:
    try:
        req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=60).status == 200
    except Exception:
        return False


def sync(col, *, pull: bool, force_build: bool, dry_run: bool, notify_url: str | None) -> SyncResult:
    started = time.time()
    result = SyncResult()
    if pull:
        result.commit_before, result.commit_after = pull_and_build(force_build)
    else:
        result.commit_before = result.commit_after = I.docs_revision()

    records, lazy = I.build_records(verbose=False)
    current = stored_pages(col)

    new_paths, old_paths = set(records), set(current)
    result.added = sorted(new_paths - old_paths)
    result.removed = sorted(old_paths - new_paths)
    result.changed = sorted(p for p in new_paths & old_paths if records[p].file_entry["hash"] != current[p]["hash"])
    result.unchanged = len(new_paths & old_paths) - len(result.changed)
    tree_outdated = stored_tree_revision(col) is None  # trees from before sync existed carry no revision

    if dry_run:
        result.seconds = time.time() - started
        return result

    # a. new + changed chunks (the only ones we embed)
    to_write = [records[p] for p in result.added + result.changed]
    if to_write:
        result.chunks_written = I.upsert_records(col, to_write, DefaultEmbeddingFunction(), verbose=False)

    # b. leftovers when a changed page shrank
    stale = [f"{p}#{i}" for p in result.changed for i in range(records[p].file_entry["chunks"], current[p]["chunks"])]
    if stale:
        col.delete(ids=stale)
        result.chunks_deleted += len(stale)

    # c. new tree -> new revision
    if result.added or result.changed or result.removed or tree_outdated:
        result.revision = I.write_tree(col, {p: r.file_entry for p, r in records.items()}, lazy)
        result.tree_written = True

    # d. removed pages, now that the tree no longer lists them
    for start in range(0, len(result.removed), 100):
        batch = result.removed[start : start + 100]
        result.chunks_deleted += sum(current[p]["chunks"] for p in batch)
        col.delete(where={"page": {"$in": batch}})

    if result.tree_written and notify_url:
        result.server_reloaded = notify_server(notify_url)
    result.seconds = time.time() - started
    return result


def report(result: SyncResult) -> None:
    def show(label: str, paths: list[str]) -> None:
        print(f"  {label:<9} {len(paths)}")
        for p in paths[:10]:
            print(f"            {p}")
        if len(paths) > 10:
            print(f"            ... {len(paths) - 10} more")

    if result.commit_before != result.commit_after:
        print(f"docs commit {result.commit_before[:10]} -> {result.commit_after[:10]}")
    show("added", result.added)
    show("changed", result.changed)
    show("removed", result.removed)
    print(f"  unchanged {result.unchanged}")
    print(f"  chunks    {result.chunks_written} written, {result.chunks_deleted} deleted")
    if result.tree_written:
        reloaded = {True: "server reloaded", False: "server not reachable (it will pick this up by polling)", None: ""}
        print(f"  tree      new revision {result.revision}  {reloaded[result.server_reloaded]}")
    else:
        print("  tree      unchanged")
    print(f"  took      {result.seconds:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-pull", action="store_true", help="don't fetch/build; sync the current build")
    parser.add_argument("--force-build", action="store_true", help="rebuild the docs even if the commit didn't change")
    parser.add_argument("--dry-run", action="store_true", help="show what would change, write nothing")
    parser.add_argument("--notify", default="http://localhost:3333/admin/reload", help="server reload URL ('' to skip)")
    parser.add_argument("--every", type=int, default=0, help="repeat every N seconds")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args()

    col = I.connect(args.host, args.port)
    while True:
        result = sync(col, pull=not args.no_pull, force_build=args.force_build, dry_run=args.dry_run, notify_url=args.notify or None)
        if args.json:
            print(json.dumps(asdict(result)))
        else:
            print(f"sync {'(dry run) ' if args.dry_run else ''}@ {time.strftime('%H:%M:%S')}")
            report(result)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
