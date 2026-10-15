"""Ingest the built LangChain docs into Chroma, chunked the way Mintlify chunks them.

Chunking follows what Mintlify's live server shows (see docs/how-mintlify-chunks.md):

* chunk 0  = page header: "# {title}\\n\\n> {description}" (shown in search as "title\\ndescription")
* chunk 1  = the page intro (text before the first heading), titled with the page title
* chunk 2+ = one chunk per heading section, at EVERY heading level 2-4 (##, ###, ####),
             running to the next heading. No size cap. `#` lines inside code fences are not headings.
* chunks are trimmed; `cat` rebuilds the page as "\\n\\n".join(chunks sorted by chunk_index)

Also stored: one `__path_tree__` record (kind="path_tree") holding the whole file tree as
gzipped, base64-encoded JSON with access metadata per path. The server loads it once and answers
`ls` / `cd` / `find` from memory.

Run the docs build first (see README), start Chroma, then:

    uv run python ingest/ingest.py --reset
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
import yaml
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "vendor" / "langchain-docs" / "build"
ACCESS_FILE = Path(__file__).resolve().parent / "access.json"
EXTRA_DIR = Path(__file__).resolve().parent / "extra"  # extra pages (e.g. the staff-only RBAC demo)
SITE_URL = "https://docs.langchain.com"
COLLECTION = "docs"
PATH_TREE_ID = "__path_tree__"
EMBED_DIM = 384  # all-MiniLM-L6-v2, Chroma's default embedding model
NORMALIZER = Path(__file__).resolve().parent / "normalize-mdx.mjs"
REMARK_CACHE = Path(__file__).resolve().parent / ".cache" / "remark"
DOCS_REPO = ROOT / "vendor" / "langchain-docs"

SNIPPET_IMPORT = re.compile(r"""^import\s+(\w+)\s+from\s+['"](/snippets/[^'"]+\.mdx?)['"];?\s*$""", re.M)
JSX_IMPORT = re.compile(r"""^import\s+\{?[\w\s,]+\}?\s+from\s+['"](/snippets/[^'"]+\.(?:jsx|js|tsx))['"];?\s*$""", re.M)
# Mintlify inlines imported JSX component source into the served page. Faithful, but it is noise
# for an agent (it can push real content past the 30k cat limit); set False for client projects.
INLINE_JSX = True
FENCE = re.compile(r"^\s*(```|~~~)")
FENCE_OPEN_WITH_LANG = re.compile(r"^(\s*(?:```|~~~))([^\s`]+)(.*)$")
HEADING = re.compile(r"^(#{2,4})\s+(.+?)\s*#*\s*$")


# --------------------------------------------------------------------------- docs.json


def nav_pages(docs_json: dict) -> tuple[list[str], list[dict]]:
    """Return navigation page slugs and OpenAPI spec entries from docs.json."""
    pages: list[str] = []
    specs: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "openapi":
                    for spec in value if isinstance(value, list) else [value]:
                        specs.append(spec if isinstance(spec, dict) else {"source": spec})
                walk(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, str):
                    pages.append(item)
                else:
                    walk(item)

    walk(docs_json.get("navigation"))
    return list(dict.fromkeys(pages)), specs


def code_theme_suffix(docs_json: dict) -> str:
    """Mintlify annotates fenced code with the site's code theme at render time."""
    theme = docs_json.get("styling", {}).get("codeblocks", {}).get("theme")
    return f" theme={json.dumps({'theme': theme}, separators=(',', ':'))}" if theme else ""


# --------------------------------------------------------------------------- rendering


def split_frontmatter(text: str) -> tuple[dict, str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            meta = yaml.safe_load(text[3:end]) or {}
            return (meta if isinstance(meta, dict) else {}), text[end + 4 :].lstrip("\n")
    return {}, text


def inline_jsx_imports(body: str) -> str:
    """Mintlify pastes imported JSX component source into the page (`export const X = ...`).

    Imports whose file is not in the repo (e.g. pattern-embed.jsx, which LangChain's build pulls
    from an npm package) are dropped instead.
    """

    def replace(m: re.Match) -> str:
        file = BUILD_DIR / m[1].lstrip("/")
        if INLINE_JSX and file.exists():
            return file.read_text(encoding="utf-8").strip("\n")
        return ""

    return JSX_IMPORT.sub(replace, body)


def inline_snippets(body: str, depth: int = 0) -> str:
    """Replace `<Snippet />` usages with the imported snippet's content, like Mintlify's render."""
    body = inline_jsx_imports(body)
    imports = dict(SNIPPET_IMPORT.findall(body))
    body = SNIPPET_IMPORT.sub("", body)
    for name, path in imports.items():
        file = BUILD_DIR / path.lstrip("/")
        if not file.exists() or depth > 3:
            continue
        _, snippet = split_frontmatter(file.read_text(encoding="utf-8"))
        snippet = inline_snippets(snippet, depth + 1).strip("\n")
        body = re.sub(rf"<{name}(\s[^>]*)?/>", lambda _m: snippet, body)
    return body


def annotate_fences(body: str, suffix: str) -> str:
    if not suffix:
        return body
    out, in_fence = [], False
    for line in body.split("\n"):
        if FENCE.match(line):
            m = FENCE_OPEN_WITH_LANG.match(line)
            if not in_fence and m and "theme=" not in line:
                line = f"{m[1]}{m[2]}{m[3]}{suffix}"
            in_fence = not in_fence
        out.append(line)
    return "\n".join(out)


def heading_text(raw: str) -> str:
    """Visible heading text: drop MDX tags like <Icon icon="wand" /> and markdown emphasis."""
    text = re.sub(r"<[^>]+>", "", raw)
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return " ".join(text.split())


def anchor_for(title: str) -> str:
    """GitHub-style slug, as used by Mintlify heading anchors."""
    slug = re.sub(r"[^\w\- ]", "", title.lower()).strip()
    return re.sub(r"[ ]+", "-", slug)


def split_sections(body: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (intro, [(heading title, section text), ...]) split at every ##/###/#### outside fences."""
    intro: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    in_fence = False
    for line in body.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING.match(line)
        if m:
            sections.append((heading_text(m[2]), [line]))
        elif sections:
            sections[-1][1].append(line)
        else:
            intro.append(line)
    return "\n".join(intro).strip("\n"), [(t, "\n".join(lines).strip("\n")) for t, lines in sections]


def read_page(slug: str, base: Path) -> dict:
    meta, body = split_frontmatter((base / f"{slug}.mdx").read_text(encoding="utf-8"))
    title = str(meta.get("title") or slug.rsplit("/", 1)[-1].replace("-", " ").title())
    description = str(meta.get("description") or "").strip()
    return {"title": title, "description": description, "body": inline_snippets(body)}


def normalize_mdx(bodies: dict[str, str], batch: int = 150) -> tuple[dict[str, str], dict[str, str]]:
    """Round-trip MDX through remark (parse -> mdast -> stringify), as Mintlify does before serving.

    Results are cached by content hash, so a sync only re-normalizes pages whose source changed.
    """
    REMARK_CACHE.mkdir(parents=True, exist_ok=True)
    version = hashlib.sha256(NORMALIZER.read_bytes()).hexdigest()[:12]
    key = {k: hashlib.sha256(f"{version}\0{v}".encode("utf-8")).hexdigest() for k, v in bodies.items()}
    out: dict[str, str] = {}
    errors: dict[str, str] = {}
    todo: list[tuple[str, str]] = []
    for k, v in bodies.items():
        hit = REMARK_CACHE / f"{key[k]}.json"
        if hit.exists():
            row = json.loads(hit.read_text(encoding="utf-8"))
            out[k] = row["text"]
            if row.get("error"):
                errors[k] = row["error"]
        else:
            todo.append((k, v))
    for start in range(0, len(todo), batch):
        payload = json.dumps([{"id": k, "text": v} for k, v in todo[start : start + batch]])
        proc = subprocess.run(
            ["node", str(NORMALIZER)], input=payload, capture_output=True, text=True, encoding="utf-8", check=True
        )
        for row in json.loads(proc.stdout):
            out[row["id"]] = row["text"]
            if row.get("error"):
                errors[row["id"]] = row["error"]
            (REMARK_CACHE / f"{key[row['id']]}.json").write_text(json.dumps(row), encoding="utf-8")
    return out, errors


def finish_page(page: dict, body: str, theme_suffix: str) -> dict:
    intro, sections = split_sections(annotate_fences(body, theme_suffix))
    header = f"# {page['title']}" + (f"\n\n> {page['description']}" if page["description"] else "")
    return {**page, "header": header, "intro": intro, "sections": sections}


# --------------------------------------------------------------------------- access


def load_access_rules() -> list[dict]:
    if ACCESS_FILE.exists():
        return json.loads(ACCESS_FILE.read_text(encoding="utf-8")).get("rules", [])
    return []


def groups_for(path: str, rules: list[dict]) -> list[str]:
    """Longest matching prefix wins; no match means public."""
    best = None
    for rule in rules:
        if path.startswith(rule["prefix"]) and (best is None or len(rule["prefix"]) > len(best["prefix"])):
            best = rule
    return best["groups"] if best else ["public"]


def spec_entry(spec: dict) -> tuple[str, str]:
    """Map a docs.json OpenAPI entry to (virtual path, lazy source), mirroring the live layout."""
    source = spec["source"]
    if source.startswith("http"):
        return "/openapi/" + source.replace("://", ":/"), source
    return "/openapi/" + source.lstrip("/"), (BUILD_DIR / source).resolve().as_uri()


# --------------------------------------------------------------------------- records


@dataclass
class PageRecords:
    """Everything stored for one page: its chunks, plus its entry in the path tree."""

    path: str
    file_entry: dict
    ids: list[str] = field(default_factory=list)
    docs: list[str] = field(default_factory=list)
    metas: list[dict] = field(default_factory=list)
    embed_inputs: list[str] = field(default_factory=list)


def page_fingerprint(docs: list[str], metas: list[dict]) -> str:
    """Hash of exactly what gets stored for a page. Same content -> same hash, on either side of a sync."""
    payload = json.dumps([[d, m] for d, m in zip(docs, metas)], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_sources(limit: int = 0) -> tuple[list[tuple[str, Path]], list[dict], str]:
    """(slug, base dir) for every navigation page plus extra pages, the OpenAPI specs, and the code theme."""
    docs_json = json.loads((BUILD_DIR / "docs.json").read_text(encoding="utf-8"))
    slugs, specs = nav_pages(docs_json)
    pages = [(s, BUILD_DIR) for s in slugs if (BUILD_DIR / f"{s}.mdx").exists()]
    if limit:
        pages = pages[:limit]
    pages += [(f.relative_to(EXTRA_DIR).with_suffix("").as_posix(), EXTRA_DIR) for f in sorted(EXTRA_DIR.rglob("*.mdx"))]
    return pages, specs, code_theme_suffix(docs_json)


def build_records(limit: int = 0, verbose: bool = True) -> tuple[dict[str, PageRecords], dict[str, dict]]:
    """Render, normalize and chunk every page. Returns (records by path, lazy file entries)."""
    pages, specs, theme_suffix = collect_sources(limit)
    rules = load_access_rules()
    raw = {f"/{slug.lstrip('/')}.mdx": read_page(slug, base) for slug, base in pages}
    if verbose:
        print(f"normalizing {len(raw)} pages through remark (cached by content)...", flush=True)
    normalized, errors = normalize_mdx({path: page["body"] for path, page in raw.items()})
    if verbose and errors:
        print(f"  ! {len(errors)} pages kept their source text (strict remark could not parse them)")

    records: dict[str, PageRecords] = {}
    for path, page in raw.items():
        page = finish_page(page, normalized[path], theme_suffix)
        groups = groups_for(path, rules)
        page_url = SITE_URL + path.removesuffix(".mdx")

        # (section title, anchor, text, kind)
        parts = [(page["title"], "", page["header"], "header")]
        if page["intro"]:
            parts.append((page["title"], "", page["intro"], "intro"))
        parts += [(t, anchor_for(t), text, "section") for t, text in page["sections"]]

        rec = PageRecords(path=path, file_entry={})
        for i, (title, anchor, text, part) in enumerate(parts):
            rec.ids.append(f"{path}#{i}")
            rec.docs.append(text)
            rec.metas.append(
                {
                    "kind": "chunk",
                    "part": part,
                    "page": path,
                    "chunk_index": i,
                    "page_title": page["title"],
                    "title": title[:300],
                    "url": page_url + (f"#{anchor}" if anchor else ""),
                    **{f"g_{g}": True for g in groups},
                }
            )
            # Header chunks embed as "title + description"; sections carry page + heading context.
            if part == "header":
                rec.embed_inputs.append(f"{page['title']}\n{page['description']}")
            else:
                rec.embed_inputs.append(f"{page['title']} - {title}\n{text}")

        full_text = "\n\n".join(rec.docs)
        rec.file_entry = {
            "size": len(full_text.encode("utf-8")),
            "title": page["title"],
            "groups": groups,
            "isPublic": "public" in groups,
            "chunks": len(rec.ids),
            "hash": page_fingerprint(rec.docs, rec.metas),
        }
        records[path] = rec

    lazy = {}
    for spec in specs:
        vpath, source = spec_entry(spec)
        lazy[vpath] = {"source": source, "groups": ["public"], "isPublic": True}
    return records, lazy


def upsert_records(col, records: list[PageRecords], embed, batch: int = 256, verbose: bool = True) -> int:
    """Embed and upsert the chunks of the given pages. Returns the number of chunks written."""
    ids = [i for r in records for i in r.ids]
    docs = [d for r in records for d in r.docs]
    metas = [m for r in records for m in r.metas]
    inputs = [e for r in records for e in r.embed_inputs]
    started = time.time()
    for start in range(0, len(ids), batch):
        end = start + batch
        col.upsert(ids=ids[start:end], documents=docs[start:end], metadatas=metas[start:end], embeddings=embed(inputs[start:end]))
        if verbose:
            print(f"  {min(end, len(ids))}/{len(ids)} chunks ({time.time() - started:.0f}s)", flush=True)
    return len(ids)


def docs_revision() -> str:
    """Git commit of the docs source (shown in the tree; change detection uses page hashes)."""
    try:
        out = subprocess.run(["git", "-C", str(DOCS_REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_tree(col, files: dict[str, dict], lazy: dict[str, dict]) -> str:
    """Store the path tree. Its `revision` metadata is what running servers poll to notice a change."""
    revision = str(int(time.time() * 1000))
    tree = {
        "version": 1,
        "site": SITE_URL,
        "generated_at": int(time.time()),
        "revision": revision,
        "source_commit": docs_revision(),
        "files": files,
        "lazy": lazy,
    }
    blob = base64.b64encode(gzip.compress(json.dumps(tree).encode("utf-8"))).decode("ascii")
    # A fixed placeholder embedding: the tree is fetched by id, never by similarity.
    col.upsert(
        ids=[PATH_TREE_ID],
        documents=[blob],
        metadatas=[{"kind": "path_tree", "files": len(files), "revision": revision}],
        embeddings=[[1.0] + [0.0] * (EMBED_DIM - 1)],
    )
    return revision


def connect(host: str, port: int, reset: bool = False):
    client = chromadb.HttpClient(host=host, port=port)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Full ingest. For incremental updates use sync.py.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reset", action="store_true", help="drop and rebuild the collection")
    parser.add_argument("--limit", type=int, default=0, help="only ingest N pages (quick test)")
    args = parser.parse_args()

    col = connect(args.host, args.port, reset=args.reset)
    records, lazy = build_records(args.limit)
    total = sum(len(r.ids) for r in records.values())
    print(f"{len(records)} pages -> {total} chunks, {len(lazy)} lazy OpenAPI specs; embedding...", flush=True)
    started = time.time()
    upsert_records(col, list(records.values()), DefaultEmbeddingFunction())
    write_tree(col, {p: r.file_entry for p, r in records.items()}, lazy)
    print(f"path tree stored. Done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
