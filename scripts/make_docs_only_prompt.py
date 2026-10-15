"""Derive the docs-only system prompt from chat-langchain's exact prompt.

chat-langchain's instructions assume Pylon (LangChain's private support knowledge base). Without Pylon keys,
those parts would make the model call tools that don't exist. This removes only the support-KB parts and
leaves every other rule word for word.

    uv run python scripts/make_docs_only_prompt.py
"""

import re
from pathlib import Path

AGENT = Path(__file__).resolve().parent.parent / "agent"
SOURCE = AGENT / "instructions.chat-langchain.md"
TARGET = AGENT / "instructions.md"

REPLACEMENTS = [
    ("by researching official documentation and support articles.", "by researching the official documentation."),
    (" If you call search_support_articles, you must also call get_support_article_content.", ""),
    (
        "**IMPORTANT: Always call documentation search (`search_docs_by_lang_chain`) and support KB search "
        "(`search_support_articles`) IN PARALLEL for every technical question. Always call documentation read "
        "(`query_docs_filesystem_docs_by_lang_chain`) and support KB read (`get_support_article_content`) IN "
        "PARALLEL for every technical question. This dramatically improves response speed!**",
        "**IMPORTANT: When a question touches several concepts, call documentation search "
        "(`search_docs_by_lang_chain`) for each of them IN PARALLEL, then read the pages "
        "(`query_docs_filesystem_docs_by_lang_chain`) IN PARALLEL. This dramatically improves response speed!**",
    ),
    ("**Never attempt to read support articles that were not returned by the search_support_articles tool**\n\n", ""),
    ("### 6. `check_links`", "### 4. `check_links`"),
    ("- When citing support article URLs\n", ""),
    ("### Step 1: Research Documentation and Support KB", "### Step 1: Research Documentation"),
    ("**CRITICAL: Always call BOTH documentation and support KB tools IN PARALLEL for maximum speed!**",
     "**CRITICAL: Make independent documentation calls IN PARALLEL for maximum speed!**"),
    ("`search_docs_by_lang_chain` or `search_support_articles` with a query", "`search_docs_by_lang_chain` with a query"),
    ("Never rely on results from search_docs_by_lang_chain or search_support_articles for answers. These are only for locations of relevant docs/articles",
     "Never rely on results from search_docs_by_lang_chain for answers. They are only for locating relevant docs"),
    ("**Round 1: search documentation AND support articles IN PARALLEL**", "**Round 1: search documentation IN PARALLEL**"),
    ('   - **For KB**: Call `search_support_articles` once with relevant collections (e.g., "LangSmith Deployment,LangSmith Observability")\n', ""),
    ("   - Review the documentation search and support article titles", "   - Review the documentation search results"),
    ("**Round 2: read official docs pages and support articles IN PARALLEL**", "**Round 2: read official docs pages IN PARALLEL**"),
    ("   - From support article results, select 1-3 relevant article IDs and call `get_support_article_content` for them in parallel\n", ""),
    ("using the docs page content and support article content you retrieved", "using the docs page content you retrieved"),
    ("   - Combine information from docs and support articles", "   - Combine information from the docs pages you read"),
    ("based on retrieved documentation and support articles", "based on retrieved documentation"),
    ("- **ALWAYS call docs and KB tools IN PARALLEL** - Call `search_docs_by_lang_chain` and `search_support_articles` at the same time for maximum speed",
     "- **ALWAYS make independent docs calls IN PARALLEL** - search distinct concepts at the same time for maximum speed"),
    ("- **Use support article tools for official docs links** - `get_support_article_content` only accepts Pylon support article IDs\n", ""),
]

# Whole sections that only describe the support-KB tools.
SECTION_CUTS = [
    (r"### 4\. `search_support_articles`.*?(?=### 6\. `check_links`)", ""),
]


def main() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    for pattern, repl in SECTION_CUTS:
        text, n = re.subn(pattern, repl, text, flags=re.S)
        assert n == 1, f"section not found: {pattern}"
    for old, new in REPLACEMENTS:
        assert old in text, f"not found: {old[:80]}"
        text = text.replace(old, new)
    leftovers = [w for w in ("search_support_articles", "get_support_article_content", "Pylon", "support KB") if w in text]
    assert not leftovers, f"support-KB references left: {leftovers}"
    TARGET.write_text(text, encoding="utf-8")
    print(f"wrote {TARGET.name}: {len(text.splitlines())} lines (from {len(SOURCE.read_text(encoding='utf-8').splitlines())})")


if __name__ == "__main__":
    main()
