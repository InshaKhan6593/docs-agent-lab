# How Mintlify chunks and serves docs (what we observed)

Mintlify hasn't published its chunking code. Everything below was measured against the live
`https://docs.langchain.com/mcp` server (read-only calls) and copied into `ingest/ingest.py`.
Where something is a guess, it says so.

## What a page looks like when you `cat` it

1. **Front-matter becomes a header**: `# {title}`, a blank line, then `> {description}`.
2. **Snippets are inlined.** `import X from '/snippets/...mdx'` plus `<X />` is replaced by the snippet's content.
3. **JSX component imports are pasted in as source** (`export const PatternEmbed = (...) => {...}`).
   This is faithful but noisy for an agent. `INLINE_JSX` in `ingest.py` turns it off.
4. **The MDX is re-serialized by remark** (parse to an mdast tree, then stringify with defaults):
   children of JSX components get 2-space indentation, bullets become `*`, rules become `***`, and `&` in URLs becomes `\&`.
   We run the same pipeline (`ingest/normalize-mdx.mjs`).
5. **Code fences get the site's code theme**: ` ```python theme={"theme":{"light":"catppuccin-latte","dark":"catppuccin-mocha"}}`.
6. **Images are rewritten to Mintlify's CDN** (`mintcdn.com/...` with signed parameters). *Not copied*: those URLs are signed by Mintlify.
7. **Output over 30,000 characters is cut** and ends with `... (output truncated)`.

## How a page is chunked

| chunk | content | search title | link |
|---|---|---|---|
| 0 | the header: `# Title` and `> description` (search shows it as `Title\ndescription`) | page title | page URL |
| 1 | the intro: everything before the first heading | page title | page URL |
| 2+ | one chunk per heading section: `##`, `###` **and** `####`, regardless of nesting, up to the next heading | heading text | page URL + `#anchor` |

* Chunks are **exact slices** of the served page, trimmed. `cat` = chunks sorted by `chunk_index`, joined with `\n\n`.
* There is **no size cap**: a section with long code stays one chunk (we've seen live chunks of 4,800+ characters).
* `#` lines inside code fences are not headings.
* *Unverified:* whether headings nested inside JSX (indented after remark) start a chunk. We don't split on them.

## Search

* Live hits look like `Title: … / Link: …#anchor / Page: oss/python/… / Content: <section text>`, 10 per query.
* Mintlify uses **hybrid search on Chroma Cloud: dense vectors + SPLADE sparse vectors, fused with RRF**
  (from Chroma's Mintlify case study). Nothing public mentions a reranker.
* Locally, Chroma refuses sparse indexes ("Sparse vector indexing is not enabled in local"), so we fuse
  dense vectors (Chroma) with an in-memory **BM25** index using the same RRF.

## Measured fidelity (images excluded)

Pages built from GitHub `HEAD` match the live pages at 98–100% for most pages. The gaps come from:
content drift (the live site was deployed from a different commit), Mintlify's CDN image rewrites, about 25 pages
that strict remark can't parse (we keep their source text), and the `pattern-embed.jsx` component, which
LangChain's build pulls from an npm package that isn't in the repo.
