export const config = {
  chromaHost: process.env.CHROMA_HOST ?? "localhost",
  chromaPort: Number(process.env.CHROMA_PORT ?? 8000),
  collection: process.env.DOCS_COLLECTION ?? "docs",
  port: Number(process.env.PORT ?? 3333),
  siteName: process.env.SITE_NAME ?? "Docs by LangChain",
  // Same suffix as the live server, so the chat-langchain prompt works against either one.
  toolSuffix: process.env.TOOL_SUFFIX ?? "docs_by_lang_chain",
  // Groups for requests that carry no identity. Production would derive these from a verified session.
  defaultGroups: (process.env.DEFAULT_GROUPS ?? "public").split(",").map((g) => g.trim()),
  // Mintlify cuts filesystem output at 30,000 characters ("... (output truncated)").
  maxOutputChars: 30_000,
  pageCacheBytes: 256 * 1024 * 1024,
  searchResults: 10,
  // How often the server checks the stored tree's revision for a sync (0 = only on /admin/reload).
  treePollSeconds: Number(process.env.TREE_POLL_SECONDS ?? 15),
};
