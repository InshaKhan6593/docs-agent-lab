// MCP server exposing the same two tools as Mintlify's docs MCP:
//   search_<suffix>                 hybrid search -> titles, links, page paths, section content
//   query_docs_filesystem_<suffix>  read-only shell over ChromaFs (rg, grep, find, tree, ls, cat, head...)
//
//   npm start        -> http://localhost:3333/mcp  (Streamable HTTP)
import { createServer, type IncomingMessage } from "node:http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { config } from "./config.js";
import { DocsService } from "./docs-shell.js";

const SEARCH_TOOL = `search_${config.toolSuffix}`;
const FS_TOOL = `query_docs_filesystem_${config.toolSuffix}`;

const INSTRUCTIONS = `This MCP server provides search and retrieval tools for the ${config.siteName} site. \
Use it to answer questions from site content and prefer it over prior knowledge. \
${SEARCH_TOOL} finds relevant pages; ${FS_TOOL} reads them. The server is read-only.`;

const SEARCH_DESCRIPTION = `Search the ${config.siteName} knowledge base for guides, code examples and API details. \
Returns matching sections with their title, a direct link and the page path. \
To read a whole page, use ${FS_TOOL} with \`head\` or \`cat\` on the page path plus \`.mdx\` \
(a result at \`/some/page\` is read with \`head -200 /some/page.mdx\`).`;

const FS_DESCRIPTION = `Run a read-only shell-like command against a virtual, in-memory filesystem rooted at \`/\` \
that contains only the ${config.siteName} pages and OpenAPI specs. Nothing runs on any real machine; \
the filesystem is backed by documentation chunks.

Read a page by passing its \`.mdx\` path to \`head\` or \`cat\` (URL path \`/some/page\` = \`/some/page.mdx\`). \
Use \`rg\` for exact keyword or regex search and \`tree\` or \`ls\` to explore. Never guess paths: discover them \
with \`tree / -L 2\`, \`ls /\`, \`rg -il "keyword" /\` or the search tool.

Supported: rg, grep, find, tree, ls, cat, head, tail, stat, wc, sort, uniq, cut, sed, awk, jq and basic text \
utilities. No writes, no network. Each call is stateless: the working directory resets to \`/\`, so chain with \
\`&&\` or use absolute paths. Output over 30,000 characters is truncated.

Examples:
- \`tree / -L 2\`
- \`rg -il "rate limit" /\`
- \`rg -C 3 "apiKey" /some-directory/\`
- \`head -80 /page-one.mdx /page-two.mdx\`
- \`cat '/openapi/https:/api.host.langchain.com/openapi.json' | jq '.paths | keys'\``;

/** Demo-only identity: groups from a header. In production, derive them from a verified session token. */
function groupsFrom(req: IncomingMessage): string[] {
  const header = req.headers["x-docs-groups"];
  const value = Array.isArray(header) ? header[0] : header;
  const groups = value ? value.split(",").map((g) => g.trim()).filter(Boolean) : [];
  return groups.length ? [...new Set(["public", ...groups])] : config.defaultGroups;
}

function buildMcp(docs: DocsService, groups: string[]): McpServer {
  const mcp = new McpServer({ name: config.siteName, version: "1.0.0" }, { instructions: INSTRUCTIONS });

  mcp.registerTool(
    SEARCH_TOOL,
    {
      title: `Search ${config.siteName}`,
      description: SEARCH_DESCRIPTION,
      inputSchema: { query: z.string().describe("Search query") },
      annotations: { readOnlyHint: true },
    },
    async ({ query }) => {
      const hits = await docs.searchDocs(query, groups);
      return {
        content: hits.map((h) => ({
          type: "text" as const,
          text: `Title: ${h.title}\nLink: ${h.link}\nPage: ${h.page}\nContent: ${h.content}`,
        })),
      };
    },
  );

  mcp.registerTool(
    FS_TOOL,
    {
      title: `Query the ${config.siteName} filesystem`,
      description: FS_DESCRIPTION,
      inputSchema: {
        command: z.string().describe("A shell command for the virtual docs filesystem, e.g. `tree / -L 2` or `head -80 /path/page.mdx`."),
      },
      annotations: { readOnlyHint: true },
    },
    async ({ command }) => {
      const { text } = await docs.run(command, groups);
      return { content: [{ type: "text" as const, text }] };
    },
  );
  return mcp;
}

async function readBody(req: IncomingMessage): Promise<unknown> {
  let raw = "";
  for await (const chunk of req) raw += chunk;
  return raw ? JSON.parse(raw) : undefined;
}

async function main() {
  const docs = await DocsService.start();
  console.log(`ChromaFs ready: ${docs.pageCount} pages, ${docs.search.chunkCount} chunks indexed for search`);

  const logReload = (why: string, r: Awaited<ReturnType<DocsService["refresh"]>>) => {
    if (r.reloaded) {
      console.log(`[sync] ${why}: revision ${r.revision}, ${r.pages} pages, ${r.invalidated} cached pages dropped (${r.ms.toFixed(0)} ms)`);
    }
  };
  if (config.treePollSeconds > 0) {
    setInterval(() => {
      docs.refresh().then((r) => logReload("tree changed", r), (err) => console.error("[sync] poll failed:", err));
    }, config.treePollSeconds * 1000).unref();
  }

  const json = (res: import("node:http").ServerResponse, status: number, body: unknown) =>
    res.writeHead(status, { "Content-Type": "application/json" }).end(JSON.stringify(body));

  const http = createServer(async (req, res) => {
    if (req.url === "/health") {
      json(res, 200, { ok: true, revision: docs.revision, pages: docs.pageCount, chunks: docs.search.chunkCount });
      return;
    }
    if (req.url === "/admin/reload") {
      // sync.py calls this right after writing a new tree. Local callers only.
      const local = ["127.0.0.1", "::1", "::ffff:127.0.0.1"].includes(req.socket.remoteAddress ?? "");
      if (req.method !== "POST" || !local) {
        json(res, 403, { error: "POST from localhost only" });
        return;
      }
      const result = await docs.refresh();
      logReload("reload requested", result);
      json(res, 200, result);
      return;
    }
    if (!req.url?.startsWith("/mcp")) {
      res.writeHead(404).end("Not found. MCP endpoint is /mcp");
      return;
    }
    try {
      // Stateless, like the live server: a fresh MCP server + transport per request.
      const mcp = buildMcp(docs, groupsFrom(req));
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
      res.on("close", () => {
        void transport.close();
        void mcp.close();
      });
      await mcp.connect(transport);
      await transport.handleRequest(req, res, req.method === "POST" ? await readBody(req) : undefined);
    } catch (err) {
      console.error(err);
      if (!res.headersSent) res.writeHead(500).end(JSON.stringify({ error: String(err) }));
    }
  });
  http.listen(config.port, () => console.log(`MCP server on http://localhost:${config.port}/mcp`));
}

void main();
