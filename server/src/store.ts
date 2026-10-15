// Everything that talks to Chroma. ChromaFs, grep acceleration and search all go through here.
import { gunzipSync } from "node:zlib";
import { ChromaClient, type Collection, type Where, type WhereDocument } from "chromadb";
import { DefaultEmbeddingFunction } from "@chroma-core/default-embed";
import { config } from "./config.js";

export const PATH_TREE_ID = "__path_tree__";

export interface TreeFile {
  size: number;
  title?: string;
  groups: string[];
  isPublic: boolean;
  chunks?: number;
  hash?: string; // fingerprint of the page's stored chunks; lets a reload invalidate only changed pages
}
export interface TreeLazy {
  source: string; // http(s):// or file:// URL, fetched only when read
  groups: string[];
  isPublic: boolean;
}
export interface PathTree {
  version: number;
  site: string;
  generated_at: number;
  revision?: string; // changes on every sync that writes a new tree
  source_commit?: string;
  files: Record<string, TreeFile>;
  lazy: Record<string, TreeLazy>;
}
export interface ChunkMeta {
  kind: "chunk";
  part: "header" | "intro" | "section";
  page: string;
  chunk_index: number;
  page_title: string;
  title: string;
  url: string;
  // plus one `g_<group>: true` flag per group that may read the chunk
  [key: string]: string | number | boolean;
}
export interface Chunk {
  id: string;
  text: string;
  meta: ChunkMeta;
}

/** Chroma filter that only matches chunks the caller's groups may see (one boolean key per group). */
export function chunkWhere(groups: string[], extra: Where[] = []): Where {
  const groupClauses = groups.map((g) => ({ [`g_${g}`]: true }) as Where);
  const access: Where = groupClauses.length === 1 ? groupClauses[0] : { $or: groupClauses };
  return { $and: [{ kind: "chunk" }, access, ...extra] } as Where;
}

export class DocsStore {
  /** Number of Chroma round trips so far (lets tests prove ls/cd/find never touch the database). */
  calls = 0;

  private constructor(private readonly chroma: Collection) {}

  get collection(): Collection {
    this.calls++;
    return this.chroma;
  }

  /** DOCS_DEBUG_QUERIES=1 prints every Chroma query the filesystem makes (to stderr). */
  private debug(op: string, query: unknown, result: string, started: number): void {
    if (!process.env.DOCS_DEBUG_QUERIES) return;
    const ms = (performance.now() - started).toFixed(0);
    console.error(`  [chroma] ${op} ${JSON.stringify(query)} -> ${result} (${ms} ms)`);
  }

  static async connect(): Promise<DocsStore> {
    const client = new ChromaClient({ host: config.chromaHost, port: config.chromaPort });
    const collection = await client.getCollection({
      name: config.collection,
      embeddingFunction: new DefaultEmbeddingFunction(),
    });
    return new DocsStore(collection);
  }

  /** Cheap poll: only the tree record's metadata. Not counted in `calls` (it runs on a timer). */
  async treeRevision(): Promise<string | null> {
    const res = await this.chroma.get({ ids: [PATH_TREE_ID], include: ["metadatas"] });
    const revision = res.metadatas[0]?.revision;
    return typeof revision === "string" ? revision : null;
  }

  /** The whole file tree: one record, gzipped + base64 JSON. */
  async loadTree(): Promise<PathTree> {
    const started = performance.now();
    const res = await this.collection.get({ ids: [PATH_TREE_ID], include: ["documents"] });
    const blob = res.documents[0];
    this.debug("get", { ids: [PATH_TREE_ID] }, `1 record, ${blob?.length ?? 0} chars of gzip+base64`, started);
    if (!blob) throw new Error(`No ${PATH_TREE_ID} in collection '${config.collection}'. Run the ingest first.`);
    return JSON.parse(gunzipSync(Buffer.from(blob, "base64")).toString("utf-8"));
  }

  /** Reassemble whole pages: fetch every chunk of each page, sort by chunk_index, join. */
  async getPages(paths: string[], groups: string[]): Promise<Map<string, string>> {
    const pages = new Map<string, string>();
    for (let i = 0; i < paths.length; i += 50) {
      const batch = paths.slice(i, i + 50);
      const started = performance.now();
      const where = chunkWhere(groups, [{ page: { $in: batch } }]);
      const res = await this.collection.get<ChunkMeta>({ where, include: ["documents", "metadatas"] });
      this.debug("get", { where }, `${res.ids.length} chunks of ${batch.length} page(s)`, started);
      const byPage = new Map<string, { index: number; text: string }[]>();
      res.ids.forEach((_, j) => {
        const meta = res.metadatas[j]!;
        const list = byPage.get(meta.page) ?? [];
        list.push({ index: meta.chunk_index, text: res.documents[j] ?? "" });
        byPage.set(meta.page, list);
      });
      for (const [page, chunks] of byPage) {
        chunks.sort((a, b) => a.index - b.index);
        pages.set(page, chunks.map((c) => c.text).join("\n\n"));
      }
    }
    return pages;
  }

  /** Grep coarse filter: which pages have at least one chunk matching? (metadata only, no text) */
  async pagesMatching(whereDocument: WhereDocument, groups: string[]): Promise<Set<string>> {
    const started = performance.now();
    const where = chunkWhere(groups);
    const res = await this.collection.get<ChunkMeta>({ where, whereDocument, include: ["metadatas"] });
    const pages = new Set(res.metadatas.map((m) => m!.page));
    this.debug("get", { where, whereDocument }, `${res.ids.length} matching chunks in ${pages.size} page(s), metadata only`, started);
    return pages;
  }

  /** Every chunk (for the in-memory BM25 index). Paged to keep responses small. */
  async allChunks(): Promise<Chunk[]> {
    const out: Chunk[] = [];
    const limit = 2000;
    for (let offset = 0; ; offset += limit) {
      const res = await this.collection.get<ChunkMeta>({
        where: { kind: "chunk" },
        include: ["documents", "metadatas"],
        limit,
        offset,
      });
      res.ids.forEach((id, j) => out.push({ id, text: res.documents[j] ?? "", meta: res.metadatas[j]! }));
      if (res.ids.length < limit) return out;
    }
  }

  /** Dense (embedding) search, restricted to the caller's groups. */
  async denseSearch(query: string, groups: string[], n: number): Promise<Chunk[]> {
    const res = await this.collection.query<ChunkMeta>({
      queryTexts: [query],
      nResults: n,
      where: chunkWhere(groups),
      include: ["documents", "metadatas"],
    });
    return res.ids[0].map((id, j) => ({ id, text: res.documents[0][j] ?? "", meta: res.metadatas[0][j]! }));
  }
}
