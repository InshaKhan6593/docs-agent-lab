// Hybrid search: dense vectors (Chroma) + keyword BM25 (in memory), fused with Reciprocal Rank Fusion.
//
// Mintlify runs dense + SPLADE sparse vectors on Chroma Cloud, fused with RRF. Sparse indexes are a
// Chroma Cloud feature ("Sparse vector indexing is not enabled in local"), so locally the keyword
// half is a plain BM25 index built from the same chunks at startup.
import type { Chunk, DocsStore } from "./store.js";

const K1 = 1.2;
const B = 0.75;
const RRF_K = 60;
const CANDIDATES = 50;

const tokenize = (text: string) => text.toLowerCase().match(/[a-z0-9_]+/g) ?? [];

class Bm25 {
  private postings = new Map<string, Map<number, number>>(); // term -> (doc -> tf)
  private lengths: number[] = [];
  private avgLength = 0;

  constructor(readonly chunks: Chunk[]) {
    chunks.forEach((chunk, doc) => {
      const terms = tokenize(`${chunk.meta.page_title} ${chunk.meta.title} ${chunk.text}`);
      this.lengths.push(terms.length);
      for (const term of terms) {
        let posting = this.postings.get(term);
        if (!posting) this.postings.set(term, (posting = new Map()));
        posting.set(doc, (posting.get(doc) ?? 0) + 1);
      }
    });
    this.avgLength = this.lengths.reduce((a, b) => a + b, 0) / Math.max(1, this.lengths.length);
  }

  search(query: string, allowed: (c: Chunk) => boolean, n: number): Chunk[] {
    const scores = new Map<number, number>();
    const total = this.chunks.length;
    for (const term of new Set(tokenize(query))) {
      const posting = this.postings.get(term);
      if (!posting) continue;
      const idf = Math.log(1 + (total - posting.size + 0.5) / (posting.size + 0.5));
      for (const [doc, tf] of posting) {
        const norm = tf * (K1 + 1) / (tf + K1 * (1 - B + (B * this.lengths[doc]) / this.avgLength));
        scores.set(doc, (scores.get(doc) ?? 0) + idf * norm);
      }
    }
    return [...scores.entries()]
      .sort((a, b) => b[1] - a[1])
      .map(([doc]) => this.chunks[doc])
      .filter(allowed)
      .slice(0, n);
  }
}

export interface SearchHit {
  title: string;
  link: string;
  page: string;
  content: string;
}

export class HybridSearch {
  private constructor(
    private readonly store: DocsStore,
    private readonly bm25: Bm25,
  ) {}

  static async build(store: DocsStore): Promise<HybridSearch> {
    return new HybridSearch(store, new Bm25(await store.allChunks()));
  }

  get chunkCount(): number {
    return this.bm25.chunks.length;
  }

  async search(query: string, groups: string[], n: number): Promise<SearchHit[]> {
    const allowed = (c: Chunk) => groups.some((g) => c.meta[`g_${g}`] === true);
    const [dense, keyword] = await Promise.all([
      this.store.denseSearch(query, groups, CANDIDATES),
      Promise.resolve(this.bm25.search(query, allowed, CANDIDATES)),
    ]);
    const fused = new Map<string, { chunk: Chunk; score: number }>();
    for (const ranking of [dense, keyword]) {
      ranking.forEach((chunk, rank) => {
        const entry = fused.get(chunk.id) ?? { chunk, score: 0 };
        entry.score += 1 / (RRF_K + rank + 1);
        fused.set(chunk.id, entry);
      });
    }
    return [...fused.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, n)
      .map(({ chunk }) => toHit(chunk));
  }
}

/** Format a chunk the way Mintlify's search tool returns it. */
function toHit(chunk: Chunk): SearchHit {
  const page = chunk.meta.page.replace(/^\//, "").replace(/\.mdx$/, "");
  let content = chunk.text;
  if (chunk.meta.part === "header") {
    // "# Title\n\n> description" is shown as "Title\ndescription"
    content = chunk.text.replace(/^# /, "").replace(/\n\n> /, "\n");
  }
  return { title: chunk.meta.title, link: chunk.meta.url, page, content };
}
