// One place that turns a command string into Mintlify-style output, shared by the MCP server and the CLI.
import { Bash } from "just-bash";
import { ChromaFs, invalidateChangedPages } from "./chroma-fs.js";
import { config } from "./config.js";
import { accelerateCommand, type AccelInfo } from "./grep-accel.js";
import { HybridSearch, type SearchHit } from "./search.js";
import { DocsStore, type PathTree } from "./store.js";

// When a command hits maxExecutionTimeMs, just-bash cancels it, but a command that was mid-await can
// still reject later ("used its context after cancellation"). The exec() result already reports the
// timeout, so log these instead of letting them crash the server.
process.on("unhandledRejection", (err) => {
  const message = err instanceof Error ? err.message : String(err);
  if (/execution aborted|after cancellation/i.test(message)) return;
  console.error("unhandled rejection:", err);
});

export interface ReloadResult {
  reloaded: boolean;
  revision: string | null;
  pages: number;
  invalidated: number;
  ms: number;
}

export class DocsService {
  private fsByGroups = new Map<string, ChromaFs>();
  private reloading: Promise<ReloadResult> | null = null;

  private constructor(
    readonly store: DocsStore,
    private tree: PathTree,
    private searchIndex: HybridSearch,
  ) {}

  static async start(): Promise<DocsService> {
    const store = await DocsStore.connect();
    const tree = await store.loadTree(); // fetched once; ls/cd/find never hit the database again
    const search = await HybridSearch.build(store);
    return new DocsService(store, tree, search);
  }

  get search(): HybridSearch {
    return this.searchIndex;
  }

  get pageCount(): number {
    return Object.keys(this.tree.files).length;
  }

  get revision(): string | null {
    return this.tree.revision ?? null;
  }

  /**
   * Pick up a sync: if the stored tree has a new revision, load it and rebuild the search index,
   * then swap both in at once. Requests already running keep their old view; new ones get the new one.
   * Only pages whose fingerprint changed are dropped from the page cache.
   */
  async refresh(force = false): Promise<ReloadResult> {
    if (this.reloading) return this.reloading;
    this.reloading = (async () => {
      const started = performance.now();
      try {
        const latest = await this.store.treeRevision();
        if (!force && latest === this.revision) {
          return { reloaded: false, revision: latest, pages: this.pageCount, invalidated: 0, ms: performance.now() - started };
        }
        const [tree, search] = await Promise.all([this.store.loadTree(), HybridSearch.build(this.store)]);
        const invalidated = invalidateChangedPages(this.tree, tree);
        this.tree = tree;
        this.searchIndex = search;
        this.fsByGroups.clear(); // trees are rebuilt (and re-pruned per group set) on next use
        return { reloaded: true, revision: this.revision, pages: this.pageCount, invalidated, ms: performance.now() - started };
      } finally {
        this.reloading = null;
      }
    })();
    return this.reloading;
  }

  /** One pruned tree per distinct group set (Mintlify caches the tree per site). */
  fsFor(groups: string[]): ChromaFs {
    const key = [...groups].sort().join(",");
    let fs = this.fsByGroups.get(key);
    if (!fs) this.fsByGroups.set(key, (fs = new ChromaFs(this.store, this.tree, groups)));
    return fs;
  }

  /** Stateless: every call gets a fresh shell rooted at "/". Output mirrors the live server. */
  async run(command: string, groups: string[]): Promise<{ text: string; exitCode: number; accel: AccelInfo; ms: number }> {
    const started = performance.now();
    const fs = this.fsFor(groups);
    const { command: effective, info } = await accelerateCommand(command, fs, this.store);
    const bash = new Bash({
      fs,
      cwd: "/",
      env: { HOME: "/", PWD: "/" },
      executionLimits: { maxExecutionTimeMs: 20_000 },
    });
    let result: { stdout: string; stderr: string; exitCode: number };
    try {
      result = await bash.exec(effective);
    } catch (err) {
      // e.g. a redirect onto the read-only filesystem: report it like a shell would
      result = { stdout: "", stderr: `bash: ${(err as Error).message}\n`, exitCode: 1 };
    }
    let stdout = result.stdout;
    if (stdout.length > config.maxOutputChars) stdout = `${stdout.slice(0, config.maxOutputChars)}\n... (output truncated)`;
    let text = `exit: ${result.exitCode}\n`;
    if (stdout) text += `--- stdout ---\n${stdout}`;
    if (result.stderr) text += `${stdout && !stdout.endsWith("\n") ? "\n" : ""}--- stderr ---\n${result.stderr}`;
    return { text, exitCode: result.exitCode, accel: info, ms: performance.now() - started };
  }

  async searchDocs(query: string, groups: string[]): Promise<SearchHit[]> {
    return this.search.search(query, groups, config.searchResults);
  }
}
