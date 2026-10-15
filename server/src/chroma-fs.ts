// ChromaFs: a read-only virtual filesystem over Chroma, in the shape of Mintlify's design.
//
//  * The path tree comes from one `__path_tree__` record, already pruned to the caller's groups,
//    so ls / cd / find / stat are answered from memory with no database calls.
//  * readFile (cat, head, rg...) reassembles a page from its chunks and caches it.
//  * Large files (OpenAPI specs) are lazy pointers: listed, but only fetched when read.
//  * Every write throws EROFS. Nothing a session does can change what another session sees.
import { readFile as readLocalFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { DefenseInDepthBox } from "just-bash";
import type { CpOptions, FileContent, FsStat, IFileSystem, MkdirOptions, RmOptions } from "just-bash";
import { config } from "./config.js";
import type { DocsStore, PathTree } from "./store.js";

type DirentEntry = Awaited<ReturnType<NonNullable<IFileSystem["readdirWithFileTypes"]>>>[number];

const FILE_MODE = 0o100444; // regular file, read-only
const DIR_MODE = 0o040555; // directory, read-only

export function normalizePath(path: string): string {
  const parts: string[] = [];
  for (const part of path.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") parts.pop();
    else parts.push(part);
  }
  return `/${parts.join("/")}`;
}

/** Shared page cache (path -> text). Access is checked against the caller's tree before any read. */
class PageCache {
  private map = new Map<string, string>();
  private bytes = 0;
  get(path: string): string | undefined {
    const value = this.map.get(path);
    if (value !== undefined) {
      this.map.delete(path); // refresh LRU position
      this.map.set(path, value);
    }
    return value;
  }
  delete(path: string): void {
    const old = this.map.get(path);
    if (old === undefined) return;
    this.map.delete(path);
    this.bytes -= old.length;
  }
  clear(): void {
    this.map.clear();
    this.bytes = 0;
  }
  get size(): number {
    return this.map.size;
  }
  set(path: string, text: string): void {
    if (this.map.has(path)) return;
    this.map.set(path, text);
    this.bytes += text.length;
    for (const [key, old] of this.map) {
      if (this.bytes <= config.pageCacheBytes) break;
      this.map.delete(key);
      this.bytes -= old.length;
    }
  }
}
const pageCache = new PageCache();

/** After a sync: drop cached copies of pages that changed, were removed, or can't be compared. */
export function invalidateChangedPages(oldTree: PathTree, newTree: PathTree): number {
  const before = pageCache.size;
  for (const [path, old] of Object.entries(oldTree.files)) {
    const next = newTree.files[path];
    if (!next || !old.hash || old.hash !== next.hash) pageCache.delete(path);
  }
  for (const [path, old] of Object.entries(oldTree.lazy)) {
    if (newTree.lazy[path]?.source !== old.source) pageCache.delete(path);
  }
  return before - pageCache.size;
}

interface FileNode {
  size: number;
  lazySource?: string;
}

export class ChromaFs implements IFileSystem {
  private files = new Map<string, FileNode>();
  private dirs = new Map<string, Set<string>>([["/", new Set()]]);
  private readonly mtime: Date;

  constructor(
    private readonly store: DocsStore,
    tree: PathTree,
    readonly groups: string[],
  ) {
    this.mtime = new Date(tree.generated_at * 1000);
    const allowed = (g: string[]) => g.some((x) => groups.includes(x));
    // RBAC happens here, before the tree is built: hidden paths simply do not exist.
    for (const [path, f] of Object.entries(tree.files)) {
      if (allowed(f.groups)) this.addFile(path, { size: f.size });
    }
    for (const [path, f] of Object.entries(tree.lazy)) {
      if (allowed(f.groups)) this.addFile(path, { size: 0, lazySource: f.source });
    }
  }

  private addFile(rawPath: string, node: FileNode): void {
    const path = normalizePath(rawPath);
    this.files.set(path, node);
    let child = path;
    while (child !== "/") {
      const parent = child.slice(0, child.lastIndexOf("/")) || "/";
      const name = child.slice(child.lastIndexOf("/") + 1);
      if (!this.dirs.has(parent)) this.dirs.set(parent, new Set());
      this.dirs.get(parent)!.add(name);
      child = parent;
    }
  }

  get fileCount(): number {
    return this.files.size;
  }

  /** Paths of all non-lazy files under a directory (used by grep acceleration). */
  pagesUnder(dir: string): string[] {
    const base = normalizePath(dir);
    const prefix = base === "/" ? "/" : `${base}/`;
    return [...this.files.entries()]
      .filter(([p, n]) => !n.lazySource && (p === base || p.startsWith(prefix)))
      .map(([p]) => p);
  }

  isFile(path: string): boolean {
    return this.files.has(normalizePath(path));
  }

  isDir(path: string): boolean {
    return this.dirs.has(normalizePath(path));
  }

  /** Warm the cache for many pages in a few batched queries (grep's fine-filter step). */
  async prefetch(paths: string[]): Promise<void> {
    const missing = paths.filter((p) => !this.files.get(p)?.lazySource && pageCache.get(p) === undefined);
    if (missing.length === 0) return;
    const pages = await this.store.getPages(missing, this.groups);
    for (const [path, text] of pages) pageCache.set(path, text);
  }

  // ------------------------------------------------------------------ reads

  async readFile(path: string): Promise<string> {
    const p = normalizePath(path);
    if (p === "/dev/null") return "";
    if (this.dirs.has(p)) throw new Error(`EISDIR: illegal operation on a directory, read '${path}'`);
    const node = this.files.get(p);
    if (!node) throw new Error(`ENOENT: no such file or directory, open '${path}'`);
    const cached = pageCache.get(p);
    if (cached !== undefined) return cached;
    // We're called from inside bash.exec(), where just-bash's defense-in-depth box blocks network
    // globals. Our Chroma/HTTP reads are host infrastructure, so run them in its trusted scope
    // (the same thing InMemoryFs does for lazy files).
    const text = await DefenseInDepthBox.runTrustedAsync(() =>
      node.lazySource ? this.fetchLazy(node.lazySource) : this.fetchPage(p),
    );
    pageCache.set(p, text);
    return text;
  }

  private async fetchPage(path: string): Promise<string> {
    const text = (await this.store.getPages([path], this.groups)).get(path);
    if (text === undefined) throw new Error(`ENOENT: no such file or directory, open '${path}'`);
    return text;
  }

  private async fetchLazy(source: string): Promise<string> {
    if (source.startsWith("file:")) return readLocalFile(fileURLToPath(source), "utf-8");
    const res = await fetch(source);
    if (!res.ok) throw new Error(`EIO: could not fetch lazy file (${res.status})`);
    return res.text();
  }

  async readFileBuffer(path: string): Promise<Uint8Array> {
    return new TextEncoder().encode(await this.readFile(path));
  }

  async exists(path: string): Promise<boolean> {
    const p = normalizePath(path);
    return this.files.has(p) || this.dirs.has(p);
  }

  async stat(path: string): Promise<FsStat> {
    const p = normalizePath(path);
    if (this.dirs.has(p)) {
      return { isFile: false, isDirectory: true, isSymbolicLink: false, mode: DIR_MODE, size: 0, mtime: this.mtime };
    }
    const node = this.files.get(p);
    if (!node) throw new Error(`ENOENT: no such file or directory, stat '${path}'`);
    // just-bash rejects reads larger than stat().size, so a lazy file is materialized on stat
    // (as InMemoryFs does). Listing a directory never stats, so `ls` still doesn't fetch it.
    const size = node.lazySource ? Buffer.byteLength(await this.readFile(p)) : node.size;
    return { isFile: true, isDirectory: false, isSymbolicLink: false, mode: FILE_MODE, size, mtime: this.mtime };
  }

  async lstat(path: string): Promise<FsStat> {
    return this.stat(path);
  }

  async readdir(path: string): Promise<string[]> {
    const p = normalizePath(path);
    const children = this.dirs.get(p);
    if (!children) {
      if (this.files.has(p)) throw new Error(`ENOTDIR: not a directory, scandir '${path}'`);
      throw new Error(`ENOENT: no such file or directory, scandir '${path}'`);
    }
    return [...children].sort();
  }

  async readdirWithFileTypes(path: string): Promise<DirentEntry[]> {
    const p = normalizePath(path);
    const names = await this.readdir(p);
    return names.map((name) => {
      const isDirectory = this.dirs.has(p === "/" ? `/${name}` : `${p}/${name}`);
      return { name, isFile: !isDirectory, isDirectory, isSymbolicLink: false };
    });
  }

  async realpath(path: string): Promise<string> {
    const p = normalizePath(path);
    if (!(await this.exists(p))) throw new Error(`ENOENT: no such file or directory, realpath '${path}'`);
    return p;
  }

  async readlink(path: string): Promise<string> {
    throw new Error(`EINVAL: invalid argument, readlink '${path}'`);
  }

  resolvePath(base: string, path: string): string {
    return path.startsWith("/") ? normalizePath(path) : normalizePath(`${base}/${path}`);
  }

  getAllPaths(): string[] {
    return [...this.dirs.keys(), ...this.files.keys()];
  }

  // ------------------------------------------------------------------ writes: read-only

  private readOnly(op: string, path: string): never {
    throw new Error(`EROFS: read-only file system, ${op} '${path}'`);
  }

  async writeFile(path: string, _content: FileContent): Promise<void> {
    if (normalizePath(path) === "/dev/null") return; // let `2>/dev/null` work
    this.readOnly("open", path);
  }
  async appendFile(path: string, _content: FileContent): Promise<void> {
    if (normalizePath(path) === "/dev/null") return;
    this.readOnly("open", path);
  }
  async mkdir(path: string, _options?: MkdirOptions): Promise<void> {
    this.readOnly("mkdir", path);
  }
  async rm(path: string, _options?: RmOptions): Promise<void> {
    this.readOnly("rm", path);
  }
  async cp(_src: string, dest: string, _options?: CpOptions): Promise<void> {
    this.readOnly("cp", dest);
  }
  async mv(src: string, _dest: string): Promise<void> {
    this.readOnly("rename", src);
  }
  async chmod(path: string, _mode: number): Promise<void> {
    this.readOnly("chmod", path);
  }
  async symlink(_target: string, linkPath: string): Promise<void> {
    this.readOnly("symlink", linkPath);
  }
  async link(_existing: string, newPath: string): Promise<void> {
    this.readOnly("link", newPath);
  }
  async utimes(path: string, _atime: Date, _mtime: Date): Promise<void> {
    this.readOnly("utime", path);
  }
}
