// Grep acceleration, Mintlify-style.
//
// A recursive `rg PATTERN /` would otherwise read every page through ChromaFs. Instead:
//   1. Coarse filter: parse the grep flags and turn the pattern into a Chroma document filter
//      ($contains for fixed strings, $regex otherwise) to find candidate pages.
//   2. Fine filter: prefetch those pages into the cache in a few batched queries, rewrite the
//      command to name only those files, and let just-bash's real grep/rg run in memory.
// Anything we can't translate safely is left untouched: it still works, just without the shortcut.
//
// Mintlify mention yargs-parser for step 1. yargs assumes an undeclared flag takes the next word as
// its value (`rg -l PATTERN` would read PATTERN as -l's value), so this uses a small getopt-style
// parser instead: a flag is a boolean switch unless it is listed as taking a value.
import type { WhereDocument } from "chromadb";
import type { ChromaFs } from "./chroma-fs.js";
import type { DocsStore } from "./store.js";

const GREP_COMMANDS = new Set(["rg", "grep", "egrep", "fgrep"]);

const VALUE_FLAGS: Record<"rg" | "grep", Set<string>> = {
  rg: new Set(["C", "A", "B", "m", "M", "e", "f", "g", "t", "T", "j", "context", "after-context",
    "before-context", "max-count", "max-columns", "regexp", "file", "glob", "iglob", "type", "type-not",
    "max-depth", "color", "colors", "sort", "sortr", "context-separator", "field-match-separator",
    "threads", "encoding", "E", "max-filesize", "replace", "r"]),
  grep: new Set(["C", "A", "B", "m", "e", "f", "context", "after-context", "before-context", "max-count",
    "regexp", "file", "include", "exclude", "exclude-dir", "include-dir", "color", "colour", "label",
    "binary-files", "devices", "directories", "group-separator"]),
};
// Flags that change which files are searched, or invert/rewrite matching: skip acceleration.
const UNSAFE_FLAGS = new Set(["v", "invert-match", "f", "file", "g", "glob", "iglob", "t", "type", "T",
  "type-not", "include", "exclude", "exclude-dir", "include-dir", "L", "files-without-match", "files",
  "P", "pcre2", "U", "multiline", "z", "search-zip", "null-data", "hidden", "u", "unrestricted",
  "directories", "type-list"]);

export interface AccelInfo {
  rewritten: boolean;
  reason?: string;
  candidates?: number;
}

interface ParsedArgs {
  flags: Map<string, string | true>;
  values: string[]; // -e/--regexp values, in order
  positionals: { word: string; index: number }[];
}

/** getopt-style parsing: `-abc` groups, `-C3` / `-C 3`, `--long=value`, `--long value`, `--`. */
function parseArgs(words: string[], valueFlags: Set<string>): ParsedArgs {
  const flags = new Map<string, string | true>();
  const values: string[] = [];
  const positionals: { word: string; index: number }[] = [];
  const set = (name: string, value: string | true) => {
    flags.set(name, value);
    if ((name === "e" || name === "regexp") && typeof value === "string") values.push(value);
  };
  let onlyPositionals = false;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    if (onlyPositionals || w === "-" || !w.startsWith("-")) {
      positionals.push({ word: w, index: i });
    } else if (w === "--") {
      onlyPositionals = true;
    } else if (w.startsWith("--")) {
      const eq = w.indexOf("=");
      const name = eq === -1 ? w.slice(2) : w.slice(2, eq);
      if (eq !== -1) set(name, w.slice(eq + 1));
      else if (valueFlags.has(name)) set(name, words[++i] ?? "");
      else set(name, true);
    } else {
      for (let j = 1; j < w.length; j++) {
        const name = w[j];
        if (valueFlags.has(name)) {
          set(name, j + 1 < w.length ? w.slice(j + 1) : (words[++i] ?? ""));
          break;
        }
        set(name, true);
      }
    }
  }
  return { flags, values, positionals };
}

/** Split a command line into stages at top-level `|`, `||`, `&&`, `;` (respecting quotes). */
function splitStages(command: string): { text: string; sep: string }[] {
  const stages: { text: string; sep: string }[] = [];
  let quote: string | null = null;
  let start = 0;
  for (let i = 0; i < command.length; i++) {
    const ch = command[i];
    if (quote) {
      if (ch === "\\" && quote === '"') i++;
      else if (ch === quote) quote = null;
    } else if (ch === "'" || ch === '"') quote = ch;
    else if (ch === "\\") i++;
    else if (ch === "|" || ch === ";" || ch === "&") {
      const sep = (ch === "|" || ch === "&") && command[i + 1] === ch ? ch + ch : ch;
      stages.push({ text: command.slice(start, i), sep });
      i += sep.length - 1;
      start = i + 1;
    }
  }
  stages.push({ text: command.slice(start), sep: "" });
  return stages;
}

/**
 * Accelerate every grep that starts a pipeline (`rg x /`, `ls; rg x /`, `rg x / | head`).
 * A grep after `|` reads stdin, so it is left alone.
 */
export async function accelerateCommand(
  command: string,
  fs: ChromaFs,
  store: DocsStore,
): Promise<{ command: string; info: AccelInfo }> {
  const stages = splitStages(command);
  let rewritten = false;
  let candidates = 0;
  let reason: string | undefined;
  for (let i = 0; i < stages.length; i++) {
    if (i > 0 && stages[i - 1].sep === "|") continue;
    const { command: stage, info } = await accelerateGrep(stages[i].text, fs, store);
    stages[i].text = stage;
    if (info.rewritten) {
      rewritten = true;
      candidates += info.candidates ?? 0;
    } else reason ??= info.reason;
  }
  const rebuilt = stages.map((s) => s.text + s.sep).join("");
  return { command: rebuilt, info: rewritten ? { rewritten, candidates } : { rewritten, reason } };
}

/** Minimal shell word splitter (quotes and backslashes; bails on expansions, globs, redirects). */
function shellWords(stage: string): string[] | null {
  const words: string[] = [];
  let cur = "";
  let started = false;
  let quote: string | null = null;
  for (let i = 0; i < stage.length; i++) {
    const ch = stage[i];
    if (quote) {
      if (ch === quote) quote = null;
      else if (ch === "\\" && quote === '"' && i + 1 < stage.length) cur += stage[++i];
      else cur += ch;
    } else if (ch === "'" || ch === '"') {
      quote = ch;
      started = true;
    } else if (ch === "\\" && i + 1 < stage.length) {
      cur += stage[++i];
      started = true;
    } else if (/\s/.test(ch)) {
      if (started) words.push(cur);
      cur = "";
      started = false;
    } else if ("$`*?[<>(){}~".includes(ch)) {
      return null;
    } else {
      cur += ch;
      started = true;
    }
  }
  if (quote) return null;
  if (started) words.push(cur);
  return words;
}

const shellQuote = (s: string) => (/^[\w./:@%+=,-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`);
const escapeRegex = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export async function accelerateGrep(
  stage: string,
  fs: ChromaFs,
  store: DocsStore,
): Promise<{ command: string; info: AccelInfo }> {
  const leading = stage.match(/^\s*/)![0];
  const trailing = stage.match(/\s*$/)![0];
  const words = shellWords(stage.trim());
  const unchanged = (reason: string) => ({ command: stage, info: { rewritten: false, reason } });
  if (!words || words.length === 0 || !GREP_COMMANDS.has(words[0])) return unchanged("not a grep stage");

  const cmd = words[0];
  const family = cmd === "rg" ? "rg" : "grep";
  const args = words.slice(1);
  const { flags, values, positionals } = parseArgs(args, VALUE_FLAGS[family]);
  const has = (...names: string[]) => names.some((n) => flags.has(n));

  const paths = [...positionals];
  const pattern = values[0] ?? paths.shift()?.word;
  if (family === "grep" && !has("r", "R", "recursive", "dereference-recursive")) return unchanged("grep without -r");
  const targets = paths.length ? paths.map((p) => p.word) : ["."];
  const absTargets = targets.map((t) => fs.resolvePath("/", t));
  const dirTargets = absTargets.filter((t) => fs.isDir(t));
  if (dirTargets.length === 0) return unchanged("only file targets");
  const fileTargets = targets.filter((_, i) => !fs.isDir(absTargets[i]));

  // Can't narrow safely: still warm every page under the targets in a few batched queries,
  // so the full in-memory scan costs ~20 round trips instead of one per page.
  const fallback = async (reason: string) => {
    await fs.prefetch([...new Set(dirTargets.flatMap((d) => fs.pagesUnder(d)))]);
    return unchanged(`${reason} (bulk-prefetched instead)`);
  };
  for (const name of flags.keys()) if (UNSAFE_FLAGS.has(name)) return fallback(`-${name} changes the file set`);
  if (family === "grep" && has("c", "count")) return fallback("grep -c prints zero counts");
  if (values.length > 1) return fallback("multiple patterns");
  if (pattern === undefined || pattern === "") return unchanged("no pattern");

  // ---- 1. coarse filter in Chroma
  const smartLower = has("S", "smart-case") && pattern === pattern.toLowerCase();
  const ignoreCase = has("i", "ignore-case") || smartLower;
  const fixed = has("F", "fixed-strings") || cmd === "fgrep";
  const extended = family === "rg" || cmd === "egrep" || has("E", "extended-regexp");
  let source: string | null = null; // null = treat as a literal string
  if (!fixed) {
    if (extended) source = pattern;
    else if (/[\^$.*[\]|+?(){}]/.test(pattern)) return fallback("basic-regex pattern");
  }
  if (has("w", "word-regexp")) source = `\b(?:${source ?? escapeRegex(pattern)})\b`;
  if (has("x", "line-regexp")) source = `(?m)^(?:${source ?? escapeRegex(pattern)})$`;

  const whereDocument: WhereDocument =
    source === null && !ignoreCase
      ? { $contains: pattern }
      : { $regex: `${ignoreCase ? "(?i)" : ""}${source ?? escapeRegex(pattern)}` };

  let matched: Set<string>;
  try {
    matched = await store.pagesMatching(whereDocument, fs.groups);
  } catch {
    return fallback("pattern not supported by Chroma's regex engine");
  }

  // ---- 2. fine filter: prefetch candidates, then name only them
  const candidates = [...new Set(dirTargets.flatMap((d) => fs.pagesUnder(d)))].filter((p) => matched.has(p)).sort();
  await fs.prefetch(candidates);

  const files = [...fileTargets, ...candidates];
  if (files.length === 0) return { command: `${leading}false${trailing}`, info: { rewritten: true, candidates: 0 } };

  const pathIndexes = new Set(paths.map((p) => p.index));
  const kept = args.filter((_, i) => !pathIndexes.has(i));
  // A directory search always prints file names; with one explicit file it wouldn't, so force -H.
  const noNames = family === "rg" ? has("I", "no-filename") : has("h", "no-filename");
  const withNames = noNames || has("H", "with-filename") ? [] : ["-H"];
  const rebuilt = [cmd, ...withNames, ...kept.map(shellQuote), ...files.map(shellQuote)].join(" ");
  return { command: `${leading}${rebuilt}${trailing}`, info: { rewritten: true, candidates: candidates.length } };
}
