// End-to-end checks of the Mintlify-style behaviours. Needs Chroma running and the ingest done.
//   npm test
import assert from "node:assert/strict";
import { DocsService } from "./docs-shell.js";

const docs = await DocsService.start();
const PUBLIC = ["public"];
let failures = 0;

async function check(name: string, fn: () => Promise<void>) {
  try {
    await fn();
    console.log(`  ok   ${name}`);
  } catch (err) {
    failures++;
    console.log(`  FAIL ${name}\n       ${(err as Error).message.replace(/\n/g, " | ").slice(0, 500)}`);
  }
}

console.log(`Loaded ${docs.pageCount} pages / ${docs.search.chunkCount} chunks\n`);

await check("tree: top level matches the live layout", async () => {
  const { text } = await docs.run("ls /", PUBLIC);
  for (const entry of ["index.mdx", "langsmith", "openapi", "oss"]) assert.match(text, new RegExp(`^${entry}$`, "m"));
});

await check("ls / cd / find / tree answered from memory (zero DB round trips)", async () => {
  const before = docs.store.calls;
  const { text } = await docs.run("cd /oss/python/langgraph && ls | head -5 && find /oss -name '*.mdx' | wc -l && tree / -L 1", PUBLIC);
  assert.match(text, /^exit: 0/);
  assert.equal(docs.store.calls - before, 0, "made Chroma calls");
});

await check("cat rebuilds a page from its chunks", async () => {
  const { text } = await docs.run("head -3 /oss/python/langgraph/streaming.mdx", PUBLIC);
  assert.match(text, /^exit: 0\n--- stdout ---\n# Streaming\n/);
});

await check("rg is accelerated: Chroma narrows candidates first", async () => {
  const { text, accel } = await docs.run('rg -l "stream_mode" /oss/python', PUBLIC);
  assert.ok(accel.rewritten && (accel.candidates ?? 0) > 0, JSON.stringify(accel));
  assert.match(text, /\/oss\/python\/langgraph\/streaming\.mdx/);
});

await check("accelerated rg gives the same files as a full scan", async () => {
  const fast = await docs.run('rg -l "interrupt_before" /oss/python/langgraph', PUBLIC);
  const slow = await docs.run('find /oss/python/langgraph -name "*.mdx" | xargs rg -l "interrupt_before"', PUBLIC);
  const files = (t: string) => t.split("\n").filter((l) => l.endsWith(".mdx")).sort();
  assert.deepEqual(files(fast.text), files(slow.text));
});

await check("pipes and flags work (just-bash)", async () => {
  const { text } = await docs.run("rg -i -c 'checkpointer' /oss/python/langgraph | sort -t: -k2 -nr | head -3", PUBLIC);
  assert.match(text, /\.mdx:\d+/);
});

await check("lazy OpenAPI spec: listed, fetched only on read", async () => {
  const { text } = await docs.run("ls /openapi/langsmith && cat /openapi/langsmith/langsmith-platform-openapi.json | jq -r '.openapi'", PUBLIC);
  assert.match(text, /langsmith-platform-openapi\.json/);
  assert.match(text, /^3\./m);
});

await check("read-only: writes fail with EROFS", async () => {
  const { text, exitCode } = await docs.run("echo hi > /oss/new.mdx", PUBLIC);
  assert.notEqual(exitCode, 0);
  assert.match(text, /EROFS|read-only/i);
});

await check("RBAC: staff page invisible to public users", async () => {
  const anon = await docs.run("ls /internal; rg -l 'BLUE-HERON' /", PUBLIC);
  assert.doesNotMatch(anon.text, /support-playbook/);
  const hits = await docs.searchDocs("support escalation playbook BLUE-HERON", PUBLIC);
  assert.ok(!hits.some((h) => h.page.startsWith("internal/")), "search leaked a staff page");
});

await check("RBAC: staff users can see it", async () => {
  const staff = await docs.run("rg -l 'BLUE-HERON' /", ["public", "staff"]);
  assert.match(staff.text, /\/internal\/support-playbook\.mdx/);
});

await check("hybrid search returns Mintlify-shaped hits", async () => {
  const hits = await docs.searchDocs("stream subgraph outputs", PUBLIC);
  assert.ok(hits.length > 0);
  assert.ok(hits.some((h) => /subgraph/i.test(h.title) && h.link.includes("#")), JSON.stringify(hits.map((h) => h.title)));
});

await check("output over 30k chars is truncated like the live server", async () => {
  const { text } = await docs.run("cat /oss/python/langgraph/streaming.mdx /oss/python/langgraph/use-subgraphs.mdx", PUBLIC);
  assert.match(text, /\.\.\. \(output truncated\)$/);
});

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
