// Try ChromaFs by hand, no MCP client needed.
//   npm run shell                       -> interactive prompt
//   npm run shell -- "tree / -L 1"      -> run one command
//   npm run shell -- --search "stream subgraph outputs"
//   DOCS_GROUPS=public,staff npm run shell   -> act as a user with extra groups
import { createInterface } from "node:readline/promises";
import { DocsService } from "./docs-shell.js";

const groups = (process.env.DOCS_GROUPS ?? "public").split(",").map((g) => g.trim());
const docs = await DocsService.start();
const args = process.argv.slice(2);

async function runOne(line: string) {
  if (line.startsWith("--search ")) {
    for (const hit of await docs.searchDocs(line.slice(9), groups)) {
      console.log(`- ${hit.title}  ->  ${hit.link}\n  ${hit.content.slice(0, 160).replace(/\n/g, " ")}`);
    }
    return;
  }
  const { text, accel, ms } = await docs.run(line, groups);
  console.log(text.trimEnd());
  const how = accel.rewritten ? `grep accelerated: ${accel.candidates} candidate pages` : "";
  console.log(`\x1b[2m(${ms.toFixed(0)} ms${how ? `, ${how}` : ""})\x1b[0m`);
}

if (args.length) {
  await runOne(args.join(" "));
} else if (!process.stdin.isTTY) {
  // Piped input: run each line in order (same process, so the page cache carries over).
  let input = "";
  for await (const chunk of process.stdin) input += chunk;
  for (const line of input.split(/\r?\n/).map((l) => l.trim()).filter(Boolean)) {
    console.log(`\n$ ${line}`);
    await runOne(line);
  }
} else {
  console.log(`ChromaFs shell: ${docs.pageCount} pages, groups=[${groups}]. Ctrl+C to quit.`);
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  for (;;) {
    const line = (await rl.question("docs$ ")).trim();
    if (line) await runOne(line);
  }
}
