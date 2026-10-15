// Normalize MDX pages the way Mintlify serves them: parse to an mdast tree, stringify back
// with remark defaults (2-space JSX indentation, "*" bullets, "***" rules, escaped "&").
// Protocol: JSON array of {id, text} on stdin -> JSON array of {id, text, error?} on stdout.
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkMdx from "remark-mdx";
import remarkGfm from "remark-gfm";
import remarkStringify from "remark-stringify";

const processor = unified().use(remarkParse).use(remarkMdx).use(remarkGfm).use(remarkStringify);

let input = "";
for await (const chunk of process.stdin) input += chunk;
const out = JSON.parse(input).map(({ id, text }) => {
  try {
    return { id, text: String(processor.processSync(text)) };
  } catch (err) {
    // Unparseable MDX: keep the source text so the page still exists.
    return { id, text, error: String(err.message ?? err).split("\n")[0] };
  }
});
process.stdout.write(JSON.stringify(out));
