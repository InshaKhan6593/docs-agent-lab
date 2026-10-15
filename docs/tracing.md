# Tracing: the chat-langchain setup, piece by piece

Everything chat-langchain does with LangSmith, where it lives in their repo, and where it lives here.

| # | What | chat-langchain | here | Writes to your workspace? |
|---|---|---|---|---|
| 1 | Turn tracing on | `.env`: `LANGCHAIN_TRACING_V2`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT=docs-agent` | `.env`: `LANGSMITH_TRACING=true`, `LANGSMITH_PROJECT=docs-agent-lab` | yes, every run |
| 2 | Root-run metadata | `define_deep_agent(metadata=...)`, applied with `.with_config` | `agent/graph.py` `.with_config(...)` | (part of 1) |
|   | `source_type` | `"Chat-LangChain"` | `"docs-agent-lab"` (`SOURCE_TYPE`) | |
|   | `graph` | sent by their frontend | set server-side | |
|   | `prompt_source` / `prompt_commit` | Prompt Hub provenance | same; `local:...` until you push | |
|   | `guardrails_prompt_source` / `_commit` | same | same | |
|   | `LANGSMITH_AGENT_VERSION` | deployment revision | deployment revision, else `local-<hash of agent code + prompts>` | |
| 3 | Run tags | frontend: `["Chat-LangChain", "docs_agent"]` | graph: `["docs-agent-lab", "docs_agent"]` | (part of 1) |
| 4 | User metadata (`user_id`, `user_email`, `user_name`, Segment id) | frontend, from their login | **not available**: Agent Chat UI has no login. Add auth, then tag users the same way | |
| 5 | Guardrails decision | `run_tree.metadata["guardrails_result" / "guardrails_explanation"]` | identical (`agent/middleware/guardrails_middleware.py`) | (part of 1) |
| 6 | Summarization call marked | `config={"metadata": {"lc_source": "summarization"}}` | identical | (part of 1) |
| 7 | Live traffic -> eval dataset | 100% blocked + 1% allowed questions -> `Chat-LangChain-Guardrails-Samples` | -> `docs-agent-lab-guardrails-samples`; **off until** `GUARDRAILS_SAMPLE_TO_DATASET=true` | yes |
| 8 | Prompts in Prompt Hub | `scripts/push_*_prompt.py`, guardrails pulled live from `:production` | `scripts/push_prompts.py --tag staging/production` | yes |
| 9 | Thumbs feedback | connector capability `langsmith:chat-feedback` (key `ux.thumb_vote`, 1/0, comment <= 2000) | `POST /langsmith/feedback` on the agent server (`agent/webapp.py`) | yes |
| 10 | Tokens / cost / share link in the UI | connector capability `langsmith:trace-viewer` | `GET /langsmith/runs/{id}/summary`, `POST /langsmith/runs/{id}/share` | share link: yes |
| 11 | Evals logged to LangSmith | `tests/evals` with `@pytest.mark.langsmith` | same tests (`tests/evals`) | yes |
| 12 | Extra trace destinations | deploy secret `LANGSMITH_RUNS_ENDPOINTS` | same env var (optional, `.env`) | yes |

Agent Chat UI doesn't call 9 or 10 (it has no feedback buttons). The routes are there for a custom UI, or use LangSmith's
own trace view to annotate runs.

## Check it

```bash
uv run python scripts/check_tracing.py               # 1 question; verifies 1, 2, 3, 5 and tool / token capture
uv run python scripts/check_tracing.py --off-topic   # also: guardrails blocks and records it
uv run python scripts/check_tracing.py --feedback    # also: leaves a ux.thumb_vote = 1 (9)
uv run pytest tests/evals                             # 11 (logs a test run to LangSmith)
```

## Where to look in LangSmith

* **Tracing -> docs-agent-lab**: one row per question. Filter by tag `docs_agent`, or by metadata
  `LANGSMITH_AGENT_VERSION` to compare code versions, or `guardrails_result = BLOCKED`.
* Open a run: the guardrails node (with its decision), each model call (Qwen, tokens, cost), each tool call
  (`search_docs_*`, `query_docs_filesystem_*`, `check_links`).
* **Prompts**: `docs-agent-lab`, `docs-agent-lab-guardrails` (after `push_prompts.py`).
* **Datasets & Experiments**: `docs-agent-lab-guardrails-samples` (if sampling is on), eval test runs.
