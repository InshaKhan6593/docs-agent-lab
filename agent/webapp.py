"""Custom HTTP routes mounted on the LangGraph server (langgraph.json -> "http.app").

The open-source stand-in for chat-langchain's managed LangSmith connector (connectors/langsmith.py): the
browser can leave thumbs feedback on a run and read its token/cost summary or a public share link, while the
LangSmith API key stays on the server.

    POST   /langsmith/feedback               {"run_id", "score": 1|0, "comment"?}  -> {"feedback_id"}
    PATCH  /langsmith/feedback/{feedback_id} {"score"?, "comment"?}
    DELETE /langsmith/feedback/{feedback_id}
    GET    /langsmith/runs/{run_id}/summary   -> status, timing, tokens, cost, url
    POST   /langsmith/runs/{run_id}/share     -> {"url"}

Same limits as their connector: key `ux.thumb_vote`, scores 1 or 0, comments up to 2,000 characters.
Production would also check the caller's identity (they bind feedback to one vote per user).

The LangSmith client is synchronous, and `langgraph dev` rejects blocking calls on its event loop, so every
call runs in a worker thread (asyncio.to_thread).
"""

import asyncio
import logging

from langsmith import Client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

logger = logging.getLogger(__name__)

FEEDBACK_KEY = "ux.thumb_vote"
MAX_COMMENT_CHARS = 2000
SUMMARY_FIELDS = [
    "id", "status", "start_time", "end_time", "prompt_tokens", "completion_tokens", "total_tokens",
    "prompt_cost", "completion_cost", "total_cost",
]

_client: Client | None = None


def client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def _score(value) -> int:
    if value not in (0, 1, "0", "1"):
        raise ValueError("score must be 1 or 0")
    return int(value)


async def create_feedback(request: Request) -> JSONResponse:
    body = await request.json()
    try:
        fb = await asyncio.to_thread(
            client().create_feedback,
            run_id=body["run_id"],
            key=FEEDBACK_KEY,
            score=_score(body["score"]),
            comment=(body.get("comment") or "")[:MAX_COMMENT_CHARS] or None,
        )
    except (KeyError, ValueError) as err:
        return JSONResponse({"error": str(err)}, status_code=400)
    return JSONResponse({"feedback_id": str(fb.id)})


async def update_feedback(request: Request) -> JSONResponse:
    body = await request.json()
    kwargs = {}
    try:
        if "score" in body:
            kwargs["score"] = _score(body["score"])
    except ValueError as err:
        return JSONResponse({"error": str(err)}, status_code=400)
    if "comment" in body:
        kwargs["comment"] = (body["comment"] or "")[:MAX_COMMENT_CHARS]
    await asyncio.to_thread(client().update_feedback, request.path_params["feedback_id"], **kwargs)
    return JSONResponse({"ok": True})


async def delete_feedback(request: Request) -> JSONResponse:
    await asyncio.to_thread(client().delete_feedback, request.path_params["feedback_id"])
    return JSONResponse({"ok": True})


async def run_summary(request: Request) -> JSONResponse:
    run = await asyncio.to_thread(client().read_run, request.path_params["run_id"])
    data = {f: getattr(run, f, None) for f in SUMMARY_FIELDS}
    data["url"] = getattr(run, "url", None)
    return JSONResponse({k: (str(v) if v is not None and not isinstance(v, (int, float)) else v) for k, v in data.items()})


async def share_run(request: Request) -> JSONResponse:
    return JSONResponse({"url": await asyncio.to_thread(client().share_run, request.path_params["run_id"])})


async def _error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("LangSmith route %s failed", request.url.path, exc_info=exc)
    return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


app = Starlette(
    exception_handlers={Exception: _error},
    routes=[
        Route("/langsmith/feedback", create_feedback, methods=["POST"]),
        Route("/langsmith/feedback/{feedback_id}", update_feedback, methods=["PATCH"]),
        Route("/langsmith/feedback/{feedback_id}", delete_feedback, methods=["DELETE"]),
        Route("/langsmith/runs/{run_id}/summary", run_summary, methods=["GET"]),
        Route("/langsmith/runs/{run_id}/share", share_run, methods=["POST"]),
    ]
)
