"""A minimal ASGI app mimicking just enough of vLLM's OpenAI-compatible server
(/health, /metrics, streaming /v1/completions) to exercise src/ctxcost/bench/ without
a real server, a real model, or a GPU.

Used via `httpx.ASGITransport(app=...)`, which drives real HTTP request/response
encoding (headers, chunked streaming body) without opening a socket -- the client
code under test can't tell the difference from a real server except that there's no
network involved.
"""

from __future__ import annotations

import asyncio
import json


async def _read_body(receive) -> bytes:
    body = b""
    more_body = True
    while more_body:
        message = await receive()
        body += message.get("body", b"")
        more_body = message.get("more_body", False)
    return body


def make_mock_vllm_app(
    token_delay_s: float = 0.0, metrics_text: str = "", fail_every: int = 0, served_model_id: str | None = None
):
    """Build a fresh mock app + its request log.

    `token_delay_s`: sleep between streamed token chunks, to produce measurable,
    non-zero inter-token gaps in tests.
    `metrics_text`: raw body served at /metrics.
    `fail_every`: if > 0, every Nth request (1-indexed) returns a 500 instead of
    streaming, to exercise client.py's per-request error handling.
    `served_model_id`: if set, GET /v1/models reports this as the one served model
    id (real vLLM's response shape), for testing external-mode model verification.

    Returns (app, request_log) -- request_log is a list this app appends each
    decoded /v1/completions request body to, so tests can assert on what was sent.
    """
    request_log: list[dict] = []

    async def app(scope, receive, send):
        assert scope["type"] == "http"
        method, path = scope["method"], scope["path"]

        if method == "GET" and path == "/health":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"OK"})
            return

        if method == "GET" and path == "/v1/models" and served_model_id is not None:
            body = json.dumps({"data": [{"id": served_model_id, "object": "model"}]}).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return

        if method == "GET" and path == "/metrics":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/plain; version=0.0.4")],
                }
            )
            await send({"type": "http.response.body", "body": metrics_text.encode()})
            return

        if method == "POST" and path == "/v1/completions":
            raw = await _read_body(receive)
            payload = json.loads(raw)
            request_log.append(payload)

            if fail_every and len(request_log) % fail_every == 0:
                await send({"type": "http.response.start", "status": 500, "headers": []})
                await send({"type": "http.response.body", "body": b"synthetic failure"})
                return

            max_tokens = payload.get("max_tokens", 1)
            prompt = payload.get("prompt", "")
            prompt_tokens = max(1, len(str(prompt).split()))
            include_usage = payload.get("stream_options", {}).get("include_usage", False)

            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                }
            )
            for i in range(max_tokens):
                if token_delay_s:
                    await asyncio.sleep(token_delay_s)
                chunk = {"choices": [{"text": f" tok{i}"}]}
                await send(
                    {"type": "http.response.body", "body": f"data: {json.dumps(chunk)}\n\n".encode(), "more_body": True}
                )
            if include_usage:
                usage_chunk = {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens}}
                await send(
                    {
                        "type": "http.response.body",
                        "body": f"data: {json.dumps(usage_chunk)}\n\n".encode(),
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
            return

        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b"not found"})

    return app, request_log
