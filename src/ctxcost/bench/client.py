"""Async load generator against any OpenAI-compatible completions endpoint.

Talks to `base_url` over HTTP only -- it has no idea whether that's a vLLM process on
this machine, a CPU container, a Colab GPU tunnel, or a rented multi-GPU node, and
must never be given a reason to care. Every other part of this module exists to
support that one constraint plus honest latency measurement:

- Streaming (`/v1/completions`, not `/v1/chat/completions` -- a chat template would
  silently change the prompt's token count out from under `workload.py`'s exactness
  guarantee) so time-to-first-token and inter-token gaps are directly observable
  rather than inferred from a single end-to-end duration.
- Both `intended_arrival_ts` and `actual_send_ts` are recorded for every request, in
  both load modes, so a gap between them -- coordinated omission -- shows up in the
  data instead of silently depressing the tail-latency numbers. In closed-loop mode,
  "intended" is when a worker became free to send its next request; in open-loop mode,
  it's the Poisson-scheduled arrival time. Either can slip behind "actual" under
  contention (connection-pool limits, event-loop scheduling delay), and that slip is
  exactly the thing worth catching.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx


@dataclass
class BenchRequest:
    """One request to send, already built to spec by workload.py."""

    prompt: str
    max_tokens: int
    ignore_eos: bool = True
    request_id: str | None = None
    prompt_tokens: int | None = None  # from workload's own count, for cross-checking


@dataclass
class RequestRecord:
    """Everything observed about one request. Never discarded, never pre-aggregated --
    analyze.py is the only place statistics get computed, and it does so from these."""

    request_id: str
    intended_arrival_ts: float
    actual_send_ts: float | None = None
    first_token_ts: float | None = None
    completion_ts: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    token_arrival_ts: list[float] = field(default_factory=list)
    is_warmup: bool = False
    error: str | None = None


DEFAULT_TIMEOUT_S = 300.0


async def _send_and_record(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    req: BenchRequest,
    intended_arrival_ts: float,
    is_warmup: bool,
) -> RequestRecord:
    record = RequestRecord(
        request_id=req.request_id or "",
        intended_arrival_ts=intended_arrival_ts,
        is_warmup=is_warmup,
    )
    payload = {
        "model": model,
        "prompt": req.prompt,
        "max_tokens": req.max_tokens,
        "ignore_eos": req.ignore_eos,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    record.actual_send_ts = time.monotonic()
    output_tokens_seen = 0
    try:
        async with client.stream("POST", f"{base_url.rstrip('/')}/v1/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                now = time.monotonic()
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    if record.first_token_ts is None:
                        record.first_token_ts = now
                    record.token_arrival_ts.append(now)
                    output_tokens_seen += 1
                usage = chunk.get("usage")
                if usage:
                    record.prompt_tokens = usage.get("prompt_tokens")
                    record.output_tokens = usage.get("completion_tokens")
        record.completion_ts = time.monotonic()
        if record.output_tokens is None:
            record.output_tokens = output_tokens_seen
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        # One bad request shouldn't kill the run -- record the failure and move on;
        # analyze.py can report error rates instead of the whole sweep cell vanishing.
        record.error = repr(exc)
        record.completion_ts = time.monotonic()
    return record


def _request_id(req: BenchRequest, index: int) -> str:
    return req.request_id if req.request_id is not None else f"req-{index}"


async def closed_loop(
    base_url: str,
    model: str,
    requests: Sequence[BenchRequest],
    concurrency: int,
    warmup: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: httpx.BaseTransport | None = None,
) -> list[RequestRecord]:
    """N workers, each: send, await full response, immediately send the next.

    Concurrency is held constant at `concurrency` for the duration of the run. The
    first `warmup` requests *drawn from the queue* (not per-worker) are tagged
    `is_warmup=True` for analyze.py to exclude from statistics -- they're still
    returned, never dropped.

    `transport` is a pure testability seam (e.g. `httpx.ASGITransport` pointed at an
    in-process mock server) -- production callers never pass it, and `base_url` is
    still what routes every request; only the socket layer underneath changes.
    """
    queue: asyncio.Queue[tuple[int, BenchRequest]] = asyncio.Queue()
    for i, req in enumerate(requests):
        queue.put_nowait((i, req))
    records: list[RequestRecord | None] = [None] * len(requests)

    async def worker(client: httpx.AsyncClient) -> None:
        ready_ts = time.monotonic()  # when this worker became free to send next
        while True:
            try:
                i, req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            record = await _send_and_record(
                client, base_url, model, req, intended_arrival_ts=ready_ts, is_warmup=i < warmup
            )
            record.request_id = _request_id(req, i)
            records[i] = record
            ready_ts = time.monotonic()

    async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
        await asyncio.gather(*(worker(client) for _ in range(max(1, concurrency))))

    return [r for r in records if r is not None]


async def open_loop(
    base_url: str,
    model: str,
    requests: Sequence[BenchRequest],
    rate: float,
    warmup: int = 0,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: httpx.BaseTransport | None = None,
) -> list[RequestRecord]:
    """Poisson arrivals at `rate` req/s, independent of completion.

    A new request is dispatched at each scheduled arrival regardless of how many
    prior requests are still in flight -- the queue of in-flight requests can grow
    unboundedly if the server can't keep up. That's the point: this mode exists to
    find the rate at which a server (or an autoscaler reacting to it) falls over,
    which closed-loop's self-throttling concurrency cap can never expose.

    `transport` is a pure testability seam, as in `closed_loop`.
    """
    rng = random.Random(seed)
    n = len(requests)
    records: list[RequestRecord | None] = [None] * n
    start = time.monotonic()

    async def dispatch(client: httpx.AsyncClient, i: int, req: BenchRequest, intended_ts: float) -> None:
        record = await _send_and_record(
            client, base_url, model, req, intended_arrival_ts=intended_ts, is_warmup=i < warmup
        )
        record.request_id = _request_id(req, i)
        records[i] = record

    async with httpx.AsyncClient(timeout=timeout_s, transport=transport) as client:
        tasks = []
        t = 0.0
        for i, req in enumerate(requests):
            t += rng.expovariate(rate)
            intended_ts = start + t
            now = time.monotonic()
            if intended_ts > now:
                await asyncio.sleep(intended_ts - now)
            tasks.append(asyncio.create_task(dispatch(client, i, req, intended_ts)))
        await asyncio.gather(*tasks)

    return [r for r in records if r is not None]
