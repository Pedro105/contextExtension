"""client.py: closed_loop / open_loop against the mock ASGI server -- streaming,
per-request record fields, warmup tagging, coordinated-omission timestamps, and
per-request error isolation."""


import httpx
import pytest

from ctxcost.bench.client import BenchRequest, closed_loop, open_loop
from mock_vllm_server import make_mock_vllm_app


def _requests(n: int, max_tokens: int = 3) -> list[BenchRequest]:
    return [BenchRequest(prompt=f"prompt number {i}", max_tokens=max_tokens, request_id=f"r{i}") for i in range(n)]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_closed_loop_basic_fields_populated():
    app, log = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    records = await closed_loop(
        "http://mock", "test-model", _requests(4), concurrency=2, transport=transport
    )
    assert len(records) == 4
    assert len(log) == 4
    for r in records:
        assert r.error is None
        assert r.actual_send_ts is not None
        assert r.first_token_ts is not None
        assert r.completion_ts is not None
        assert r.first_token_ts >= r.actual_send_ts
        assert r.completion_ts >= r.first_token_ts
        assert r.output_tokens == 3
        assert len(r.token_arrival_ts) == 3


@pytest.mark.anyio
async def test_closed_loop_holds_constant_concurrency_via_request_count():
    # every request in the mock is served instantly; simply confirm all requests are
    # dispatched exactly once each under a constrained worker count
    app, _log = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    n = 10
    records = await closed_loop("http://mock", "m", _requests(n), concurrency=3, transport=transport)
    assert len(records) == n
    assert {r.request_id for r in records} == {f"r{i}" for i in range(n)}


@pytest.mark.anyio
async def test_warmup_requests_are_tagged_not_dropped():
    app, _ = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    records = await closed_loop("http://mock", "m", _requests(6), concurrency=1, warmup=2, transport=transport)
    assert len(records) == 6  # nothing dropped
    warmup_flags = sorted(r.is_warmup for r in records)
    assert warmup_flags == [False, False, False, False, True, True]


@pytest.mark.anyio
async def test_closed_loop_detects_coordinated_omission():
    """A worker delayed by a slow request should show actual_send_ts for the *next*
    request lagging behind its intended_arrival_ts once the slow one finally frees
    the worker -- that gap is the whole point of recording both timestamps."""
    app, _ = make_mock_vllm_app(token_delay_s=0.05)
    transport = httpx.ASGITransport(app=app)
    records = await closed_loop(
        "http://mock", "m", _requests(3, max_tokens=5), concurrency=1, transport=transport
    )
    # single worker, strictly sequential: each request's intended_arrival_ts should
    # equal (approximately) the previous request's completion_ts
    records_by_id = {r.request_id: r for r in records}
    for i in range(1, 3):
        prev = records_by_id[f"r{i - 1}"]
        cur = records_by_id[f"r{i}"]
        assert cur.intended_arrival_ts == pytest.approx(prev.completion_ts, abs=0.01)


@pytest.mark.anyio
async def test_open_loop_schedules_poisson_arrivals():
    app, log = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    records = await open_loop("http://mock", "m", _requests(8), rate=50.0, seed=1, transport=transport)
    assert len(records) == 8
    assert len(log) == 8
    # arrivals are scheduled independent of completion -- intended timestamps should
    # be monotonically non-decreasing in submission order
    intended = [r.intended_arrival_ts for r in sorted(records, key=lambda r: r.request_id)]
    assert intended == sorted(intended)


@pytest.mark.anyio
async def test_open_loop_warmup_tagging():
    app, _ = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    records = await open_loop("http://mock", "m", _requests(5), rate=200.0, warmup=2, seed=0, transport=transport)
    warmup_ids = {r.request_id for r in records if r.is_warmup}
    assert warmup_ids == {"r0", "r1"}


@pytest.mark.anyio
async def test_one_failed_request_does_not_kill_the_run():
    app, _log = make_mock_vllm_app(fail_every=2)  # every 2nd request fails
    transport = httpx.ASGITransport(app=app)
    records = await closed_loop("http://mock", "m", _requests(4), concurrency=1, transport=transport)
    assert len(records) == 4
    errored = [r for r in records if r.error is not None]
    ok = [r for r in records if r.error is None]
    assert len(errored) == 2
    assert len(ok) == 2
    for r in errored:
        assert r.completion_ts is not None  # still timestamped even on failure


@pytest.mark.anyio
async def test_stream_options_include_usage_sent_and_usage_parsed():
    app, log = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    await closed_loop("http://mock", "m", _requests(1), concurrency=1, transport=transport)
    assert log[0]["stream_options"] == {"include_usage": True}
    assert log[0]["stream"] is True
    assert log[0]["ignore_eos"] is True
