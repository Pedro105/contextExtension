"""metrics.py: Prometheus text parsing (against a captured real /metrics fixture) and
the polling loop (against the mock server, offline)."""

import asyncio

import httpx
import pytest

from ctxcost.bench.metrics import (
    MetricsTimeSeries,
    parse_prometheus_text,
    poll_metrics,
    scrape_once,
)
from mock_vllm_server import make_mock_vllm_app

FIXTURE = "tests/fixtures/vllm_metrics/vllm_0_27_1_metrics.txt"


def _load_fixture() -> str:
    with open(FIXTURE) as f:
        return f.read()


def test_parses_gauges_counters_and_histogram_buckets():
    samples = parse_prometheus_text(_load_fixture())
    names = {s.name for s in samples}
    # confirmed-real vLLM 0.27.1 names -- not the possibly-stale names this project's
    # spec assumed (vllm:gpu_cache_usage_perc, vllm:prefix_cache_hit_rate)
    assert "vllm:num_requests_running" in names
    assert "vllm:num_requests_waiting" in names
    assert "vllm:kv_cache_usage_perc" in names
    assert "vllm:gpu_cache_usage_perc" not in names  # renamed away in this version
    assert "vllm:prefix_cache_queries" in names
    assert "vllm:prefix_cache_hits" in names
    assert "vllm:prefix_cache_hit_rate" not in names  # never existed as one metric
    assert "vllm:time_to_first_token_seconds_bucket" in names


def test_parses_labels_correctly():
    samples = parse_prometheus_text(_load_fixture())
    running = [s for s in samples if s.name == "vllm:num_requests_running"]
    assert len(running) == 1
    assert running[0].labels == {"model_name": "HuggingFaceTB/SmolLM2-1.7B"}
    assert running[0].value == 12.0


def test_ignores_help_and_type_comments():
    samples = parse_prometheus_text(_load_fixture())
    assert all(not s.name.startswith("#") for s in samples)


def test_handles_inf_bucket_value():
    samples = parse_prometheus_text(_load_fixture())
    inf_bucket = [
        s
        for s in samples
        if s.name == "vllm:time_to_first_token_seconds_bucket" and s.labels.get("le") == "+Inf"
    ]
    assert len(inf_bucket) == 1
    assert inf_bucket[0].value == 55.0


def test_malformed_lines_are_skipped_not_fatal():
    text = "not a valid prometheus line\nvllm:num_requests_running 5.0\n### junk ###"
    samples = parse_prometheus_text(text)
    assert len(samples) == 1
    assert samples[0].name == "vllm:num_requests_running"
    assert samples[0].value == 5.0


def test_time_series_metric_names_and_values_for():
    series = MetricsTimeSeries()
    from ctxcost.bench.metrics import MetricsSnapshot

    series.snapshots.append(MetricsSnapshot(poll_ts=1.0, samples=tuple(parse_prometheus_text(_load_fixture()))))
    series.snapshots.append(MetricsSnapshot(poll_ts=2.0, samples=tuple(parse_prometheus_text(_load_fixture()))))
    assert "vllm:num_requests_running" in series.metric_names()
    values = series.values_for("vllm:num_requests_running")
    assert values == [(1.0, 12.0), (2.0, 12.0)]


def test_to_dataframe_shape():
    series = MetricsTimeSeries()
    from ctxcost.bench.metrics import MetricsSnapshot

    samples = parse_prometheus_text(_load_fixture())
    series.snapshots.append(MetricsSnapshot(poll_ts=1.0, samples=tuple(samples)))
    df = series.to_dataframe()
    assert len(df) == len(samples)
    assert list(df.columns) == ["poll_ts", "metric_name", "labels", "value"]


@pytest.mark.anyio
async def test_scrape_once_against_mock_server():
    app, _ = make_mock_vllm_app(metrics_text=_load_fixture())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        snap = await scrape_once(client, "http://mock/metrics")
    assert len(snap.samples) > 0
    assert any(s.name == "vllm:num_requests_running" for s in snap.samples)


@pytest.mark.anyio
async def test_poll_metrics_collects_multiple_snapshots():
    app, _ = make_mock_vllm_app(metrics_text=_load_fixture())
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://mock")
    stop_event = asyncio.Event()

    async def stop_after(delay):
        await asyncio.sleep(delay)
        stop_event.set()

    poll_task = asyncio.create_task(poll_metrics("http://mock/metrics", interval_s=0.05, stop_event=stop_event, client=client))
    await stop_after(0.22)
    series = await poll_task
    await client.aclose()

    assert len(series.snapshots) >= 3  # ~0.22s / 0.05s interval


@pytest.fixture
def anyio_backend():
    return "asyncio"
