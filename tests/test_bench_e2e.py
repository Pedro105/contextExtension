"""End-to-end smoke test: workload -> client -> metrics -> analyze, wired together
against the mock OpenAI-compatible server. No GPU, no real model, no network --
proves the harness works as a pipeline, not just as isolated units."""

import asyncio

import httpx
import pytest

from ctxcost.bench.analyze import (
    aggregate_run,
    clock_offset,
    join_with_metrics,
    records_to_dataframe,
    write_run_outputs,
)
from ctxcost.bench.client import BenchRequest, closed_loop
from ctxcost.bench.metrics import poll_metrics
from ctxcost.bench.workload import WorkloadConfig, WorkloadGenerator
from fake_tokenizer import WordTokenizer
from mock_vllm_server import make_mock_vllm_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_full_pipeline_smoke(tmp_path):
    metrics_text = (
        '# HELP vllm:num_requests_running x\n# TYPE vllm:num_requests_running gauge\n'
        'vllm:num_requests_running{model_name="m"} 1.0\n'
    )
    app, request_log = make_mock_vllm_app(token_delay_s=0.005, metrics_text=metrics_text)
    transport = httpx.ASGITransport(app=app)

    # 1. workload: build a handful of exact-length, distinct-prefix prompts
    tokenizer = WordTokenizer()
    wc = WorkloadConfig(target_prompt_tokens=40, max_tokens=5, prefix_tokens=8, seed=0)
    generator = WorkloadGenerator(tokenizer=tokenizer, config=wc)
    prompts = generator.generate(6)
    assert all(p.prompt_tokens == 40 for p in prompts)
    assert len({p.token_ids[:8] for p in prompts}) == 6  # distinct prefixes

    requests = [
        BenchRequest(prompt=p.text, max_tokens=wc.max_tokens, ignore_eos=wc.ignore_eos, request_id=f"req-{i}")
        for i, p in enumerate(prompts)
    ]

    # 2. client + metrics run concurrently, as a real sweep cell would do it
    stop_event = asyncio.Event()
    metrics_client = httpx.AsyncClient(transport=transport, base_url="http://mock")
    metrics_task = asyncio.create_task(
        poll_metrics("http://mock/metrics", interval_s=0.01, stop_event=stop_event, client=metrics_client)
    )
    offset = clock_offset()
    records = await closed_loop("http://mock", "test-model", requests, concurrency=2, warmup=1, transport=transport)
    stop_event.set()
    metrics_series = await metrics_task
    await metrics_client.aclose()

    assert len(records) == 6
    assert len(request_log) == 6
    assert len(metrics_series.snapshots) >= 1

    # 3. analyze: aggregate + join + write
    summary = aggregate_run(records, run_id="e2e-smoke")
    assert summary.n_requests == 6
    assert summary.n_warmup_excluded == 1
    assert summary.n_errors == 0
    assert summary.output_tokens_per_s > 0
    assert summary.ttft_p50_s > 0

    requests_df = records_to_dataframe(records)
    joined = join_with_metrics(requests_df, metrics_series, clock_offset_s=offset)
    assert "vllm:num_requests_running" in joined.columns

    req_path, summary_path = write_run_outputs(records, run_id="e2e-smoke", out_dir=tmp_path, metrics=metrics_series, clock_offset_s=offset)
    assert req_path.exists()
    assert summary_path.exists()


@pytest.mark.anyio
async def test_shared_prefix_confound_is_measurable():
    """With enable_shared_prefix=True, every request's prompt shares an identical
    leading block -- the harness must be able to represent this deliberately, not
    just avoid it."""
    app, _request_log = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)

    tokenizer = WordTokenizer()
    wc = WorkloadConfig(target_prompt_tokens=30, max_tokens=3, prefix_tokens=10, enable_shared_prefix=True, seed=5)
    generator = WorkloadGenerator(tokenizer=tokenizer, config=wc)
    prompts = generator.generate(4)
    assert len({p.token_ids[:10] for p in prompts}) == 1  # confound present on purpose

    requests = [BenchRequest(prompt=p.text, max_tokens=wc.max_tokens, request_id=f"r{i}") for i, p in enumerate(prompts)]
    records = await closed_loop("http://mock", "m", requests, concurrency=2, transport=transport)
    assert len(records) == 4
    assert all(r.error is None for r in records)
