"""analyze.py: aggregation against known synthetic RequestRecord inputs, the tidy
per-request + run-summary CSV output, and the client/server metrics join."""

from pathlib import Path

import pandas as pd
import pytest

from ctxcost.bench.analyze import (
    aggregate_run,
    clock_offset,
    join_with_metrics,
    records_to_dataframe,
    write_run_outputs,
)
from ctxcost.bench.client import RequestRecord
from ctxcost.bench.metrics import MetricSample, MetricsSnapshot, MetricsTimeSeries


def _record(request_id, send, first_token, completion, output_tokens, token_gaps, is_warmup=False, error=None):
    """Build a RequestRecord with token_arrival_ts spaced by `token_gaps` (list of
    inter-token deltas) starting at first_token."""
    ts = first_token
    arrivals = [ts]
    for gap in token_gaps:
        ts += gap
        arrivals.append(ts)
    return RequestRecord(
        request_id=request_id,
        intended_arrival_ts=send,
        actual_send_ts=send,
        first_token_ts=first_token if error is None else None,
        completion_ts=completion,
        prompt_tokens=100,
        output_tokens=output_tokens,
        token_arrival_ts=arrivals if error is None else [],
        is_warmup=is_warmup,
        error=error,
    )


def test_aggregate_run_known_percentiles():
    # 4 requests, TTFT = 1,2,3,4 seconds exactly (send at t=0 for all, first token at
    # t=1..4) -- percentiles over this small, known set are checkable by hand.
    records = [_record(f"r{i}", send=0.0, first_token=float(i + 1), completion=float(i + 5), output_tokens=2, token_gaps=[0.5]) for i in range(4)]
    summary = aggregate_run(records, run_id="test-run")
    assert summary.n_requests == 4
    assert summary.n_errors == 0
    assert summary.n_warmup_excluded == 0
    # p50 of [1,2,3,4] (numpy linear interpolation) == 2.5
    assert summary.ttft_p50_s == pytest.approx(2.5)
    assert summary.ttft_p99_s == pytest.approx(3.97, abs=0.05)


def test_aggregate_run_excludes_warmup_and_errors():
    good = [_record(f"g{i}", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[]) for i in range(3)]
    warm = [_record("w0", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[], is_warmup=True)]
    bad = [_record("b0", send=0.0, first_token=None, completion=2.0, output_tokens=None, token_gaps=[], error="boom")]
    summary = aggregate_run(good + warm + bad, run_id="run")
    assert summary.n_requests == 5
    assert summary.n_warmup_excluded == 1
    assert summary.n_errors == 1
    # only the 3 "good" records feed the percentiles
    assert summary.ttft_p50_s == pytest.approx(1.0)


def test_output_tokens_per_second_and_requests_per_second():
    # 2 requests, each 1 output token, send at t=0, complete at t=0 and t=2 ->
    # duration = 2s, 2 requests, 2 tokens total
    records = [
        _record("a", send=0.0, first_token=0.5, completion=0.0, output_tokens=1, token_gaps=[]),
        _record("b", send=0.0, first_token=1.5, completion=2.0, output_tokens=1, token_gaps=[]),
    ]
    summary = aggregate_run(records, run_id="run")
    assert summary.duration_s == pytest.approx(2.0)
    assert summary.output_tokens_per_s == pytest.approx(1.0)
    assert summary.requests_per_s == pytest.approx(1.0)


def test_inter_token_latency_pooled_across_requests():
    # request 1: gaps of 0.1, 0.1 (2 samples); request 2: gap of 0.5 (1 sample) ->
    # pooled itl values = [0.1, 0.1, 0.5], not one-mean-per-request
    records = [
        _record("a", send=0.0, first_token=1.0, completion=3.0, output_tokens=3, token_gaps=[0.1, 0.1]),
        _record("b", send=0.0, first_token=1.0, completion=3.0, output_tokens=2, token_gaps=[0.5]),
    ]
    summary = aggregate_run(records, run_id="run")
    # p50 of [0.1, 0.1, 0.5] == 0.1
    assert summary.itl_p50_s == pytest.approx(0.1)


def test_records_to_dataframe_row_count_and_derived_fields():
    records = [_record("a", send=0.0, first_token=1.0, completion=3.0, output_tokens=2, token_gaps=[0.3])]
    df = records_to_dataframe(records)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ttft_s"] == pytest.approx(1.0)
    assert row["e2e_s"] == pytest.approx(3.0)
    assert row["coordinated_omission_delay_s"] == pytest.approx(0.0)
    assert row["mean_inter_token_latency_s"] == pytest.approx(0.3)
    assert row["n_tokens_streamed"] == 2


def test_records_to_dataframe_keeps_errored_and_warmup_rows():
    records = [
        _record("ok", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[]),
        _record("bad", send=0.0, first_token=None, completion=2.0, output_tokens=None, token_gaps=[], error="x"),
        _record("warm", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[], is_warmup=True),
    ]
    df = records_to_dataframe(records)
    assert len(df) == 3  # nothing dropped, per the DO NOT rule


def test_join_with_metrics_matches_nearest_preceding_snapshot():
    offset = 1000.0  # pretend clock_offset: wall = mono + 1000
    records = [_record("a", send=0.0, first_token=1.0, completion=5.0, output_tokens=1, token_gaps=[])]
    df = records_to_dataframe(records)

    series = MetricsTimeSeries(
        snapshots=[
            MetricsSnapshot(poll_ts=1000.0 + 0.0, samples=(MetricSample("vllm:num_requests_running", {}, 2.0),)),
            MetricsSnapshot(poll_ts=1000.0 + 4.0, samples=(MetricSample("vllm:num_requests_running", {}, 9.0),)),
            MetricsSnapshot(poll_ts=1000.0 + 100.0, samples=(MetricSample("vllm:num_requests_running", {}, 99.0),)),
        ]
    )
    joined = join_with_metrics(df, series, clock_offset_s=offset)
    # completion_ts (mono) = 5.0 -> wall = 1005.0 -> nearest preceding snapshot is at
    # wall=1004.0 (value 9.0), not the later one at wall=1100.0
    assert joined.loc[0, "vllm:num_requests_running"] == 9.0


def test_write_run_outputs_creates_requests_and_summary_csv(tmp_path: Path):
    records = [_record("a", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[])]
    req_path, summary_path = write_run_outputs(records, run_id="cell1", out_dir=tmp_path)
    assert req_path.exists()
    assert summary_path.exists()
    req_df = pd.read_csv(req_path)
    assert len(req_df) == 1
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["run_id"] == "cell1"


def test_write_run_outputs_appends_across_multiple_runs(tmp_path: Path):
    records = [_record("a", send=0.0, first_token=1.0, completion=2.0, output_tokens=1, token_gaps=[])]
    write_run_outputs(records, run_id="cell1", out_dir=tmp_path)
    write_run_outputs(records, run_id="cell2", out_dir=tmp_path)
    summary_df = pd.read_csv(tmp_path / "summary.csv")
    assert list(summary_df["run_id"]) == ["cell1", "cell2"]


def test_clock_offset_converts_monotonic_to_wall():
    import time

    offset = clock_offset()
    mono = time.monotonic()
    wall_approx = mono + offset
    assert wall_approx == pytest.approx(time.time(), abs=0.05)
