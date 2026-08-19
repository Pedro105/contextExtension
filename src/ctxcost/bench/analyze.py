"""Aggregate raw request records (and, optionally, a server metrics time series) from
one load-test run into tidy per-request and run-summary tables.

This is the only place statistics get computed. client.py and metrics.py both keep
raw, unaggregated records for exactly this reason -- collapsing to percentiles at
collection time would make the choice of statistic irreversible.

Clock note: client.py timestamps are `time.monotonic()`-based (correct for durations,
immune to wall-clock adjustments mid-run). metrics.py's Prometheus poll timestamps are
`time.time()`-based (wall clock, since that's what Prometheus scraping gives you).
Joining the two needs a conversion, done once via `clock_offset()` sampled at the same
instant by whatever orchestrates a run (scripts/serve_sweep.py) and passed in here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from .client import RequestRecord
from .metrics import MetricsTimeSeries


def clock_offset() -> float:
    """Sample once, at the same instant, to later convert a `time.monotonic()`
    timestamp into its `time.time()` (wall-clock) equivalent: wall_ts = mono_ts +
    clock_offset()."""
    return time.time() - time.monotonic()


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    n_requests: int
    n_errors: int
    n_warmup_excluded: int
    duration_s: float
    output_tokens_per_s: float
    requests_per_s: float
    ttft_p50_s: float
    ttft_p90_s: float
    ttft_p99_s: float
    e2e_p50_s: float
    e2e_p90_s: float
    e2e_p99_s: float
    itl_p50_s: float
    itl_p90_s: float
    itl_p99_s: float


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def records_to_dataframe(records: list[RequestRecord]) -> pd.DataFrame:
    """One row per request -- the per-request half of the tidy output. Kept even for
    warmup and errored requests; `is_warmup` and `error` mark them for filtering
    downstream instead of dropping data at generation time."""
    rows = []
    for r in records:
        ttft_s = (
            (r.first_token_ts - r.actual_send_ts)
            if r.first_token_ts is not None and r.actual_send_ts is not None
            else None
        )
        e2e_s = (
            (r.completion_ts - r.actual_send_ts)
            if r.completion_ts is not None and r.actual_send_ts is not None
            else None
        )
        coordinated_omission_delay_s = (
            (r.actual_send_ts - r.intended_arrival_ts) if r.actual_send_ts is not None else None
        )
        itl_values = [b - a for a, b in pairwise(r.token_arrival_ts)]
        mean_itl_s = float(np.mean(itl_values)) if itl_values else None
        rows.append(
            {
                "request_id": r.request_id,
                "is_warmup": r.is_warmup,
                "error": r.error,
                "intended_arrival_ts": r.intended_arrival_ts,
                "actual_send_ts": r.actual_send_ts,
                "first_token_ts": r.first_token_ts,
                "completion_ts": r.completion_ts,
                "prompt_tokens": r.prompt_tokens,
                "output_tokens": r.output_tokens,
                "ttft_s": ttft_s,
                "e2e_s": e2e_s,
                "coordinated_omission_delay_s": coordinated_omission_delay_s,
                "mean_inter_token_latency_s": mean_itl_s,
                "n_tokens_streamed": len(r.token_arrival_ts),
            }
        )
    columns = [
        "request_id",
        "is_warmup",
        "error",
        "intended_arrival_ts",
        "actual_send_ts",
        "first_token_ts",
        "completion_ts",
        "prompt_tokens",
        "output_tokens",
        "ttft_s",
        "e2e_s",
        "coordinated_omission_delay_s",
        "mean_inter_token_latency_s",
        "n_tokens_streamed",
    ]
    return pd.DataFrame(rows, columns=columns)


def aggregate_run(records: list[RequestRecord], run_id: str) -> RunSummary:
    """Compute the one-row run summary: throughput + latency percentiles over the
    non-warmup, non-errored requests."""
    n_warmup = sum(1 for r in records if r.is_warmup)
    stats_records = [r for r in records if not r.is_warmup and r.error is None]
    n_errors = sum(1 for r in records if not r.is_warmup and r.error is not None)

    send_ts = [r.actual_send_ts for r in stats_records if r.actual_send_ts is not None]
    completion_ts = [r.completion_ts for r in stats_records if r.completion_ts is not None]
    duration_s = (max(completion_ts) - min(send_ts)) if send_ts and completion_ts else float("nan")

    total_output_tokens = sum(r.output_tokens or 0 for r in stats_records)
    output_tokens_per_s = total_output_tokens / duration_s if duration_s and duration_s > 0 else float("nan")
    requests_per_s = len(stats_records) / duration_s if duration_s and duration_s > 0 else float("nan")

    ttft = [
        r.first_token_ts - r.actual_send_ts
        for r in stats_records
        if r.first_token_ts is not None and r.actual_send_ts is not None
    ]
    e2e = [
        r.completion_ts - r.actual_send_ts
        for r in stats_records
        if r.completion_ts is not None and r.actual_send_ts is not None
    ]
    # Inter-token latency is pooled across all requests: every consecutive-token gap
    # is one sample, not one sample per request, so a single slow request can't be
    # outvoted by many fast short ones in the percentile.
    itl = [b - a for r in stats_records for a, b in pairwise(r.token_arrival_ts)]

    return RunSummary(
        run_id=run_id,
        n_requests=len(records),
        n_errors=n_errors,
        n_warmup_excluded=n_warmup,
        duration_s=duration_s,
        output_tokens_per_s=output_tokens_per_s,
        requests_per_s=requests_per_s,
        ttft_p50_s=_percentile(ttft, 50),
        ttft_p90_s=_percentile(ttft, 90),
        ttft_p99_s=_percentile(ttft, 99),
        e2e_p50_s=_percentile(e2e, 50),
        e2e_p90_s=_percentile(e2e, 90),
        e2e_p99_s=_percentile(e2e, 99),
        itl_p50_s=_percentile(itl, 50),
        itl_p90_s=_percentile(itl, 90),
        itl_p99_s=_percentile(itl, 99),
    )


def join_with_metrics(
    requests_df: pd.DataFrame, metrics: MetricsTimeSeries, clock_offset_s: float
) -> pd.DataFrame:
    """Attach the nearest-preceding server metrics snapshot (by wall-clock time) to
    each request row, keyed on `completion_ts` -- what was the scheduler/cache state
    doing right as this request finished.

    Adds one column per metric name actually present in `metrics` (discovered, not
    assumed) plus `metrics_poll_ts` recording which snapshot was matched.
    """
    if not metrics.snapshots or requests_df.empty:
        return requests_df.copy()

    snap_wall_ts = [s.poll_ts for s in metrics.snapshots]
    metric_names = sorted(metrics.metric_names())
    snap_values: list[dict[str, float]] = []
    for snap in metrics.snapshots:
        by_name: dict[str, float] = {}
        for s in snap.samples:
            by_name[s.name] = by_name.get(s.name, 0.0) + s.value
        snap_values.append(by_name)

    out = requests_df.copy()
    matched_idx = []
    for completion_ts in out["completion_ts"]:
        if completion_ts is None or (isinstance(completion_ts, float) and np.isnan(completion_ts)):
            matched_idx.append(None)
            continue
        wall_ts = completion_ts + clock_offset_s
        # index of the latest snapshot at or before wall_ts; falls back to the first
        # snapshot if the request completed before any scrape landed.
        idx = None
        for i, poll_ts in enumerate(snap_wall_ts):
            if poll_ts <= wall_ts:
                idx = i
            else:
                break
        matched_idx.append(idx if idx is not None else 0)

    out["metrics_poll_ts"] = [snap_wall_ts[i] if i is not None else None for i in matched_idx]
    for name in metric_names:
        out[name] = [snap_values[i].get(name) if i is not None else None for i in matched_idx]
    return out


def write_run_outputs(
    records: list[RequestRecord],
    run_id: str,
    out_dir: str | Path,
    metrics: MetricsTimeSeries | None = None,
    clock_offset_s: float | None = None,
) -> tuple[Path, Path]:
    """Write `<run_id>_requests.csv` (one row per request, joined with server metrics
    if provided) and append the run's one summary row to `summary.csv` in `out_dir`
    (creating it with a header on first write). Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requests_df = records_to_dataframe(records)
    if metrics is not None and clock_offset_s is not None:
        requests_df = join_with_metrics(requests_df, metrics, clock_offset_s)
    requests_path = out_dir / f"{run_id}_requests.csv"
    requests_df.to_csv(requests_path, index=False)

    summary = aggregate_run(records, run_id)
    summary_path = out_dir / "summary.csv"
    summary_row = pd.DataFrame([summary.__dict__])
    header = not summary_path.exists()
    summary_row.to_csv(summary_path, mode="a", index=False, header=header)

    return requests_path, summary_path
