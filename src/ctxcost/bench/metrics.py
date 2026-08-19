"""Prometheus /metrics scraping: poll vLLM's exporter during a run, keep the whole
time series.

Only the aggregate throughput numbers come out of client.py's own timestamps -- the
*shape* of what the scheduler and KV cache were doing while those requests were in
flight only exists in the server's own metrics, sampled over time. A single
end-of-run scrape can't tell a steady-state run apart from one that thrashed for the
first half and recovered, so this stores every poll, not just the last one or an
average.

The exact metric names exposed depend on the installed vLLM version -- vLLM has
renamed and restructured its metrics across releases (e.g. the v1 engine's metrics
rework). This module does not hardcode a fixed set of names to look for; it parses
whatever the endpoint actually returns and lets callers query by name. See
`results/vllm_metrics_report.md` (written by verifying against a real running server)
for the names actually confirmed for this project's pinned vLLM version.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

import httpx
import pandas as pd

# Prometheus text exposition format: `name{label="value",...} number [timestamp]`.
# Labels are optional; so is the trailing sample timestamp (we stamp our own poll
# time instead, since that's what we actually want to correlate against client-side
# request timestamps).
_SAMPLE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


@dataclass(frozen=True)
class MetricSample:
    name: str
    labels: dict[str, str]
    value: float


@dataclass(frozen=True)
class MetricsSnapshot:
    poll_ts: float
    samples: tuple[MetricSample, ...]


@dataclass
class MetricsTimeSeries:
    snapshots: list[MetricsSnapshot] = field(default_factory=list)

    def metric_names(self) -> set[str]:
        """Every distinct metric name seen across all snapshots -- i.e. what this
        vLLM instance actually exposes, discovered rather than assumed."""
        return {s.name for snap in self.snapshots for s in snap.samples}

    def values_for(self, metric_name: str) -> list[tuple[float, float]]:
        """(poll_ts, value) pairs for `metric_name`, summed across label sets at each
        poll (e.g. per-model or per-finish-reason labels collapse to one series)."""
        out = []
        for snap in self.snapshots:
            total = sum(s.value for s in snap.samples if s.name == metric_name)
            matched = any(s.name == metric_name for s in snap.samples)
            if matched:
                out.append((snap.poll_ts, total))
        return out

    def to_dataframe(self) -> pd.DataFrame:
        """One row per (poll_ts, metric_name, label set, value) -- the full time
        series, unaggregated, so shape-over-time analysis has everything to work with."""
        rows = []
        for snap in self.snapshots:
            for s in snap.samples:
                rows.append(
                    {
                        "poll_ts": snap.poll_ts,
                        "metric_name": s.name,
                        "labels": ",".join(f"{k}={v}" for k, v in sorted(s.labels.items())),
                        "value": s.value,
                    }
                )
        return pd.DataFrame(rows, columns=["poll_ts", "metric_name", "labels", "value"])


def parse_prometheus_text(text: str) -> list[MetricSample]:
    """Parse a raw Prometheus exposition-format response body into samples.

    Ignores `# HELP` / `# TYPE` comment lines and blank lines; anything else that
    doesn't match the sample grammar is skipped rather than raising, since exporters
    occasionally emit metric families this project has no use for and a formatting
    quirk there shouldn't take down the whole scrape.
    """
    samples = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        name, label_block, raw_value = m.group(1), m.group(2), m.group(3)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        labels = dict(_LABEL_RE.findall(label_block)) if label_block else {}
        samples.append(MetricSample(name=name, labels=labels, value=value))
    return samples


async def scrape_once(client: httpx.AsyncClient, metrics_url: str) -> MetricsSnapshot:
    resp = await client.get(metrics_url)
    resp.raise_for_status()
    return MetricsSnapshot(poll_ts=time.time(), samples=tuple(parse_prometheus_text(resp.text)))


async def poll_metrics(
    metrics_url: str,
    interval_s: float,
    stop_event: asyncio.Event,
    client: httpx.AsyncClient | None = None,
) -> MetricsTimeSeries:
    """Scrape `metrics_url` every `interval_s` until `stop_event` is set.

    Run this as a background task alongside a load-test coroutine; set `stop_event`
    once the load test finishes. A transient scrape failure (server briefly
    unreachable) is logged into the series as a gap rather than aborting the poll --
    one missed scrape shouldn't cost the rest of the run's time series.
    """
    series = MetricsTimeSeries()
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(timeout=10.0)
    try:
        while not stop_event.is_set():
            try:
                series.snapshots.append(await scrape_once(client, metrics_url))
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            except TimeoutError:
                pass
    finally:
        if owns_client:
            await client.aclose()
    return series
