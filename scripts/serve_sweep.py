#!/usr/bin/env python3
"""Drive a serving benchmark sweep from a config YAML (configs/bench/*.yaml).

For each (model, context_length, concurrency) cell: make sure a server is up with
the right flags, warm up, run the load test, scrape metrics for the duration, and
write results -- traceable to a commit/config/machine via run_manifest.

`server.mode: external` in the config means "a server is already running somewhere
(local CPU, a Colab GPU tunnel, a rented node) at this base_url" and this script
never touches process management -- it only ever talks HTTP, same as
src/ctxcost/bench/client.py itself. `server.mode: docker` is a convenience for local
runs: this script launches/stops a container per (model, context_length) group,
reusing it across that group's concurrency levels rather than restarting per cell.

Resumable: a cell is skipped if its `manifest.json` already exists under the output
directory, so an interrupted sweep can be re-run with the same --out-dir cheaply.

Example:
    python scripts/serve_sweep.py --config configs/bench/smoke.yaml
    python scripts/serve_sweep.py --config configs/bench/baseline.yaml \\
        --out-dir results/serving/20260101-120000  # resume a specific run
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ctxcost.bench import (
    BenchRequest,
    WorkloadConfig,
    WorkloadGenerator,
    clock_offset,
    closed_loop,
    poll_metrics,
    write_run_outputs,
)
from ctxcost.run_manifest import write_manifest


@dataclass(frozen=True)
class ServerConfig:
    mode: str  # "docker" | "external"
    base_url: str | None = None
    image: str | None = None
    gpu: bool = False
    host_port: int = 8000
    startup_timeout_s: float = 180.0
    extra_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("docker", "external"):
            raise ValueError(f"server.mode must be 'docker' or 'external', got {self.mode!r}")
        if self.mode == "external" and not self.base_url:
            raise ValueError("server.mode: external requires server.base_url")
        if self.mode == "docker" and not self.image:
            raise ValueError("server.mode: docker requires server.image")


@dataclass(frozen=True)
class WorkloadSweepConfig:
    n_requests_per_cell: int
    max_output_tokens: int
    prefix_tokens: int = 64
    enable_shared_prefix: bool = False
    warmup_requests: int = 0
    seed: int = 0


@dataclass(frozen=True)
class SweepConfig:
    server: ServerConfig
    models: tuple[dict, ...]
    context_lengths: tuple[int, ...]
    concurrency_levels: tuple[int, ...]
    workload: WorkloadSweepConfig
    metrics_poll_interval_s: float = 1.0
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            server=ServerConfig(**raw["server"]),
            models=tuple(raw["models"]),
            context_lengths=tuple(raw["context_lengths"]),
            concurrency_levels=tuple(raw["concurrency_levels"]),
            workload=WorkloadSweepConfig(**raw["workload"]),
            metrics_poll_interval_s=raw.get("metrics_poll_interval_s", 1.0),
            gpu_memory_utilization=raw.get("gpu_memory_utilization", 0.9),
            enable_prefix_caching=raw.get("enable_prefix_caching", False),
        )


class SimpleTokenizer:
    """Fallback duck-typed tokenizer used only if `transformers` isn't installed in
    the current environment -- real sweeps should have it (it's a base dependency);
    this exists so --dry-run and config validation work without the multi-hundred-MB
    download some dev machines won't have pulled yet."""

    def encode(self, text: str) -> list[int]:
        return [hash(w) % 50000 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in ids)


def _load_tokenizer(hf_model_id: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(hf_model_id)
    except ImportError:
        print(f"WARNING: transformers not installed; using a fake tokenizer for {hf_model_id}", file=sys.stderr)
        return SimpleTokenizer()


def _docker_run(
    image: str,
    hf_model_id: str,
    max_model_len: int,
    server: ServerConfig,
    gpu_memory_utilization: float,
    enable_prefix_caching: bool,
) -> str:
    name = f"ctxcost-serve-{uuid.uuid4().hex[:8]}"
    env = {
        "MODEL": hf_model_id,
        "MAX_MODEL_LEN": str(max_model_len),
        "GPU_MEMORY_UTILIZATION": str(gpu_memory_utilization),
        "ENABLE_PREFIX_CACHING": "true" if enable_prefix_caching else "false",
        **server.extra_env,
    }
    cmd = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{server.host_port}:8000"]
    if server.gpu:
        cmd += ["--gpus", "all"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return name


def _docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


async def _wait_healthy(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url.rstrip('/')}/health")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(f"server at {base_url} did not become healthy within {timeout_s}s")


def _cell_id(model_slug: str, ctx_len: int, concurrency: int) -> str:
    return f"{model_slug}__ctx{ctx_len}__conc{concurrency}"


async def _run_cell(
    base_url: str,
    hf_model_id: str,
    tokenizer,
    ctx_len: int,
    concurrency: int,
    cfg: SweepConfig,
    cell_dir: Path,
) -> None:
    wc = WorkloadConfig(
        target_prompt_tokens=ctx_len,
        max_tokens=cfg.workload.max_output_tokens,
        prefix_tokens=cfg.workload.prefix_tokens,
        enable_shared_prefix=cfg.workload.enable_shared_prefix,
        ignore_eos=True,
        seed=cfg.workload.seed,
    )
    generator = WorkloadGenerator(tokenizer=tokenizer, config=wc)
    prompts = generator.generate(cfg.workload.n_requests_per_cell)
    requests = [
        BenchRequest(prompt=p.text, max_tokens=wc.max_tokens, ignore_eos=wc.ignore_eos, request_id=f"req-{i}")
        for i, p in enumerate(prompts)
    ]

    stop_event = asyncio.Event()
    metrics_task = asyncio.create_task(
        poll_metrics(f"{base_url.rstrip('/')}/metrics", cfg.metrics_poll_interval_s, stop_event)
    )
    offset = clock_offset()
    try:
        records = await closed_loop(
            base_url, hf_model_id, requests, concurrency=concurrency, warmup=cfg.workload.warmup_requests
        )
    finally:
        stop_event.set()
        metrics = await metrics_task

    write_run_outputs(records, run_id=cell_dir.name, out_dir=cell_dir, metrics=metrics, clock_offset_s=offset)

    manifest_config = {
        "hf_model_id": hf_model_id,
        "context_length": ctx_len,
        "concurrency": concurrency,
        "workload": cfg.workload.__dict__,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "server_mode": cfg.server.mode,
        "base_url": base_url,
    }
    write_manifest(manifest_config, cell_dir / "manifest.json")


async def run_sweep(cfg: SweepConfig, out_dir: Path, dry_run: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_entry in cfg.models:
        hf_model_id = model_entry["hf_model_id"]
        model_slug = model_entry.get("served_model_name", hf_model_id).replace("/", "__")
        tokenizer = None if dry_run else _load_tokenizer(hf_model_id)

        for ctx_len in cfg.context_lengths:
            container_name = None
            base_url = cfg.server.base_url

            cells_pending = [
                c
                for c in cfg.concurrency_levels
                if not (out_dir / _cell_id(model_slug, ctx_len, c) / "manifest.json").exists()
            ]
            if not cells_pending:
                print(f"skip {model_slug} ctx={ctx_len}: all concurrency levels already have results")
                continue

            if dry_run:
                for c in cells_pending:
                    print(f"[dry-run] would run {_cell_id(model_slug, ctx_len, c)}")
                continue

            try:
                if cfg.server.mode == "docker":
                    print(f"starting container for {hf_model_id} (max_model_len={ctx_len})")
                    container_name = _docker_run(
                        cfg.server.image,
                        hf_model_id,
                        ctx_len,
                        cfg.server,
                        cfg.gpu_memory_utilization,
                        cfg.enable_prefix_caching,
                    )
                    base_url = f"http://localhost:{cfg.server.host_port}"
                    await _wait_healthy(base_url, cfg.server.startup_timeout_s)

                for concurrency in cells_pending:
                    cell_dir = out_dir / _cell_id(model_slug, ctx_len, concurrency)
                    print(f"running {cell_dir.name}")
                    await _run_cell(base_url, hf_model_id, tokenizer, ctx_len, concurrency, cfg, cell_dir)
            finally:
                if container_name is not None:
                    _docker_stop(container_name)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="path to a configs/bench/*.yaml sweep config")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="output directory; defaults to a new results/serving/<UTC timestamp>/. "
        "Pass an existing one to resume an interrupted sweep.",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the planned cells without running anything")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = SweepConfig.from_yaml(args.config)
    out_dir = args.out_dir or Path("results") / "serving" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    asyncio.run(run_sweep(cfg, out_dir, args.dry_run))
    if not args.dry_run:
        print(f"sweep results written to {out_dir}")


if __name__ == "__main__":
    main()
