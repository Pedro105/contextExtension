#!/usr/bin/env python3
"""Drive a serving benchmark sweep from a config YAML (configs/bench/*.yaml).

For each (model, context_length, concurrency) cell: make sure a server is up with
the right flags, warm up, run the load test, scrape metrics for the duration, and
write results -- traceable to a commit/config/machine via run_manifest.

Server lifecycle is a strategy selected by `server.mode`, and the rest of the driver
never knows which one is in play -- each mode is an async context manager yielding a
base_url:

  external   -- a server is already running somewhere (Colab, a rented node, one you
                started by hand). This script never starts or stops anything; it
                health-checks the endpoint and verifies /v1/models actually reports
                the model the config expects before running a single request.
  subprocess -- launches `vllm serve <model> ...` as a child process. No Docker
                involved. The default for real runs (Colab, a rented GPU box).
  docker     -- launches a container from a prebuilt image (deploy/docker/). Kept for
                Phase 5; checks the image exists locally before trying to run it.

Resumable: a cell is skipped if its `manifest.json` already exists under the output
directory, so an interrupted sweep can be re-run with the same --out-dir cheaply.

Example:
    python scripts/serve_sweep.py --config configs/bench/smoke.yaml
    CTXCOST_EXTERNAL_BASE_URL=https://xxxx.ngrok.io \\
        python scripts/serve_sweep.py --config configs/bench/smoke_external.yaml
    python scripts/serve_sweep.py --config configs/bench/baseline.yaml \\
        --out-dir results/serving/20260101-120000  # resume a specific run
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _expand_env_vars(value: str) -> str:
    """Expand `${VAR}` references against os.environ, raising if unset rather than
    silently substituting an empty string -- a forgotten env var producing a broken
    base_url should fail loudly, not connect to the wrong (or no) host."""

    def _sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in os.environ:
            raise ValueError(f"config references ${{{name}}}, but that environment variable is not set")
        return os.environ[name]

    return _ENV_VAR_RE.sub(_sub, value)


@dataclass(frozen=True)
class ServerConfig:
    mode: str  # "external" | "subprocess" | "docker"
    base_url: str | None = None  # external
    image: str | None = None  # docker
    gpu: bool = False  # docker: pass --gpus all; subprocess: informational only
    host_port: int = 8000  # subprocess, docker
    startup_timeout_s: float = 180.0
    vllm_binary: str = "vllm"  # subprocess: override e.g. for a venv-specific path
    extra_env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("external", "subprocess", "docker"):
            raise ValueError(f"server.mode must be 'external', 'subprocess', or 'docker', got {self.mode!r}")
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
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False
    # Fake tokenizers make prompt token counts meaningless -- exactly what the smoke
    # test exists to verify -- so they're refused unless a config opts in explicitly,
    # and every cell run under one records that fact in its manifest.
    allow_fake_tokenizer: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> SweepConfig:
        with open(path) as f:
            raw = yaml.safe_load(f)
        server_raw = dict(raw["server"])
        if isinstance(server_raw.get("base_url"), str):
            server_raw["base_url"] = _expand_env_vars(server_raw["base_url"])
        return cls(
            server=ServerConfig(**server_raw),
            models=tuple(raw["models"]),
            context_lengths=tuple(raw["context_lengths"]),
            concurrency_levels=tuple(raw["concurrency_levels"]),
            workload=WorkloadSweepConfig(**raw["workload"]),
            metrics_poll_interval_s=raw.get("metrics_poll_interval_s", 1.0),
            max_num_seqs=raw.get("max_num_seqs", 256),
            gpu_memory_utilization=raw.get("gpu_memory_utilization", 0.9),
            enable_prefix_caching=raw.get("enable_prefix_caching", False),
            allow_fake_tokenizer=raw.get("allow_fake_tokenizer", False),
        )


class SimpleTokenizer:
    """Fake duck-typed tokenizer. Only ever used when a config explicitly sets
    allow_fake_tokenizer: true (see _load_tokenizer) -- token counts it produces are
    NOT the real model's token counts, so any cell run under it has that fact
    recorded in its manifest.json rather than silently passing as a real result."""

    def encode(self, text: str) -> list[int]:
        return [hash(w) % 50000 for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        return " ".join(f"tok{i}" for i in ids)


def _load_tokenizer(hf_model_id: str, allow_fake_tokenizer: bool) -> tuple[object, bool]:
    """Returns (tokenizer, used_fake_tokenizer). Fails loudly by default: prompt
    token-count exactness is the whole point of workload.py, so silently swapping in
    a fake tokenizer would make every downstream number meaningless while still
    looking like a real result."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(hf_model_id), False
    except ImportError as exc:
        if not allow_fake_tokenizer:
            raise RuntimeError(
                f"transformers is not installed, so a real tokenizer for {hf_model_id!r} "
                "cannot be loaded -- prompt token counts would be meaningless if this run "
                "continued. Install transformers (it's a base project dependency: "
                "`pip install -e .`), or set `allow_fake_tokenizer: true` in this sweep "
                "config to explicitly opt into a fake tokenizer for plumbing tests only "
                "(this gets recorded in every cell's manifest.json when used)."
            ) from exc
        print(
            f"WARNING: transformers not installed; using a fake tokenizer for {hf_model_id} "
            "(allow_fake_tokenizer=true) -- prompt token counts are NOT real",
            file=sys.stderr,
        )
        return SimpleTokenizer(), True


# ---------------------------------------------------------------------------
# HTTP helpers shared by the server-lifecycle strategies. `transport` is a pure
# testability seam (httpx.ASGITransport against the mock server), as in client.py.
# ---------------------------------------------------------------------------


async def _wait_healthy(base_url: str, timeout_s: float, transport: httpx.BaseTransport | None = None) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0, transport=transport) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url.rstrip('/')}/health")
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
    raise TimeoutError(
        f"server at {base_url} did not become healthy within {timeout_s}s "
        "(is it actually running there? check server.base_url / port mapping)"
    )


async def _verify_served_model(
    base_url: str, hf_model_id: str, transport: httpx.BaseTransport | None = None
) -> None:
    """Query /v1/models and abort if it isn't serving the model this config expects.
    Silently benchmarking the wrong model is a failure mode worth guarding against,
    not just an inconvenience."""
    async with httpx.AsyncClient(timeout=10.0, transport=transport) as client:
        resp = await client.get(f"{base_url.rstrip('/')}/v1/models")
        resp.raise_for_status()
        data = resp.json()
    served_ids = {m.get("id") for m in data.get("data", [])}
    if hf_model_id not in served_ids:
        raise RuntimeError(
            f"refusing to benchmark: server at {base_url} is not serving {hf_model_id!r}. "
            f"/v1/models reports: {sorted(served_ids) or ['<none>']}. "
            "Check server.base_url or the model entry in your sweep config."
        )


def _warn_if_external_mode_cannot_honor_sweep(cfg: SweepConfig) -> None:
    """external mode can't restart the server, so max_model_len / max_num_seqs can't
    actually change between cells even though the sweep config varies them -- warn
    (don't fail) so results aren't silently mislabelled with a context length the
    server was never actually configured for."""
    if cfg.server.mode != "external":
        return
    if len(cfg.context_lengths) > 1:
        print(
            f"WARNING: server.mode: external cannot change max_model_len between cells, "
            f"but context_lengths sweeps {list(cfg.context_lengths)}. Every cell will hit "
            f"the same already-running server -- results will be labelled with the sweep's "
            f"intended context length, not a value this driver can guarantee the server "
            f"actually has.",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# docker mode
# ---------------------------------------------------------------------------


def _check_docker_image_exists(image: str, gpu: bool) -> None:
    result = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        dockerfile = "deploy/docker/serve.Dockerfile" if gpu else "deploy/docker/serve.cpu.Dockerfile"
        raise RuntimeError(
            f"docker image {image!r} not found locally. Build it first:\n"
            f"  docker build -f {dockerfile} -t {image} ."
        )


def _docker_run(image: str, hf_model_id: str, max_model_len: int, cfg: SweepConfig) -> str:
    name = f"ctxcost-serve-{uuid.uuid4().hex[:8]}"
    env = {
        "MODEL": hf_model_id,
        "MAX_MODEL_LEN": str(max_model_len),
        "MAX_NUM_SEQS": str(cfg.max_num_seqs),
        "GPU_MEMORY_UTILIZATION": str(cfg.gpu_memory_utilization),
        "ENABLE_PREFIX_CACHING": "true" if cfg.enable_prefix_caching else "false",
        **cfg.server.extra_env,
    }
    cmd = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{cfg.server.host_port}:8000"]
    if cfg.server.gpu:
        cmd += ["--gpus", "all"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"`docker run` failed (exit {result.returncode}): {result.stderr.strip()}")
    return name


def _docker_stop(name: str) -> None:
    subprocess.run(["docker", "stop", name], check=False, capture_output=True)


@asynccontextmanager
async def _docker_session(
    cfg: SweepConfig, hf_model_id: str, ctx_len: int, transport: httpx.BaseTransport | None = None
) -> AsyncIterator[str]:
    _check_docker_image_exists(cfg.server.image, cfg.server.gpu)
    print(f"starting container for {hf_model_id} (max_model_len={ctx_len})")
    container_name = _docker_run(cfg.server.image, hf_model_id, ctx_len, cfg)
    base_url = f"http://localhost:{cfg.server.host_port}"
    try:
        await _wait_healthy(base_url, cfg.server.startup_timeout_s, transport=transport)
        yield base_url
    finally:
        _docker_stop(container_name)


# ---------------------------------------------------------------------------
# subprocess mode
# ---------------------------------------------------------------------------


def _build_vllm_serve_cmd(hf_model_id: str, ctx_len: int, cfg: SweepConfig) -> list[str]:
    cmd = [
        cfg.server.vllm_binary,
        "serve",
        hf_model_id,
        "--max-model-len",
        str(ctx_len),
        "--max-num-seqs",
        str(cfg.max_num_seqs),
        "--port",
        str(cfg.server.host_port),
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    cmd.append("--enable-prefix-caching" if cfg.enable_prefix_caching else "--no-enable-prefix-caching")
    return cmd


async def _spawn_vllm_subprocess(hf_model_id: str, ctx_len: int, cfg: SweepConfig) -> asyncio.subprocess.Process:
    cmd = _build_vllm_serve_cmd(hf_model_id, ctx_len, cfg)
    env = {**os.environ, **cfg.server.extra_env}
    print(f"launching: {' '.join(cmd)}")
    return await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env
    )


async def _terminate_subprocess(proc: asyncio.subprocess.Process, grace_s: float = 15.0) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=grace_s)
    except TimeoutError:
        proc.kill()
        await proc.wait()


@asynccontextmanager
async def _subprocess_session(
    cfg: SweepConfig, hf_model_id: str, ctx_len: int, transport: httpx.BaseTransport | None = None
) -> AsyncIterator[str]:
    proc = await _spawn_vllm_subprocess(hf_model_id, ctx_len, cfg)
    base_url = f"http://localhost:{cfg.server.host_port}"
    try:
        await _wait_healthy(base_url, cfg.server.startup_timeout_s, transport=transport)
        yield base_url
    finally:
        await _terminate_subprocess(proc)


# ---------------------------------------------------------------------------
# external mode
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _external_session(
    cfg: SweepConfig, hf_model_id: str, ctx_len: int, transport: httpx.BaseTransport | None = None
) -> AsyncIterator[str]:
    base_url = cfg.server.base_url
    await _wait_healthy(base_url, cfg.server.startup_timeout_s, transport=transport)
    await _verify_served_model(base_url, hf_model_id, transport=transport)
    yield base_url
    # nothing to tear down -- this script never started it


_SESSION_BY_MODE = {
    "external": _external_session,
    "subprocess": _subprocess_session,
    "docker": _docker_session,
}


def _cell_id(model_slug: str, ctx_len: int, concurrency: int) -> str:
    return f"{model_slug}__ctx{ctx_len}__conc{concurrency}"


async def _run_cell(
    base_url: str,
    hf_model_id: str,
    tokenizer,
    used_fake_tokenizer: bool,
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
        "max_num_seqs": cfg.max_num_seqs,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "enable_prefix_caching": cfg.enable_prefix_caching,
        "server_mode": cfg.server.mode,
        "base_url": base_url,
        "used_fake_tokenizer": used_fake_tokenizer,
    }
    write_manifest(manifest_config, cell_dir / "manifest.json")


async def run_sweep(
    cfg: SweepConfig, out_dir: Path, dry_run: bool, transport: httpx.BaseTransport | None = None
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _warn_if_external_mode_cannot_honor_sweep(cfg)
    session_fn = _SESSION_BY_MODE[cfg.server.mode]

    for model_entry in cfg.models:
        hf_model_id = model_entry["hf_model_id"]
        model_slug = model_entry.get("served_model_name", hf_model_id).replace("/", "__")
        used_fake_tokenizer = False
        if not dry_run:
            tokenizer, used_fake_tokenizer = _load_tokenizer(hf_model_id, cfg.allow_fake_tokenizer)
        else:
            tokenizer = None

        for ctx_len in cfg.context_lengths:
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
                    print(f"[dry-run] would run {_cell_id(model_slug, ctx_len, c)} (server.mode={cfg.server.mode})")
                continue

            async with session_fn(cfg, hf_model_id, ctx_len, transport) as base_url:
                for concurrency in cells_pending:
                    cell_dir = out_dir / _cell_id(model_slug, ctx_len, concurrency)
                    print(f"running {cell_dir.name}")
                    await _run_cell(
                        base_url, hf_model_id, tokenizer, used_fake_tokenizer, ctx_len, concurrency, cfg, cell_dir
                    )


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
