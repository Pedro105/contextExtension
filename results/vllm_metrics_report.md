# vLLM `/metrics` verification (Phase 1)

Pinned serving version: **vLLM 0.27.1** (latest stable on PyPI/Docker Hub as of this
report; `deploy/docker/serve.Dockerfile` is pinned to `vllm/vllm-openai:v0.27.1`).

## How this was verified

Not by scraping a live server. The Docker daemon in this dev environment has no
working network path to pull base images or clone source (`docker pull` hangs
indefinitely regardless of image; confirmed on `python:3.11-slim`, ~10+ min with zero
progress) even though the host shell's own network is fine (`curl`/`pip` to PyPI,
Docker Hub's registry API, and GitHub all work normally) -- the failure is specific to
the Docker Desktop VM's networking, not general unavailability. Building vLLM's CPU
backend from source was consequently not attempted end-to-end here either; see
`deploy/docker/README.md` for what that would take and why it wasn't verified.

Instead, metric names were read directly from vLLM's own source
(`vllm/v1/metrics/loggers.py`) at git tag `v0.27.1` -- the literal strings passed to
`Counter`/`Gauge`/`Histogram` constructors, i.e. ground truth for what `/metrics`
actually exposes, just sourced from the code that generates the endpoint rather than
a live scrape of it. `src/ctxcost/bench/metrics.py`'s parser doesn't hardcode any of
these names regardless -- it discovers whatever a real scrape returns -- so runtime
behavior is correct independent of this verification method.

## Findings vs. this project's original assumptions

| Assumed name | Actual (v0.27.1) | Note |
|---|---|---|
| `vllm:gpu_cache_usage_perc` | `vllm:kv_cache_usage_perc` | Renamed, presumably to be backend-agnostic (also meaningful for the CPU backend) |
| `vllm:prefix_cache_hit_rate` | *(does not exist)* | No single ratio metric. Exposed as two raw counters instead: `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits` (token counts). Hit rate must be computed downstream as `hits / queries`; the ratio itself only ever appears in vLLM's own log line, never as a Prometheus metric. |
| `vllm:num_requests_running` | confirmed, unchanged | |
| `vllm:num_requests_waiting` | confirmed, unchanged | |

## Full metric set confirmed present (v0.27.1)

Gauges: `num_requests_running`, `num_requests_waiting`,
`num_requests_waiting_by_reason`, `engine_sleep_state`, `kv_cache_usage_perc`,
`lora_requests_info`, `cache_config_info`.

Counters: `corrupted_requests`, `prefix_cache_queries`, `prefix_cache_hits`,
`external_prefix_cache_queries`, `external_prefix_cache_hits`, `mm_cache_queries`,
`mm_cache_hits`, `num_preemptions`, `prompt_tokens`, `prompt_tokens_by_source`,
`prompt_tokens_cached`, `generation_tokens`, `request_success`.

Histograms: `request_prompt_tokens`, `request_generation_tokens`,
`iteration_tokens_total`, `request_max_num_generation_tokens`, `request_params_n`,
`request_params_max_tokens`, `time_to_first_token_seconds`,
`inter_token_latency_seconds`, `request_time_per_output_token_seconds`,
`e2e_request_latency_seconds`, `request_queue_time_seconds`,
`request_inference_time_seconds`, `request_prefill_time_seconds`,
`request_decode_time_seconds`, `request_prefill_kv_computed_tokens`,
`kv_block_lifetime_seconds`, `kv_block_idle_before_evict_seconds`,
`kv_block_reuse_gap_seconds`.

(All prefixed `vllm:`.) `tests/fixtures/vllm_metrics/vllm_0_27_1_metrics.txt` carries
a representative excerpt (real names/types/HELP text, synthetic values) used to test
`parse_prometheus_text` offline.

## vLLM flag/behavior differences found while building `deploy/docker/`

- **`enable_prefix_caching` defaults to `True`** as of 0.27.x (confirmed in
  `vllm/config/cache.py`), not an opt-in. `ENABLE_PREFIX_CACHING=false` must pass
  `--no-enable-prefix-caching` explicitly; omitting a flag would silently leave
  caching on.
- **CLI entrypoint is `vllm serve <model> ...`**, not the older
  `python3 -m vllm.entrypoints.openai.api_server` invocation style.
- **CPU backend has no `--gpu-memory-utilization` equivalent flag.** KV cache size is
  set via the `VLLM_CPU_KVCACHE_SPACE` *environment variable* (GiB), with no CLI
  counterpart -- confirmed against the CPU installation docs, not guessed.

## CPU backend build status

`deploy/docker/serve.cpu.Dockerfile` was written from vLLM's documented CPU
source-install steps (adapted from `uv` to plain `pip`) but is **not build-verified**
in this environment, for the network reason above. Treat it as a faithful first
draft, not a tested build; `vllm-project/vllm`'s own `docker/Dockerfile.cpu` at the
pinned tag is the canonical reference if it needs debugging.
