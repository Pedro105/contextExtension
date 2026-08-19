# Serving images

Two images, one env-var contract (`MODEL`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`,
`GPU_MEMORY_UTILIZATION`, `ENABLE_PREFIX_CACHING`, `PORT`), one shared
`entrypoint.sh`. `scripts/serve_sweep.py` and the load-test harness in
`src/ctxcost/bench/` don't know or care which one is running behind a given
`base_url` -- they only ever see an OpenAI-compatible endpoint.

| | `serve.Dockerfile` | `serve.cpu.Dockerfile` |
|---|---|---|
| Base | `vllm/vllm-openai:v0.27.1` (prebuilt) | `ubuntu:22.04`, vLLM built from source |
| Requires | NVIDIA GPU + `--gpus all` | Nothing but CPU + RAM |
| Build time | seconds (image pull) | long -- real C++/Python compile |
| Use for | real sweep cells (`configs/bench/baseline.yaml`) | plumbing tests (`configs/bench/smoke.yaml`) |

## Metrics endpoint

Both images serve the OpenAI API (`/v1/...`) and Prometheus metrics (`/metrics`) on
the same `PORT` -- there's no separate metrics port. Confirmed metric names for
`v0.27.1` are recorded in the project's cost-model/serving report; verify against a
running instance yourself with:

```
curl -s http://localhost:8000/metrics | grep '^vllm:'
```

## `ENABLE_PREFIX_CACHING` note

vLLM 0.27.x enables prefix caching **by default**. `ENABLE_PREFIX_CACHING=false`
passes `--no-enable-prefix-caching` explicitly rather than just omitting a flag --
if your workload config also sets `enable_shared_prefix=True` to study the caching
confound on purpose, make sure this env var is `true` for that cell, or you'll be
measuring shared prompts against a server that's ignoring the shared prefix.

## CPU image: before you build

Docker Desktop's default VM resource allocation is usually too small for a from-source
build. Raise it first: Docker Desktop -> Settings -> Resources -> raise Memory to at
least 8 GiB and CPUs to at least 4, then Apply & Restart.

```
docker build -f deploy/docker/serve.cpu.Dockerfile -t ctxcost-serve:cpu .
docker run -p 8000:8000 \
  -e MODEL=hf-internal-testing/tiny-random-gpt2 \
  -e MAX_MODEL_LEN=512 -e MAX_NUM_SEQS=4 -e ENABLE_PREFIX_CACHING=false \
  -e CPU_KV_CACHE_SPACE_GIB=2 \
  ctxcost-serve:cpu
```

This CPU Dockerfile was written from vLLM's documented source-install steps but was
**not** build-verified in this environment -- the Docker daemon here has no working
network path to pull base layers or clone the vLLM source (`docker pull` hangs
indefinitely; direct `curl`/`pip` from the host shell work fine, so this is specific
to the Docker VM's networking, not general network unavailability). Treat it as a
faithful first draft of the documented steps, not a build verified end-to-end; if it
breaks, `vllm-project/vllm`'s own `docker/Dockerfile.cpu` at the pinned tag is the
canonical reference to diff against.
