#!/bin/sh
# Shared entrypoint for both serve.Dockerfile (GPU) and serve.cpu.Dockerfile (CPU).
# Every experimental axis is an environment variable, never a value baked into an
# image layer, so the same image serves every cell of a sweep.
set -eu

: "${MODEL:?MODEL environment variable is required, e.g. HuggingFaceTB/SmolLM2-1.7B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
PORT="${PORT:-8000}"

set -- vllm serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --port "$PORT"

# vLLM (as of 0.27.x) enables prefix caching by DEFAULT -- ENABLE_PREFIX_CACHING is
# an explicit tri-state toggle, not an opt-in, so "false" must pass the negating
# flag rather than simply omitting one.
if [ "$ENABLE_PREFIX_CACHING" = "true" ]; then
    set -- "$@" --enable-prefix-caching
else
    set -- "$@" --no-enable-prefix-caching
fi

# GPU_MEMORY_UTILIZATION only means something with a real GPU behind it. The CPU
# backend has no such flag at all -- it takes a fixed GiB budget via the
# VLLM_CPU_KVCACHE_SPACE *environment* variable instead (no CLI equivalent), which
# serve.cpu.Dockerfile's base image already expects; CPU_KV_CACHE_SPACE_GIB here is
# just this image's name for that same knob, kept consistent with the other
# *_GIB-style vars in this project's configs.
if [ "${VLLM_CPU_BACKEND:-false}" = "true" ]; then
    export VLLM_CPU_KVCACHE_SPACE="${CPU_KV_CACHE_SPACE_GIB:-4}"
    # Preload tcmalloc if it's present, as vLLM's CPU docs recommend for stable
    # throughput measurements -- resolved at runtime since the library path differs
    # by architecture (x86_64 vs aarch64) and a missing library shouldn't be fatal,
    # just quietly worse for the numbers this project exists to measure.
    tcmalloc_path="$(find /usr/lib -name 'libtcmalloc_minimal.so.4' 2>/dev/null | head -n1 || true)"
    if [ -n "$tcmalloc_path" ]; then
        export LD_PRELOAD="${tcmalloc_path}${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
else
    set -- "$@" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
fi

exec "$@"
