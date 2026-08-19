# CPU-backend vLLM OpenAI-compatible serving image, for local plumbing tests on a
# dev machine with no CUDA GPU (this repo's own dev machine included). Small models
# only -- CPU prefill is slow enough that anything past ~1-2B params and a few
# hundred tokens of context is impractical for interactive testing.
#
# vLLM's CPU backend is source-only: there is no prebuilt CPU wheel or official
# prebuilt CPU image on Docker Hub, so this builds from source following vLLM's own
# documented CPU install steps (docs/getting_started/installation/cpu.html), adapted
# from `uv` to plain `pip` so it needs nothing beyond what's already in this image.
# Single-stage on purpose: build tooling stays in the final image because a "local
# plumbing test" tool optimizes for a build that actually completes over image size.
#
# Expect the build itself to take a long time and a lot of RAM/CPU -- this is a real
# C++/Python source build, not a wheel install. Docker Desktop's default resource
# allocation is usually not enough; raise it in Settings -> Resources first.
#
# Build:
#   docker build -f deploy/docker/serve.cpu.Dockerfile -t ctxcost-serve:cpu .
#
# Run (example, smoke config):
#   docker run -p 8000:8000 \
#     -e MODEL=hf-internal-testing/tiny-random-gpt2 \
#     -e MAX_MODEL_LEN=512 \
#     -e MAX_NUM_SEQS=4 \
#     -e ENABLE_PREFIX_CACHING=false \
#     -e CPU_KV_CACHE_SPACE_GIB=2 \
#     ctxcost-serve:cpu

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        git ca-certificates curl python3 python3-pip \
        gcc-12 g++-12 libnuma-dev libtcmalloc-minimal4 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ARG VLLM_REF=v0.27.1
RUN git clone --depth 1 --branch ${VLLM_REF} https://github.com/vllm-project/vllm.git vllm_source
WORKDIR /workspace/vllm_source

# vLLM's own docs use `uv pip install --torch-backend cpu`; plain pip's equivalent
# is pointing at PyTorch's CPU wheel index directly.
RUN pip install --upgrade pip \
    && pip install -r requirements/build/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && VLLM_TARGET_DEVICE=cpu pip install . --no-build-isolation

COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MAX_MODEL_LEN="512" \
    MAX_NUM_SEQS="4" \
    ENABLE_PREFIX_CACHING="true" \
    PORT="8000" \
    VLLM_CPU_BACKEND="true" \
    CPU_KV_CACHE_SPACE_GIB="4"

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
