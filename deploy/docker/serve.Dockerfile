# GPU-backed vLLM OpenAI-compatible serving image.
#
# Every axis that changes across sweep cells -- which model, how much context, how
# many concurrent sequences, how much of the GPU vLLM may claim, whether prefix
# caching is on -- is an environment variable read by entrypoint.sh, never a value
# baked into an image layer. The same built image serves every arm of the sweep.
#
# Build:
#   docker build -f deploy/docker/serve.Dockerfile -t ctxcost-serve:gpu .
#
# Run (example):
#   docker run --gpus all -p 8000:8000 \
#     -e MODEL=HuggingFaceTB/SmolLM2-1.7B \
#     -e MAX_MODEL_LEN=8192 \
#     -e MAX_NUM_SEQS=256 \
#     -e GPU_MEMORY_UTILIZATION=0.9 \
#     -e ENABLE_PREFIX_CACHING=false \
#     ctxcost-serve:gpu
#
# /v1/... (OpenAI-compatible API) and /metrics (Prometheus) are both served on PORT
# -- vLLM does not expose metrics on a separate port.
#
# No suitable GPU on the dev machine? See serve.cpu.Dockerfile -- same env-var
# contract, CPU backend, small models only.

FROM vllm/vllm-openai:v0.27.1

COPY deploy/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MAX_MODEL_LEN="4096" \
    MAX_NUM_SEQS="256" \
    GPU_MEMORY_UTILIZATION="0.9" \
    ENABLE_PREFIX_CACHING="true" \
    PORT="8000"

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
