# CLAUDE.md

Project conventions for this repo. Read this before making changes.

## Commit attribution

- Every commit's author must be Pedro (pintopedrop@gmail.com), and no one else.
- Never add `Co-Authored-By` trailers, "Generated with Claude Code", or any similar
  AI-attribution line to commit messages, PR descriptions, or file headers — in this
  repo or any tool operating on it. This applies from the initial commit onward, with
  no exceptions.

## Design principle: config axes, not hardcoding

Model choice, RoPE scaling method, context length, and evaluation mode are experimental
variables, not constants. They must be selected via YAML configs under `configs/`, never
hardcoded into a script or module. A later phase adds a retrieval baseline and a
model-scale sweep, so code should not assume a single model, a single RoPE method, or a
single context length anywhere.

## Structure

```
configs/models/         one YAML per model (arch params, native ctx, HF id)
configs/gpus/            one YAML per GPU spec (VRAM, name) — feeds the cost model CLI
configs/train/           training run configs
configs/serve/           vLLM serving configs
configs/bench/           load profiles for benchmarking
src/ctxcost/costmodel/   analytical KV/memory/throughput model
src/ctxcost/data/        corpus prep, sequence packing
src/ctxcost/model/       RoPE scaling and model patching
src/ctxcost/train/       hand-written torch.distributed training loop
src/ctxcost/eval/        synthetic long-context probe generation and scoring
src/ctxcost/bench/       load generator, Prometheus scraping, metrics
src/ctxcost/run_manifest.py  provenance manifest written by every experiment run
deploy/docker/
deploy/k8s/
scripts/
results/                 committed metric outputs — NOT gitignored
report/
```

Only scaffold a module or directory when there is real code or a real config to put in
it — no empty placeholder packages, no TODO stub files.

## Environment

- Package `ctxcost`, Python 3.11+, src layout, editable install (`pip install -e .`).
- `vllm` lives only in the `serve` optional extra — the base install must work on
  machines without CUDA.
- `results/` is committed on purpose; everything else generated (checkpoints, `.venv`,
  wandb, `__pycache__`, notebook outputs) is gitignored.

## Provenance

Every experiment run should call `ctxcost.run_manifest` to record a JSON manifest
(git commit SHA, dirty-tree flag, resolved config, hostname, GPU, library versions,
UTC timestamp) alongside its outputs.
