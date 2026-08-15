"""Max concurrent sequences a serving deployment can hold, KV- vs scheduler-limited.

These two ceilings are a core experimental variable, not an implementation detail:
- KV-limited: how many sequences of `ctx_len` fit in the memory set aside for KV cache.
- scheduler-limited: the deployment's configured cap on concurrent sequences
  (vLLM's `max_num_seqs`), independent of whether memory allows more.
Which one binds determines whether growing context length costs you concurrency
directly (KV-bound regime) or whether you're leaving memory on the table (scheduler-
bound regime) — collapsing them into a single number would hide exactly the
distinction this project is measuring.
"""

from __future__ import annotations

from dataclasses import dataclass

from .archspec import ArchSpec
from .kv import kv_bytes_per_token

GIB = 1024**3


@dataclass(frozen=True)
class ConcurrencyResult:
    kv_limited: int
    scheduler_limited: int
    effective: int
    binding: str  # "kv" or "scheduler"


def max_concurrency(
    spec: ArchSpec,
    gpu_vram_gib: float,
    util: float,
    ctx_len: int,
    max_num_seqs: int,
) -> ConcurrencyResult:
    """Compute KV- and scheduler-limited concurrency and report which one binds.

    `gpu_vram_gib * util` is taken as the memory pool available for KV cache (mirroring
    vLLM's `gpu_memory_utilization` budget once model weights and activations are
    already accounted for elsewhere) — this cost model does not attempt to estimate
    weight memory from ArchSpec, since the fields it tracks (no vocab_size or
    intermediate_size) aren't enough to do that accurately.
    """
    if not 0 < util <= 1:
        raise ValueError(f"util must be in (0, 1], got {util}")

    usable_bytes = gpu_vram_gib * util * GIB
    per_seq_bytes = kv_bytes_per_token(spec, ctx_len)
    kv_limited = int(usable_bytes // per_seq_bytes) if per_seq_bytes > 0 else 0

    effective = min(kv_limited, max_num_seqs)
    binding = "kv" if kv_limited <= max_num_seqs else "scheduler"
    return ConcurrencyResult(
        kv_limited=kv_limited,
        scheduler_limited=max_num_seqs,
        effective=effective,
        binding=binding,
    )
