"""KV cache memory cost as a function of context length."""

from __future__ import annotations

from .archspec import DTYPE_BYTES, ArchSpec


def kv_bytes_per_token(spec: ArchSpec, ctx_len: int, dtype: str | None = None) -> int:
    """Total KV cache bytes to hold a single sequence of length `ctx_len`.

    Despite the name (kept to match the field it answers: bytes attributable to each
    token position under `spec`'s attention pattern), this is NOT constant in
    `ctx_len` for full-attention models — it returns the total cache size for the
    whole sequence, which is:

      - full attention: linear in ctx_len (every layer caches every token)
      - uniform sliding window: linear up to the window, then constant
        (each layer only ever caches the last `sliding_window` tokens)
      - interleaved SWA: full-attention layers keep growing linearly while
        sliding-window layers cap out, so the total is a mix of both

    Per stored token, per layer, the cache holds one K and one V vector of width
    `num_key_value_heads * head_dim` (GQA/MQA already accounted for since the KV
    projection — not the query projection — is what's cached).
    """
    dtype = dtype or spec.torch_dtype
    bytes_per_elem = DTYPE_BYTES.get(dtype)
    if bytes_per_elem is None:
        raise ValueError(f"unknown dtype {dtype!r}; known dtypes: {sorted(DTYPE_BYTES)}")

    per_token_per_layer = 2 * spec.num_key_value_heads * spec.head_dim * bytes_per_elem

    total = 0
    for layer_idx in range(spec.num_hidden_layers):
        if spec.is_layer_sliding(layer_idx):
            stored_tokens = min(ctx_len, spec.sliding_window)
        else:
            stored_tokens = ctx_len
        total += stored_tokens * per_token_per_layer
    return total
