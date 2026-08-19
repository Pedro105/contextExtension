"""Architecture specs pulled from HF `config.json`, normalized for cost modeling.

The HF config schema is not uniform across model families: RoPE parameters show up
under a top-level `rope_theta` on older configs and under a unified `rope_parameters`
(or `rope_scaling`) dict on newer ones; sliding-window attention is expressed as a
present-but-possibly-disabled `sliding_window` + `use_sliding_window` pair on some
families (e.g. Qwen2) and as a per-layer `layer_types` list on others (interleaved
SWA, e.g. Gemma3-style models); multimodal wrappers (e.g. Mistral3) nest the actual
text-decoder config under a `text_config` key. Gemma3 additionally expresses its
interleaved local/global attention as a `sliding_window_pattern` period rather than an
explicit `layer_types` list, and uses a different RoPE base frequency for local layers
(`rope_local_base_freq`) than for global ones (`rope_theta`). `fetch_arch` absorbs all
of that so the rest of the cost model only ever deals with `ArchSpec`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "ctxcost" / "hf_configs"

# Attention-pattern strings used by interleaved-SWA models (e.g. Gemma3) in `layer_types`.
_SLIDING_LAYER_TYPES = {"sliding_attention", "sliding_window_attention", "sliding"}

# Bytes per element for the dtypes we're likely to see in `torch_dtype`/`dtype`.
DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "int8": 1,
}


@dataclass(frozen=True)
class ArchSpec:
    """Normalized architecture spec used throughout the cost model."""

    hf_model_id: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    max_position_embeddings: int
    rope_theta: float
    torch_dtype: str

    # None when the model has no RoPE scaling applied (trained natively at
    # max_position_embeddings). When set, holds the raw scaling dict (rope_type/type,
    # factor, original_max_position_embeddings, and any method-specific params).
    rope_scaling: dict | None = None

    # Sliding-window attention, represented explicitly rather than assumed away:
    #   sliding_window=None                          -> no SWA support in this arch
    #   sliding_window=N, sliding_window_enabled=True -> uniform SWA, window N
    #   sliding_window=N, sliding_window_enabled=False -> arch supports SWA but it's
    #                                                      off (e.g. Qwen2.5 base configs)
    sliding_window: int | None = None
    sliding_window_enabled: bool = False

    # Per-layer attention pattern for interleaved SWA (e.g. Gemma3: alternating
    # "sliding_attention"/"full_attention"). None means uniform attention across layers
    # (either all-full, or all-sliding per sliding_window_enabled above).
    layer_types: tuple[str, ...] | None = None

    # Alternate way of expressing interleaved SWA, used by Gemma3 instead of an explicit
    # layer_types list: a period N where layer_idx is global (full-attention) iff
    # (layer_idx + 1) % N == 0, and sliding otherwise. E.g. N=6 -> 5 sliding layers then
    # 1 global layer, repeating. Only consulted when layer_types is None.
    sliding_window_pattern: int | None = None

    # Some interleaved-SWA models (Gemma3) use a different RoPE base frequency for local
    # (sliding-window) layers than for global (full-attention) layers -- e.g. Gemma3's
    # rope_theta=1e6 for global layers vs. rope_local_base_freq=10000 for local layers.
    # None means every layer shares `rope_theta` (the common case).
    rope_theta_local: float | None = None

    family: str = ""
    description: str = ""

    @property
    def native_max_position_embeddings(self) -> int:
        """Context length the model was actually trained at.

        Equal to max_position_embeddings unless RoPE scaling stretches the position
        range beyond training (e.g. YaRN/linear/NTK scaling with an explicit
        `original_max_position_embeddings`), in which case that original value is the
        native length and max_position_embeddings is the post-scaling extended length.
        """
        if self.rope_scaling and "original_max_position_embeddings" in self.rope_scaling:
            return int(self.rope_scaling["original_max_position_embeddings"])
        return self.max_position_embeddings

    def is_layer_sliding(self, layer_idx: int) -> bool:
        """Whether layer `layer_idx` uses sliding-window (vs full) attention."""
        if self.layer_types is not None:
            return self.layer_types[layer_idx] in _SLIDING_LAYER_TYPES
        if self.sliding_window_pattern is not None:
            return (layer_idx + 1) % self.sliding_window_pattern != 0
        return self.sliding_window_enabled and self.sliding_window is not None

    def rope_theta_for_layer(self, layer_idx: int) -> float:
        """RoPE base frequency for `layer_idx`, honoring a local/global theta split."""
        if self.rope_theta_local is not None and self.is_layer_sliding(layer_idx):
            return self.rope_theta_local
        return self.rope_theta

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.layer_types is not None:
            d["layer_types"] = list(self.layer_types)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ArchSpec:
        d = dict(d)
        if d.get("layer_types") is not None:
            d["layer_types"] = tuple(d["layer_types"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> ArchSpec:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _text_config(raw: dict) -> dict:
    """Unwrap the decoder config from a multimodal wrapper if present (e.g. Mistral3)."""
    text_cfg = raw.get("text_config")
    return text_cfg if isinstance(text_cfg, dict) else raw


def _extract_rope(raw: dict, text_cfg: dict) -> tuple[float, dict | None]:
    """Return (rope_theta, rope_scaling) handling both old and unified config shapes.

    Older configs: top-level `rope_theta` (float) + optional `rope_scaling` (dict).
    Newer (transformers >=5.0) configs: a single `rope_parameters` dict carrying both
    `rope_theta` and the scaling method/params together.
    """
    rope_params = text_cfg.get("rope_parameters") or text_cfg.get("rope_scaling")
    if isinstance(rope_params, dict):
        rope_theta = float(rope_params.get("rope_theta", text_cfg.get("rope_theta", 10000.0)))
        rope_type = rope_params.get("rope_type") or rope_params.get("type")
        if rope_type in (None, "default"):
            return rope_theta, None
        return rope_theta, rope_params
    return float(text_cfg.get("rope_theta", 10000.0)), None


def _extract_dtype(raw: dict, text_cfg: dict) -> str:
    for cfg in (text_cfg, raw):
        dtype = cfg.get("torch_dtype") or cfg.get("dtype")
        if dtype:
            return dtype
    return "float32"


def _extract_sliding_window(text_cfg: dict) -> tuple[int | None, bool]:
    window = text_cfg.get("sliding_window")
    if window is None:
        return None, False
    # Some families (Qwen2) declare a sliding_window value but gate it with a separate
    # flag; if that flag is present and false, the window is inert. If the flag is
    # absent, a non-null sliding_window is taken to mean SWA is active.
    enabled = bool(text_cfg.get("use_sliding_window", True))
    return int(window), enabled


def _extract_interleaved_pattern(text_cfg: dict) -> tuple[int | None, float | None]:
    """Gemma3-style interleaved local/global attention fields.

    `sliding_window_pattern` (int period) and `rope_local_base_freq` (local-layer RoPE
    theta) are how Gemma3 expresses interleaved SWA instead of an explicit `layer_types`
    list. Both are absent on families that don't use this scheme.
    """
    pattern = text_cfg.get("sliding_window_pattern")
    local_theta = text_cfg.get("rope_local_base_freq")
    return (
        int(pattern) if pattern is not None else None,
        float(local_theta) if local_theta is not None else None,
    )


def parse_config(raw: dict, hf_model_id: str) -> ArchSpec:
    """Parse a raw HF `config.json` dict into an ArchSpec."""
    text_cfg = _text_config(raw)

    num_attention_heads = int(text_cfg["num_attention_heads"])
    hidden_size = text_cfg.get("hidden_size")
    head_dim = text_cfg.get("head_dim")
    if head_dim is None:
        if hidden_size is None:
            raise ValueError(
                f"{hf_model_id}: config has neither head_dim nor hidden_size to derive it from"
            )
        head_dim = hidden_size // num_attention_heads
    if hidden_size is None:
        hidden_size = head_dim * num_attention_heads

    rope_theta, rope_scaling = _extract_rope(raw, text_cfg)
    sliding_window, sliding_window_enabled = _extract_sliding_window(text_cfg)
    layer_types = text_cfg.get("layer_types")
    sliding_window_pattern, rope_theta_local = _extract_interleaved_pattern(text_cfg)
    if sliding_window_pattern is not None:
        sliding_window_enabled = True

    return ArchSpec(
        hf_model_id=hf_model_id,
        num_hidden_layers=int(text_cfg["num_hidden_layers"]),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(text_cfg.get("num_key_value_heads", num_attention_heads)),
        head_dim=int(head_dim),
        hidden_size=int(hidden_size),
        max_position_embeddings=int(text_cfg["max_position_embeddings"]),
        rope_theta=rope_theta,
        torch_dtype=_extract_dtype(raw, text_cfg),
        rope_scaling=rope_scaling,
        sliding_window=sliding_window,
        sliding_window_enabled=sliding_window_enabled,
        layer_types=tuple(layer_types) if layer_types is not None else None,
        sliding_window_pattern=sliding_window_pattern,
        rope_theta_local=rope_theta_local,
    )


def fetch_arch(
    hf_model_id: str,
    cache_dir: str | Path | None = None,
    force_refresh: bool = False,
) -> ArchSpec:
    """Fetch `config.json` for `hf_model_id` from the HF Hub and parse it into an ArchSpec.

    Fetched configs are cached to disk as raw JSON under `cache_dir` (default
    `~/.cache/ctxcost/hf_configs/`, overridable so tests can point it at fixtures) so
    repeat calls, including test runs, don't hit the network.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_file = cache_dir / f"{hf_model_id.replace('/', '__')}.json"

    if cache_file.exists() and not force_refresh:
        raw = json.loads(cache_file.read_text())
    else:
        config_path = hf_hub_download(repo_id=hf_model_id, filename="config.json")
        raw = json.loads(Path(config_path).read_text())
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(raw, indent=2))

    return parse_config(raw, hf_model_id)


def load_model_config(path: str | Path) -> ArchSpec:
    """Load an ArchSpec from a `configs/models/*.yaml` file."""
    return ArchSpec.load_yaml(path)
