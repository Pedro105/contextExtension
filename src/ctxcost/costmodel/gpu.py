"""GPU hardware specs, kept as config just like models — see `configs/gpus/`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GPUSpec:
    name: str
    vram_gib: float

    @classmethod
    def load_yaml(cls, path: str | Path) -> GPUSpec:
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(name=d["name"], vram_gib=float(d["vram_gib"]))
