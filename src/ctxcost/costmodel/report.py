"""Build the cost-model sweep table consumed by `scripts/costmodel_report.py`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .archspec import ArchSpec
from .concurrency import max_concurrency
from .gpu import GPUSpec
from .kv import kv_bytes_per_token

GIB = 1024**3

DEFAULT_CTX_LENS = (4096, 8192, 16384, 32768, 65536)


def build_report(
    models: list[ArchSpec],
    gpus: list[GPUSpec],
    ctx_lens: list[int] = DEFAULT_CTX_LENS,
    util: float = 0.9,
    max_num_seqs: int = 256,
) -> pd.DataFrame:
    """Sweep {model x gpu x ctx_len}, returning one row of cost-model figures each."""
    rows = []
    for spec in models:
        for gpu in gpus:
            for ctx_len in ctx_lens:
                kv_bytes = kv_bytes_per_token(spec, ctx_len)
                conc = max_concurrency(spec, gpu.vram_gib, util, ctx_len, max_num_seqs)
                rows.append(
                    {
                        "model": spec.hf_model_id,
                        "gpu": gpu.name,
                        "ctx_len": ctx_len,
                        "kv_cache_gib_per_seq": kv_bytes / GIB,
                        "kv_limited_concurrency": conc.kv_limited,
                        "scheduler_limited_concurrency": conc.scheduler_limited,
                        "effective_concurrency": conc.effective,
                        "binding": conc.binding,
                    }
                )
    return pd.DataFrame(rows)


def _to_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored markdown table without extra deps."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame, results_dir: str | Path, basename: str = "costmodel_report") -> tuple[Path, Path]:
    """Write `df` to `<results_dir>/<basename>.csv` and `.md`. Returns (csv_path, md_path)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{basename}.csv"
    md_path = results_dir / f"{basename}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_to_markdown_table(df) + "\n")
    return csv_path, md_path
