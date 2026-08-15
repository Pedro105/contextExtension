#!/usr/bin/env python3
"""Sweep the analytical cost model over models x GPUs x context lengths.

Example:
    python scripts/costmodel_report.py \\
        --model configs/models/smollm2-1_7b.yaml \\
        --model configs/models/qwen2_5-1_5b.yaml \\
        --model configs/models/ministral-3-3b-base.yaml \\
        --gpu configs/gpus/l4-24gb.yaml \\
        --gpu configs/gpus/a100-80gb.yaml

Writes results/<basename>.csv and results/<basename>.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ctxcost.costmodel import GPUSpec, build_report, load_model_config, write_report
from ctxcost.costmodel.report import DEFAULT_CTX_LENS


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", action="append", required=True, dest="models",
        help="path to a configs/models/*.yaml file (repeatable)",
    )
    parser.add_argument(
        "--gpu", action="append", required=True, dest="gpus",
        help="path to a configs/gpus/*.yaml file (repeatable)",
    )
    parser.add_argument(
        "--ctx-len", type=int, action="append", dest="ctx_lens",
        help="context length to sweep (repeatable); defaults to 4k/8k/16k/32k/64k",
    )
    parser.add_argument("--util", type=float, default=0.9, help="GPU memory utilization fraction for KV cache (default: 0.9)")
    parser.add_argument("--max-num-seqs", type=int, default=256, help="scheduler concurrency cap (default: 256)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="output directory (default: results/)")
    parser.add_argument("--basename", default="costmodel_report", help="output file basename (default: costmodel_report)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    models = [load_model_config(p) for p in args.models]
    gpus = [GPUSpec.load_yaml(p) for p in args.gpus]
    ctx_lens = args.ctx_lens if args.ctx_lens else list(DEFAULT_CTX_LENS)

    df = build_report(models, gpus, ctx_lens=ctx_lens, util=args.util, max_num_seqs=args.max_num_seqs)
    csv_path, md_path = write_report(df, args.results_dir, basename=args.basename)

    print(f"wrote {len(df)} rows to {csv_path} and {md_path}")


if __name__ == "__main__":
    main()
