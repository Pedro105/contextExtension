"""Provenance manifest: every experiment run writes one of these alongside its outputs.

Captures what ran (resolved config), where (hostname, GPU), against which code
(git commit SHA + dirty flag), with which library versions, and when (UTC timestamp) —
so a results file found later can be traced back to exactly what produced it.
"""

from __future__ import annotations

import json
import socket
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _git_info(repo_dir: Path) -> dict:
    sha = _run(["git", "rev-parse", "HEAD"], repo_dir)
    status = _run(["git", "status", "--porcelain"], repo_dir)
    return {
        "commit_sha": sha,
        "dirty": bool(status) if status is not None else None,
    }


def _gpu_info() -> dict:
    out = _run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        Path.cwd(),
    )
    if not out:
        return {"model": None, "count": 0}
    names = [line.strip() for line in out.splitlines() if line.strip()]
    return {"model": names[0] if names else None, "count": len(names)}


def _library_versions() -> dict:
    versions = {}
    for pkg in ("torch", "transformers", "datasets", "vllm", "ctxcost"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_manifest(config: dict, repo_dir: str | Path | None = None) -> dict:
    """Assemble the provenance manifest for `config` without writing it anywhere."""
    repo_dir = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parents[2]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": _git_info(repo_dir),
        "hostname": socket.gethostname(),
        "gpu": _gpu_info(),
        "library_versions": _library_versions(),
        "config": config,
    }


def write_manifest(config: dict, path: str | Path, repo_dir: str | Path | None = None) -> Path:
    """Build the manifest for `config` and write it as JSON to `path`. Returns `path`."""
    manifest = build_manifest(config, repo_dir=repo_dir)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
