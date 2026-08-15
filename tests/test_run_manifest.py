"""run_manifest: provenance JSON written by every experiment."""

import json

from ctxcost.run_manifest import build_manifest, write_manifest


def test_build_manifest_shape_and_types():
    config = {"model": "synthetic/test", "ctx_len": 4096}
    manifest = build_manifest(config)

    assert manifest["config"] == config
    assert isinstance(manifest["timestamp_utc"], str)
    assert manifest["timestamp_utc"].endswith("+00:00") or "Z" in manifest["timestamp_utc"]
    assert isinstance(manifest["hostname"], str) and manifest["hostname"]

    assert "commit_sha" in manifest["git"]
    assert "dirty" in manifest["git"]

    assert "model" in manifest["gpu"]
    assert "count" in manifest["gpu"]
    assert isinstance(manifest["gpu"]["count"], int)

    assert isinstance(manifest["library_versions"], dict)


def test_build_manifest_picks_up_this_repos_commit():
    manifest = build_manifest({})
    # this repo is a real git checkout, so a commit SHA must be found (not None)
    assert manifest["git"]["commit_sha"] is not None
    assert len(manifest["git"]["commit_sha"]) == 40


def test_write_manifest_writes_valid_json(tmp_path):
    path = tmp_path / "nested" / "manifest.json"
    result_path = write_manifest({"foo": "bar"}, path)

    assert result_path == path
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["config"] == {"foo": "bar"}
