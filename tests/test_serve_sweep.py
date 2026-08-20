"""scripts/serve_sweep.py: the three server-lifecycle strategies (external,
subprocess, docker) behind one shared interface, env-var config expansion, and the
loud-by-default tokenizer fallback."""

import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from mock_vllm_server import make_mock_vllm_app

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import serve_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# ServerConfig validation
# ---------------------------------------------------------------------------


def test_external_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        serve_sweep.ServerConfig(mode="external")


def test_docker_requires_image():
    with pytest.raises(ValueError, match="image"):
        serve_sweep.ServerConfig(mode="docker")


def test_subprocess_requires_neither():
    cfg = serve_sweep.ServerConfig(mode="subprocess")
    assert cfg.base_url is None
    assert cfg.image is None


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="external.*subprocess.*docker"):
        serve_sweep.ServerConfig(mode="carrier-pigeon")


# ---------------------------------------------------------------------------
# env-var expansion in config
# ---------------------------------------------------------------------------


def test_expand_env_vars_substitutes(monkeypatch):
    monkeypatch.setenv("CTXCOST_TEST_URL", "http://example.com:1234")
    assert serve_sweep._expand_env_vars("${CTXCOST_TEST_URL}/v1") == "http://example.com:1234/v1"


def test_expand_env_vars_unset_raises(monkeypatch):
    monkeypatch.delenv("CTXCOST_DEFINITELY_UNSET", raising=False)
    with pytest.raises(ValueError, match="CTXCOST_DEFINITELY_UNSET"):
        serve_sweep._expand_env_vars("${CTXCOST_DEFINITELY_UNSET}")


def test_sweep_config_expands_base_url_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("CTXCOST_TEST_URL", "http://example.com:9999")
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "server": {"mode": "external", "base_url": "${CTXCOST_TEST_URL}"},
                "models": [{"hf_model_id": "m"}],
                "context_lengths": [64],
                "concurrency_levels": [1],
                "workload": {"n_requests_per_cell": 1, "max_output_tokens": 1},
            }
        )
    )
    cfg = serve_sweep.SweepConfig.from_yaml(config_path)
    assert cfg.server.base_url == "http://example.com:9999"


# ---------------------------------------------------------------------------
# real repo configs parse correctly
# ---------------------------------------------------------------------------


def test_smoke_yaml_is_subprocess_mode():
    cfg = serve_sweep.SweepConfig.from_yaml(REPO_ROOT / "configs" / "bench" / "smoke.yaml")
    assert cfg.server.mode == "subprocess"


def test_baseline_yaml_is_docker_mode():
    cfg = serve_sweep.SweepConfig.from_yaml(REPO_ROOT / "configs" / "bench" / "baseline.yaml")
    assert cfg.server.mode == "docker"
    assert cfg.max_num_seqs == 256


def test_smoke_external_yaml_is_external_mode(monkeypatch):
    monkeypatch.setenv("CTXCOST_EXTERNAL_BASE_URL", "http://localhost:9999")
    cfg = serve_sweep.SweepConfig.from_yaml(REPO_ROOT / "configs" / "bench" / "smoke_external.yaml")
    assert cfg.server.mode == "external"
    assert cfg.server.base_url == "http://localhost:9999"
    assert len(cfg.context_lengths) == 1  # can't vary max_model_len against a fixed server


# ---------------------------------------------------------------------------
# tokenizer: loud by default, opt-in fake, recorded when used
# ---------------------------------------------------------------------------


def test_missing_transformers_fails_loudly_by_default(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # force ImportError deterministically
    with pytest.raises(RuntimeError, match="allow_fake_tokenizer"):
        serve_sweep._load_tokenizer("some/model", allow_fake_tokenizer=False)


def test_missing_transformers_allows_fake_when_opted_in(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "transformers", None)
    tokenizer, used_fake = serve_sweep._load_tokenizer("some/model", allow_fake_tokenizer=True)
    assert used_fake is True
    assert isinstance(tokenizer, serve_sweep.SimpleTokenizer)
    assert "WARNING" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# HTTP helpers against the ASGI mock (external mode's only interface to a server)
# ---------------------------------------------------------------------------


async def _unhealthy_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 503, "headers": []})
    await send({"type": "http.response.body", "body": b"nope"})


@pytest.mark.anyio
async def test_wait_healthy_succeeds_against_mock():
    app, _ = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    await serve_sweep._wait_healthy("http://mock", timeout_s=5.0, transport=transport)  # should not raise


@pytest.mark.anyio
async def test_wait_healthy_times_out_with_clear_error():
    transport = httpx.ASGITransport(app=_unhealthy_app)
    with pytest.raises(TimeoutError, match="did not become healthy"):
        await serve_sweep._wait_healthy("http://mock", timeout_s=0.2, transport=transport)


def _models_app_factory(served_id: str):
    async def app(scope, receive, send):
        if scope["path"] == "/v1/models":
            import json

            body = json.dumps({"data": [{"id": served_id}]}).encode()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    return app


@pytest.mark.anyio
async def test_verify_served_model_matches():
    transport = httpx.ASGITransport(app=_models_app_factory("expected/model"))
    await serve_sweep._verify_served_model("http://mock", "expected/model", transport=transport)  # no raise


@pytest.mark.anyio
async def test_verify_served_model_mismatch_aborts():
    transport = httpx.ASGITransport(app=_models_app_factory("wrong/model"))
    with pytest.raises(RuntimeError, match="refusing to benchmark"):
        await serve_sweep._verify_served_model("http://mock", "expected/model", transport=transport)


# ---------------------------------------------------------------------------
# external mode: warning about cells it can't honor
# ---------------------------------------------------------------------------


def _sweep_config(mode: str, context_lengths=(64,), **server_kwargs) -> "serve_sweep.SweepConfig":
    server_defaults = {"mode": mode}
    if mode == "external":
        server_defaults["base_url"] = "http://mock"
    elif mode == "docker":
        server_defaults["image"] = "some-image"
    server_defaults.update(server_kwargs)
    return serve_sweep.SweepConfig(
        server=serve_sweep.ServerConfig(**server_defaults),
        models=({"hf_model_id": "m"},),
        context_lengths=tuple(context_lengths),
        concurrency_levels=(1,),
        workload=serve_sweep.WorkloadSweepConfig(n_requests_per_cell=1, max_output_tokens=1),
    )


def test_warns_when_external_mode_varies_context_length(capsys):
    cfg = _sweep_config("external", context_lengths=(64, 128))
    serve_sweep._warn_if_external_mode_cannot_honor_sweep(cfg)
    assert "WARNING" in capsys.readouterr().err


def test_no_warning_for_external_mode_single_context_length(capsys):
    cfg = _sweep_config("external", context_lengths=(64,))
    serve_sweep._warn_if_external_mode_cannot_honor_sweep(cfg)
    assert capsys.readouterr().err == ""


def test_no_warning_for_docker_mode_multi_context_length(capsys):
    cfg = _sweep_config("docker", context_lengths=(64, 128))
    serve_sweep._warn_if_external_mode_cannot_honor_sweep(cfg)
    assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# external_session context manager
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_external_session_yields_base_url_on_match():
    app, _ = make_mock_vllm_app(served_model_id="test-model")
    transport = httpx.ASGITransport(app=app)
    cfg = serve_sweep.SweepConfig(
        server=serve_sweep.ServerConfig(mode="external", base_url="http://mock", startup_timeout_s=5.0),
        models=({"hf_model_id": "test-model"},),
        context_lengths=(64,),
        concurrency_levels=(1,),
        workload=serve_sweep.WorkloadSweepConfig(n_requests_per_cell=1, max_output_tokens=1),
    )

    async with serve_sweep._external_session(cfg, "test-model", 64, transport=transport) as base_url:
        assert base_url == "http://mock"


@pytest.mark.anyio
async def test_external_session_aborts_on_model_mismatch():
    app, _ = make_mock_vllm_app(served_model_id="other/model")
    transport = httpx.ASGITransport(app=app)
    cfg = serve_sweep.SweepConfig(
        server=serve_sweep.ServerConfig(mode="external", base_url="http://mock", startup_timeout_s=5.0),
        models=({"hf_model_id": "expected/model"},),
        context_lengths=(64,),
        concurrency_levels=(1,),
        workload=serve_sweep.WorkloadSweepConfig(n_requests_per_cell=1, max_output_tokens=1),
    )
    with pytest.raises(RuntimeError, match="refusing to benchmark"):
        async with serve_sweep._external_session(cfg, "expected/model", 64, transport=transport):
            pytest.fail("should not have entered the session body")


# ---------------------------------------------------------------------------
# docker mode: image-existence check gives an actionable error, not a raw
# CalledProcessError from a later `docker run`
# ---------------------------------------------------------------------------


def test_check_docker_image_missing_names_build_command(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="No such image")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="docker build -f deploy/docker/serve.Dockerfile"):
        serve_sweep._check_docker_image_exists("ctxcost-serve:gpu", gpu=True)


def test_check_docker_image_missing_names_cpu_dockerfile(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="No such image")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="serve.cpu.Dockerfile"):
        serve_sweep._check_docker_image_exists("ctxcost-serve:cpu", gpu=False)


def test_check_docker_image_present_does_not_raise(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    serve_sweep._check_docker_image_exists("ctxcost-serve:gpu", gpu=True)  # should not raise


# ---------------------------------------------------------------------------
# subprocess mode: command construction + session lifecycle (process spawn mocked,
# health check against the ASGI mock)
# ---------------------------------------------------------------------------


def test_build_vllm_serve_cmd_contains_expected_flags():
    cfg = _sweep_config("subprocess")
    cfg = serve_sweep.SweepConfig(
        **{**cfg.__dict__, "max_num_seqs": 42, "gpu_memory_utilization": 0.75, "enable_prefix_caching": False}
    )
    cmd = serve_sweep._build_vllm_serve_cmd("some/model", 8192, cfg)
    assert cmd[:3] == ["vllm", "serve", "some/model"]
    assert "--max-model-len" in cmd and cmd[cmd.index("--max-model-len") + 1] == "8192"
    assert "--max-num-seqs" in cmd and cmd[cmd.index("--max-num-seqs") + 1] == "42"
    assert "--gpu-memory-utilization" in cmd and cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.75"
    assert "--no-enable-prefix-caching" in cmd


def test_build_vllm_serve_cmd_prefix_caching_true():
    cfg = _sweep_config("subprocess")
    cfg = serve_sweep.SweepConfig(**{**cfg.__dict__, "enable_prefix_caching": True})
    cmd = serve_sweep._build_vllm_serve_cmd("some/model", 4096, cfg)
    assert "--enable-prefix-caching" in cmd
    assert "--no-enable-prefix-caching" not in cmd


class _FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminate_called = False
        self.kill_called = False

    def terminate(self):
        self.terminate_called = True
        self.returncode = -15

    def kill(self):
        self.kill_called = True

    async def wait(self):
        return self.returncode or 0


@pytest.mark.anyio
async def test_subprocess_session_terminates_process_on_exit(monkeypatch):
    fake_proc = _FakeProcess()

    async def fake_spawn(hf_model_id, ctx_len, cfg):
        return fake_proc

    monkeypatch.setattr(serve_sweep, "_spawn_vllm_subprocess", fake_spawn)

    app, _ = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    cfg = _sweep_config("subprocess")

    async with serve_sweep._subprocess_session(cfg, "some/model", 64, transport=transport) as base_url:
        assert base_url == f"http://localhost:{cfg.server.host_port}"
        assert not fake_proc.terminate_called

    assert fake_proc.terminate_called


@pytest.mark.anyio
async def test_subprocess_session_terminates_process_even_on_exception(monkeypatch):
    fake_proc = _FakeProcess()

    async def fake_spawn(hf_model_id, ctx_len, cfg):
        return fake_proc

    monkeypatch.setattr(serve_sweep, "_spawn_vllm_subprocess", fake_spawn)

    app, _ = make_mock_vllm_app()
    transport = httpx.ASGITransport(app=app)
    cfg = _sweep_config("subprocess")

    with pytest.raises(ValueError, match="boom"):
        async with serve_sweep._subprocess_session(cfg, "some/model", 64, transport=transport):
            raise ValueError("boom")

    assert fake_proc.terminate_called
