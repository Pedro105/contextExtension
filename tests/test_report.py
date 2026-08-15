"""build_report / write_report: the sweep table behind the costmodel_report CLI."""

from ctxcost.costmodel import ArchSpec, GPUSpec, build_report, write_report

SPEC = ArchSpec(
    hf_model_id="synthetic/test",
    num_hidden_layers=4,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    hidden_size=512,
    max_position_embeddings=8192,
    rope_theta=10000.0,
    torch_dtype="bfloat16",
)
GPU = GPUSpec(name="Fake GPU 40GB", vram_gib=40)


def test_build_report_shape():
    ctx_lens = [4096, 8192]
    df = build_report([SPEC], [GPU], ctx_lens=ctx_lens, util=0.9, max_num_seqs=64)
    assert len(df) == len(ctx_lens)
    assert set(df["ctx_len"]) == set(ctx_lens)
    assert (df["model"] == "synthetic/test").all()
    assert (df["gpu"] == "Fake GPU 40GB").all()
    assert set(df["binding"]) <= {"kv", "scheduler"}


def test_write_report_produces_csv_and_markdown(tmp_path):
    df = build_report([SPEC], [GPU], ctx_lens=[4096], util=0.9, max_num_seqs=64)
    csv_path, md_path = write_report(df, tmp_path, basename="report")
    assert csv_path.exists() and csv_path.suffix == ".csv"
    assert md_path.exists() and md_path.suffix == ".md"

    csv_text = csv_path.read_text()
    assert "synthetic/test" in csv_text

    md_text = md_path.read_text()
    assert md_text.startswith("| model")
    assert "synthetic/test" in md_text
