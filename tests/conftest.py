import matplotlib
import pytest

matplotlib.use("Agg")  # headless test environment

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "hf_configs"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR
