import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tokokita.data import generate_dataset  # noqa: E402


@pytest.fixture(scope="session")
def small_ds():
    """A small but realistic dataset shared by all tests (generated once)."""
    return generate_dataset(n_customers=400, seed=7)
