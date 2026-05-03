from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def test_cases_dir() -> Path:
    return REPO_ROOT / "test_cases"


@pytest.fixture(scope="session")
def case3_balanced(test_cases_dir: Path) -> str:
    return str(test_cases_dir / "case3_balanced.dss")


@pytest.fixture(scope="session")
def ieee13(test_cases_dir: Path) -> str:
    return str(test_cases_dir / "13Bus" / "IEEE13Nodeckt.dss")
