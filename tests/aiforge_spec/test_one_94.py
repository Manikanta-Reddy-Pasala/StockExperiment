"""Acceptance tests for ONE-94: PIPELINE_SMOKE_LOCAL constant."""
from src.version_smoke_local import PIPELINE_SMOKE_LOCAL


def test_01_src_version_smoke_local_py_contains_pipeline_smoke_local() -> None:
    assert PIPELINE_SMOKE_LOCAL == "local-pass-2026-06-11"


def test_02_a_pytest_test_asserts_the_constant_value() -> None:
    assert PIPELINE_SMOKE_LOCAL == "local-pass-2026-06-11"


def test_03_existing_tests_stay_green() -> None:
    import src.version_smoke_local as vs
    assert hasattr(vs, "PIPELINE_SMOKE_LOCAL")
    assert isinstance(vs.PIPELINE_SMOKE_LOCAL, str)
