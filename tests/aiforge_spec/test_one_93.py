"""Acceptance tests for ONE-93: PIPELINE_SMOKE constant."""
from src.version_smoke import PIPELINE_SMOKE


def test_01_version_smoke_py_contains_pipeline_smoke_v6_workflow_2026() -> None:
    assert PIPELINE_SMOKE == "v6-workflow-2026-06-11"


def test_02_a_pytest_test_asserts_the_constant_value() -> None:
    assert PIPELINE_SMOKE == "v6-workflow-2026-06-11"


def test_03_existing_tests_stay_green() -> None:
    import src.version_smoke as vs
    assert hasattr(vs, "PIPELINE_SMOKE")
    assert isinstance(vs.PIPELINE_SMOKE, str)
