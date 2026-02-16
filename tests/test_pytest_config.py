import pytest


def test_rpy2_marker_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.strip().startswith("rpy2") for m in markers), (
        "pytest marker 'rpy2' must be registered in pytest.ini to avoid "
        "PytestUnknownMarkWarning"
    )
