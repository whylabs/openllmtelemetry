import os

import pytest

import openllmtelemetry


def test_instrument_checks_for_keys():
    with pytest.raises(ValueError):
        openllmtelemetry.instrument()


def test_instrument():
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = "fake-string-for-testing-org-id"
    os.environ["WHYLABS_API_KEY"] = "fake-string-for-testing-key"
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "fake-string-for-testing-model-id"
    openllmtelemetry.instrument("my-test-application")
    os.environ.pop("WHYLABS_DEFAULT_ORG_ID", None)
    os.environ.pop("WHYLABS_API_KEY", None)
    os.environ.pop("WHYLABS_DEFAULT_DATASET_ID", None)


def test_version():
    from openllmtelemetry.version import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__.startswith("0.0")
