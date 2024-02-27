import openllmtelemetry
import os
import pytest


def test_instrument_checks_for_keys():
    with pytest.raises(ValueError):
        openllmtelemetry.instrument()

def test_instrument():
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = "fake-string-for-testing-org-id"
    os.environ["WHYLABS_API_KEY"] = "fake-string-for-testing-key"
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "fake-string-for-testing-model-id"
    openllmtelemetry.instrument("my-test-application", console_out=True)
    os.environ.pop("WHYLABS_DEFAULT_ORG_ID", None)
    os.environ.pop("WHYLABS_API_KEY", None)
    os.environ.pop("WHYLABS_DEFAULT_DATASET_ID", None)


def test_instrument_langkit():
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = "fake-string-for-testing-org-id"
    os.environ["WHYLABS_API_KEY"] = "fake-string-for-testing-key"
    os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "fake-string-for-testing-model-id"
    openllmtelemetry.instrument(extract_metrics=True)
    os.environ.pop("WHYLABS_DEFAULT_ORG_ID", None)
    os.environ.pop("WHYLABS_API_KEY", None)
    os.environ.pop("WHYLABS_DEFAULT_DATASET_ID", None)


def test_version():
    version = openllmtelemetry.version.__version__
    assert version is not None
    assert isinstance(version, str)
    assert version.startswith("0.0")
