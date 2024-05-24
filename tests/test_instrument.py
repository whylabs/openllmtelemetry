import os

import openllmtelemetry


def test_instrument():
    os.environ["WHYLABS_DEFAULT_ORG_ID"] = "fake-string-for-testing-org-id"
    os.environ["WHYLABS_API_KEY"] = "fake-string-for-testing-key"
    os.environ["WHYLABS_GUARDRAILS_CONFIG"] = "/tmp/fake-config/file/does/not/exist"
    try:
        openllmtelemetry.instrument(
            "my-test-application",
            dataset_id="model-1"
        )
    finally:
        os.environ.pop("WHYLABS_DEFAULT_ORG_ID", None)
        os.environ.pop("WHYLABS_API_KEY", None)
        os.environ.pop("WHYLABS_DEFAULT_DATASET_ID", None)
        os.environ.pop("WHYLABS_GUARDRAILS_CONFIG", None)


def test_version():
    from openllmtelemetry.version import __version__

    assert __version__ is not None
    assert isinstance(__version__, str)
    assert __version__.startswith("0.0")
