import pytest
import llm_traceguard


def test_instrument():
    with pytest.raises(NotImplementedError):
        llm_traceguard.instrument()
