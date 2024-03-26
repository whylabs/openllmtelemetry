import importlib
from typing import Any, Dict

from opentelemetry.trace import TracerProvider


def init_openai_instrumentor(trace_provider: TracerProvider, **kwargs: Dict[str, Any]):
    if importlib.util.find_spec("openai") is not None:  # type: ignore
        from openllmtelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument(trace_provider=trace_provider, **kwargs)  # type: ignore
    else:
        raise ValueError("Need to install openai to instrument openai with OpenLLMTelemetry!")
