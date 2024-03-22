import importlib


def init_openai_instrumentor(trace_provider):
    if importlib.util.find_spec("openai") is not None:
        from openllmtelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor()
        instrumentor.instrument(trace_provider=trace_provider)
    else:
        raise ValueError("Need to install openai to instrument openai with OpenLLMTelemetry!")
