import importlib
import logging

from opentelemetry.sdk.trace import Tracer

from openllmtelemetry.secure import WhyLabsSecureApi

LOGGER = logging.getLogger(__name__)


def init_instrumentors(trace_provider: Tracer, secure_api: WhyLabsSecureApi):
    for instrumentor in [init_openai_instrumentor, init_bedrock_instrumentor]:
        instrumentor(trace_provider=trace_provider, secure_api=secure_api)


def init_openai_instrumentor(trace_provider: Tracer, secure_api: WhyLabsSecureApi):
    if importlib.util.find_spec("openai") is not None:
        from openllmtelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(secure_api=secure_api)
        instrumentor.instrument(trace_provider=trace_provider, guard=secure_api)
    else:
        LOGGER.warning("OpenAPI not found, skipping instrumentation")


def init_bedrock_instrumentor(trace_provider: Tracer, secure_api: WhyLabsSecureApi):
    if importlib.util.find_spec("boto3") is not None:
        from openllmtelemetry.instrumentation.bedrock import BedrockInstrumentor

        instrumentor = BedrockInstrumentor(secure_api=secure_api)
        instrumentor.instrument(trace_provider=trace_provider, guard=secure_api)
    else:
        LOGGER.warning("Boto3 not found, skipping instrumentation")
