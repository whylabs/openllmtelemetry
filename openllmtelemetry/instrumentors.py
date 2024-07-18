import importlib
import logging
from typing import Optional

from opentelemetry.trace import Tracer

from openllmtelemetry.guardrails import GuardrailsApi

LOGGER = logging.getLogger(__name__)


def init_instrumentors(trace_provider: Tracer, secure_api: Optional[GuardrailsApi]):
    for instrumentor in [init_openai_instrumentor, init_bedrock_instrumentor, init_watsonx_instrumentor]:
        instrumentor(trace_provider=trace_provider, secure_api=secure_api)


def init_openai_instrumentor(trace_provider: Tracer, secure_api: Optional[GuardrailsApi]):
    if importlib.util.find_spec("openai") is not None:  # type: ignore
        from openllmtelemetry.instrumentation.openai import OpenAIInstrumentor

        instrumentor = OpenAIInstrumentor(secure_api=secure_api)
        instrumentor.instrument(trace_provider=trace_provider, guard=secure_api)  # type: ignore
    else:
        LOGGER.info("OpenAPI not found, skipping instrumentation")


def init_bedrock_instrumentor(trace_provider: Tracer, secure_api: Optional[GuardrailsApi]):
    if importlib.util.find_spec("boto3") is not None:  # type: ignore
        from openllmtelemetry.instrumentation.bedrock import BedrockInstrumentor

        instrumentor = BedrockInstrumentor(secure_api=secure_api)
        instrumentor.instrument(trace_provider=trace_provider, guard=secure_api)  # type: ignore
    else:
        LOGGER.info("boto3 not found, skipping instrumenting Amazon Bedrock")


def init_watsonx_instrumentor(trace_provider: Tracer, secure_api: Optional[GuardrailsApi]):
    if importlib.util.find_spec("ibm_watsonx_ai") is not None:  # type: ignore
        from openllmtelemetry.instrumentation.watsonx import WatsonxInstrumentor

        instrumentor = WatsonxInstrumentor(guardrails_api=secure_api)
        instrumentor.instrument(trace_provider=trace_provider, guard=secure_api)  # type: ignore
    else:
        LOGGER.info("watsonx not found, skipping instrumentation")
