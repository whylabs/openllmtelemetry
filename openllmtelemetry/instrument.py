import os
from logging import getLogger
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from openllmtelemetry.config import load_config, load_dataset_id
from openllmtelemetry.intrument_openai import init_instrumentors
from openllmtelemetry.version import __version__

LOGGER = getLogger(__name__)


def instrument(
    application_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    tracer_name: Optional[str] = None,
    service_name: Optional[str] = None,
    disable_batching: bool = False,
    debug: bool = False,
) -> Tracer:
    config = load_config()
    dataset_id = load_dataset_id(dataset_id)
    guardrails_api = config.guardrail_client(default_dataset_id=dataset_id)

    if application_name is None:
        otel_service_name = os.environ.get("OTEL_SERVICE_NAME")
        if otel_service_name:
            application_name = otel_service_name
        else:
            application_name = "unknown-llm-app"
    if tracer_name is None:
        tracer_name = os.environ.get("WHYLABS_TRACER_NAME") or "openllmtelemetry"
    if service_name is None:
        service_name = os.environ.get("WHYLABS_TRACER_SERVICE_NAME") or "openllmtelemetry-instrumented-service"
    resource = Resource(
        attributes={
            "service.name": service_name,
            "application.name": application_name,
            "version": __version__,
            "resource.id": dataset_id or "",  # TODO: resource id probably should be at the span level
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    config.config_tracer_provider(tracer_provider, dataset_id=dataset_id, disable_batching=disable_batching, debug=debug)

    tracer = trace.get_tracer(tracer_name)
    trace.set_tracer_provider(tracer_provider)

    init_instrumentors(tracer, guardrails_api)
    return tracer
