import os
from logging import getLogger
from typing import Dict, Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from openllmtelemetry.config import load_config, load_dataset_id
from openllmtelemetry.instrumentors import init_instrumentors
from openllmtelemetry.version import __version__

LOGGER = getLogger(__name__)

_tracer_cache: Dict[str, trace.Tracer] = {}
_last_added_tracer: Optional[Tuple[str, trace.Tracer]] = None


def instrument(
    application_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    tracer_name: Optional[str] = None,
    service_name: Optional[str] = None,
    disable_batching: bool = False,
    debug: bool = False,
) -> Tracer:
    global _tracer_cache, _last_added_tracer

    config = load_config()
    dataset_id = load_dataset_id(dataset_id)
    if dataset_id is None:
        raise ValueError(
            "dataset_id must be specified in a parameter or in env var: e.g. " 'os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-1"'
        )
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
    _tracer_cache[tracer_name] = tracer
    _last_added_tracer = (tracer_name, tracer)

    init_instrumentors(tracer, guardrails_api)
    return tracer


def get_tracer(name: Optional[str] = None) -> Optional[Tracer]:
    if _last_added_tracer is None:
        return None
    elif name is None:
        return _last_added_tracer[1]
    else:
        return _tracer_cache.get(name)
