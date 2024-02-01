from logging import getLogger
from typing import List, Optional
from .intrument_openai import init_openai_instrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Span, StatusCode
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter


diagnostic_logger = getLogger(__name__)


# this will log as warnings the spans on export
class DebugOTLPSpanExporter(OTLPSpanExporter):
    def export(self, spans: List[Span]):
        diagnostic_logger.debug("Exporting spans...")
        for span in spans:
            diagnostic_logger.debug(f"Exporting span: {span.name}")
        response = None
        try:
            response = super().export(spans)
            diagnostic_logger.debug(f"Done exporting spans for {span.context.trace_id}")
        except Exception as e:
            diagnostic_logger.error(f"Error exporting: {e}")
        return response


def instrument(application_name: str = "llm-traceguard-application", console_out: bool = False):
    whylabs_api_key = os.environ.get("WHYLABS_API_KEY")
    dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
    tracer_name = os.environ.get("WHYLABS_TRACER_NAME") or "llm-traceguard"
    service_name = os.environ.get("WHYLABS_TRACER_SERVICE_NAME") or "llm-traceguard-instrumented-service"
    if whylabs_api_key is None or dataset_id is None:
        raise ValueError("In order to export traces to WhyLabs you must set the env variables: WHYLABS_API_KEY and WHYLABS_DEFAULT_DATASET_ID")

    resource = Resource(attributes={
        "service.name": service_name,
        "application.name": application_name,
        "version": "0.0.1",
        "resource.id": dataset_id,
    })

    otlp_exporter = DebugOTLPSpanExporter(
            endpoint="https://songbird.development.whylabsdev.com/v1/traces",
            headers={"X-API-Key": whylabs_api_key, "X-WHYLABS-RESOURCE": dataset_id})

    span_processor = BatchSpanProcessor(otlp_exporter)

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(span_processor)
    tracer = trace.get_tracer(tracer_name)

    # Debug console output
    if console_out:
        processor2 = BatchSpanProcessor(ConsoleSpanExporter())
        tracer_provider.add_span_processor(processor2)
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(meter_provider)

    trace.set_tracer_provider(tracer_provider)

    init_openai_instrumentor(tracer)
    return tracer
