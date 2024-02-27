import os
from logging import getLogger
from typing import List

from openllmtelemetry.instrument_langkit import init_langkit_instrumentor
from openllmtelemetry.intrument_openai import init_openai_instrumentor
from openllmtelemetry.version import __version__
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter


diagnostic_logger = getLogger(__name__)


# this will log as warnings the spans on export
class DebugOTLPSpanExporter(OTLPSpanExporter):
    def export(self, spans: List[ReadableSpan]):
        diagnostic_logger.debug("Exporting spans...")
        for span in spans:
            diagnostic_logger.debug(f"Exporting span: {span.name}")
        response = None
        try:
            response = super().export(spans)
            for span in spans:
                diagnostic_logger.debug(f"Done exporting spans for {span.context.trace_id}")
        except Exception as e:
            diagnostic_logger.error(f"Error exporting: {e}")
        return response


def instrument(application_name: str = "llm-traceguard-application", extract_metrics: bool = False, console_out: bool = False):
    whylabs_api_key = os.environ.get("WHYLABS_API_KEY")
    dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID")
    tracer_name = os.environ.get("WHYLABS_TRACER_NAME") or "llm-traceguard"
    service_name = os.environ.get("WHYLABS_TRACER_SERVICE_NAME") or "llm-traceguard-instrumented-service"
    if whylabs_api_key is None or dataset_id is None:
        raise ValueError("In order to export traces to WhyLabs you must set the env variables: WHYLABS_API_KEY and WHYLABS_DEFAULT_DATASET_ID")
    traces_endpoint = os.environ.get("WHYLABS_TRACES_ENDPOINT") or "https://songbird.development.whylabsdev.com/v1/traces"

    resource = Resource(attributes={
        "service.name": service_name,
        "application.name": application_name,
        "version": __version__,
        "resource.id": dataset_id,
    })

    otlp_exporter = DebugOTLPSpanExporter(
            endpoint=traces_endpoint,
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
    if extract_metrics:
        diagnostic_logger.info("Attempting to add instrumentation to `langkit` metrics extraction to LLM traces")
        init_langkit_instrumentor(tracer)
    else:
        diagnostic_logger.info("Not adding `langkit` metrics to LLM traces but you can do so by passing the parameter into instrument like this: instrument(extract_metrics=True)")
    return tracer
