from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.zipkin.json import ZipkinExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from .intrument_openai import init_openai_instrumentor

resource = Resource(attributes={
    SERVICE_NAME: "llm-traceguard"
})

tracer_provider = TracerProvider(resource=resource)


trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("llm-traceguard")

init_openai_instrumentor(trace_provider=tracer_provider)
