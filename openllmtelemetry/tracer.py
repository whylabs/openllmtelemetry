from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from .intrument_openai import init_openai_instrumentor

resource = Resource(attributes={SERVICE_NAME: "llm-traceguard"})

tracer_provider = TracerProvider(resource=resource)


trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer("llm-traceguard")

init_openai_instrumentor(tracer_provider=tracer_provider)
