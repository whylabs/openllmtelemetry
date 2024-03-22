from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from .utils import is_openai_v1
from .v0 import OpenAIV0Instrumentor
from .v1 import OpenAIV1Instrumentor

_instruments = ("openai>=0.27.0",)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        if is_openai_v1():
            OpenAIV1Instrumentor().instrument(**kwargs)
        else:
            OpenAIV0Instrumentor().instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        if is_openai_v1():
            OpenAIV1Instrumentor().uninstrument(**kwargs)
        else:
            OpenAIV0Instrumentor().uninstrument(**kwargs)
