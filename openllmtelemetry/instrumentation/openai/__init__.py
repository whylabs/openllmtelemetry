import logging
from typing import Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from openllmtelemetry.guard import WhyLabsGuard

from .utils import is_openai_v1
from .v0 import OpenAIV0Instrumentor
from .v1 import OpenAIV1Instrumentor

_instruments = ("openai>=0.27.0",)

LOGGER = logging.getLogger(__name__)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(self, guard: WhyLabsGuard):
        LOGGER.info("Instrumenting Bedrock")

        self._guard = guard
        if is_openai_v1():
            self._instrumentor = OpenAIV1Instrumentor(self._guard)
        else:
            self._instrumentor = OpenAIV0Instrumentor(self._guard)

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        self._instrumentor.instrument(**kwargs)

    def _uninstrument(self, **kwargs):
        self._instrumentor.uninstrument(**kwargs)