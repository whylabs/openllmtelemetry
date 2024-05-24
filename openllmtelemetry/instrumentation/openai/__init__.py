"""
Copyright 2024 traceloop

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Changes made: customization for WhyLabs

Original source: openllmetry: https://github.com/traceloop/openllmetry
"""
import logging
from typing import Collection, Optional

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor

from openllmtelemetry.guardrails import GuardrailsApi

from .utils import is_openai_v1
from .v0 import OpenAIV0Instrumentor
from .v1 import OpenAIV1Instrumentor

_instruments = ("openai>=0.27.0",)

LOGGER = logging.getLogger(__name__)


class OpenAIInstrumentor(BaseInstrumentor):
    """An instrumentor for OpenAI's client library."""

    def __init__(self, secure_api: Optional[GuardrailsApi]):
        LOGGER.info("Instrumenting OpenAI")

        self._guard = secure_api
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
