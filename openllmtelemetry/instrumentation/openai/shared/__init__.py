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
import json
import logging
import os
import types
from importlib.metadata import version

import openai

from openllmtelemetry.instrumentation.openai.utils import is_openai_v1
from openllmtelemetry.semantic_conventions.gen_ai import SpanAttributes

OPENAI_API_VERSION = "openai.api_version"
OPENAI_API_BASE = "openai.api_base"
OPENAI_API_TYPE = "openai.api_type"

logger = logging.getLogger(__name__)


def should_send_prompts():
    return (os.getenv("TRACE_PROMPT_AND_RESPONSE") or "false").lower() == "true"


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _set_api_attributes(span, instance=None):
    if not span.is_recording():
        return

    try:
        base_url = getattr(getattr(instance, "_client", None), "base_url", None)

        _set_span_attribute(span, "llm.base_url", str(base_url))
        _set_span_attribute(span, OPENAI_API_TYPE, openai.api_type)
        _set_span_attribute(span, OPENAI_API_VERSION, openai.api_version)
        _set_span_attribute(span, "openapi.client.version", openai.__version__)
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set api attributes for openai span, error: %s", str(ex))

    return


def _set_functions_attributes(span, functions):
    if not functions:
        return

    for i, function in enumerate(functions):
        prefix = f"{SpanAttributes.LLM_REQUEST_FUNCTIONS}.{i}"
        _set_span_attribute(span, f"{prefix}.name", function.get("name"))
        _set_span_attribute(span, f"{prefix}.description", function.get("description"))
        _set_span_attribute(span, f"{prefix}.parameters", json.dumps(function.get("parameters")))


def _set_request_attributes(span, kwargs, vendor="unknown", instance=None):
    if not span.is_recording():
        return

    try:
        _set_api_attributes(span, instance)
        _set_span_attribute(span, SpanAttributes.LLM_VENDOR, vendor)
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, kwargs.get("model"))
        _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
        _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, kwargs.get("temperature"))
        _set_span_attribute(span, SpanAttributes.LLM_TOP_P, kwargs.get("top_p"))
        _set_span_attribute(span, SpanAttributes.LLM_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))
        _set_span_attribute(span, SpanAttributes.LLM_PRESENCE_PENALTY, kwargs.get("presence_penalty"))
        _set_span_attribute(span, SpanAttributes.LLM_USER, kwargs.get("user"))
        _set_span_attribute(span, SpanAttributes.LLM_HEADERS, str(kwargs.get("headers")))
        _set_span_attribute(span, SpanAttributes.LLM_STREAMING, str(kwargs.get("stream")))
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set input attributes for openai span, error: %s", str(ex))


def _set_response_attributes(response, span):
    if not span.is_recording():
        return

    try:
        _set_span_attribute(span, SpanAttributes.LLM_RESPONSE_MODEL, response.get("model"))

        usage = response.get("usage")
        if not usage:
            return

        if is_openai_v1() and not isinstance(usage, dict):
            usage = usage.__dict__

        _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, usage.get("total_tokens"))
        _set_span_attribute(
            span,
            SpanAttributes.LLM_USAGE_COMPLETION_TOKENS,
            usage.get("completion_tokens"),
        )
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, usage.get("prompt_tokens"))

        return
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set response attributes for openai span, error: %s", str(ex))


def is_streaming_response(response):
    if is_openai_v1():
        return isinstance(response, openai.Stream)

    return isinstance(response, types.GeneratorType) or isinstance(response, types.AsyncGeneratorType)


def model_as_dict(model):
    if version("pydantic") < "2.0.0":
        return model.dict()

    return model.model_dump()
