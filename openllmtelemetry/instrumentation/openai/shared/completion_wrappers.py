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
from typing import Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace.status import Status, StatusCode

from openllmtelemetry.guardrails import GuardrailsApi
from openllmtelemetry.guardrails.handlers import async_wrapper, sync_wrapper
from openllmtelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    is_streaming_response,
    model_as_dict,
    should_send_prompts,
)
from openllmtelemetry.instrumentation.openai.utils import (
    _with_tracer_wrapper,
    is_openai_v1,
)
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes

SPAN_NAME = "openai.completion"
LLM_REQUEST_TYPE = LLMRequestTypeValues.COMPLETION

logger = logging.getLogger(__name__)


def create_prompt_provider(kwargs):
    def prompt_provider():
        return kwargs.get("prompt")

    return prompt_provider


@_with_tracer_wrapper
def completion_wrapper(tracer, guardrails_api: Optional[GuardrailsApi], wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    prompt_provider = create_prompt_provider(kwargs)

    def call_llm(span):
        r = wrapped(*args, **kwargs)
        if not kwargs.get("stream"):
            _handle_response(r, span)
            if is_openai_v1():
                response_dict = model_as_dict(r)
            else:
                response_dict = r

            _set_response_attributes(response_dict, span)
        return r

    def prompt_attributes_setter(span):
        _set_request_attributes(span, kwargs, instance=instance)

    def response_extractor(r):
        if is_openai_v1():
            response_dict = model_as_dict(r)
        else:
            response_dict = r
        return response_dict["choices"][0]["text"]

    return sync_wrapper(
        tracer, guardrails_api, prompt_provider, call_llm, response_extractor, prompt_attributes_setter, LLMRequestTypeValues.COMPLETION
    )


@_with_tracer_wrapper
async def acompletion_wrapper(tracer, guardrails_api: Optional[GuardrailsApi], wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    prompt_provider = create_prompt_provider(kwargs)

    async def call_llm(span):
        r = await wrapped(*args, **kwargs)
        if not kwargs.get("stream"):
            _handle_response(r, span)
        return r

    def prompt_attributes_setter(span):
        _set_request_attributes(span, kwargs, instance=instance)

    def response_extractor(r):
        if is_openai_v1():
            response_dict = model_as_dict(r)
        else:
            response_dict = r
        return response_dict["choices"][0]["text"]

    return async_wrapper(
        tracer,
        guardrails_api,  # guardrails_client,
        prompt_provider,
        call_llm,
        response_extractor,
        kwargs,
        prompt_attributes_setter,
        _build_from_streaming_response,
        is_streaming_response,
        LLMRequestTypeValues.COMPLETION,
    )


def _handle_response(response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response

    _set_response_attributes(response_dict, span)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))


def _set_prompts(span, prompt):
    if not span.is_recording() or not prompt:
        return

    try:
        _set_span_attribute(
            span,
            f"{SpanAttributes.LLM_PROMPTS}.0.user",
            prompt[0] if isinstance(prompt, list) else prompt,
        )
    except Exception as ex:  # pylint: disable=broad-except
        logger.warning("Failed to set prompts for openai span, error: %s", str(ex))


def _set_completions(span, choices):
    if not span.is_recording() or not choices:
        return

    try:
        for choice in choices:
            index = choice.get("index")
            prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
            _set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))
            _set_span_attribute(span, f"{prefix}.content", choice.get("text"))
    except Exception as e:
        logger.warning("Failed to set completion attributes, error: %s", str(e))


def _build_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    for item in response:
        item_to_yield = item
        if is_openai_v1():
            item = model_as_dict(item)

        for choice in item.get("choices"):
            index = choice.get("index")
            if len(complete_response.get("choices")) <= index:
                complete_response["choices"].append({"index": index, "text": ""})
            complete_choice = complete_response.get("choices")[index]
            if choice.get("finish_reason"):
                complete_choice["finish_reason"] = choice.get("finish_reason")

            complete_choice["text"] += choice.get("text")

        yield item_to_yield

    _set_response_attributes(complete_response, span)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()
