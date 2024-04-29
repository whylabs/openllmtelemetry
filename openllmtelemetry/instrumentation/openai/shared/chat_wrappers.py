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

from opentelemetry import context as context_api

# noinspection PyProtectedMember
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import Status, StatusCode

from openllmtelemetry.instrumentation.openai.shared import (
    _set_functions_attributes,
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    is_streaming_response,
    model_as_dict,
    should_send_prompts,
)
from openllmtelemetry.instrumentation.openai.utils import _with_tracer_wrapper, is_openai_v1
from openllmtelemetry.secure import GuardrailsApi  # noqa: E402
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT

LOGGER = logging.getLogger(__name__)


@_with_tracer_wrapper
def chat_wrapper(tracer, secure_api: GuardrailsApi, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    # span needs to be opened and closed manually because the response is a generator
    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value, "span.type": "completion"},
    )

    (prompt, prompt_metrics) = _handle_request(secure_api, span, kwargs)
    if prompt_metrics:
        LOGGER.debug(prompt_metrics)
        metrics = prompt_metrics.metrics[0]

        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                span.set_attribute(f"langkit.metrics.{k}", metrics.additional_properties[k])

    response = wrapped(*args, **kwargs)

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _build_from_streaming_response(span, response)

    _handle_response(secure_api, prompt, response, span)
    span.end()

    return response


@_with_tracer_wrapper
async def achat_wrapper(tracer, guard: GuardrailsApi, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    span = tracer.start_span(
        SPAN_NAME,
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: LLM_REQUEST_TYPE.value, "span.type": "completion"},
    )
    prompt, prompt_metrics = _handle_request(guard, span, kwargs)
    LOGGER.debug("Prompt metrics: ", prompt_metrics)
    response = await wrapped(*args, **kwargs)
    LOGGER.debug(f"Async type response: {type(response)}")

    if is_streaming_response(response):
        # span will be closed after the generator is done
        return _abuild_from_streaming_response(span, response)

    _handle_response(guard, prompt, response, span)
    span.end()

    return response


def _handle_request(secure_api: GuardrailsApi, span, kwargs):
    _set_request_attributes(span, kwargs)
    stream = kwargs.get("stream")
    LOGGER.debug("Stream: %s. ", stream)
    LOGGER.debug(f"Stream: {stream}")
    messages = kwargs.get("messages")
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    prompt = user_messages[-1]
    prompt_metrics = secure_api.eval_prompt(prompt) if secure_api is not None else None

    if should_send_prompts():
        _set_prompts(span, messages)
        _set_functions_attributes(span, kwargs.get("functions"))

    return prompt, prompt_metrics


def _handle_response(secure_api: GuardrailsApi, prompt, response, span):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response
    response = response_dict["choices"][0]["message"]["content"]
    response_metrics = secure_api.eval_response(prompt=prompt, response=response) if secure_api is not None else None
    if response_metrics:
        LOGGER.debug(response_metrics)
        metrics = response_metrics.metrics[0]

        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                span.set_attribute(f"langkit.metrics.{k}", metrics.additional_properties[k])

    _set_response_attributes(span, response_dict)

    if should_send_prompts():
        _set_completions(span, response_dict.get("choices"))

    return response


def _set_prompts(span, messages):
    if not span.is_recording() or messages is None:
        return

    try:
        for i, msg in enumerate(messages):
            prefix = f"{SpanAttributes.LLM_PROMPTS}.{i}"
            content = None
            if isinstance(msg.get("content"), str):
                content = msg.get("content")
            elif isinstance(msg.get("content"), list):
                content = json.dumps(msg.get("content"))

            _set_span_attribute(span, f"{prefix}.role", msg.get("role"))
            if content:
                _set_span_attribute(span, f"{prefix}.content", content)
    except Exception as ex:  # pylint: disable=broad-except
        LOGGER.warning("Failed to set prompts for openai span, error: %s", str(ex))


def _set_completions(span, choices):
    if choices is None:
        return

    for choice in choices:
        index = choice.get("index")
        prefix = f"{SpanAttributes.LLM_COMPLETIONS}.{index}"
        _set_span_attribute(span, f"{prefix}.finish_reason", choice.get("finish_reason"))

        message = choice.get("message")
        if not message:
            return

        _set_span_attribute(span, f"{prefix}.role", message.get("role"))
        _set_span_attribute(span, f"{prefix}.content", message.get("content"))

        function_call = message.get("function_call")
        if not function_call:
            return

        _set_span_attribute(span, f"{prefix}.function_call.name", function_call.get("name"))
        _set_span_attribute(span, f"{prefix}.function_call.arguments", function_call.get("arguments"))


def _build_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    for item in response:
        item_to_yield = item
        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    async for item in response:
        item_to_yield = item
        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(span, complete_response)

    if should_send_prompts():
        _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


def _accumulate_stream_items(item, complete_response):
    if is_openai_v1():
        item = model_as_dict(item)

    for choice in item.get("choices"):
        index = choice.get("index")
        if len(complete_response.get("choices")) <= index:
            complete_response["choices"].append({"index": index, "message": {"content": "", "role": ""}})
        complete_choice = complete_response.get("choices")[index]
        if choice.get("finish_reason"):
            complete_choice["finish_reason"] = choice.get("finish_reason")

        delta = choice.get("delta")

        if delta.get("content"):
            complete_choice["message"]["content"] += delta.get("content")
        if delta.get("role"):
            complete_choice["message"]["role"] = delta.get("role")
