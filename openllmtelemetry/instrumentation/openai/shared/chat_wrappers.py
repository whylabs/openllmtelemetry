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
import time
from typing import Optional

# Get current datetime in epoch seconds and convert to int
from opentelemetry import context as context_api

# noinspection PyProtectedMember
from opentelemetry.instrumentation.utils import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace.status import Status, StatusCode
from whylogs_container_client.models import EvaluationResult

from openllmtelemetry.guardrails import GuardrailsApi  # noqa: E402
from openllmtelemetry.guardrails.handlers import async_wrapper, sync_wrapper
from openllmtelemetry.instrumentation.openai.shared import (
    _set_request_attributes,
    _set_response_attributes,
    _set_span_attribute,
    is_streaming_response,
    model_as_dict,
    should_send_prompts,
)
from openllmtelemetry.instrumentation.openai.utils import _with_tracer_wrapper, is_openai_v1
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes

SPAN_TYPE = "span.type"

SPAN_NAME = "openai.chat"
LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT

LOGGER = logging.getLogger(__name__)


def create_prompt_provider(kwargs):
    def prompt_provider():
        messages = kwargs.get("messages")
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        prompt = user_messages[-1]
        return prompt

    return prompt_provider


def _handle_response(response):
    if is_openai_v1():
        response_dict = model_as_dict(response)
    else:
        response_dict = response
    response = response_dict["choices"][0]["message"]["content"]

    return response


@_with_tracer_wrapper
def chat_wrapper(tracer, guardrails_api: GuardrailsApi, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    prompt_provider = create_prompt_provider(kwargs)
    host = getattr(getattr(getattr(instance, "_client", None), "base_url", None), "host", None)
    vendor = "GenericOpenAI"
    span_name = "llm.chat"
    if host and host.endswith(".openai.com"):
        vendor = "OpenAI"
        span_name = "openai.chat"
    elif host.endswith(".azure.com"):
        vendor = "AzureOpenAI"
        span_name = "azureopenai.chat"
    elif host.endswith(".nvidia.com"):
        vendor = "Nvidia"
        span_name = "nvidia.nim.chat"

    def call_llm(span):
        r = wrapped(*args, **kwargs)
        is_streaming = kwargs.get("stream")
        if not is_streaming:
            if is_openai_v1():
                response_dict = model_as_dict(r)
            else:
                response_dict = r

            _set_response_attributes(response_dict, span)
        return r, is_streaming

    def blocked_message_factory(eval_result: Optional[EvaluationResult] = None, is_prompt=True, open_api_v1=True, is_streaming=False):
        if open_api_v1:
            from openai.types.chat.chat_completion import ChatCompletion, Choice
            from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
            from openai.types.chat.chat_completion_message import ChatCompletionMessage
            from openai.types.completion_usage import CompletionUsage

            if is_prompt:
                content = f"Prompt blocked by WhyLabs: {eval_result.action.block_message}"
            else:
                content = f"Response blocked by WhyLabs: {eval_result.action.block_message}"
            choice = Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content=content,  #
                    role="assistant",
                ),
            )
            current_epoch_time = int(time.time())

            if not is_streaming:
                return ChatCompletion(
                    id="whylabs-guardrails-blocked",
                    choices=[
                        choice,
                    ],
                    created=current_epoch_time,
                    model="whylabs-guardrails",
                    object="chat.completion",
                    system_fingerprint=None,
                    usage=CompletionUsage(completion_tokens=0, prompt_tokens=0, total_tokens=0),
                )
            else:
                return ChatCompletionChunk(
                    id="whylabs-guardrails-blocked",
                    created=current_epoch_time,
                    choices=[choice],
                    model="whylabs-guardrails",
                    object="chat.completion.chunk",
                )

    def prompt_attributes_setter(span):
        _set_request_attributes(span, kwargs, vendor=vendor, instance=instance)

    def response_extractor(r):
        if is_openai_v1():
            response_dict = model_as_dict(r)
        else:
            response_dict = r
        return response_dict["choices"][0]["message"]["content"]

    return sync_wrapper(
        tracer,
        guardrails_api,
        prompt_provider,
        call_llm,
        response_extractor,
        prompt_attributes_setter,
        LLMRequestTypeValues.CHAT,
        blocked_message_factory=blocked_message_factory,
        completion_span_name=span_name,
    )


@_with_tracer_wrapper
async def achat_wrapper(tracer, guardrails_api: GuardrailsApi, wrapped, instance, args, kwargs):
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    prompt_provider = create_prompt_provider(kwargs)

    async def call_llm(span):
        r = await wrapped(*args, **kwargs)
        is_streaming = kwargs.get("stream")
        if not is_streaming:
            _handle_response(r)
            if is_openai_v1():
                response_dict = model_as_dict(r)
            else:
                response_dict = r

            _set_response_attributes(response_dict, span)
            return True, r

        else:
            # TODO: handle streaming response. Where does guard response live?
            res = _abuild_from_streaming_response(span, r)
            return False, res

    def prompt_attributes_setter(span):
        _set_request_attributes(span, kwargs, instance=instance)

    def response_extractor(r):
        if is_openai_v1():
            response_dict = model_as_dict(r)
        else:
            response_dict = r
        return response_dict["choices"][0]["message"]["content"]

    await async_wrapper(
        tracer,
        guardrails_api,  # guardrails_client,
        prompt_provider,
        call_llm,
        response_extractor,
        kwargs,
        prompt_attributes_setter,
        _abuild_from_streaming_response,
        is_streaming_response,
        LLMRequestTypeValues.CHAT,
    )


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

    _set_response_attributes(complete_response, span)

    # if should_send_prompts():
    #     _set_completions(span, complete_response.get("choices"))

    span.set_status(Status(StatusCode.OK))
    span.end()


async def _abuild_from_streaming_response(span, response):
    complete_response = {"choices": [], "model": ""}
    async for item in response:
        item_to_yield = item
        _accumulate_stream_items(item, complete_response)

        yield item_to_yield

    _set_response_attributes(complete_response, span)

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
