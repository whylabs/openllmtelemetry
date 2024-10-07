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
import io
import json
import logging
import os
import uuid
from functools import wraps
from typing import Any, Collection, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.trace import SpanKind, get_tracer, set_span_in_context
from opentelemetry.trace.span import Span
from whylogs_container_client.models import EvaluationResult
from wrapt import wrap_function_wrapper

from openllmtelemetry.guardrails import GuardrailsApi  # noqa: E402
from openllmtelemetry.guardrails.handlers import _create_guardrail_span, generate_event
from openllmtelemetry.instrumentation.bedrock.reusable_streaming_body import ReusableStreamingBody
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes
from openllmtelemetry.version import __version__

LOGGER = logging.getLogger(__name__)

_instruments = ("boto3 >= 1.28.57",)
SPAN_TYPE = "span.type"

WRAPPED_METHODS = [{"package": "botocore.client", "object": "ClientCreator", "method": "create_client"}]

def should_send_prompts():
    return (os.getenv("TRACE_PROMPT_AND_RESPONSE") or "false").lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span: Span, name: str, value: Any) -> None:
    if value is not None:
        if value != "":
            span.set_attribute(name, value)


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, guardrails_api: GuardrailsApi, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, guardrails_api, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


class BlockedMessageStream(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)

    def read(self, amt=None):
        return super().read(amt)


def _create_blocked_response_streaming_body(content):
    content_stream = BlockedMessageStream(content)
    content_length = len(content)
    return ReusableStreamingBody(content_stream, content_length)


def _handle_request(guardrails_api: Optional[GuardrailsApi], prompt: str, span: Span):
    evaluation_results = None
    if prompt is not None:
        guardrail_response = guardrails_api.eval_prompt(prompt, context=set_span_in_context(span), span=span) if guardrails_api is not None else None
        if hasattr(guardrail_response, "parsed"):
            evaluation_results = getattr(guardrail_response, "parsed")
        else:
            evaluation_results = guardrail_response
    if evaluation_results and span is not None:
        client_side_metrics = os.environ.get("INCLUDE_CLIENT_SIDE_GUARDRAILS_METRICS", None)
        if not client_side_metrics:
            span.set_attribute("guardrails.client_side_metrics.tracing", 0)
            return evaluation_results
        span.set_attribute("guardrails.client_side_metrics.tracing", 1)
        LOGGER.debug(evaluation_results)
        metrics = evaluation_results
        metrics = evaluation_results.metrics[0]
        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                metric_value = metrics.additional_properties[k]
                if not str(k).endswith(".redacted"):
                    span.set_attribute(f"langkit.metrics.{k}", metric_value)
    return evaluation_results


def _handle_response(secure_api: Optional[GuardrailsApi], prompt, response, span):
    response_text: Optional[str] = None
    response_metrics = None
    # Titan
    if "results" in response:
        results = response.get("results")
        if results and results[0]:
            response_message = results[0]
            response_text = response_message.get("outputText")
    # Claude
    elif "content" in response:
        content = response.get("content")
        if content:
            response_message = content[0]
            if response_message and "text" in response_message:
                response_text = response_message.get("text")
    # LLama 2/3 instruct/chat
    elif "generation" in response:
        response_text = response.get("generation")
    response_metrics = None
    if response_text is not None:
        guardrail_response = secure_api.eval_response(prompt=prompt, response=response_text, context=set_span_in_context(span), span=span) if secure_api is not None else None
        if hasattr(guardrail_response, "parsed"):
            response_metrics = getattr(guardrail_response, "parsed")
        else:
            response_metrics = guardrail_response
    if response_metrics:
        LOGGER.debug(response_metrics)
        metrics = response_metrics.metrics[0]
        client_side_metrics = os.environ.get("INCLUDE_CLIENT_SIDE_GUARDRAILS_METRICS", None)
        if not client_side_metrics:
            span.set_attribute("guardrails.client_side_metrics.tracing", 0)
            return response_metrics
        span.set_attribute("guardrails.client_side_metrics.tracing", 1)
        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                metric_value = metrics.additional_properties[k]
                if not str(k).endswith(".redacted"):
                    span.set_attribute(f"langkit.metrics.{k}", metric_value)
    else:
        LOGGER.debug("response metrics is none, skipping")

    return response_metrics


@_with_tracer_wrapper
def _wrap(tracer, secure_api: GuardrailsApi, to_wrap, wrapped, instance, args, kwargs):
    """Instruments and calls every function defined in TO_WRAP."""
    if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
        return wrapped(*args, **kwargs)

    if kwargs.get("service_name") == "bedrock-runtime":
        client = wrapped(*args, **kwargs)
        client.invoke_model = _instrumented_model_invoke(client.invoke_model, tracer, secure_api)

        return client
    response = wrapped(*args, **kwargs)
    return response


def _instrumented_model_invoke(fn, tracer, secure_api: GuardrailsApi):
    @wraps(fn)
    def with_instrumentation(*args, **kwargs):
        with tracer.start_as_current_span(name="interaction",
                                          kind=SpanKind.CLIENT,
                                          attributes={SpanAttributes.SPAN_TYPE: "interaction"}) as span:
            request_body = json.loads(kwargs.get("body"))
            (vendor, model) = kwargs.get("modelId").split(".")
            is_titan_text = model.startswith("titan-text-")

            prompt = None
            if vendor == "cohere":
                prompt = request_body.get("prompt")
            elif vendor == "anthropic":
                messages = request_body.get("messages")
                last_message = messages[-1]
                if last_message:
                    prompt = last_message.get('content')
            elif vendor == "ai21":
                prompt = request_body.get("prompt")
            elif vendor == "meta":
                prompt = request_body.get("prompt")
            elif vendor == "amazon":
                if is_titan_text:
                    prompt = request_body["inputText"]
                else:
                    LOGGER.debug("LLM not suppported yet")
            LOGGER.debug(f"extracted prompt: {prompt}")

            with _create_guardrail_span(tracer) as guardrail_span:
                eval_result = _handle_request(secure_api, prompt, guardrail_span)

            def blocked_message_factory(eval_result: Optional[EvaluationResult] = None, is_prompt=True, is_streaming=False, request_id = None):
                message = None
                if eval_result and hasattr(eval_result, "action"):
                    action = eval_result.action
                    if hasattr(action, "message"):
                        message = action.message
                    elif hasattr(action, "block_message"):
                        message = action.block_message

                if is_prompt:
                    content = f"Prompt blocked by WhyLabs: {message}"
                else:
                    content = f"Response blocked by WhyLabs: {message}"
                blocked_message = os.environ.get("GUARDRAILS_BLOCKED_MESSAGE_OVERRIDE", content)
                # default to Amazon's response format
                response_content = json.dumps({
                            "inputTextTokenCount": 0,
                            "results": [
                                {
                                    "tokenCount": 0,
                                    "outputText": blocked_message,
                                    "completionReason": "FINISH"
                                }
                            ]
                        }).encode('utf-8')
                if vendor == "meta":
                     response_content = json.dumps({
                        "generation": blocked_message,
                        "prompt_token_count": 0,
                        "generation_token_count": 0,
                        }).encode('utf-8')
                elif vendor == "anthropic":
                    response_content = json.dumps({
                        'id': str(uuid.uuid4()),
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [
                            {'type': 'text', 'text': blocked_message}
                        ],
                        'stop_reason': 'end_turn',
                        'stop_sequence': None,
                        'usage': {'input_tokens': 0, 'output_tokens': 0}
                        }).encode('utf-8')
                blocked_message_body = _create_blocked_response_streaming_body(response_content)
                if request_id is None:
                    request_id = str(uuid.uuid4())
                blocked_response = {
                    'ResponseMetadata': {
                        'RequestId': request_id,
                        'HTTPStatusCode': 200,
                        'HTTPHeaders': {},
                        'RetryAttempts': 0
                    },
                    'contentType': 'application/json',
                    'body': blocked_message_body
                }

                return blocked_response

            if eval_result and eval_result.action and eval_result.action.action_type == 'block':
                blocked_prompt_response = blocked_message_factory(eval_result=eval_result)
                if eval_result.validation_results:
                    eval_metadata = eval_result.metadata.additional_properties
                    generate_event(eval_result.validation_results.report, eval_metadata, span) # LOGGER.debug(f"blocked prompt: {eval_metadata}")
                return blocked_prompt_response

            response = None
            with tracer.start_as_current_span("bedrock.completion", kind=SpanKind.CLIENT) as completion_span:
                _set_span_attribute(completion_span, SpanAttributes.LLM_VENDOR, vendor)
                _set_span_attribute(completion_span, SpanAttributes.LLM_REQUEST_MODEL, model)
                response = fn(*args, **kwargs) # need to copy for a fake response
                response["body"] = ReusableStreamingBody(response["body"]._raw_stream, response["body"]._content_length)
                response_body = json.loads(response.get("body").read())

                if vendor == "cohere":
                    _set_cohere_span_attributes(completion_span, request_body, response_body)
                elif vendor == "anthropic":
                    _set_anthropic_span_attributes(completion_span, request_body, response_body)
                elif vendor == "ai21":
                    _set_ai21_span_attributes(completion_span, request_body, response_body)
                elif vendor == "meta":
                    _set_llama_span_attributes(completion_span, request_body, response_body)
                elif vendor == "amazon":
                    _set_amazon_titan_span_attributes(completion_span, request_body, response_body)

            with _create_guardrail_span(tracer, "guardrails.response") as guard_response_span:
                response_eval_result = _handle_response(secure_api, prompt, response_body, guard_response_span)
            if response_eval_result and response_eval_result.action and response_eval_result.action.action_type == 'block':
                request_id = None
                if 'ResponseMetadata' in response:
                    request_id = response['ResponseMetadata'].get('RequestId')
                blocked_response_response = blocked_message_factory(eval_result=response_eval_result, is_prompt=False, request_id=request_id)
                if response_eval_result.validation_results:
                    response_eval_metadata = response_eval_result.metadata.additional_properties
                    generate_event(response_eval_result.validation_results.report, response_eval_metadata, span)
                return blocked_response_response

            return response

    return with_instrumentation


def _set_amazon_titan_span_attributes(span, request_body, response_body):
    try:
        _set_span_attribute(span, "span.type", "completion")
        input_token_count = response_body.get("inputTextTokenCount") if response_body else None
        if response_body:
            results = response_body.get("results")
            total_tokens = results[0].get("tokenCount") if results else None
            completions_tokens = max(total_tokens - input_token_count, 0)
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
            _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completions_tokens)

        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, input_token_count)

        if should_send_prompts():
            _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("inputText"))
            contents = response_body.get("results")
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", contents[0].get("outputText") if contents else "")
    except Exception as ex:  # pylint: disable=broad-except
        LOGGER.warning(f"Failed to set input attributes for openai span, error:{str(ex)}")


def _set_cohere_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_tokens"))

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))

        for i, generation in enumerate(response_body.get("generations")):
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", generation.get("text"))


def _set_anthropic_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    max_tokens = request_body.get("max_tokens") or request_body.get("max_tokens_to_sample")
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, max_tokens)
    _set_span_attribute(span, "anthropic_version", request_body.get("anthropic_version"))
    _set_span_attribute(span, "response.id", response_body.get("id"))
    usage = response_body.get("usage")
    if usage:
        prompt_tokens = usage.get("input_tokens")
        completion_tokens = usage.get("output_tokens")
        total_tokens = prompt_tokens + completion_tokens
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)
    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))
        _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.0.content", response_body.get("completion"))


def _set_ai21_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("topP"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("maxTokens"))

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))

        for i, completion in enumerate(response_body.get("completions")):
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", completion.get("data").get("text"))


def _set_llama_span_attributes(span, request_body, response_body):
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_TYPE, LLMRequestTypeValues.COMPLETION.value)
    _set_span_attribute(span, SpanAttributes.LLM_TOP_P, request_body.get("top_p"))
    _set_span_attribute(span, SpanAttributes.LLM_TEMPERATURE, request_body.get("temperature"))
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_gen_len"))

    if response_body:
        prompt_tokens = response_body.get("prompt_token_count")
        completion_tokens = response_body.get("generation_token_count")
        total_tokens = prompt_tokens + completion_tokens
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_PROMPT_TOKENS, prompt_tokens)
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_COMPLETION_TOKENS, completion_tokens)
        _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

    if should_send_prompts():
        _set_span_attribute(span, f"{SpanAttributes.LLM_PROMPTS}.0.user", request_body.get("prompt"))

        for i, generation in enumerate(response_body.get("generations")):
            _set_span_attribute(span, f"{SpanAttributes.LLM_COMPLETIONS}.{i}.content", response_body)


class BedrockInstrumentor(BaseInstrumentor):
    """An instrumentor for Bedrock's client library."""

    def __init__(self, secure_api: Optional[GuardrailsApi]):
        self._secure_api = secure_api

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)
        LOGGER.info("Instrumenting Bedrock")
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            wrap_method = wrapped_method.get("method")
            wrap_function_wrapper(
                wrap_package,
                f"{wrap_object}.{wrap_method}",
                _wrap(tracer, self._secure_api, wrapped_method),
            )

    def _uninstrument(self, **kwargs):
        for wrapped_method in WRAPPED_METHODS:
            wrap_package = wrapped_method.get("package")
            wrap_object = wrapped_method.get("object")
            unwrap(
                f"{wrap_package}.{wrap_object}",
                wrapped_method.get("method"),
            )
