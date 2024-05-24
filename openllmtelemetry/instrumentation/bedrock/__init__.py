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
from functools import wraps
from typing import Collection, Optional

from opentelemetry import context as context_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    unwrap,
)
from opentelemetry.trace import SpanKind, get_tracer
from wrapt import wrap_function_wrapper

from openllmtelemetry.guardrails import GuardrailsApi  # noqa: E402
from openllmtelemetry.instrumentation.bedrock.reusable_streaming_body import ReusableStreamingBody
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes
from openllmtelemetry.version import __version__

LOGGER = logging.getLogger(__name__)

_instruments = ("boto3 >= 1.28.57",)

WRAPPED_METHODS = [{"package": "botocore.client", "object": "ClientCreator", "method": "create_client"}]


def should_send_prompts():
    return (os.getenv("TRACE_PROMPT_AND_RESPONSE") or "false").lower() == "true" or context_api.get_value("override_enable_content_tracing")


def _set_span_attribute(span, name, value):
    if value is not None:
        if value != "":
            span.set_attribute(name, value)
    return


def _with_tracer_wrapper(func):
    """Helper for providing tracer for wrapper functions."""

    def _with_tracer(tracer, guardrails_api: GuardrailsApi, to_wrap):
        def wrapper(wrapped, instance, args, kwargs):
            return func(tracer, guardrails_api, to_wrap, wrapped, instance, args, kwargs)

        return wrapper

    return _with_tracer


def _handle_request(guardrails_api: Optional[GuardrailsApi], prompt: str, span):
    prompt_metrics = None
    if prompt is not None:
        prompt_metrics = guardrails_api.eval_prompt(prompt) if guardrails_api is not None else None
    if prompt_metrics and span is not None:
        LOGGER.debug(prompt_metrics)
        metrics = prompt_metrics.metrics[0]
        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                metric_value = metrics.additional_properties[k]
                span.set_attribute(f"langkit.metrics.{k}", metric_value)
    return prompt


def _handle_response(secure_api: Optional[GuardrailsApi], prompt, response, span):
    response_text: Optional[str] = None
    response_metrics = None
    results = response.get("results")
    if results:
        response_message = results[0]
        if response_message:
            response_text = response_message.get("outputText")
    if response_text is not None:
        response_metrics = secure_api.eval_response(prompt=prompt, response=response_text) if secure_api is not None else None
    if response_metrics:
        LOGGER.debug(response_metrics)
        metrics = response_metrics.metrics[0]

        for k in metrics.additional_keys:
            if metrics.additional_properties[k] is not None:
                metric_value = metrics.additional_properties[k]
                span.set_attribute(f"langkit.metrics.{k}", metric_value)
    else:
        LOGGER.debug("response metrics is none, skipping")

    return response


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
        with tracer.start_as_current_span("bedrock.completion", kind=SpanKind.CLIENT) as span:
            request_body = json.loads(kwargs.get("body"))
            (vendor, model) = kwargs.get("modelId").split(".")
            is_titan_text = model.startswith("titan-text-")

            prompt = None
            if vendor == "cohere":
                prompt = request_body.get("prompt")
            elif vendor == "anthropic":
                prompt = request_body.get("inputText")
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

            def prompt_provider():
                prompt = None
                if vendor == "cohere":
                    prompt = request_body.get("prompt")
                elif vendor == "anthropic":
                    prompt = request_body.get("inputText")
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
                return prompt

            def call_llm(span):
                response = fn(*args, **kwargs)
                response["body"] = ReusableStreamingBody(response["body"]._raw_stream, response["body"]._content_length)
                return response

            def prompt_attributes_setter(span):
                _set_span_attribute(span, SpanAttributes.LLM_VENDOR, vendor)
                _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)

                if vendor == "cohere":
                    _set_cohere_span_attributes(span, request_body, {})
                elif vendor == "anthropic":
                    _set_anthropic_span_attributes(span, request_body, {})
                elif vendor == "ai21":
                    _set_ai21_span_attributes(span, request_body, {})
                elif vendor == "meta":
                    _set_llama_span_attributes(span, request_body, {})
                elif vendor == "amazon":
                    _set_amazon_titan_span_attributes(span, request_body, {})

            def response_extractor(r):
                if is_openai_v1():
                    response_dict = model_as_dict(r)
                else:
                    response_dict = r
                return response_dict["choices"][0]["text"]

            # TODO: check for input text first
            prompt = _handle_request(secure_api, prompt, span)
            response = fn(*args, **kwargs)

            response["body"] = ReusableStreamingBody(response["body"]._raw_stream, response["body"]._content_length)
            response_body = json.loads(response.get("body").read())
            response_body = _handle_response(secure_api, prompt, response_body, span)
            # noinspection PyProtectedMember

            _set_span_attribute(span, SpanAttributes.LLM_VENDOR, vendor)
            _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MODEL, model)

            if vendor == "cohere":
                _set_cohere_span_attributes(span, request_body, response_body)
            elif vendor == "anthropic":
                _set_anthropic_span_attributes(span, request_body, response_body)
            elif vendor == "ai21":
                _set_ai21_span_attributes(span, request_body, response_body)
            elif vendor == "meta":
                _set_llama_span_attributes(span, request_body, response_body)
            elif vendor == "amazon":
                _set_amazon_titan_span_attributes(span, request_body, response_body)

            return response

    return with_instrumentation


def _set_amazon_titan_span_attributes(span, request_body, response_body):
    try:
        _set_span_attribute(span, "span.type", "completion")
        input_token_count = response_body.get("inputTextTokenCount") if response_body else None
        if response_body:
            results = response_body.get("results")
            total_tokens = results[0].get("tokenCount") if results else None

            _set_span_attribute(span, SpanAttributes.LLM_USAGE_TOTAL_TOKENS, total_tokens)

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
    _set_span_attribute(span, SpanAttributes.LLM_REQUEST_MAX_TOKENS, request_body.get("max_tokens_to_sample"))

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
