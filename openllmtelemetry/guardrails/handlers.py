import logging
from typing import Dict, List, Optional, Union

from opentelemetry.trace import Span, SpanKind
from opentelemetry.util.types import Attributes
from whylogs_container_client.models import EvaluationResult
from whylogs_container_client.models.validation_failure import ValidationFailure
from whylogs_container_client.types import Unset

from openllmtelemetry.guardrails import GuardrailsApi
from openllmtelemetry.semantic_conventions.gen_ai import LLMRequestTypeValues, SpanAttributes

LOGGER = logging.getLogger(__name__)
SPAN_TYPE = "span.type"

SPAN_NAME = "openai.chat"

LLM_REQUEST_TYPE = LLMRequestTypeValues.CHAT
_LANGKIT_METRIC_PREFIX = "langkit.metrics"
_RESPONSE_SCORE_PREFIX = "response.score."
_PROMPT_SCORE_PREFIX = "prompt.score."


def generate_event(report: List[ValidationFailure], eval_metadata: Dict[str, Union[str, float, int]], span: Span):
    policy_version = eval_metadata.get("policy_id")
    if not report:
        return
    for validation_failure in report:
        rule = validation_failure.metric.replace(_RESPONSE_SCORE_PREFIX, "").replace(_PROMPT_SCORE_PREFIX, "")
        validation_id = validation_failure.id
        event_attributes: Attributes = dict()
        if policy_version is not None:
            event_attributes["langkit.metrics.policy"] = policy_version
        event_attributes["rule_id"] = rule
        event_attributes["explanation"] = validation_failure.details
        event_attributes["id"] = validation_id
        event_attributes["metrics"] = [validation_failure.metric]

        action = validation_failure.additional_properties.get("failure_level")
        if action is not None:
            event_attributes["action"] = (action,)

        if validation_failure.allowed_values is not None:
            event_attributes["allowed_values"] = str(validation_failure.allowed_values)
        if validation_failure.lower_threshold is not None and not isinstance(validation_failure.lower_threshold, Unset):
            event_attributes["lower_threshold"] = validation_failure.lower_threshold
        if validation_failure.must_be_non_none is not None and not isinstance(validation_failure.must_be_non_none, Unset):
            event_attributes["must_be_non_none"] = validation_failure.must_be_non_none
        if validation_failure.must_be_none is not None and not isinstance(validation_failure.must_be_none, Unset):
            event_attributes["must_be_none"] = validation_failure.must_be_none
        if validation_failure.upper_threshold is not None and not isinstance(validation_failure.upper_threshold, Unset):
            event_attributes["upper_threshold"] = validation_failure.upper_threshold
        if validation_failure.value is not None:
            event_attributes["metric_value"] = validation_failure.value
        name = "guardrails.api.validation_failure"
        span.add_event(name, event_attributes)


def sync_wrapper(
    tracer,
    guardrails_client,
    prompt_provider,
    llm_caller,
    response_extractor,
    prompt_attributes_setter,
    request_type: LLMRequestTypeValues,
    streaming_response_handler=None,
    blocked_message_factory=None,
    completion_span_name=SPAN_NAME,
):
    """
    Wrapper for synchronous calls to an LLM API.
    :param blocked_message_factory:
    :param streaming_response_handler:
    :param request_type:
    :param tracer: the trace provider
    :param guardrails_client:
    :param prompt_provider:
    :param llm_caller:
    :param response_extractor:
    :param prompt_attributes_setter:
    :return:
    """
    with start_span(request_type, tracer):
        prompt = prompt_provider()
        prompt_eval = _evaluate_prompt(tracer, guardrails_client, prompt)

        if prompt_eval and prompt_eval.action and prompt_eval.action.action_type == "block":
            if blocked_message_factory:
                return blocked_message_factory(prompt_eval, True)
            else:
                LOGGER.warning("Prompt blocked but no blocked message factory provided")

        with tracer.start_span(
            completion_span_name,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: request_type.value, SPAN_TYPE: "completion"},
        ) as span:
            prompt_attributes_setter(span)
            response, is_streaming = llm_caller(span)
            if is_streaming:
                if streaming_response_handler:
                    # TODO: handle streaming response. Where does guard response live?
                    return streaming_response_handler(span, response)
                else:
                    return response

        response_text = response_extractor(response)

        response_result = _guard_response(guardrails_client, prompt, response_text, tracer)
        if response_result and response_result.action and response_result.action.action_type == "block":
            if blocked_message_factory:
                return blocked_message_factory(response_result, False)
            else:
                LOGGER.warning("Response blocked but no blocked message factory provided")

        return response


def start_span(request_type, tracer):
    return tracer.start_as_current_span(
        "interaction",
        kind=SpanKind.CLIENT,
        attributes={SpanAttributes.LLM_REQUEST_TYPE: request_type.value, SPAN_TYPE: "interaction"},
    )


def _create_guardrail_span(tracer, name="guardrails.request"):
    return tracer.start_span(
        name,
        kind=SpanKind.CLIENT,
        attributes={SPAN_TYPE: "guardrails"},
    )


async def async_wrapper(
    tracer,
    guardrails_api,
    prompt_provider,
    llm_caller,
    response_extractor,
    kwargs,
    prompt_attributes_setter,
    streaming_response_handler,
    is_streaming_response_checker,
    request_type: LLMRequestTypeValues,
):
    """
    Wrapper for synchronous calls to an LLM API.
    :param request_type:
    :param tracer: the trace provider
    :param guardrails_api:
    :param prompt_provider:
    :param llm_caller:
    :param response_extractor:
    :param kwargs:
    :param prompt_attributes_setter:
    :param streaming_response_handler:
    :param is_streaming_response_checker:
    :return:
    """
    with start_span(request_type, tracer):
        prompt = prompt_provider()
        # TODO: need async version
        _evaluate_prompt(tracer, guardrails_api, prompt)

        with tracer.start_as_current_span(
            SPAN_NAME,
            kind=SpanKind.CLIENT,
            attributes={SpanAttributes.LLM_REQUEST_TYPE: request_type.value, SPAN_TYPE: "completion"},
        ) as span:
            prompt_attributes_setter(span)
            response = await llm_caller(span)
            # response_attributes_setter(response, span)

        # response_text = response_extractor(response)
        # _guard_response(guardrails_api, prompt_provider(), response_text, tracer)

        return response


def _evaluate_prompt(tracer, guardrails_api: GuardrailsApi, prompt: str) -> Optional[EvaluationResult]:
    if guardrails_api:
        with _create_guardrail_span(tracer, "guardrails.request") as span:
            # noinspection PyBroadException
            try:
                evaluation_result = guardrails_api.eval_prompt(prompt)
                LOGGER.debug("Prompt evaluated: %s", evaluation_result)
                if evaluation_result:
                    # The underlying API can handle batches of inputs, so we always get a list of metrics
                    metrics = evaluation_result.metrics[0]

                    for k in metrics.additional_keys:
                        if metrics.additional_properties[k] is not None:
                            span.set_attribute(f"{_LANGKIT_METRIC_PREFIX}.{k}", metrics.additional_properties[k])
                    scores = evaluation_result.scores
                    if scores and len(scores) > 0:
                        score_dictionary = scores[0].additional_properties
                        for score_key in score_dictionary:
                            if score_dictionary[score_key] is not None:
                                slim_score_key = score_key.replace("response.score.", "").replace("prompt.score.", "")
                                span.set_attribute(f"{_LANGKIT_METRIC_PREFIX}.{slim_score_key}", score_dictionary[score_key])
                    eval_metadata = evaluation_result.metadata.additional_properties
                    if eval_metadata:
                        for metadata_key in eval_metadata:
                            span.set_attribute(f"guardrails.api.{metadata_key}", eval_metadata[metadata_key])
                    tags = []
                    if evaluation_result.action.action_type == "block":
                        tags.append("BLOCKED")
                        if evaluation_result.validation_results:
                            generate_event(evaluation_result.validation_results.report, eval_metadata, span)

                    for r in evaluation_result.validation_results.report:
                        tags.append(r.metric.replace("response.score.", "").replace("prompt.score.", ""))
                    if len(tags) > 0:
                        span.set_attribute("langkit.insights.tags", tags)
                return evaluation_result
            except Exception as e:  # noqa: E722
                LOGGER.warning("Error evaluating prompt")
                logging.exception(f"Error evaluating prompt: {e}")
                span.set_attribute("guardrails.error", 1)
                # TODO: set more attributes to help us diagnose in our side
                return None

    return None


def _guard_response(guardrails, prompt, response, tracer):
    if guardrails:
        with _create_guardrail_span(tracer, "guardrails.response") as span:
            # noinspection PyBroadException
            try:
                result: Optional[EvaluationResult] = guardrails.eval_response(prompt=prompt, response=response)
                if result:
                    LOGGER.debug("Response evaluated: %s", result)
                    # The underlying API can handle batches of inputs, so we always get a list of metrics
                    metrics = result.metrics[0]

                    for k in metrics.additional_keys:
                        if metrics.additional_properties[k] is not None:
                            span.set_attribute(f"{_LANGKIT_METRIC_PREFIX}.{k}", metrics.additional_properties[k])

                    scores = result.scores
                    if scores and len(scores) > 0:
                        score_dictionary = scores[0].additional_properties
                        for score_key in score_dictionary:
                            if score_dictionary[score_key] is not None:
                                slim_score_key = score_key.replace("response.score.", "").replace("prompt.score.", "")
                                span.set_attribute(f"{_LANGKIT_METRIC_PREFIX}.{slim_score_key}", score_dictionary[score_key])

                    eval_metadata = result.metadata.additional_properties
                    if eval_metadata:
                        for metadata_key in eval_metadata:
                            span.set_attribute(f"guardrails.api.{metadata_key}", eval_metadata[metadata_key])

                    tags = []
                    if result.action.action_type == "block":
                        tags.append("BLOCKED")
                    if result.validation_results and result.validation_results.report:
                        generate_event(result.validation_results.report, eval_metadata, span)

                    for r in result.validation_results.report:
                        tags.append(r.metric.replace("response.score.", "").replace("prompt.score.", ""))
                    if len(tags) > 0:
                        span.set_attribute("langkit.insights.tags", tags)
                return result
            except:  # noqa: E722
                LOGGER.warning("Error evaluating response")
                span.set_attribute("guardrails.error", 1)
                return None
