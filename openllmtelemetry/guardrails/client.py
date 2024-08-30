import logging
import os
from typing import Optional

import whylogs_container_client.api.llm.evaluate as Evaluate
from httpx import Timeout
from opentelemetry.context.context import Context
from opentelemetry.propagate import inject
from whylogs_container_client import AuthenticatedClient
from whylogs_container_client.models import EvaluationResult, HTTPValidationError, LLMValidateRequest
from whylogs_container_client.models.metric_filter_options import MetricFilterOptions
from whylogs_container_client.models.run_options import RunOptions

from openllmtelemetry.content_id import ContentIdProvider

LOGGER = logging.getLogger(__name__)


def pass_otel_context(client: AuthenticatedClient, context: Optional[Context] = None) -> AuthenticatedClient:
    headers = dict()
    inject(headers, context=context)
    return client.with_headers(headers)


class GuardrailsApi(object):
    def __init__(
        self,
        guardrails_endpoint: str,
        guardrails_api_key: str,
        dataset_id: Optional[str] = None,
        timeout: Optional[float] = 15.0,
        auth_header_name: str = "X-API-Key",
        log_profile: bool = True,
        content_id_provider: Optional[ContentIdProvider] = None,
    ):
        """
        Construct a new WhyLabs Guard client

        :param guardrails_endpoint: the endpoint for the guard client
        :param guardrails_api_key: the API key to authorize with the endpoint
        :param dataset_id: the default dataset ID
        :param timeout: timeout in second
        :param auth_header_name: the name of the auth header. Shouldn't be set normally
        """
        self._api_key = guardrails_api_key
        self._dataset_id = dataset_id
        self._log = log_profile
        default_timeout = 15.0
        env_timeout = os.environ.get("GUARDRAILS_API_TIMEOUT")
        if timeout is None:
            if env_timeout is None:
                timeout = default_timeout
            else:
                try:
                    timeout = float(env_timeout)
                except Exception as error:
                    LOGGER.warning(f"Failure reading and paring GUARDRAILS_API_TIMEOUT as float: {error}, "
                                   f"default timeout of {default_timeout} used.")
                    timeout = default_timeout
        self._client = AuthenticatedClient(
            base_url=guardrails_endpoint,  # type: ignore
            token=guardrails_api_key,  #
            prefix="",  #
            auth_header_name=auth_header_name,  # type: ignore
            timeout=Timeout(timeout, read=timeout),  # type: ignore
        )  # type: ignore
        self._content_id_provider = content_id_provider

    def _generate_content_id(self, messages: list[str]) -> Optional[str]:
        content_id = None
        if self._content_id_provider is not None:
            try:
                content_id = self._content_id_provider(messages)
            except Exception as error:
                LOGGER.warning(f"Error generating the content_id of on the prompt, error: {error}")
        return content_id

    def eval_prompt(self, prompt: str, context: Optional[Context] = None) -> Optional[EvaluationResult]:
        dataset_id = self._dataset_id
        LOGGER.info(f"Evaluate prompt for dataset_id: {dataset_id}")
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_prompt requires a dataset_id but dataset_id is None.")
            return None
        content_id = self._generate_content_id([prompt])

        profiling_request = LLMValidateRequest(prompt=prompt, dataset_id=dataset_id, id=content_id)
        client = pass_otel_context(self._client, context=context)
        parsed = None
        try:
            res = Evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
            parsed = res.parsed
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_prompt error: {error}")
            if res and hasattr(res, "headers"):
                version_constraint = res.headers.get('whylabssecureheaders.client_version_constraint')
                LOGGER.warning(f"GuardRail requires whylabs-container-client version: {version_constraint}")
            return None

        if isinstance(parsed, HTTPValidationError):
            # TODO: log out the client version and the API endpoint version
            LOGGER.warning(f"GuardRail request validation failure detected. result was: {res} Possible version mismatched.")
            return None

        LOGGER.debug(f"Done calling eval_prompt on prompt: {prompt} -> res: {res}")

        return res

    def eval_response(self, prompt: str, response: str,
                      context: Optional[Context] = None) -> Optional[EvaluationResult]:
        # nested array so you can model a metric requiring multiple inputs. That line says "only run the metrics
        # that require response OR (prompt and response)", which would cover the input similarity metric
        metric_filter_option = MetricFilterOptions(
            by_required_inputs=[["response"], ["prompt", "response"]],
        )
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_response requires a dataset_id but dataset_id is None.")
            return None
        content_id = self._generate_content_id([prompt, response])

        profiling_request = LLMValidateRequest(
            prompt=prompt,
            response=response,
            dataset_id=dataset_id,
            id=content_id,
            options=RunOptions(metric_filter=metric_filter_option)
        )
        client = pass_otel_context(self._client, context=context)
        res = None
        parsed = None
        try:
            res = Evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
            parsed = res.parsed
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_response error: {error}")
            if res and hasattr(res, "headers"):
                version_constraint = res.headers.get('whylabssecureheaders.client_version_constraint')
                LOGGER.warning(f"GuardRail requires whylabs-container-client version: {version_constraint}")

            return None
        if isinstance(parsed, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_response on [prompt: {prompt}, response: {response}] -> res: {res}")

        return res

    def eval_chunk(self, chunk: str, context: Optional[Context] = None) -> Optional[EvaluationResult]:
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_chunk requires a dataset_id but dataset_id is None.")
            return None
        content_id = self._generate_content_id([chunk])
        profiling_request = LLMValidateRequest(response=chunk, dataset_id=dataset_id, id=content_id)
        client = pass_otel_context(self._client, context=context)
        try:
            res = Evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_chunk error: {error}")
            return None
        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_chunk on prompt: {chunk} -> res: {res}")
        return res
