import logging
import os
from typing import Optional

import whylogs_container_client.api.llm.evaluate as Evaluate
from httpx import Timeout
from whylogs_container_client import AuthenticatedClient
from whylogs_container_client.models import EvaluationResult, HTTPValidationError, LLMValidateRequest
from whylogs_container_client.models.metric_filter_options import MetricFilterOptions
from whylogs_container_client.models.run_options import RunOptions

LOGGER = logging.getLogger(__name__)


class GuardrailsApi(object):
    def __init__(
        self,
        guardrails_endpoint: str,
        guardrails_api_key: str,
        dataset_id: Optional[str] = None,
        timeout: Optional[float] = 1.0,
        auth_header_name: str = "X-API-Key",
        log_profile: bool = True,
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
        self._client = AuthenticatedClient(
            base_url=guardrails_endpoint,  # type: ignore
            token=guardrails_api_key,  #
            prefix="",  #
            auth_header_name=auth_header_name,  # type: ignore
            timeout=Timeout(timeout, read=timeout),  # type: ignore
        )  # type: ignore

    def eval_prompt(self, prompt: str) -> Optional[EvaluationResult]:
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        LOGGER.info(f"Evaluate prompt for dataset_id: {dataset_id}")
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_prompt requires a dataset_id but dataset_id is None.")
            return None
        profiling_request = LLMValidateRequest(prompt=prompt, dataset_id=dataset_id)
        res = Evaluate.sync(client=self._client, body=profiling_request, log=self._log)

        if isinstance(res, HTTPValidationError):
            # TODO: log out the client version and the API endpoint version
            LOGGER.warning(f"GuardRail request validation failure detected. result was: {res} Possible version mismatched.")
            return None
        LOGGER.debug(f"Done calling eval_prompt on prompt: {prompt} -> res: {res}")
        return res

    def eval_response(self, prompt: str, response: str) -> Optional[EvaluationResult]:
        # nested array so you can model a metric requiring multiple inputs. That line says "only run the metrics
        # that require response OR (prompt and response)", which would cover the input similarity metric
        metric_filter_option = MetricFilterOptions(
            by_required_inputs=[["response"], ["prompt", "response"]],
        )
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_response requires a dataset_id but dataset_id is None.")
            return None
        profiling_request = LLMValidateRequest(
            prompt=prompt,
            response=response,
            dataset_id=dataset_id,
            options=RunOptions(metric_filter=metric_filter_option),
        )
        res = Evaluate.sync(client=self._client, body=profiling_request, log=self._log, perf_info=True)
        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_response on [prompt: {prompt}, response: {response}] -> res: {res}")
        return res

    def eval_chunk(self, chunk: str) -> Optional[EvaluationResult]:
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_chunk requires a dataset_id but dataset_id is None.")
            return None
        profiling_request = LLMValidateRequest(response=chunk, dataset_id=dataset_id)
        res = Evaluate.sync(client=self._client, body=profiling_request, log=self._log)

        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_chunk on prompt: {chunk} -> res: {res}")
        return res
