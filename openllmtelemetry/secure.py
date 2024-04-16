import logging
from typing import Optional

import whylogs_container_client.api.llm.evaluate as Evaluate
from httpx import Timeout
from whylogs_container_client import AuthenticatedClient
from whylogs_container_client.models import EvaluationResult, HTTPValidationError, LLMValidateRequest

LOGGER = logging.getLogger(__name__)


class WhyLabsSecureApi(object):
    def __init__(
        self,
        guard_endpoint: str,
        guard_api_key: str,
        dataset_id: str,
        timeout: Optional[float] = 1.0,
        auth_header_name: str = "X-API-Key",
    ):
        """
        Construct a new WhyLabs Guard client

        :param guard_endpoint: the endpoint for the guard client
        :param guard_api_key: the API key to authorize with the endpoint
        :param dataset_id: the default dataset ID
        :param timeout: timeout in second
        :param auth_header_name: the name of the auth header. Shouldn't be set normally
        """
        self._api_key = guard_api_key
        self._dataset_id = dataset_id
        self._client = AuthenticatedClient(
            base_url=guard_endpoint,  # type: ignore
            token=guard_api_key,  #
            prefix="",  #
            auth_header_name=auth_header_name,  # type: ignore
            timeout=Timeout(timeout, read=timeout),  # type: ignore
        )  # type: ignore

    def eval_prompt(self, prompt: str) -> Optional[EvaluationResult]:
        profiling_request = LLMValidateRequest(prompt=prompt, dataset_id=self._dataset_id)
        res = Evaluate.sync(client=self._client, body=profiling_request, log=False)

        if isinstance(res, HTTPValidationError):
            # TODO: log out the client version and the API endpoint version
            LOGGER.warning(f"GuardRail request validation failure detected. result was: {res} Possible version mismatched.")
            return None
        LOGGER.debug(f"Done calling eval_prompt on prompt: {prompt} -> res: {res}")
        return res

    def eval_response(self, prompt: str, response: str) -> Optional[EvaluationResult]:
        profiling_request = LLMValidateRequest(prompt=prompt, response=response, dataset_id=self._dataset_id)
        res = Evaluate.sync(client=self._client, body=profiling_request, log=False, perf_info=True)
        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_response on [prompt: {prompt}, response: {response}] -> res: {res}")
        return res

    def eval_chunk(self, chunk: str) -> Optional[EvaluationResult]:
        profiling_request = LLMValidateRequest(response=chunk, dataset_id=self._dataset_id)
        res = Evaluate.sync(client=self._client, body=profiling_request, log=False)

        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_chunk on prompt: {chunk} -> res: {res}")
        return res
