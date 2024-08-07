import logging
import os
import time
from typing import Optional

from httpx import Timeout
from opentelemetry.propagate import inject
from univers import version_range, versions
from whylogs_container_client import AuthenticatedClient
from whylogs_container_client.api.llm import evaluate
from whylogs_container_client.models import EvaluationResult, HTTPValidationError, LLMValidateRequest
from whylogs_container_client.models.metric_filter_options import MetricFilterOptions
from whylogs_container_client.models.run_options import RunOptions

from openllmtelemetry.version import __version__

_SERVER_SIDE_TRACE_ENABLED = "x-wls-sst"

_SERVER_SIDE_VERSION_CONSTRAINT = "x-wls-verconstr"

LOGGER = logging.getLogger(__name__)
_VERSION_CHECK_DEADLINE = None
_VERSION_CHECK_FREQUENCY_SECONDS = os.environ.get("VERSION_CHECK_FREQUENCY_SECONDS", 60)
_VERSION = versions.PypiVersion(__version__)


class GuardrailsApi(object):
    def __init__(
        self,
        guardrails_endpoint: str,
        guardrails_api_key: str,
        dataset_id: Optional[str] = None,
        timeout: Optional[float] = 300.0,
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
        dataset_id = self._dataset_id
        LOGGER.info(f"Evaluate prompt for dataset_id: {dataset_id}")
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_prompt requires a dataset_id but dataset_id is None.")
            return None
        profiling_request = LLMValidateRequest(prompt=prompt, dataset_id=dataset_id)

        headers = dict()
        inject(headers)

        client = self._client.with_headers(headers)
        res = evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
        sst_enabled = res.headers.get(_SERVER_SIDE_TRACE_ENABLED) is not None
        global _VERSION_CHECK_DEADLINE

        should_check = _VERSION_CHECK_DEADLINE is None or _VERSION_CHECK_DEADLINE < time.time()
        if should_check:
            vr_constr = res.headers.get(_SERVER_SIDE_VERSION_CONSTRAINT)

            _VERSION_CHECK_DEADLINE = time.time() + 60
            if vr_constr is None:
                LOGGER.warning(f"GuardRails service did not provide a version range (missing {_SERVER_SIDE_VERSION_CONSTRAINT} header.")
            else:
                ver_constr = version_range.PypiVersionRange.from_native(vr_constr)
                if _VERSION not in ver_constr:
                    LOGGER.warning(
                        f"OpenLLMTelemetry version not matching supported versions by GuardRails service. Current: {_VERSION}. Expected: {ver_constr}. Unexpected behaviors might arise."
                    )

        result = res.parsed

        if isinstance(result, HTTPValidationError):
            # TODO: log out the client version and the API endpoint version
            LOGGER.warning(f"GuardRail request validation failure detected. result was: {res}.")
            return None
        # LOGGER.debug(f"Done calling eval_prompt on prompt: {prompt} -> res: {res}")
        result.additional_properties["server_side_trace_enabled"] = sst_enabled
        return result

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

        headers = {}
        inject(headers)
        client = self._client.with_headers(headers)

        res = evaluate.sync(client=client, body=profiling_request, log=self._log, perf_info=True)
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
        res = evaluate.sync(client=self._client, body=profiling_request, log=self._log)

        if isinstance(res, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_chunk on prompt: {chunk} -> res: {res}")
        return res
