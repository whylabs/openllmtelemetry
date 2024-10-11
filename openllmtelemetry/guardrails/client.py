import logging
import os
from importlib.metadata import version
from typing import Any, Dict, Optional, Union

import whylogs_container_client.api.llm.evaluate as Evaluate
from httpx import Timeout
from opentelemetry.context.context import Context
from opentelemetry.propagate import inject
from opentelemetry.trace.span import Span
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from whylogs_container_client import AuthenticatedClient
from whylogs_container_client.models import EvaluationResult, HTTPValidationError, LLMValidateRequest
from whylogs_container_client.models.metric_filter_options import MetricFilterOptions
from whylogs_container_client.models.run_options import RunOptions
from whylogs_container_client.types import Response

from openllmtelemetry.content_id import ContentIdProvider

LOGGER = logging.getLogger(__name__)
_KNOWN_VERSION_HEADER_NAMES = ["x-wls-version", "whylabssecureheaders.version"]
_KNOWN_VERSION_CONSTRAINT_HEADER_NAMES = ["x-wls-verconstr", "whylabssecureheaders.client_version_constraint"]
_CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT = ">=1.0.23, <3.0.0"


def pass_otel_context(client: AuthenticatedClient, context: Optional[Context] = None) -> AuthenticatedClient:
    headers: Dict[str, Any] = dict()
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
        try:
            self._whylogs_client_version: str = version("whylogs-container-client")
        except Exception as error:
            LOGGER.warning(f"Error checking the version of the whylogs-container-client package: {error}")
            self._whylogs_client_version = "None"

    def _generate_content_id(self, messages: list[str]) -> Optional[str]:
        content_id = None
        if self._content_id_provider is not None:
            try:
                content_id = self._content_id_provider(messages)
            except Exception as error:
                LOGGER.warning(f"Error generating the content_id of on the prompt, error: {error}")
        return content_id

    def _check_version_headers(self, res: Optional[Response[Union[EvaluationResult, HTTPValidationError]]],
                               span: Optional[Span] = None) -> bool:
        if not res:
            LOGGER.warning(f"GuardRail endpoint response is empty: {res}")
            if span:
                span.set_attribute("guardrail.response", "empty")
            return False
        if hasattr(res, "headers"):
            version_constraint = None
            for version_constraint_header_name in _KNOWN_VERSION_CONSTRAINT_HEADER_NAMES:
                if version_constraint_header_name in res.headers:
                    version_constraint = res.headers.get(version_constraint_header_name)
                    if span:
                        span.set_attribute("guardrail.headers." + version_constraint_header_name, str(version_constraint))
                    break
            container_version = None
            for version_header_name in _KNOWN_VERSION_HEADER_NAMES:
                if version_header_name in res.headers:
                    container_version = res.headers.get(version_header_name)
                    if span:
                        span.set_attribute("guardrail.headers." + version_header_name, str(container_version))
                    break

            if version_constraint is None:
                LOGGER.warning("No version constraint found in header from GuardRail endpoint response, "
                               "upgrade to whylabs-container-python container version 2.0.0 or later to "
                               "enable version constraint checks to pass and avoid this warning.")
                if span:
                    span.set_attribute("guardrail.response.version_constraint", "empty")
                return False
            specifier = SpecifierSet(version_constraint)
            version = Version(self._whylogs_client_version)

            # Check if whylogs-container-client version is compatible with the guardrail endpoint constraint
            if version in specifier:
                LOGGER.debug(f"whylabs-container-client version: {self._whylogs_client_version} "
                             f"satisfies the GuardRail endpoints version constraint: {version_constraint}")
                if span:
                    span.set_attribute("guardrail.response.client_version_constraint", version_constraint)
                    span.set_attribute("guardrail.response.client_version", str(version))
            else:
                LOGGER.warning(f"GuardRail endpoint reports running version {container_version} and "
                               f"requires whylabs-container-client version: {version_constraint}, "
                               f"currently we have whylabs-container-client version: {self._whylogs_client_version}")
                if span:
                    span.set_attribute("guardrail.response.client_version_constraint", version_constraint)
                    span.set_attribute("guardrail.response.client_version", str(version))
                return False

            if container_version is None:
                LOGGER.warning("No version header in GuardRail endpoint response, unknown compatibility.")
                if span:
                    span.set_attribute("guardrail.response.container_version", "empty")
                return False
            client_specifier = SpecifierSet(_CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT)
            guardrail_version = Version(container_version)

            # Check if the whylogs-container-python container's reported version is compatible with this package
            if guardrail_version in client_specifier:
                LOGGER.debug(f"whylabs-container-python container GuardRail has version: {self._whylogs_client_version} "
                             f"which satisfies this package's version constraint: {version_constraint}")
                if span:
                    span.set_attribute("guardrail.response.container_client_version_constraint", str(_CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT))
                    span.set_attribute("guardrail.response.client_version", str(self._whylogs_client_version))
            else:
                LOGGER.warning(f"whylabs-container-python container GuardRail has version: {guardrail_version} "
                               f"which fails this package's version constrain: {_CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT}, "
                               f"upgrade the whylabs-container-python container to a version "
                               f"{_CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT}")
                if span:
                    span.set_attribute("guardrail.response.container_version", str(guardrail_version))
                    span.set_attribute("guardrail.container_version_constraint", _CONTAINER_VERSION_COMPATIBILITY_CONSTRAINT)
                return False
            return True
        else:
            LOGGER.warning("GuardRail endpoint is missing or headers, response was: {res}")
            if span:
                span.set_attribute("guardrail.response.headers", "empty: unknown compatibility.")
        return False

    def eval_prompt(self, prompt: str,
                    context: Optional[Context] = None,
                    span: Optional[Span] = None) -> Optional[Response[Union[EvaluationResult, HTTPValidationError]]]:
        dataset_id = self._dataset_id
        LOGGER.info(f"Evaluate prompt for dataset_id: {dataset_id}")
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_prompt requires a dataset_id but dataset_id is None.")
            return None
        content_id = self._generate_content_id([prompt])

        profiling_request = LLMValidateRequest(prompt=prompt, dataset_id=dataset_id, id=content_id)
        client = pass_otel_context(self._client, context=context)
        parsed = None
        res = None
        try:
            res = Evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
            self._check_version_headers(res, span)
            parsed = res.parsed
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_prompt error: {error}")
            if res:
                self._check_version_headers(res, span)
            return None

        if isinstance(parsed, HTTPValidationError):
            # TODO: log out the client version and the API endpoint version
            LOGGER.warning(f"GuardRail request validation failure detected. result was: {res} Possible version mismatched.")
            return None

        LOGGER.debug(f"Done calling eval_prompt on prompt: {prompt} -> res: {res}")

        return res

    def eval_response(self, prompt: str, response: str,
                      context: Optional[Context] = None,
                      span: Optional[Span] = None) -> Optional[Response[Union[EvaluationResult, HTTPValidationError]]]:
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
            self._check_version_headers(res, span)
            parsed = res.parsed
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_response error: {error}")
            if res:
                self._check_version_headers(res, span)

            return None
        if isinstance(parsed, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_response on [prompt: {prompt}, response: {response}] -> res: {res}")

        return res

    def eval_chunk(self, chunk: str, context: Optional[Context] = None,
                   span: Optional[Span] = None) -> Optional[Response[Union[EvaluationResult, HTTPValidationError]]]:
        dataset_id = os.environ.get("CURRENT_DATASET_ID") or self._dataset_id
        if dataset_id is None:
            LOGGER.warning("GuardRail eval_chunk requires a dataset_id but dataset_id is None.")
            return None
        content_id = self._generate_content_id([chunk])
        profiling_request = LLMValidateRequest(response=chunk, dataset_id=dataset_id, id=content_id)
        client = pass_otel_context(self._client, context=context)
        res = None
        try:
            res = Evaluate.sync_detailed(client=client, body=profiling_request, log=self._log)
            parsed = res.parsed
            self._check_version_headers(res, span)
        except Exception as error:  # noqa
            LOGGER.warning(f"GuardRail eval_chunk error: {error}")
            if res:
                self._check_version_headers(res, span)
            return None
        if isinstance(parsed, HTTPValidationError):
            LOGGER.warning(f"GuardRail request validation failure detected. Possible version mismatched: {res}")
            return None
        LOGGER.debug(f"Done calling eval_chunk on prompt: {chunk} -> res: {res}")
        return res
