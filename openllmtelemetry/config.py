import configparser
import logging
import os
from dataclasses import dataclass, fields
from getpass import getpass
from pathlib import Path
from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from openllmtelemetry.guardrails import GuardrailsApi
from openllmtelemetry.span_exporter import DebugOTLSpanExporter

CFG_API_KEY = "api_key"

CFG_ENDPOINT_KEY = "endpoint"

CFG_LOG_PROFILE_KEY = "log_profile"

CFG_WHYLABS_SECTION = "whylabs"

CFG_GUARDRAILS_SECTION = "guardrails"

LOGGER = logging.getLogger(__name__)
_DEFAULT_ENDPOINT = "https://api.whylabsapp.com"
_CONFIG_DIR = os.path.join(Path.home(), ".whylabs")
_DEFAULT_CONFIG_FILE = os.path.join(_CONFIG_DIR, "guardrails-config.ini")

_in_ipython_session = False
try:
    # noinspection PyStatementEffect
    __IPYTHON__  # pyright: ignore[reportUndefinedVariable,reportUnusedExpression]
    _in_ipython_session = True
except NameError:
    pass


@dataclass
class GuardrailConfig(object):
    whylabs_endpoint: Optional[str] = None
    whylabs_api_key: Optional[str] = None
    guardrails_endpoint: Optional[str] = None
    guardrails_api_key: Optional[str] = None
    log_profile: Optional[str] = None

    @property
    def is_partial(self):
        return (
            self.whylabs_endpoint is None
            or self.whylabs_api_key is None
            or self.guardrails_endpoint is None
            or self.guardrails_api_key is None
        )

    @property
    def whylabs_traces_endpoint(self) -> str:
        assert self.whylabs_endpoint is not None, "WhyLabs endpoint is not set."
        return f"{self.whylabs_endpoint.rstrip('/')}/v1/traces"

    def guardrail_client(self, default_dataset_id: Optional[str]) -> Optional[GuardrailsApi]:
        if self.guardrails_endpoint and self.guardrails_api_key:
            return GuardrailsApi(
                guardrails_endpoint=self.guardrails_endpoint,
                guardrails_api_key=self.guardrails_api_key,
                dataset_id=default_dataset_id,
                log_profile=True if self.log_profile is None or self.log_profile.lower() == "true" else False,
            )
        LOGGER.warning("GuardRails endpoint is not set.")
        return None

    def config_tracer_provider(
        self,
        tracer_provider: TracerProvider,
        dataset_id: str,
        disable_batching: bool = False,
        debug: bool = False,
    ):
        if self.whylabs_traces_endpoint and self.whylabs_api_key:
            debug_enabled = os.environ.get("WHYLABS_DEBUG_TRACE") or debug
            whylabs_api_key_header = {"X-API-Key": self.whylabs_api_key, "X-WHYLABS-RESOURCE": dataset_id}
            # TODO: support other kinds of exporters
            if debug_enabled:
                otlp_exporter = DebugOTLSpanExporter(
                    endpoint=self.whylabs_traces_endpoint,
                    headers=whylabs_api_key_header,  # noqa: F821
                )
            else:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.whylabs_traces_endpoint,
                    headers=whylabs_api_key_header,  # noqa: F821
                )
            if disable_batching:
                span_processor = SimpleSpanProcessor(otlp_exporter)
            else:
                span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)

        pass

    def write(self, config_path: str):
        config = configparser.ConfigParser()
        if self.whylabs_endpoint is not None and self.whylabs_api_key is not None:
            config[CFG_WHYLABS_SECTION] = {
                CFG_ENDPOINT_KEY: self.whylabs_endpoint,
                CFG_API_KEY: self.whylabs_api_key,
            }
        if self.guardrails_endpoint:
            config[CFG_GUARDRAILS_SECTION] = {
                CFG_ENDPOINT_KEY: self.guardrails_endpoint,
                CFG_API_KEY: self.guardrails_api_key or "",
                CFG_LOG_PROFILE_KEY: self.log_profile or "",
            }
        with open(config_path, "w") as configfile:
            config.write(configfile)

    def __repr__(self):
        # hide the api_key from output
        field_strs = [
            f"{field.name}='***key***'" if "key" in field.name else f"{field.name}={getattr(self, field.name)}" for field in fields(self)
        ]
        return f"{self.__class__.__name__}({', '.join(field_strs)})"

    @classmethod
    def read(cls, config_path: str) -> "GuardrailConfig":
        config = configparser.ConfigParser()
        ok_files = config.read(config_path)
        if len(ok_files) == 0:
            raise IOError("Failed to read the configuration file.")

        whylabs_endpoint = config.get(CFG_WHYLABS_SECTION, CFG_ENDPOINT_KEY)
        whylabs_api_key = config.get(CFG_WHYLABS_SECTION, CFG_API_KEY)
        guardrails_endpoint = config.get(CFG_GUARDRAILS_SECTION, CFG_ENDPOINT_KEY, fallback=None)
        guardrails_api_key = config.get(CFG_GUARDRAILS_SECTION, CFG_API_KEY, fallback=None)
        log_profile = config.get(CFG_GUARDRAILS_SECTION, CFG_LOG_PROFILE_KEY, fallback=None)

        return GuardrailConfig(whylabs_endpoint, whylabs_api_key, guardrails_endpoint, guardrails_api_key, log_profile)


def load_config() -> GuardrailConfig:
    config_path = os.environ.get("WHYLABS_GUARDRAILS_CONFIG")
    if config_path is None:
        config_path = _DEFAULT_CONFIG_FILE
    config = GuardrailConfig(None, None, None, None)
    try:
        config = GuardrailConfig.read(config_path)
    except:  # noqa
        LOGGER.warning("Failed to parse the configuration file")
    if config.is_partial:
        config = _load_config_from_env(config)
    if config.is_partial and _in_ipython_session:
        config = _interactive_config(config)

    return config


def load_dataset_id(dataset_id: Optional[str]) -> Optional[str]:
    effective_dataset_id = os.environ.get("WHYLABS_DEFAULT_DATASET_ID", dataset_id)
    if effective_dataset_id is None:
        if _in_ipython_session:
            effective_dataset_id = input("Set the default dataset ID: ").strip()
            if len(effective_dataset_id) > 0:
                print("Using dataset ID: ", effective_dataset_id)
            else:
                print("Dataset ID is not set. Skip tracing...")
                effective_dataset_id = None
    return effective_dataset_id


def _load_config_from_env(config: Optional[GuardrailConfig] = None) -> GuardrailConfig:
    if config is None:
        config = GuardrailConfig()
    whylabs_endpoint = os.environ.get("WHYLABS_ENDPOINT", config.whylabs_endpoint) or _DEFAULT_ENDPOINT
    whylabs_api_key = os.environ.get("WHYLABS_API_KEY", config.whylabs_api_key)
    guardrails_endpoint = os.environ.get("GUARDRAILS_ENDPOINT", config.guardrails_endpoint)
    guardrails_api_key = os.environ.get("GUARDRAILS_API_KEY", config.guardrails_api_key)
    log_profile = os.environ.get("GUARDRAILS_LOG_PROFILE", config.log_profile)
    guardrail_config = GuardrailConfig(whylabs_endpoint, whylabs_api_key, guardrails_endpoint, guardrails_api_key, log_profile)
    return guardrail_config


def _interactive_config(config: GuardrailConfig) -> GuardrailConfig:
    whylabs_endpoint = config.whylabs_endpoint or _DEFAULT_ENDPOINT
    whylabs_api_key = config.whylabs_api_key
    guardrails_endpoint = config.guardrails_endpoint
    guardrails_api_key = config.guardrails_api_key
    if whylabs_api_key is None:
        whylabs_api_key = getpass("Set WhyLabs API Key: ").strip()
        print("Using WhyLabs API key with ID: ", whylabs_api_key[:10])
    if guardrails_endpoint is None:
        guardrails_endpoint = input("Set GuardRails endpoint (leave blank to skip guardrail): ").strip()
        if len(guardrails_endpoint) == 0:
            guardrails_endpoint = None
        if guardrails_endpoint is None:
            print("GuardRails endpoint is not set. Only tracing is enabled.")
    if guardrails_endpoint is not None and guardrails_api_key is None:
        guardrails_api_key = getpass("Set GuardRails API Key: ").strip()
        if len(guardrails_api_key) > 15:
            print("Using GuardRails API key with prefix: ", guardrails_api_key[:6])

    guardrail_config = GuardrailConfig(
        whylabs_endpoint=whylabs_endpoint,
        whylabs_api_key=whylabs_api_key,
        guardrails_endpoint=guardrails_endpoint,
        guardrails_api_key=guardrails_api_key,
    )

    save_config = input("Do you want to save these settings to a configuration file? [y/n]: ").strip().lower()
    if save_config == "y" or save_config == "yes":
        try:
            os.makedirs(_CONFIG_DIR, exist_ok=True)
            guardrail_config.write(_DEFAULT_CONFIG_FILE)
        except Exception as e:  # noqa
            LOGGER.exception(f"Failed to write the configuration file: {e}")

            print("Failed to write the configuration file.")

    print(f"Set config: {guardrail_config}")
    return guardrail_config
