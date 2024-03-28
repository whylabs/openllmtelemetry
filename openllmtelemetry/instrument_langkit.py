import importlib
import logging
from typing import Any, Callable, Collection, Dict, Optional

import wrapt
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import SpanKind, Tracer, TracerProvider, get_tracer
from opentelemetry.trace.span import Span
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)
_instruments = ("langkit>=0.0.29",)


def _set_attribute(span: Span, name: str, value: Optional[AttributeValue]):
    if value is None:
        value = "None"
    span.set_attribute(name, value)


class LangKitInstrumentor(BaseInstrumentor):
    _langkit_instrumented = False

    @property
    def is_instrumented(self):
        return self._langkit_instrumented

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Dict[str, Any]):
        if self._langkit_instrumented:
            logger.warning("info already instrumented, returning")
            return

        _tracer_provider = kwargs.get("tracer_provider")
        if not _tracer_provider:
            raise ValueError("No tracer_provider was specified in _instrument, must supply a provider")
        if not isinstance(_tracer_provider, TracerProvider):
            raise ValueError(f"No tracer_provider has type {type(_tracer_provider)} but requires a TracerProvider")
        _tracer: Tracer = get_tracer(instrumenting_module_name=__name__, tracer_provider=_tracer_provider)

        def instrumented_extract(wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any) -> Any:
            with _tracer.start_as_current_span(name="guards", kind=SpanKind.SERVER) as langkit_span:
                logger.debug("Calling wrapped langkit.extract!")
                user_llm_interaction_with_metrics = wrapped(*args, **kwargs)
                for metric_name in user_llm_interaction_with_metrics:
                    metric_name_string = str(metric_name)
                    # This works on langkit metrics, but skips the prompt/response, and it only works if the langkit metric is a scalar.
                    if "." in metric_name_string:
                        _set_attribute(langkit_span, "langkit.metrics." + metric_name_string, user_llm_interaction_with_metrics[metric_name])

                return user_llm_interaction_with_metrics

        logger.info("instrumenting LangKit extract")
        if not hasattr(self, "_schema"):
            from langkit import light_metrics, vader_sentiment  # type: ignore

            vader_sentiment.init()  # type: ignore
            self._schema = light_metrics.init()  # type: ignore
        wrapt.wrap_function_wrapper(  # type: ignore
            "langkit",
            "extract",
            instrumented_extract,
        )

        self._langkit_instrumented = True

    def _uninstrument(self, **kwargs: Dict[str, Any]):
        logger.warning("uninstrument called on LangKit extract, doing nothing.")


def init_langkit_instrumentor(tracer_provider: TracerProvider, schema: Optional[Any] = None, **kwargs: Dict[str, Any]) -> None:
    if importlib.util.find_spec("langkit") is not None:  # type: ignore
        instrumentor = LangKitInstrumentor(schema=schema)
        if not instrumentor.is_instrumented:
            instrumentor.instrument(tracer_provider=tracer_provider, **kwargs)  # type: ignore
    else:
        logger.warning("langkit not installed, run `pip install langkit` first if you want extended LLM metrics traced with OpenLLMTelemetry!")
