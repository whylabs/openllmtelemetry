import importlib
import logging
from typing import Collection
import wrapt

from opentelemetry.trace import get_tracer, SpanKind
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor


logger = logging.getLogger(__name__)
_LANGKIT_INSTRUMENTED = False
_instruments = ("langkit >= 0.0.29",)


def _set_attribute(span, name, value):
    if value is None:
        value = "None"
    span.set_attribute(name, value)


class LangKitInstrumentor(BaseInstrumentor):

    @property
    def is_instrumented(self):
        return _LANGKIT_INSTRUMENTED

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        global _LANGKIT_INSTRUMENTED
        if _LANGKIT_INSTRUMENTED:
            logger.warning("info already instrumented, returning")
            return

        _tracer_provider = kwargs.get("tracer_provider")
        _tracer = get_tracer(__name__, _tracer_provider)

        def instrumented_extract(wrapped, instance, args, kwargs):
            with _tracer.start_as_current_span(name="guards", kind=SpanKind.SERVER) as langkit_span:
                logger.debug("Calling wrapped langkit.extract!")
                user_llm_interaction_with_metrics = wrapped(*args, **kwargs)
                for metric_name in user_llm_interaction_with_metrics:
                    # This works on langkit metrics, but skips the prompt/response, and it only works if the langkit metric is a scalar.
                    if "." in metric_name:
                        _set_attribute(langkit_span, "langkit.metrics." + metric_name, user_llm_interaction_with_metrics[metric_name])

                return user_llm_interaction_with_metrics

        logger.info("instrumenting LangKit extract")
        wrapt.wrap_function_wrapper(
            "langkit",
            "extract",
            instrumented_extract,
        )
        _LANGKIT_INSTRUMENTED = True

    def _uninstrument(self, **kwargs):
        logger.warning("uninstrument called on LangKit extract, doing nothing.")


def init_langkit_instrumentor(trace_provider):
    if importlib.util.find_spec("langkit") is not None:
        instrumentor = LangKitInstrumentor()
        if not instrumentor.is_instrumented:
            instrumentor.instrument(trace_provider=trace_provider)
    else:
        logger.warning("langkit not installed, run `pip install langkit` first if you want extended LLM metrics traced with OpenLLMTelemetry!")
