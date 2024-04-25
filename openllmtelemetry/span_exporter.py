import logging
from typing import Sequence

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult

LOGGER = logging.getLogger(__name__)


class DebugOTLSpanExporter(OTLPSpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        LOGGER.debug(f"Exporting spans: {len(spans)} spans...")
        for span in spans:
            LOGGER.debug(f"Exporting span: {span.name}")
        try:
            response = super().export(spans)
            LOGGER.debug("Done exporting spans")
            return response
        except Exception as e:
            LOGGER.error(f"Error exporting spans: {e}")
            return SpanExportResult.FAILURE
