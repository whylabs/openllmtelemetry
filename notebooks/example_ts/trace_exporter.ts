import { BasicTracerProvider, SimpleSpanProcessor, ReadableSpan, SpanExporter } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { ExportResultCode } from '@opentelemetry/core';
import { Resource } from '@opentelemetry/resources';
import { Config } from './config';

class DebugOTLPTraceExporter extends OTLPTraceExporter {
  export(spans: ReadableSpan[], resultCallback: (result: { code: ExportResultCode }) => void): void {
    console.log(`Exporting spans: ${spans.length} spans...`);
    for (const span of spans) {
      console.log(`Exporting span: ${span.name}`);
    }
    try {
      super.export(spans, (result: { code: ExportResultCode }) => {
        if (result.code === ExportResultCode.SUCCESS) {
          console.log("Done exporting spans");
        } else {
          console.error("Failed exporting spans");
          console.error(result)
        }
        resultCallback(result);
      });
    } catch (e) {
      console.error(`Error exporting spans: ${e}`);
      resultCallback({ code: ExportResultCode.FAILED });
    }
  }
}

class LoggingSpanProcessor extends SimpleSpanProcessor {
  constructor(exporter: SpanExporter) {
    super(exporter);
  }

  onEnd(span: ReadableSpan): void {
    console.log('Span ended:', span.name);
    super.onEnd(span);
  }

  forceFlush(): Promise<void> {
    console.log('Flushing spans...');
    return super.forceFlush();
  }

  shutdown(): Promise<void> {
    console.log('Shutting down span processor...');
    return super.shutdown();
  }
}

export function initialize_tracing(config: Config, datasetId: string, serviceName = 'openllmtelemetry-instrumented-service') {
  const whylabs_api_key_header = {
    "X-API-Key": config.whylabs.api_key,
    "X-WHYLABS-RESOURCE": datasetId
  };

  const collectorOptions = {
    url: 'https://api.whylabsapp.com/v1/traces',
    headers: whylabs_api_key_header
  };

  const resource = new Resource({
    "service.name": serviceName,
  });

  const provider = new BasicTracerProvider({resource});

  const useDebug = process.env.WHYLABS_DEBUG_TRACE === 'true';

  const exporter = useDebug ? new DebugOTLPTraceExporter(collectorOptions) : new OTLPTraceExporter(collectorOptions);
  const spanProcessor = useDebug ? new LoggingSpanProcessor(exporter) : new SimpleSpanProcessor(exporter);

  provider.addSpanProcessor(spanProcessor);

  provider.register();

  ['SIGINT', 'SIGTERM'].forEach(signal => {
    process.on(signal, () => provider.shutdown().catch(console.error));
  });

  return provider;  // Return the provider
}
