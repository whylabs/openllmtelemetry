import { Span, context, trace, Tracer } from '@opentelemetry/api';
import { Config } from './config';
import { callGuardrailApi } from './guardrail_api';
import { GuardrailApiResponse, Metric, Score, Metadata, ValidationReportEntry } from './guardrail_types';


export async function wrap_guard_prompt({ prompt, id, datasetId, response, config, parentSpan, tracer }: { prompt: string; id: string; datasetId: string; response?: string; config: Config, parentSpan?: Span, tracer: Tracer}): Promise<GuardrailApiResponse> {
  const span_name =  response ? `guardrails.response` : `guardrails.request`;
  const ctx = parentSpan ? trace.setSpan(context.active(), parentSpan) : context.active();
  const span = tracer.startSpan(span_name, undefined, ctx);

  try {
    // Add an attribute to the span
    span.setAttribute('span.type', 'guardrails');

    const guardrailRequestData = {
      prompt: prompt,
      id: id,
      datasetId: datasetId,
      ...(response && { response }),
      config
    };

    const guardrailResponse: GuardrailApiResponse = await callGuardrailApi(guardrailRequestData);

    // Set metrics as span attributes
    guardrailResponse.metrics.forEach((metric: Metric) => {
      for (const [key, value] of Object.entries(metric)) {
        if (value !== null && !key.endsWith("redacted")) {
          span.setAttribute(`langkit.metrics.${key}`, value);
        }
      }
    });

    // Set scores as span attributes
    guardrailResponse.scores.forEach((score: Score) => {
      for (const [key, value] of Object.entries(score)) {
        if (value !== null) {
          const slimKey = key.replace("response.score.", "").replace("prompt.score.", "");
          span.setAttribute(`langkit.metrics.${slimKey}`, value);
        }
      }
    });

    // Set metadata as span attributes
    const metadata: Metadata = guardrailResponse.metadata;
    for (const [key, value] of Object.entries(metadata)) {
      span.setAttribute(`guardrail.api.${key}`, value);
    }

    // Set tags based on action type and validation results
    const tags: string[] = [];
    if (guardrailResponse.action.action_type === "block") {
      tags.push("BLOCKED");
    }

    // Add validation results to tags
    guardrailResponse.validation_results.report.forEach((report: ValidationReportEntry) => {
      const metric = report.metric.replace("response.score.", "").replace("prompt.score.", "");
      tags.push(metric);
    });

    if (tags.length > 0) {
      span.setAttribute("langkit.insights.tags", tags);
    }

    return guardrailResponse;
  } catch (error) {
    console.error('Error in wrap_guard_prompt:', error);
    span.setAttribute("guardrails.error", 1);
    throw error;
  } finally {
    span.end();
  }
}