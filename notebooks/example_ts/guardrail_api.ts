import { GuardrailApiResponse } from './guardrail_types';
import { Config } from './config';

export async function callGuardrailApi({ prompt, id, datasetId, response, config }: { prompt: string; id: string; datasetId: string; response?: string; config: Config }) {
  const postData = {
    prompt,
    id,
    datasetId,
    ...(response && { response })  // You can guardrail the response/prompt pair or just the prompt
  };

  const fetchOptions: RequestInit = {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'X-API-Key': config.guardrails.api_key,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(postData)
  };

  const url = `${config.guardrails.endpoint}/evaluate?log=${config.guardrails.log_profile}&perf_info=false`;
  // console.log("url: " + url)
  const result = fetch(url, fetchOptions)
    .then(response => response.json())
    .then(data => data as GuardrailApiResponse)
    .catch((error) => {
      console.error('Error:', error);
      throw error;
    });

  return result;
}