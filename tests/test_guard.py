from openllmtelemetry.secure import GuardrailsApi

guard = GuardrailsApi(
    guardrails_endpoint="http://4.149.153.66",
    guardrails_api_key="wjYDn8K0jZRoWkuALgjr5TGuX/NaaWzSy7ntMxRsI6M+FwcEp9aNQcaEs/q9e5dn",
    dataset_id="model-0",
)


def test_eval_prompt():
    print(guard.eval_prompt("Hello world"))
