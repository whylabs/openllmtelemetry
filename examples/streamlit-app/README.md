# Streamlit Chatbot with `openllmtelemetry`
This example packages a chatbot UI example using `streamlit` along with `openllmtelemetry` instrumentation to demonstrate how users can benefit from securing AI applications with Guardrails and also automatically sending prompt and response traces to WhyLabs.

## Configuration
We have put together a `.env` file containing all relevant environment variables for this app to integrate seamlessly with a running container and WhyLabs. 

**Checklist:**
- [ ] You have already setup an account organization and a model in WhyLabs
- [ ] You have a `langkit-container` up and running on your environment
- [ ] You have defined Guardrails policies for each `dataset-id` you want to guardrail

With those checked out, make sure you correctly fill in the `.env` file on this package and set them up on your environment, by running: 

```sh
set -a && source .env && set +a
```

## Getting started

1. Clone the repo

2. Install dependencies (preferably in a virtual environment)

```sh
pip install -r requirements.txt
```

3. Start the app:

```sh
streamlit run app.py
```
