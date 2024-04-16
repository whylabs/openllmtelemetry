# OpenLLMTelemetry

`openllmtelemetry` is an open-source Python library that provides Open Telemetry integration with Large Language Models (LLMs). It is designed to facilitate tracing applications that leverage LLMs and Generative AI, ensuring better observability and monitoring.

## Features

- Easy integration with Open Telemetry for LLM applications.
- Real-time tracing and monitoring of LLM-based systems.
- Enhanced safeguards and insights for your LLM applications.

## Installation

To install `openllmtelemetry` simply use pip:

```bash
pip install openllmtelemetry
```

## Usage 🚀

Here's a basic example of how to use **OpenLLMTelemetry** in your project:

First you need to setup a few environment variables to specify where you want your LLM telemetry to be sent, and make sure you also have any API keys set for interacting with your LLM and for sending the telemetry to [WhyLabs](https://hub.whylabsapp.com/signup)

```python
import os

os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "your-model-id" #  e.g. model-1 
os.environ["WHYLABS_API_KEY"] = "replace-with-your-whylabs-api-key"

```

After you verify your env variables are set you can now instrument your app by running the following:

```python
import openllmtelemetry

openllmtelemetry.instrument()
```

This will automatically instrument your calls to LLMs to gather open telemetry traces and send these to WhyLabs.

## Integration: OpenAI
Integration with an OpenAI application is straightforward with `openllmtelemetry` package.

First, you need to set a few environment variables. This can be done via your container set up or via code. 

```python
import os 

os.environ["WHYLABS_API_KEY"] = "<your-whylabs-api-key>"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "<your-llm-resource-id>"
os.environ["WHYLABS_GUARD_ENDPOINT"] = "<your container endpoint>"
os.environ["WHYLABS_GUARD_API_KEY"] = "internal-secret-for-whylabs-Secure"
```

Once this is done, all of your OpenAI interactions will be automatically traced. If you have rulesets enabled for blocking in WhyLabs Secure policy, the library will block requests accordingly

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You are a helpful chatbot. "
    },
    {
      "role": "user",
      "content": "Aren't noodles amazing?"
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)
```



## Requirements 📋

- Python 3.8 or higher
- opentelemetry-api
- opentelemetry-sdk

## Contributing 👐

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License 📄

**OpenLLMTelemetry** is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) for more details.

## Contact 📧

For support or any questions, feel free to contact us at support@whylabs.ai.