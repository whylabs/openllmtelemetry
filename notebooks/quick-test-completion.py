import asyncio
import logging
import os

from openai import AsyncOpenAI

import openllmtelemetry

logging.getLogger().setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.DEBUG)

# os.environ["TRACE_PROMPT_AND_RESPONSE"] = "true"
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-2271"  #
os.environ["WHYLABS_ENDPOINT"] = "https://songbird.development.whylabsdev.com/"
# os.environ["WHYLABS_API_ENDPOINT"] = ""
# os.environ["WHYLABS_TRACES_ENDPOINT"] =""

openllmtelemetry.instrument(application_name="andy-test-5", disable_batching=True, debug=True)

client = AsyncOpenAI()


async def do_it():
    response = await client.completions.create(
        model="gpt-3.5-turbo-instruct",  # You can choose different engines based on your needs
        prompt="Write a tagline for an ice cream shop.",
        max_tokens=50,  # You can adjust the max tokens to control the length of the generated text
        temperature=0.7,  # You can adjust the temperature to control the creativity of the generated text
        n=1,  # Number of completions to generate
        stop=None,  # You can provide a list of stop sequences to stop generation early
    )
    print(response)


asyncio.run(do_it())
