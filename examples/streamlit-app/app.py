import os
import logging

from openai import OpenAI
import streamlit as st
import openllmtelemetry

st.set_page_config(page_title="ACME Customer Service", page_icon="👩🏽‍💻")

ORG_ID = os.environ["WHYLABS_API_KEY"][-10:]
PROMPT_EXAMPLES_FOLDER = "./examples/demo"


@st.cache_resource
def instrument(model_id):
    openllmtelemetry.instrument(application_name="chatbot-guardrails", dataset_id=model_id)


def run_app():
    logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)

    instrument(os.environ["WHYLABS_DEFAULT_DATASET_ID"])

    with st.sidebar:
        st.header("WhyLabs Secure")
        st.markdown("GuardRails: **Enabled**")
        st.markdown(
            f"[Policy](https://hub.whylabsapp.com/{ORG_ID}/{os.environ['WHYLABS_DEFAULT_DATASET_ID']}/llm-secure/policy?presetRange=daily-relative-7)")

        st.markdown(
            f"[Trace Dashboard](https://hub.whylabsapp.com/{ORG_ID}/{os.environ['WHYLABS_DEFAULT_DATASET_ID']}/llm-secure/traces?presetRange=daily"
            f"-relative-7)")

        prompt_examples = [f for f in os.listdir(PROMPT_EXAMPLES_FOLDER) if f.endswith(".txt")]
        prompt_examples.sort()
        st_prompt_example = st.selectbox("Select prompt example", prompt_examples, index=0)
        with open(os.path.join(PROMPT_EXAMPLES_FOLDER, st_prompt_example), "r") as file:
            prompt_example_text = file.read() or ""
        st.text_area(
            label="Example", value=prompt_example_text, height=300, key="output_text_input", disabled=True,
        )

    st.subheader("ACME Customer Service Chatbot")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Set a default model if one is not set
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = [{"role": "system",
                                     "content": """
                                You are a helpful customer service chatbot. You strive to make customer happy. Here are 
                                the topics you can discuss:
                                - Shipping time
                                - Order status
                                - Returns links to policy documents
                                - Product information
                                - Shipping time
                                """}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        prompt_entry = {"role": "user", "content": prompt}
        st.session_state.messages.append(prompt_entry)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            context = st.session_state.context.copy()
            context.append(prompt_entry)
            result = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=context,
                stream=False,
            )
            msg = result.choices[0].message.content
            st.write(msg)
            assistant_entry = {"role": "assistant", "content": msg}
            st.session_state.messages.append(assistant_entry)
            if not result.model.startswith("whylabs"):
                st.session_state.context.append(prompt_entry)
                st.session_state.context.append(assistant_entry)


if __name__ == "__main__":
    run_app()
