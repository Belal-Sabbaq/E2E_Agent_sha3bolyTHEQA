# ui_utils.py
import streamlit as st

def config_model_selector():
    with st.sidebar:
        mode = st.selectbox(
            "Model Mode",
            ["Local (Ollama)", "API (OpenAI)"],
            key="model_mode"
        )

        if mode == "Local (Ollama)":
            st.selectbox(
                "Local Model",
                ["qwen3:4b", "llama3", "mistral"],
                key="model_name"
            )

        else:
            st.selectbox(
                "OpenAI Model",
                ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
                key="model_name"
            )

            st.text_input(
                "OpenAI API Key",
                type="password",
                key="api_key"
            )
