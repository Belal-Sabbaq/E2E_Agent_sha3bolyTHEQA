import streamlit as st

def config_model_selector():
    st.sidebar.header("Model Configuration")

    # ✅ Use Groq models 
    groq_models = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]

    # Dropdown for model selection
    model_name = st.sidebar.selectbox("Choose Model", groq_models, index=0)

    # Dropdown for mode selection
    model_mode = st.sidebar.selectbox("Mode", ["Local (Ollama)", "API (Groq)"], index=1)

    # Input for Groq API key
    api_key = st.sidebar.text_input("Groq API Key", type="password")

    # Save into session state
    st.session_state["model_name"] = model_name
    st.session_state["model_mode"] = model_mode
    st.session_state["api_key"] = api_key
