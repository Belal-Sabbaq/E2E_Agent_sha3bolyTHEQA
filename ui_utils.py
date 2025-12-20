import streamlit as st


def config_model_selector():
    st.sidebar.header("Model Configuration")

    # -----------------------------
    # Mode selection FIRST
    # -----------------------------
    model_mode = st.sidebar.selectbox(
        "Mode",
        ["Local (Ollama)", "API (Groq)"],
        index=1
    )

    # -----------------------------
    # Model lists
    # -----------------------------
    groq_models = [
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]

    local_models = [
        "qwen3:4b",
        "llama3:8b",
    ]

    # -----------------------------
    # Conditional model selector
    # -----------------------------
    if model_mode == "API (Groq)":
        model_name = st.sidebar.selectbox(
            "Choose Groq Model",
            groq_models,
            index=0
        )

        api_key = st.sidebar.text_input(
            "Groq API Key",
            type="password"
        )

        if not api_key:
            st.sidebar.warning("Groq API key required")

    else:
        model_name = st.sidebar.selectbox(
            "Choose Local Ollama Model",
            local_models,
            index=0
        )
        api_key = None
        st.sidebar.info("Using local Ollama model")

    # -----------------------------
    # Persist in session state
    # -----------------------------
    st.session_state["model_name"] = model_name
    st.session_state["model_mode"] = model_mode
    st.session_state["api_key"] = api_key
