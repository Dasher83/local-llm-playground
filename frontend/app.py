import os
import time
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv
from models import (
    ChatResponse,
    ChatResult,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    ModelSuggestionsResponse,
    PullResponse,
)

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "90"))
MODEL_PULL_TIMEOUT_SECONDS = float(os.getenv("MODEL_PULL_TIMEOUT_SECONDS", "3600"))


st.set_page_config(
    page_title="Local LLM Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def check_backend_health() -> Tuple[bool, HealthResponse]:
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = HealthResponse(**response.json())
            return True, health_data
        else:
            return False, HealthResponse(status="unhealthy", error="HTTP error")
    except Exception as e:
        return False, HealthResponse(status="unhealthy", error=str(e))


def get_available_models() -> List[ModelInfo]:
    """Get list of available models"""
    try:
        response = requests.get(f"{BACKEND_URL}/models", timeout=10)
        if response.status_code == 200:
            models_data = response.json()
            return [ModelInfo(**model) for model in models_data]
        return []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []


def get_model_suggestions() -> ModelSuggestionsResponse:
    """Get model suggestions"""
    try:
        response = requests.get(f"{BACKEND_URL}/model-suggestions", timeout=5)
        if response.status_code == 200:
            return ModelSuggestionsResponse(**response.json())
        return ModelSuggestionsResponse(lightweight=[], medium=[], large=[])
    except Exception as e:
        st.error(f"Error fetching model suggestions: {str(e)}")
        return ModelSuggestionsResponse(lightweight=[], medium=[], large=[])


def pull_model(model_name: str) -> Tuple[bool, PullResponse]:
    """Pull a model"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/pull",
            params={"model_name": model_name},
            timeout=MODEL_PULL_TIMEOUT_SECONDS,
        )
        if response.status_code == 200:
            return True, PullResponse(**response.json())
        else:
            return False, PullResponse(message=f"HTTP error: {response.status_code}")
    except Exception as e:
        return False, PullResponse(message=f"Error: {str(e)}")


def chat_with_model(
    model: str,
    message: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> ChatResult:
    """Send chat request to model"""
    try:
        payload = {
            "model": model,
            "message": message,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = requests.post(
            f"{BACKEND_URL}/chat", json=payload, timeout=CHAT_TIMEOUT_SECONDS
        )
        if response.status_code == 200:
            return ChatResponse(**response.json())
        else:
            return ErrorResponse(error=f"HTTP {response.status_code}: {response.text}")
    except Exception as e:
        return ErrorResponse(error=str(e))


# Main app
def main() -> None:
    st.title("🧠 Local LLM Playground")
    st.markdown("Experiment with different LLMs running locally via Ollama")

    # Sidebar for model management
    with st.sidebar:
        st.header("🛠️ Model Management")

        # Health check
        health_ok, health_info = check_backend_health()
        if health_ok:
            st.success("✅ Backend Connected")
        else:
            st.error("❌ Backend Disconnected")
            st.json(health_info.model_dump())

        # Available models
        st.subheader("📦 Installed Models")
        models = get_available_models()

        if models:
            for model in models:
                with st.expander(f"📊 {model.name}"):
                    st.write(f"**Size:** {model.size}")
                    st.write(f"**Family:** {model.family}")
                    st.write(f"**Format:** {model.format}")
                    st.write(f"**Modified:** {model.modified_at[:10]}")
        else:
            st.info("No models installed yet")

        # Model suggestions
        st.subheader("💡 Suggested Models")
        suggestions = get_model_suggestions()

        # Convert suggestions to dict for iteration
        suggestions_dict = {
            "lightweight": suggestions.lightweight,
            "medium": suggestions.medium,
            "large": suggestions.large,
        }

        for category, model_list in suggestions_dict.items():
            st.write(f"**{category.title()}:**")
            for suggestion in model_list:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {suggestion.name}")
                    st.caption(f"{suggestion.description} ({suggestion.size})")
                with col2:
                    if st.button("Pull", key=f"pull_{suggestion.name}"):
                        # Show warning for large models
                        if "large" in category.lower():
                            st.warning("⚠️ Large model - this may take 10-30 minutes!")

                        with st.spinner(
                            (
                                f"Downloading {suggestion.name}... "
                                f"This may take several minutes for large models."
                            )
                        ):
                            st.info(
                                "💡 Tip: Check the Docker logs to see download progress"
                            )
                            success, result = pull_model(suggestion.name)
                            if success:
                                st.success("✅ Pulled!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result.message}")

        # Custom model pull
        st.subheader("🔧 Pull Custom Model")
        custom_model = st.text_input("Model name (e.g., llama3.2:1b)")
        if st.button("Pull Custom Model") and custom_model:
            # Show info about download time
            st.info(
                (
                    "ℹ️ Download time depends on model size. "
                    "Large models (>10GB) can take 10-30 minutes."
                )
            )
            with st.spinner(
                f"Downloading {custom_model}... Please be patient for large models."
            ):
                st.info("💡 Tip: Check the Docker logs to see download progress")
                success, result = pull_model(custom_model)
                if success:
                    st.success("✅ Model pulled successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to pull model: {result.message}")

    # Main chat interface
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("⚙️ Chat Settings")

        # Model selection
        if models:
            selected_model = st.selectbox(
                "Choose Model:", options=[m.name for m in models], index=0
            )
        else:
            st.warning("No models available. Please pull a model first.")
            selected_model = None

        # System prompt
        system_prompt = st.text_area(
            "System Prompt (optional):",
            value="You are a helpful AI assistant.",
            height=100,
        )

        # Parameters
        temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens:", 100, 4000, 1000, 100)

        # Performance tracking
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

    with col1:
        st.subheader("💬 Chat Interface")

        # Chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and "metadata" in message:
                    with st.expander("📈 Response Details"):
                        meta = message["metadata"]
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Response Time",
                                f"{meta.get('total_duration', 0) / 1e9:.2f}s",
                            )
                            st.metric("Model", meta.get("model", "Unknown"))
                        with col_b:
                            st.metric("Tokens Generated", meta.get("eval_count", 0))
                            st.metric(
                                "Load Time",
                                f"{meta.get('load_duration', 0) / 1e9:.2f}s",
                            )

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            if not selected_model:
                st.error("Please select a model first!")
                st.stop()

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response from model
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    start_time = time.time()
                    response = chat_with_model(
                        model=selected_model,
                        message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    end_time = time.time()

                if isinstance(response, ErrorResponse):
                    st.error(f"Error: {response.error}")
                else:
                    st.markdown(response.response)

                    # Store response with metadata
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": response.response,
                            "metadata": response.model_dump(),
                        }
                    )

                    # Add to performance tracking
                    st.session_state.chat_history.append(
                        {
                            "timestamp": datetime.now(),
                            "model": selected_model,
                            "total_duration": response.total_duration,
                            "load_duration": response.load_duration,
                            "eval_count": response.eval_count,
                            "prompt_eval_count": response.prompt_eval_count,
                            "wall_time": end_time - start_time,
                        }
                    )

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
