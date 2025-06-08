import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Local LLM Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_backend_health():
    """Check if backend is accessible"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get(f"{BACKEND_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return []

def get_model_suggestions():
    """Get model suggestions"""
    try:
        response = requests.get(f"{BACKEND_URL}/model-suggestions", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error fetching model suggestions: {str(e)}")
        return {}

def pull_model(model_name):
    """Pull a model"""
    try:
        # Use a very long timeout for model downloads (1 hour)
        response = requests.post(f"{BACKEND_URL}/pull", params={"model_name": model_name}, timeout=3600)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def chat_with_model(model, message, system_prompt=None, temperature=0.7, max_tokens=1000):
    """Send chat request to model"""
    try:
        payload = {
            "model": model,
            "message": message,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        # Use a very long timeout for chat requests (20 minutes) for large models
        response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=1200)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

# Main app
def main():
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
            st.json(health_info)

        # Available models
        st.subheader("📦 Installed Models")
        models = get_available_models()

        if models:
            for model in models:
                with st.expander(f"📊 {model['name']}"):
                    st.write(f"**Size:** {model['size']}")
                    st.write(f"**Family:** {model['family']}")
                    st.write(f"**Format:** {model['format']}")
                    st.write(f"**Modified:** {model['modified_at'][:10]}")
        else:
            st.info("No models installed yet")

        # Model suggestions
        st.subheader("💡 Suggested Models")
        suggestions = get_model_suggestions()
        
        for category, model_list in suggestions.items():
            st.write(f"**{category.title()}:**")
            for model in model_list:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {model['name']}")
                    st.caption(f"{model['description']} ({model['size']})")
                with col2:
                    if st.button("Pull", key=f"pull_{model['name']}"):
                        # Show warning for large models
                        if "large" in category.lower():
                            st.warning("⚠️ Large model - this may take 10-30 minutes!")

                        with st.spinner(f"Downloading {model['name']}... This may take several minutes for large models."):
                            st.info("💡 Tip: Check the Docker logs to see download progress")
                            success, result = pull_model(model['name'])
                            if success:
                                st.success("✅ Pulled!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result}")

        # Custom model pull
        st.subheader("🔧 Pull Custom Model")
        custom_model = st.text_input("Model name (e.g., llama3.2:1b)")
        if st.button("Pull Custom Model") and custom_model:
            # Show info about download time
            st.info("ℹ️ Download time depends on model size. Large models (>10GB) can take 10-30 minutes.")
            with st.spinner(f"Downloading {custom_model}... Please be patient for large models."):
                st.info("💡 Tip: Check the Docker logs to see download progress")
                success, result = pull_model(custom_model)
                if success:
                    st.success("✅ Model pulled successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to pull model: {result}")

    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("⚙️ Chat Settings")
        
        # Model selection
        if models:
            selected_model = st.selectbox(
                "Choose Model:",
                options=[m['name'] for m in models],
                index=0
            )
        else:
            st.warning("No models available. Please pull a model first.")
            selected_model = None

        # System prompt
        system_prompt = st.text_area(
            "System Prompt (optional):",
            value="You are a helpful AI assistant.",
            height=100
        )
        
        # Parameters
        temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Tokens:", 100, 4000, 1000, 100)
        
        # Performance tracking
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if st.session_state.chat_history:
            st.subheader("📊 Performance")
            df = pd.DataFrame(st.session_state.chat_history)
            
            # Response time chart
            if 'total_duration' in df.columns:
                df['response_time_sec'] = df['total_duration'] / 1e9  # Convert nanoseconds to seconds
                fig = px.line(df, y='response_time_sec', title="Response Time Over Time")
                st.plotly_chart(fig, use_container_width=True)
            
            # Model usage
            if 'model' in df.columns:
                model_counts = df['model'].value_counts()
                fig = px.pie(values=model_counts.values, names=model_counts.index, title="Model Usage")
                st.plotly_chart(fig, use_container_width=True)

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
                            st.metric("Response Time", f"{meta.get('total_duration', 0) / 1e9:.2f}s")
                            st.metric("Model", meta.get('model', 'Unknown'))
                        with col_b:
                            st.metric("Tokens Generated", meta.get('eval_count', 0))
                            st.metric("Load Time", f"{meta.get('load_duration', 0) / 1e9:.2f}s")

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
                        max_tokens=max_tokens
                    )
                    end_time = time.time()

                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    st.markdown(response["response"])
                    
                    # Store response with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"],
                        "metadata": response
                    })
                    
                    # Add to performance tracking
                    st.session_state.chat_history.append({
                        "timestamp": datetime.now(),
                        "model": selected_model,
                        "total_duration": response.get("total_duration", 0),
                        "load_duration": response.get("load_duration", 0),
                        "eval_count": response.get("eval_count", 0),
                        "prompt_eval_count": response.get("prompt_eval_count", 0),
                        "wall_time": end_time - start_time
                    })

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
