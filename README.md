# Local LLM Playground with Docker Compose

This repository provides a ready-to-use environment for experimenting with locally served Large Language Models (LLMs) using [Ollama](https://ollama.com/), a [FastAPI](https://fastapi.tiangolo.com/) backend, and a [Streamlit](https://streamlit.io/) frontend. The setup allows you to:

- Run and manage LLMs locally on your machine
- Interact with models via a modern chat UI
- Experiment with different models to see what your hardware can handle

---

## 🐳 Quick Start

### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/) installed
- Sufficient disk space and RAM for the models you want to try

### 2. Clone the Repository
```bash
git clone <this-repo-url>
cd dumb_it_down
```

### 3. Start All Services
```bash
docker compose up --build
```
- This will start three services:
  - **Ollama** (serves LLMs on port 11434)
  - **FastAPI backend** (API on port 8000)
  - **Streamlit frontend** (UI on port 8501)

### 4. Access the UI
Open your browser and go to: [http://localhost:8501](http://localhost:8501)

---

## 🧠 Features
- **Model Management:**
  - View installed models
  - Pull new models from Ollama registry (from sidebar)
  - See suggestions for lightweight, medium, and large models
- **Chat Interface:**
  - Select model, set system prompt, temperature, and max tokens
  - View chat history and clear it anytime
  - See response time and model usage statistics
- **Backend API:**
  - `/models` — List available models
  - `/pull` — Pull a new model
  - `/chat` — Send a prompt and get a response
  - `/model-suggestions` — Get recommended models

---

## ⚡ Example Usage
1. **Pull a Model:** Use the sidebar to pull a suggested or custom model (e.g., `llama3.2:1b` for fast testing).
2. **Chat:** Select the model, enter your prompt, and chat!
3. **Experiment:** Try larger models to see what your laptop can handle. Monitor performance in the sidebar.

---

## 🛠️ Customization
- **Add More Models:**
  - Use the sidebar to pull any model available in Ollama's registry.
- **Change Backend/Frontend:**
  - Edit `backend/main.py` or `frontend/app.py` as needed.
- **GPU Support:**
  - Uncomment the `deploy` section in `docker-compose.yml` for NVIDIA GPU support (requires NVIDIA Docker runtime).

---

## 🧩 Project Structure
```
docker-compose.yml
backend/
  Dockerfile
  main.py
  requirements.txt
frontend/
  Dockerfile
  app.py
  requirements.txt
```

---

## 📝 Notes
- The first time you pull a model, it may take a while to download.
- For best performance with large models, use a machine with a modern GPU and plenty of RAM.
- Ollama supports many open LLMs. See [Ollama's model library](https://ollama.com/library) for options.

---

## 📚 References
- [Ollama Documentation](https://github.com/ollama/ollama)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🚀 License
MIT License. See [LICENSE](LICENSE) for details.
