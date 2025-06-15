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
cd local_llm_playground
```

### 2.1. Environment Setup (Optional)
Copy the example environment file and customize if needed:
```bash
cp .env.example .env
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
3. **Experiment:** Try larger models to see what your laptop can handle.

---

## 🛠️ Development & Collaboration

### Setting Up for Development

#### 1. Prerequisites
- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/)
- [Python 3.13+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

#### 2. Clone and Setup
```bash
# Clone the repository
git clone <this-repo-url>
cd local_llm_playground

# Set up environment variables
cp .env.example .env
# Edit .env file as needed

# Create and activate Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make install-dev-dependencies
# Or manually: pip install -r dev.requirements.txt

# Install pre-commit hooks
pre-commit install
```

#### 3. Start Development Environment
```bash
docker compose up --build
```

### Available Make Commands

#### 🧠 Ollama Model Management
```bash
# List installed models
make ollama-list

# Pull/download a new model
make ollama-pull MODEL='llama3.2:1b'
make ollama-pull MODEL='mistral:7b'
make ollama-pull MODEL='codellama:13b'

# Show detailed model information
make ollama-show MODEL='llama3.2:1b'

# Remove a model to free up space
make ollama-remove MODEL='llama3.2:1b'

# Clean up unused model data
make ollama-cleanup
```

#### 📦 Dependency Management
```bash
# Install/upgrade development dependencies
make install-dev-dependencies
make upgrade-dev-dependencies

# Compile requirements from .in files
make compile-backend     # Updates backend/requirements.txt
make compile-frontend    # Updates frontend/requirements.txt

# Upgrade all dependencies to latest versions
make upgrade-backend     # Upgrade backend dependencies
make upgrade-frontend    # Upgrade frontend dependencies

# Update pre-commit hooks
make pre-commit-update
```

#### 📋 Logs & Monitoring
```bash
make logs            # View logs from all services
make logs-backend    # View only backend logs
make logs-frontend   # View only frontend logs
make logs-ollama     # View only Ollama logs
```

---

## 🧩 Project Structure
```
.
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore patterns
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── Makefile                 # Development commands
├── README.md               # This file
├── dev.requirements.txt    # Development dependencies
├── docker-compose.yml      # Docker services configuration
├── backend/                # FastAPI backend service
│   ├── Dockerfile
│   ├── main.py
│   ├── models.py
│   ├── requirements.in
│   └── requirements.txt
└── frontend/               # Streamlit frontend service
    ├── Dockerfile
    ├── app.py
    ├── models.py
    ├── requirements.in
    └── requirements.txt
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

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
