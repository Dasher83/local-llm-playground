import logging
import os
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ModelInfo,
    ModelSuggestion,
    ModelSuggestionsResponse,
    PullResponse,
)

load_dotenv()

CHAT_TIMEOUT_SECONDS = float(os.getenv("CHAT_TIMEOUT_SECONDS", "90"))
MODEL_PULL_TIMEOUT_SECONDS = float(os.getenv("MODEL_PULL_TIMEOUT_SECONDS", "3600"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Inference API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting LLM Inference API")
    logger.info(f"Chat timeout: {CHAT_TIMEOUT_SECONDS}s")
    logger.info(f"Model pull timeout: {MODEL_PULL_TIMEOUT_SECONDS}s")
    logger.info(f"Ollama base URL: {OLLAMA_BASE_URL}")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")


def format_bytes(bytes_size: int) -> str:
    """Convert bytes to human readable format"""
    if bytes_size == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size_value = float(bytes_size)  # Convert to float for division
    while size_value >= 1024 and i < len(size_names) - 1:
        size_value /= 1024
        i += 1

    return f"{size_value:.1f} {size_names[i]}"


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "LLM Inference API is running"}


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if Ollama is accessible"""
    logger.info("Health check requested")
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Checking Ollama connection at {OLLAMA_BASE_URL}")
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            logger.info(f"Ollama response status: {response.status_code}")
            if response.status_code == 200:
                return HealthResponse(status="healthy", ollama="connected")
            else:
                logger.warning(
                    f"Ollama health check failed with status {response.status_code}"
                )
                return HealthResponse(status="unhealthy", ollama="disconnected")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(status="unhealthy", error=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def list_models() -> List[ModelInfo]:
    """Get list of available models from Ollama"""
    logger.info("Models list requested")
    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Fetching models from {OLLAMA_BASE_URL}/api/tags")
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            logger.info(f"Ollama models response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.info(f"Raw Ollama response: {data}")
                models = []
                for model in data.get("models", []):
                    logger.debug(f"Processing model: {model}")
                    # Convert size from int (bytes) to human-readable string
                    size_bytes = model.get("size", 0)
                    size_str = (
                        format_bytes(size_bytes)
                        if isinstance(size_bytes, int)
                        else str(size_bytes)
                    )

                    models.append(
                        ModelInfo(
                            name=model["name"],
                            size=size_str,
                            family=model.get("details", {}).get("family", "unknown"),
                            format=model.get("details", {}).get("format", "unknown"),
                            modified_at=model["modified_at"],
                        )
                    )
                logger.info(f"Successfully processed {len(models)} models")
                return models
            else:
                logger.error(
                    f"Failed to fetch models: HTTP {response.status_code}, "
                    f"Response: {response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=(
                        f"Failed to fetch models from Ollama: "
                        f"{response.status_code}"
                    ),
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting to Ollama: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error connecting to Ollama: {str(e)}"
        )


@app.post("/pull", response_model=PullResponse)
async def pull_model(model_name: str) -> PullResponse:
    """Pull a model from Ollama registry"""
    logger.info(f"Pull request for model: {model_name}")
    try:
        timeout = httpx.Timeout(MODEL_PULL_TIMEOUT_SECONDS, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            logger.info(f"Sending pull request to {OLLAMA_BASE_URL}/api/pull")
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/pull", json={"name": model_name}
            )
            logger.info(f"Pull response status: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Model {model_name} pulled successfully")
                return PullResponse(message=f"Model {model_name} pulled successfully")
            else:
                logger.error(
                    (
                        f"Failed to pull model: HTTP {response.status_code}, "
                        f"Response: {response.text}"
                    )
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to pull model: {response.status_code}",
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pulling model {model_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error pulling model: {str(e)}")


@app.post("/chat")
async def chat_with_model(request: ChatRequest) -> ChatResponse:
    """Chat with a specific model"""
    logger.info(f"Chat request for model: {request.model}")
    try:
        # Prepare the prompt
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})

        logger.debug(f"Prepared messages: {messages}")

        # Make request to Ollama
        timeout = httpx.Timeout(CHAT_TIMEOUT_SECONDS, connect=60.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            ollama_request = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                },
            }

            logger.info(f"Sending chat request to {OLLAMA_BASE_URL}/api/chat")
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=ollama_request
            )

            logger.info(f"Chat response status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                logger.info("Chat request successful")
                return ChatResponse(
                    response=data["message"]["content"],
                    model=request.model,
                    total_duration=data.get("total_duration", 0),
                    load_duration=data.get("load_duration", 0),
                    prompt_eval_count=data.get("prompt_eval_count", 0),
                    eval_count=data.get("eval_count", 0),
                )
            else:
                logger.error(
                    (
                        f"Failed to get response from model: "
                        f"HTTP {response.status_code}, Response: {response.text}"
                    )
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get response from model: {response.status_code}",
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error chatting with model {request.model}: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail=f"Error chatting with model: {str(e)}"
        )


@app.get("/model-suggestions", response_model=ModelSuggestionsResponse)
async def get_model_suggestions() -> ModelSuggestionsResponse:
    """Get suggestions for popular models to try"""
    logger.info("Model suggestions requested")

    return ModelSuggestionsResponse(
        lightweight=[
            ModelSuggestion(
                name="llama3.2:1b",
                description="Very fast, good for testing",
                size="~1.3GB",
            ),
            ModelSuggestion(
                name="llama3.2:3b",
                description="Good balance of speed and quality",
                size="~2.0GB",
            ),
            ModelSuggestion(
                name="phi3:mini",
                description="Microsoft's small but capable model",
                size="~2.3GB",
            ),
        ],
        medium=[
            ModelSuggestion(
                name="llama3.1:8b",
                description="Great general purpose model",
                size="~4.7GB",
            ),
            ModelSuggestion(
                name="mistral:7b",
                description="Fast and efficient",
                size="~4.1GB",
            ),
            ModelSuggestion(
                name="gemma2:9b",
                description="Google's latest model",
                size="~5.4GB",
            ),
        ],
        large=[
            ModelSuggestion(
                name="llama3.1:70b",
                description="High quality, needs lots of RAM",
                size="~40GB",
            ),
            ModelSuggestion(
                name="mixtral:8x7b",
                description="Mixture of experts model",
                size="~26GB",
            ),
            ModelSuggestion(
                name="codellama:34b",
                description="Specialized for coding",
                size="~19GB",
            ),
        ],
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)  # nosec B104
