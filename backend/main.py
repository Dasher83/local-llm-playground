import json
import os
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="LLM Inference API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

class ChatRequest(BaseModel):
    model: str
    message: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ModelInfo(BaseModel):
    name: str
    size: str
    family: str
    format: str
    modified_at: str

@app.get("/")
async def root():
    return {"message": "LLM Inference API is running"}

@app.get("/health")
async def health_check():
    """Check if Ollama is accessible"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                return {"status": "healthy", "ollama": "connected"}
            else:
                return {"status": "unhealthy", "ollama": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """Get list of available models from Ollama"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = []
                for model in data.get("models", []):
                    models.append(ModelInfo(
                        name=model["name"],
                        size=model["size"],
                        family=model.get("details", {}).get("family", "unknown"),
                        format=model.get("details", {}).get("format", "unknown"),
                        modified_at=model["modified_at"]
                    ))
                return models
            else:
                raise HTTPException(status_code=500, detail="Failed to fetch models from Ollama")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")

@app.post("/pull")
async def pull_model(model_name: str):
    """Pull a model from Ollama registry"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/pull",
                json={"name": model_name}
            )
            if response.status_code == 200:
                return {"message": f"Model {model_name} pulled successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to pull model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pulling model: {str(e)}")

@app.post("/chat")
async def chat_with_model(request: ChatRequest):
    """Chat with a specific model"""
    try:
        # Prepare the prompt
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        
        # Make request to Ollama
        async with httpx.AsyncClient(timeout=120.0) as client:
            ollama_request = {
                "model": request.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_request
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data["message"]["content"],
                    "model": request.model,
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0)
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to get response from model")
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error chatting with model: {str(e)}")

@app.get("/model-suggestions")
async def get_model_suggestions():
    """Get suggestions for popular models to try"""
    return {
        "lightweight": [
            {"name": "llama3.2:1b", "description": "Very fast, good for testing", "size": "~1.3GB"},
            {"name": "llama3.2:3b", "description": "Good balance of speed and quality", "size": "~2.0GB"},
            {"name": "phi3:mini", "description": "Microsoft's small but capable model", "size": "~2.3GB"}
        ],
        "medium": [
            {"name": "llama3.1:8b", "description": "Great general purpose model", "size": "~4.7GB"},
            {"name": "mistral:7b", "description": "Fast and efficient", "size": "~4.1GB"},
            {"name": "gemma2:9b", "description": "Google's latest model", "size": "~5.4GB"}
        ],
        "large": [
            {"name": "llama3.1:70b", "description": "High quality, needs lots of RAM", "size": "~40GB"},
            {"name": "mixtral:8x7b", "description": "Mixture of experts model", "size": "~26GB"},
            {"name": "codellama:34b", "description": "Specialized for coding", "size": "~19GB"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
