"""
Pydantic models for the LLM Inference API.

This module contains all the data models used for request/response validation
and serialization throughout the API.
"""

from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for chat endpoints."""

    model: str
    message: str
    system_prompt: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""

    response: str
    model: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    eval_count: int


class ModelInfo(BaseModel):
    """Information about an available model."""

    name: str
    size: str
    family: str
    format: str
    modified_at: str


class ModelSuggestion(BaseModel):
    """A suggested model with metadata."""

    name: str
    description: str
    size: str


class ModelSuggestionsResponse(BaseModel):
    """Response containing categorized model suggestions."""

    lightweight: List[ModelSuggestion]
    medium: List[ModelSuggestion]
    large: List[ModelSuggestion]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ollama: Optional[str] = None
    error: Optional[str] = None


class PullResponse(BaseModel):
    """Response for model pull operations."""

    message: str
