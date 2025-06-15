from typing import List, Optional, Union

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Response model for chat requests."""

    response: str
    model: str
    total_duration: int
    load_duration: int
    prompt_eval_count: int
    eval_count: int


class ErrorResponse(BaseModel):
    """Error response model for failed requests."""

    error: str


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


ChatResult = Union[ChatResponse, ErrorResponse]
