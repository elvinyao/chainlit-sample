"""Data models and state definitions."""

from typing import Any, Dict, TypedDict

from pydantic import BaseModel, Field


class FormData(BaseModel):
    """Structured form data extracted by the LLM."""

    name: str = Field(..., description="User name")
    email: str = Field(..., description="User email")
    message: str = Field(..., description="User message")


class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    input: str
    parsed: Dict[str, Any]
    result_text: str
    screenshot_path: str
