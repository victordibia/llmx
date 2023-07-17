from typing import Any, Optional
from pydantic.dataclasses import dataclass

@dataclass
class TextGenerationConfig:
    """Configuration for a text generation"""
    n: int = 1
    # prompt: Optional[str] = None
    # suffix: str = None
    temperature: float = 0.5
    max_tokens: int = None
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    model_type: str = "openai"
    model: str = "gpt-3.5-turbo-0301"
    messages: list[dict] = None


@dataclass
class TextGenerationResponse:
    """Response from a text generation"""
    text: list[Any]
    config: TextGenerationConfig
    logprobs: Optional[Any] = None
    usage: Optional[Any] = None