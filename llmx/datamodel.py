from dataclasses import asdict
from typing import Any, Optional, Union
from pydantic.dataclasses import dataclass


@dataclass
class Message:
    role: str
    content: str

    def __post_init__(self):
        self._fields_dict = asdict(self)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self._fields_dict.get(key)

    def to_dict(self):
        return self._fields_dict


@dataclass
class TextGenerationConfig:
    n: int = 1
    temperature: float = 0.1
    max_tokens: Union[int, None] = None
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    model_type: str = "openai"
    model: str = None
    stop: Union[list[str], str, None] = None 

    def __post_init__(self):
        self._fields_dict = asdict(self)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self._fields_dict.get(key)


@dataclass
class TextGenerationResponse:
    """Response from a text generation"""

    text: list[Message]
    config: TextGenerationConfig
    logprobs: Optional[Any] = None
    usage: Optional[Any] = None

    def __post_init__(self):
        self._fields_dict = asdict(self)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self._fields_dict.get(key)
