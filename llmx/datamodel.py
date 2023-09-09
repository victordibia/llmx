from dataclasses import asdict
from typing import Any, Optional, Union, List
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

    def __iter__(self):
        for key, value in self._fields_dict.items():
            yield key, value


@dataclass
class TextGenerationConfig:
    n: int = 1
    temperature: float = 0.1
    max_tokens: Union[int, None] = None
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    provider: Union[str, None] = None
    model: Optional[str] = None
    stop: Union[List[str], str, None] = None
    use_cache: bool = True

    def __post_init__(self):
        self._fields_dict = asdict(self)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self._fields_dict.get(key)

    def __iter__(self):
        for key, value in self._fields_dict.items():
            yield key, value


@dataclass
class TextGenerationResponse:
    """Response from a text generation"""

    text: List[Message]
    config: Any
    logprobs: Optional[Any] = None  # logprobs if available
    usage: Optional[Any] = None  # usage statistics from the provider
    response: Optional[Any] = None  # full response from the provider

    def __post_init__(self):
        self._fields_dict = asdict(self)

    def __getitem__(self, key: Union[str, int]) -> Any:
        return self._fields_dict.get(key)

    def __iter__(self):
        for key, value in self._fields_dict.items():
            yield key, value

    def to_dict(self):
        return self._fields_dict

    def __json__(self):
        return self._fields_dict
