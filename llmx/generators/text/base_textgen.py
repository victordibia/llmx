import os
from typing import Union
from diskcache import Cache
from ...utils import get_user_cache_dir
from ...datamodel import TextGenerationConfig, TextGenerationResponse
from ...version import APP_NAME
from abc import ABC, abstractmethod

class BaseTextGenerator(ABC):

    def __init__(self, provider: str = "openai", **kwargs):
        self.provider = provider
        self.model_name = kwargs.get("model_name", "gpt-3.5-turbo")

        app_name = APP_NAME
        cache_dir_default = get_user_cache_dir(app_name)
        cache_dir_based_on_model = os.path.join(cache_dir_default, self.provider, self.model_name)
        self.cache_dir = kwargs.get("cache_dir", cache_dir_based_on_model)
        self.cache = Cache(self.cache_dir)

    @abstractmethod
    def generate(
        self, messages: Union[list[dict], str], config: TextGenerationConfig = TextGenerationConfig(), use_cache=True, **kwargs
    ) -> TextGenerationResponse:
        pass

    @abstractmethod
    def count_tokens(self, text) -> int:
        pass