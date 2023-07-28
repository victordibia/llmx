import os
from diskcache import Cache
from ...utils import get_user_cache_dir
from ...datamodel import TextGenerationConfig, TextGenerationResponse
from ...version import APP_NAME


class BaseTextGenerator:
    def __init__(self, provider: str = "openai", **kwargs):
        self.provider = provider
        self.model_name = kwargs.get("model_name", "gpt-3.5-turbo")

        app_name = APP_NAME
        cache_dir_default = get_user_cache_dir(app_name)
        cache_dir_based_on_model = os.path.join(
            cache_dir_default, self.provider, self.model_name
        )
        self.cache_dir = kwargs.get("cache_dir", cache_dir_based_on_model)
        self.cache = Cache(self.cache_dir)

    def generate(
        self, config: TextGenerationConfig, use_cache=True, **kwargs
    ) -> TextGenerationResponse:
        raise NotImplementedError(
            "This method should be implemented by all subclasses."
        )

    def count_tokens(self, text) -> int:
        raise NotImplementedError(
            "This method should be implemented by all subclasses."
        )
