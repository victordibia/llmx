from .hf_textgen import HFTextGenerator
from .openai_textgen import OpenAITextGenerator
from .palm_textgen import PalmTextGenerator
from .cohere_textgen import CohereTextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse


class TextGenerator:
    def __init__(self, provider: str = "openai", **kwargs):
        self.provider = provider
        self.kwargs = kwargs

        self.generator_instance = self.get_instance()

    def get_instance(self):
        if self.provider == "openai" or self.provider == "default":
            return OpenAITextGenerator(**self.kwargs)
        elif self.provider == "hf" or self.provider == "huggingface":
            return HFTextGenerator(provider=self.provider, **self.kwargs)
        elif self.provider == "palm" or self.provider == "google":
            return PalmTextGenerator(provider=self.provider, **self.kwargs)
        elif self.provider == "cohere":
            return CohereTextGenerator(provider=self.provider, **self.kwargs)
        else:
            raise ValueError(
                f"Invalid provider '{self.provider}'.  Supported providers are 'openai', 'hf', 'palm', and 'cohere'."
            )

    def generate(
        self, config: TextGenerationConfig, use_cache=True, **kwargs
    ) -> TextGenerationResponse:
        return self.generator_instance.generate(
            config=config, use_cache=use_cache, **kwargs
        )

    def count_tokens(self, text) -> int:
        return self.generator_instance.count_tokens(text=text)
