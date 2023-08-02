from .openai_textgen import OpenAITextGenerator
from .palm_textgen import PalmTextGenerator
from .cohere_textgen import CohereTextGenerator


def text_generator(provider: str = "openai", **kwargs):
    if provider == "openai" or provider == "default":
        return OpenAITextGenerator(**kwargs)
    elif provider == "palm" or provider == "google":
        return PalmTextGenerator(provider=provider, **kwargs)
    elif provider == "cohere":
        return CohereTextGenerator(provider=provider, **kwargs)
    elif provider == "hf" or provider == "huggingface":
        try:
            import transformers
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                GenerationConfig,
            )
        except ImportError:
            raise ImportError(
                "Please install the `transformers` package to use the HFTextGenerator class. pip install llmx[transformers]"
            )

        # Check if torch package is installed
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Please install the `torch` package to use the HFTextGenerator class.  pip install llmx[transformers]"
            )

        from .hf_textgen import HFTextGenerator

        return HFTextGenerator(provider=provider, **kwargs)

    else:
        raise ValueError(
            f"Invalid provider '{provider}'.  Supported providers are 'openai', 'hf', 'palm', and 'cohere'."
        )
