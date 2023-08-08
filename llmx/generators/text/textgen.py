from .openai_textgen import OpenAITextGenerator
from .palm_textgen import PalmTextGenerator
from .cohere_textgen import CohereTextGenerator


def llm(provider: str = "openai", **kwargs):
    if provider.lower() == "openai" or provider.lower() == "default":
        return OpenAITextGenerator(**kwargs)
    elif provider.lower() == "palm" or provider.lower() == "google":
        return PalmTextGenerator(provider=provider, **kwargs)
    elif provider.lower() == "cohere":
        return CohereTextGenerator(provider=provider, **kwargs)
    elif provider.lower() == "hf" or provider.lower() == "huggingface":
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
