from ...utils import load_config
from .openai_textgen import OpenAITextGenerator
from .palm_textgen import PalmTextGenerator
from .cohere_textgen import CohereTextGenerator
import logging

logger = logging.getLogger(__name__)


def llm(provider: str = None, **kwargs):

    # load config
    if provider is None:
        # attempt to load config from environment variable LLMX_CONFIG_PATH
        config = load_config()
        if config:
            provider = config["model"]["provider"]
            kwargs = config["model"]["parameters"]
    if provider is None:
        logger.info("No provider specified. Defaulting to 'openai'.")
        provider = "openai"
    if provider.lower() == "openai" or provider.lower() == "default" or provider.lower(
    ) == "azureopenai" or provider.lower() == "azureoai":
        return OpenAITextGenerator(**kwargs)
    elif provider.lower() == "palm" or provider.lower() == "google":
        return PalmTextGenerator(provider=provider, **kwargs)
    elif provider.lower() == "cohere":
        return CohereTextGenerator(provider=provider, **kwargs)
    elif provider.lower() == "hf" or provider.lower() == "huggingface":
        try:
            import transformers
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
