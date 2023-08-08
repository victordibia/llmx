import sys
from .generators.text.textgen import llm
from .datamodel import TextGenerationConfig, TextGenerationResponse, Message
from .generators.text.base_textgen import TextGenerator


if sys.version_info < (3, 10):
    raise RuntimeError("llmx requires Python 3.10+")
