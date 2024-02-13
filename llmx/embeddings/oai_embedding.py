import os
import numpy as np
import openai
from typing import Union, List
from .text_embedding import TextEmbedding
from ..utils import cache_request


class OpenAITextEmbedding(TextEmbedding):
    """Text embedding using OpenAI models."""

    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_base: str = None,
        api_version: str = None,
        api_type: str = None,
        api_key: str = os.environ.get("OPENAI_API_KEY", None),
    ):
        super().__init__(model)

        if api_key is None:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        self.api_key = api_key
        openai.api_key = self.api_key
        openai.api_base = api_base or openai.api_base
        openai.api_version = api_version or openai.api_version
        openai.api_type = api_type or openai.api_type
        self.model = model
        self._size = 1536

    def embed(
        self, text: Union[str, List[str]], use_cache: bool = True, **kwargs
    ) -> np.ndarray:
        """Compute embedding for text."""
        # if text is numpy.ndarray, convert to list
        if isinstance(text, np.ndarray):
            text = text.tolist()

        cache_params = dict(text=text, model=self.model, **kwargs)
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_params)
            if response:
                return np.array(response).astype(np.float32)
        response = openai.Embedding.create(input=text, model=self.model)
        embeddings = [x["embedding"] for x in response["data"]]

        cache_request(cache=self.cache, params=cache_params, values=embeddings)
        return np.array(embeddings).astype(np.float32)
