import cohere
from typing import Union, List
import numpy as np
import os
from .text_embedding import TextEmbedding
from ..utils import cache_request


class CohereTextEmbedding(TextEmbedding):
    """
    Text embedding using Cohere models.

    Attributes:
        model (str): Name of the Cohere embedding model to use.
        api_key (str): API key for accessing the Cohere service. Will use the
                       COHERE_API_KEY environment variable if not provided.
    """

    def __init__(
        self,
        model: str = "embed-english-v2.0",
        api_key: str = os.environ.get("COHERE_API_KEY", None),
    ):
        super().__init__(model)

        if api_key is None:
            raise ValueError(
                "Cohere API key is not set. Please set the COHERE_API_KEY environment variable."
            )

        self.client = cohere.Client(api_key)
        self.model = model
        self._size = self.get_embedding_size_from_model()

    def get_embedding_size_from_model(self) -> int:
        """
        Get the embedding size for the current model.

        Returns:
            int: Size of the embeddings returned by the model.
        """
        size_map = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768,
        }
        size = size_map.get(self.model)
        if not size:
            raise ValueError("Invalid model name. Please provide a valid Cohere embed model.")
        return size

    def embed(
        self, text: Union[str, List[str]], use_cache: bool = True, **kwargs
    ) -> np.ndarray:
        """
        Compute embedding for the given text using the Cohere model.

        Args:
            text (Union[str, List[str]]): The input text(s) to compute embeddings for.
            use_cache (bool, optional): Whether to use cached embeddings if available.
                                         Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The computed embeddings for the input text.
        """
        # if text is numpy.ndarray, convert to list
        if isinstance(text, np.ndarray):
            text = text.tolist()

        cache_params = dict(text=text, model=self.model, **kwargs)
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_params)
            if response:
                return np.array(response).astype(np.float32)

        co_response = self.client.embed(model=self.model, texts=text)
        embeddings = co_response.embeddings

        cache_request(cache=self.cache, params=cache_params, values=embeddings)
        return np.array(embeddings).astype(np.float32)
