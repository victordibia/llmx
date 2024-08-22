import os
import numpy as np
from typing import Union, List
from diskcache import Cache
from abc import ABC, abstractmethod
from ..utils import get_user_cache_dir
from ..version import APP_NAME


class TextEmbedding(ABC):
    """Interface for computing Text Embeddings."""

    def __init__(self, model_name: str, **kwargs):
        """
        Initialize the text embedding model.

        :param model_name: str, the name of the model
        :param cache_dir: str, optional, the path to the directory where cache files are stored
        """
        self.model_name = model_name
        self._size = None

        app_name = APP_NAME
        cache_dir_default = get_user_cache_dir(app_name)
        cache_dir_based_on_model = os.path.join(cache_dir_default, model_name)
        self.cache_dir = kwargs.get("cache_dir", cache_dir_based_on_model)
        self.cache = Cache(self.cache_dir)

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Compute embedding for text.

        :param text: Union[str, List[str]], the input text or a list of texts
        :return: np.ndarray, the computed embedding(s)
        """
        pass

    @property
    def size(self):
        """Return the embedding size."""
        return self._size
