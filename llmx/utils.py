from dataclasses import asdict
import logging
import json
from typing import Any, List, Tuple, Union, Dict
import tiktoken
from diskcache import Cache
import hashlib
import os
import platform
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests
import yaml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

logger = logging.getLogger("llmx")


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if (
        model == "gpt-3.5-turbo-0301" or True
    ):  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            if not isinstance(message, dict):
                message = asdict(message)

            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )

            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens


def cache_request(cache: Cache, params: dict, values: Union[Dict, None] = None) -> Any:
    # Generate a unique key for the request

    key = hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    # Check if the request is cached
    if key in cache and values is None:
        # print("retrieving from cache")
        return cache[key]

    # Cache the provided values and return them
    if values:
        # print("saving to cache")
        cache[key] = values
    return values


def get_user_cache_dir(app_name: str) -> str:
    system = platform.system()
    if system == "Windows":
        cache_path = os.path.join(os.getenv("LOCALAPPDATA"), app_name, "Cache")
    elif system == "Darwin":
        cache_path = os.path.join(os.path.expanduser("~/Library/Caches"), app_name)
    else:  # Linux and other UNIX-like systems
        cache_path = os.path.join(
            os.getenv("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), app_name
        )
    os.makedirs(cache_path, exist_ok=True)
    return cache_path


def get_gcp_credentials(
    service_account_key_file: str = None,
    scopes: list[str] = ["https://www.googleapis.com/auth/cloud-platform"],
):
    try:
        # Attempt to use Application Default Credentials
        credentials, project_id = google.auth.default(scopes=scopes)
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials
    except google.auth.exceptions.DefaultCredentialsError:
        # Fall back to using service account key
        if service_account_key_file is None:
            raise ValueError(
                "Service account key file is not set. Please set the PALM_SERVICE_ACCOUNT_KEY_FILE environment variable."
            )
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key_file, scopes=scopes
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials


def gcp_request(
    url: str,
    method: str = "POST",
    body: dict = None,
    headers: dict = None,
    credentials: google.auth.credentials.Credentials = None,
    request_timeout: int = 60,
    **kwargs,
):
    headers = headers or {}

    if "key" not in url:
        if credentials is None:
            credentials = get_gcp_credentials()
        auth_req = google.auth.transport.requests.Request()
        if credentials.expired:
            credentials.refresh(auth_req)
        headers["Authorization"] = f"Bearer {credentials.token}"
    headers["Content-Type"] = "application/json"

    response = requests.request(
        method=method,
        url=url,
        json=body,
        headers=headers,
        timeout=request_timeout,
        **kwargs,
    )

    if response.status_code not in range(200, 300):
        try:
            error_message = response.json().get("error", {}).get("message", "")
        except json.JSONDecodeError:
            error_message = response.content
        raise Exception(
            f"Request failed with status code {response.status_code}: {error_message}"
        )

    return response.json()


def load_config():
    try:
        config_path = os.environ.get("LLMX_CONFIG_PATH", None)
        if config_path is None or os.path.exists(config_path) is False:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs/config.default.yml"
            )
            logger.info(
                "Info: LLMX_CONFIG_PATH environment variable is not set to a valid config file. Using default config file at '%s'.",
                config_path,
            )
        if config_path is not None:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    logger.info("Loaded config from '%s'.", config_path)
                    return config
            except FileNotFoundError as file_not_found:
                logger.info(
                    "Error: Config file not found at '%s'. Please check the LLMX_CONFIG_PATH environment variable. %s",
                    config_path,
                    str(file_not_found),
                )
            except IOError as io_error:
                logger.info(
                    "Error: Could not read the config file at '%s'. %s",
                    config_path,
                    str(io_error),
                )
            except yaml.YAMLError as yaml_error:
                logger.info(
                    "Error: Malformed YAML in config file at '%s'. %s",
                    config_path,
                    str(yaml_error),
                )
        else:
            logger.info(
                "Info:LLMX_CONFIG_PATH environment variable is not set. Please set it to the path of your config file to setup your default model."
            )
    except Exception as error:
        logger.info("Error: An unexpected error occurred: %s", str(error))

    return None


def get_models_maxtoken_dict(models_list):
    models_dict = {}
    for model in models_list:
        if "model" in model and "parameters" in model["model"]:
            details = model["model"]["parameters"]
            models_dict[details["model"]] = model["max_tokens"]
    return models_dict


def reduce_dimensions(
    embeddings: np.ndarray, method: str = "pca", n_components: int = 2, **kwargs
) -> np.ndarray:
    """
    Reduce the dimensionality of the embeddings using the specified method.

    :param embeddings: The input embeddings as a NumPy array of shape (n_samples, n_features).
    :param method: The dimensionality reduction method to use ('pca' or 'tsne').
    :param n_components: The number of dimensions to reduce the embeddings to.
    :param kwargs: Additional keyword arguments specific to the chosen method.
    :return: The reduced embeddings as a NumPy array of shape (n_samples, n_components).
    """
    if method.lower() == "pca":
        reducer = PCA(n_components=n_components, **kwargs)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    reduced_embeddings = reducer.fit_transform(embeddings)
    return reduced_embeddings


def cluster_embeddings(
    reduced_embeddings: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 3,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster the reduced embeddings using the specified method and return the centroids.

    :param reduced_embeddings: The reduced embeddings as a NumPy array of shape (n_samples, n_components).
    :param method: The clustering method to use ('kmeans').
    :param n_clusters: The number of clusters for KMeans clustering.
    :param kwargs: Additional keyword arguments specific to the chosen method.
    :return: A tuple containing the cluster labels (shape (n_samples,)) and the centroids (shape (n_clusters, n_components)).
    """
    if method.lower() == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
        clusters = clusterer.fit_predict(reduced_embeddings)
        centroids = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    return clusters, centroids


def closest_samples_to_centroids(
    embeddings: np.ndarray, clusters: np.ndarray, centroids: np.ndarray, n: int = 1
) -> List[List[int]]:
    """
    Find the indices of n samples closest to the centroid of each cluster.

    :param embeddings: The embeddings (reduced or not) as a NumPy array of shape (n_samples, n_components).
    :param clusters: The cluster labels as a NumPy array of shape (n_samples,).
    :param centroids: The centroids of the clusters as a NumPy array of shape (n_clusters, n_components).
    :param n: The number of samples to find for each cluster.
    :return: A list of lists containing the indices of the n closest samples for each cluster.
    """
    cluster_samples = [[] for _ in range(centroids.shape[0])]

    for i, cluster in enumerate(clusters):
        cluster_samples[cluster].append(i)

    closest_samples = []
    for i, centroid in enumerate(centroids):
        distances = np.linalg.norm(embeddings[cluster_samples[i]] - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n]
        closest_samples.append([cluster_samples[i][index] for index in closest_indices])

    return closest_samples
