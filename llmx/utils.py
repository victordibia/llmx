from dataclasses import asdict
import sys
import logging
import json
from typing import Any, Union, Dict
import tiktoken
from diskcache import Cache
import hashlib
import os
import platform
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account
import requests

logger = logging.getLogger(__name__)


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


def get_gcp_credentials(service_account_key_file: str = None, scopes: list[str] = [
        'https://www.googleapis.com/auth/cloud-platform']):
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
            service_account_key_file, scopes=scopes)
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        return credentials


def gcp_request(
    url: str,
    method: str = "POST",
    body: dict = None,
    headers: dict = None,
    credentials: google.auth.credentials.Credentials = None,
    **kwargs,
):
    if credentials is None:
        credentials = get_gcp_credentials()
    auth_req = google.auth.transport.requests.Request()
    if credentials.expired:
        credentials.refresh(auth_req)
    headers = headers or {}
    headers["Authorization"] = f"Bearer {credentials.token}"
    headers["Content-Type"] = "application/json"

    response = requests.request(
        method=method, url=url, json=body, headers=headers, **kwargs
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
