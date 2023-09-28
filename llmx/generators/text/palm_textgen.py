from dataclasses import asdict
import os
import logging
from typing import Dict, Union
from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import (
    cache_request,
    gcp_request,
    get_models_maxtoken_dict,
    num_tokens_from_messages,
    get_gcp_credentials,
)

logger = logging.getLogger("llmx")


class PalmTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("PALM_API_KEY", None),
        palm_key_file: str = os.environ.get("PALM_SERVICE_ACCOUNT_KEY_FILE", None),
        project_id: str = os.environ.get("PALM_PROJECT_ID", None),
        project_location=os.environ.get("PALM_PROJECT_LOCATION", "us-central1"),
        provider: str = "palm",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)

        if api_key is None and palm_key_file is None:
            raise ValueError(
                "PALM_API_KEY or PALM_SERVICE_ACCOUNT_KEY_FILE  must be set."
            )
        if api_key:
            self.api_key = api_key
            self.credentials = None
            self.project_id = None
            self.project_location = None
        else:
            self.project_id = project_id
            self.project_location = project_location
            self.api_key = None
            self.credentials = get_gcp_credentials(palm_key_file) if palm_key_file else None

        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "chat-bison"

    def format_messages(self, messages):
        palm_messages = []
        system_messages = ""
        for message in messages:
            if message["role"] == "system":
                system_messages += message["content"] + "\n"
            else:
                if not palm_messages or palm_messages[-1]["author"] != message["role"]:
                    palm_message = {
                        "author": message["role"],
                        "content": message["content"],
                    }
                    palm_messages.append(palm_message)
                else:
                    palm_messages[-1]["content"] += "\n" + message["content"]

        if palm_messages and len(palm_messages) % 2 == 0:
            merged_content = (
                palm_messages[-2]["content"] + "\n" + palm_messages[-1]["content"]
            )
            palm_messages[-2]["content"] = merged_content
            palm_messages.pop()

        if len(palm_messages) == 0:
            logger.info("No messages to send to PALM")

        return system_messages, palm_messages

    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name

        system_messages, messages = self.format_messages(messages)
        self.model_name = model

        max_tokens = self.model_max_token_dict[model] if model in self.model_max_token_dict else 1024
        palm_config = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens or max_tokens,
            "candidateCount": config.n,
        }

        api_url = ""
        if self.api_key:
            api_url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateMessage?key={self.api_key}"

            palm_payload = {
                "prompt": {"messages": messages, "context": system_messages},
                "temperature": config.temperature,
                "candidateCount": config.n,
                "topP": config.top_p,
                "topK": config.top_k,
            }

        else:
            api_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.project_location}/publishers/google/models/{model}:predict"

            palm_payload = {
                "instances": [
                    {
                        "messages": messages,
                        "context": system_messages,
                        "examples": [],
                    }
                ],
                "parameters": palm_config,
            }

        cache_key_params = {**palm_payload, "model": model, "api_url": api_url}

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        palm_response = gcp_request(
            url=api_url, body=palm_payload, method="POST", credentials=self.credentials
        )

        candidates = palm_response["candidates"] if self.api_key else palm_response["predictions"][
            0]["candidates"]

        response_text = [
            Message(
                role="assistant" if x["author"] == "1" else x["author"],
                content=x["content"],
            )
            for x in candidates
        ]

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=palm_config,
            usage={
                "total_tokens": num_tokens_from_messages(
                    response_text, model=self.model_name
                )
            },
            response=palm_response,
        )

        cache_request(
            cache=self.cache, params=(cache_key_params), values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
