from dataclasses import asdict
import os
from typing import Union
from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import (
    cache_request,
    gcp_request,
    num_tokens_from_messages,
    get_gcp_credentials,
)
from ..text.providers import providers


class PalmTextGenerator(TextGenerator):
    def __init__(
        self,
        palm_key_file: str = os.environ.get("PALM_SERVICE_ACCOUNT_KEY_FILE", None),
        project_id: str = os.environ.get("PALM_PROJECT_ID", None),
        project_location=os.environ.get("PALM_PROJECT_LOCATION", "us-central1"),
        provider: str = "google",
    ):
        super().__init__(provider=provider)

        self.project_id = project_id
        self.project_location = project_location
        self.credentials = get_gcp_credentials(palm_key_file)
        self.model_list = providers[provider]["models"] if provider in providers else {}

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

        return system_messages, palm_messages

    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or "codechat-bison"
        system_messages, messages = self.format_messages(messages)
        self.model_name = model

        # print("*********", messages)

        api_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.project_location}/publishers/google/models/{model}:predict"

        max_tokens = self.model_list[config.model] if model in self.model_list else 1024
        palm_config = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_tokens or max_tokens,
            "candidateCount": config.n,
        }
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
        # print("*********", use_cache, palm_payload)

        cache_key_params = {**palm_payload, "model": model}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        palm_response = gcp_request(
            url=api_url, body=palm_payload, method="POST", credentials=self.credentials
        )

        response_text = [
            Message(
                role="assistant" if x["author"] == "1" else x["author"],
                content=x["content"],
            )
            for x in palm_response["predictions"][0]["candidates"]
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
