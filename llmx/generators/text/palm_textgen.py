from dataclasses import asdict
import os
from typing import Union
import google.generativeai as palm
from .base_textgen import BaseTextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, num_tokens_from_messages


class PalmTextGenerator(BaseTextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("PALM_API_KEY", None),
        provider: str = "google",
    ):
        super().__init__(provider=provider)

        if api_key is None:
            raise ValueError(
                "Palm API key is not set. Please set the PALM_API_KEY environment variable."
            )

        palm.configure(api_key=api_key)
        self.api_key = api_key

    def format_messages(self, messages):
        palm_messages = []
        system_messages = ""
        for message in messages:
            if message["role"] == "system":
                system_messages += message["content"] + "\n"
            else:
                palm_message = {
                    "author": message["role"],
                    "content": message["content"],
                }
                palm_messages.append(palm_message)
        return system_messages, palm_messages

    def generate(
            self, messages: Union[list[dict],
                                  str],
            config: TextGenerationConfig = TextGenerationConfig(),
            use_cache=True, **kwargs) -> TextGenerationResponse:
        model = config.model or "models/chat-bison-001"
        self.model_name = model
        system_messages, messages = self.format_messages(messages)
        palm_config = {
            "model": model,
            "context": system_messages,
            "examples": None,
            "candidate_count": max(1, min(8, config.n)),  # 1 <= n <= 8
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "messages": messages,
        }
        # print("*********", config)
        cache_key_params = palm_config | {"messages": messages}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        try:
            palm_response = palm.chat(**palm_config)
        except Exception as e:
            raise ValueError(f"Error generating text: {e}")

        response_text = [
            Message(
                role="assistant" if x["author"] == "1" else x["author"],
                content=x["content"],
            )
            for x in palm_response.candidates
        ]

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=palm_config,
            usage={
                "total_tokens": num_tokens_from_messages(
                    response_text, model=palm_config["model"]
                )
            },
        )

        cache_request(cache=self.cache, params=(cache_key_params), values=asdict(response))
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
