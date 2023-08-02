from typing import Union
from .base_textgen import BaseTextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, num_tokens_from_messages
import os
import cohere
from dataclasses import asdict


class CohereTextGenerator(BaseTextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("COHERE_API_KEY", None),
        provider: str = "cohere",
        organization: str = None,
    ):
        super().__init__(provider=provider)
        if api_key is None:
            raise ValueError(
                "Cohere API key is not set. Please set the COHERE_API_KEY environment variable."
            )
        self.client = cohere.Client(api_key)

    def format_messages(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += message["content"] + "\n"
            else:
                prompt += message["role"] + ": " + message["content"] + "\n"

        return prompt

    def generate(
        self,  messages: Union[list[dict], str], config: TextGenerationConfig = TextGenerationConfig(), use_cache=True, **kwargs
    ) -> TextGenerationResponse:
         
        messages = self.format_messages(messages)
        self.model_name = config.model

        cohere_config = {
            "model": config.model or "command",
            "prompt": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "k": config.top_k,
            "p": config.top_p,
            "num_generations": config.n,
            "stop_sequences": config.stop,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }
 
        cache_key_params = cohere_config | {"messages": messages}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        co_response = self.client.generate(**cohere_config)

        response_text = [
            Message(
                role="system",
                content=x.text,
            )
            for x in co_response.generations
        ]

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],  # You may need to extract log probabilities from the response if needed
            config=cohere_config,
            usage={},  # You may need to extract usage metrics from the response if needed
        )

        cache_request(cache=self.cache, params=cache_key_params, values=asdict(response))
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
