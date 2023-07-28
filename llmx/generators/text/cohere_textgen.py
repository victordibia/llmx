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
        self, config: TextGenerationConfig, use_cache=True, **kwargs
    ) -> TextGenerationResponse:
        config.model = config.model or "command"
        config.messages = self.format_messages(config.messages)

        self.model_name = config.model

        cohere_config = {
            "model": config.model,
            "prompt": config.messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "k": config.top_k,
            "p": config.top_p,
            "num_generations": config.n,
            "stop_sequences": config.stop,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }

        print(cohere_config)

        if use_cache:
            response = cache_request(cache=self.cache, params=(cohere_config))
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
            config=config,
            usage={},  # You may need to extract usage metrics from the response if needed
        )

        if use_cache:
            cache_request(
                cache=self.cache, params=asdict(config), values=(cohere_config)
            )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
