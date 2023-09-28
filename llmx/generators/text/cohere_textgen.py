from typing import Dict, Union
import os
import cohere
from dataclasses import asdict

from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
from ..text.providers import providers


class CohereTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = None,
        provider: str = "cohere",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        api_key = api_key or os.environ.get("COHERE_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Cohere API key is not set. Please set the COHERE_API_KEY environment variable."
            )
        self.client = cohere.Client(api_key)
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "command"

    def format_messages(self, messages):
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += message["content"] + "\n"
            else:
                prompt += message["role"] + ": " + message["content"] + "\n"

        return prompt

    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        messages = self.format_messages(messages)
        self.model_name = config.model or self.model_name

        max_tokens = (
            self.model_max_token_dict[self.model_name]
            if config.model in self.model_max_token_dict else 1024)

        cohere_config = {
            "model": self.model_name,
            "prompt": messages,
            "max_tokens": config.max_tokens or max_tokens,
            "temperature": config.temperature,
            "k": config.top_k,
            "p": config.top_p,
            "num_generations": config.n,
            "stop_sequences": config.stop,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
        }

        # print("calling cohere ***************", config)

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
            usage={},
            response=co_response,
        )

        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
