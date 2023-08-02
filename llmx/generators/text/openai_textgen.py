from typing import Union
from .base_textgen import BaseTextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, num_tokens_from_messages
import os
import openai
from dataclasses import asdict

context_lengths = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-0613": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0613": 32768,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-3.5-turbo-0613": 4096,
    "gpt-3.5-turbo-16k-0613": 16384,
}


class OpenAITextGenerator(BaseTextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("OPENAI_API_KEY", None),
        provider: str = "openai",
        organization: str = None,
    ):
        super().__init__(provider=provider)
        if api_key is None:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        self.api_key = api_key

    def generate(
        self,  messages: Union[list[dict], str], config: TextGenerationConfig = TextGenerationConfig(), use_cache=True, **kwargs
    ) -> TextGenerationResponse:

        model = config.model  or "gpt-3.5-turbo-0301"
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(context_lengths.get(model, 4096) - prompt_tokens - 10, 200)

        oai_config = {
            "model": model,
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "n": config.n,
            "messages": messages,
        }
         
        self.model_name = model
        cache_key_params = (oai_config) | {"messages": messages} 
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        
        oai_response = openai.ChatCompletion.create(**oai_config)

        response = TextGenerationResponse(
            text=[dict(x.message) for x in oai_response.choices],
            logprobs=[],
            config=oai_config,
            usage=dict(oai_response.usage),
        )
        # if use_cache:
        cache_request(cache=self.cache, params=cache_key_params, values=asdict(response))
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
