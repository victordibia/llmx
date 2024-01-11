from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import os
import openai
from dataclasses import asdict


class OpenAITextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("OPENAI_API_KEY", None),
        provider: str = "openai",
        organization: str = None,
        api_type: str = None,
        api_base: str = None,
        api_version: str = None,
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        api_key = api_key or os.environ.get("OPENAI_API_KEY", None)

        if api_key is None:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        if api_version:
            openai.api_version = api_version
        if api_type:
            openai.api_type = api_type

        self.model_name = model or "gpt-3.5-turbo"

        self.model_max_token_dict = get_models_maxtoken_dict(models)
        # print("context lengths", self.model_max_token_dict)

    def generate(
        self,
        messages: Union[List[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(self.model_max_token_dict.get(model, 4096) - prompt_tokens - 10, 200)

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

        if openai.api_type and openai.api_type == "azure":
            oai_config["engine"] = model

        self.model_name = model
        cache_key_params = (oai_config) | {"messages": messages}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        oai_response = openai.ChatCompletion.create(**oai_config)

        response = TextGenerationResponse(
            text=[Message(**x.message) for x in oai_response.choices],
            logprobs=[],
            config=oai_config,
            usage=dict(oai_response.usage),
        )
        # if use_cache:
        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
