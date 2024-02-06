from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
import os
from openai import AzureOpenAI, OpenAI
from dataclasses import asdict


class OpenAITextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("OPENAI_API_KEY", None),
        provider: str = "openai",
        organization: str = None,
        api_type: str = None,
        api_version: str = None,
        azure_endpoint: str = None,
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", None)

        if self.api_key is None:
            raise ValueError(
                "OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
            )

        self.client_args = {
            "api_key": self.api_key,
            "organization": organization,
            "api_version": api_version,
            "azure_endpoint": azure_endpoint,
        }
        # remove keys with None values
        self.client_args = {k: v for k,
                            v in self.client_args.items() if v is not None}

        if api_type:
            if api_type == "azure":
                self.client = AzureOpenAI(**self.client_args)
            else:
                raise ValueError(f"Unknown api_type: {api_type}")
        else:
            self.client = OpenAI(**self.client_args)

        self.model_name = model or "gpt-3.5-turbo"
        self.model_max_token_dict = get_models_maxtoken_dict(models)

    def generate(
        self,
        messages: Union[List[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(
                model, 4096) - prompt_tokens - 10, 200
        )

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

        oai_response = self.client.chat.completions.create(**oai_config)

        response = TextGenerationResponse(
            text=[Message(**x.message.model_dump())
                  for x in oai_response.choices],
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
