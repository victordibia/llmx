from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, get_models_contextwindow_dict, num_tokens_from_messages
import os
import copy
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
        self.model_context_window_dict = get_models_contextwindow_dict(models)

    def get_oai_response(self, use_cache, config, max_tokens, messages, num_choices=1):
        oai_config = {
            "model": self.model_name,
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "n": num_choices,
            "messages": messages,
        }

        cache_key_params = (oai_config) | {"messages": messages} | {"continue_status": config.continue_until_finish}
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                is_cached = True
                return is_cached, TextGenerationResponse(**response)

        is_cached = False
        oai_response = self.client.chat.completions.create(**oai_config)
        return is_cached, oai_response

    def generate(
        self,
        messages: Union[List[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        self.model_name = model

        prompt_tokens = num_tokens_from_messages(messages)
        model_max_completion_tokens = self.model_max_token_dict.get(model, 4096)
        model_context_window = self.model_context_window_dict.get(model, 4096)
        
        # max_tokens = max(
        #     self.model_max_token_dict.get(
        #         model, 4096) - prompt_tokens - 10, 200
        # )
        max_tokens = min([
            model_context_window - prompt_tokens - 10,
            model_max_completion_tokens,
            config.max_tokens if config.max_tokens else 1000000
            ])

        is_cached, main_oai_response = self.get_oai_response(
            use_cache,
            config,
            max_tokens,
            messages,
            num_choices=config.n)
        oai_response = main_oai_response

        if is_cached:
            response = oai_response
            return response

        # for nth_choice in range(config.n):
        continuation_messages = [oai_response.choices[0].message.content]
        while config.continue_until_finish and oai_response.choices[0].finish_reason == "length":

            print("Continuing Generation! ")
            new_messages = [
                {"role": "assistant", "content": oai_response.choices[0].message.content},
                {"role": "user", "content": config.continue_prompt}
            ]
            extended_messages = messages + new_messages
            prompt_tokens = num_tokens_from_messages(extended_messages)
            max_tokens = min([
                model_context_window - prompt_tokens - 10,
                model_max_completion_tokens,
                config.max_tokens if config.max_tokens else 1000000
                ])
            _, oai_response = self.get_oai_response(
                use_cache,
                config,
                max_tokens,
                extended_messages,
                num_choices=1)

            continuation_messages.append(oai_response.choices[0].message.content)

        main_oai_response.choices[0].message.content = "".join(continuation_messages)

        oai_config = {
            "model": self.model_name,
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "top_p": config.top_p,
            "frequency_penalty": config.frequency_penalty,
            "presence_penalty": config.presence_penalty,
            "n": config.n,
            "messages": messages,
        }

        response = TextGenerationResponse(
            text=[Message(**x.message.model_dump())
                  for x in main_oai_response.choices],
            logprobs=[],
            config=oai_config,
            usage=dict(oai_response.usage),
        )

        cache_key_params = (oai_config) | {"messages": messages} | {"continue_status": config.continue_until_finish}
        # if use_cache:
        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
