from typing import Union, List, Dict
import os
import anthropic
from dataclasses import asdict

from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages


class AnthropicTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = None,
        provider: str = "anthropic",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", None)
        if api_key is None:
            raise ValueError(
                "Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable."
            )
        self.client = anthropic.Anthropic(
            api_key=api_key,
            default_headers={"anthropic-beta": "max-tokens-3-5-sonnet-2024-07-15"},
        )
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "claude-3-5-sonnet-20240620"

    def format_messages(self, messages):
        formatted_messages = []
        for message in messages:
            formatted_message = {"role": message["role"], "content": message["content"]}
            formatted_messages.append(formatted_message)
        return formatted_messages


    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(model, 8192) - prompt_tokens - 10, 200
        )

        # Process messages
        system_message = None
        other_messages = []
        for message in messages:
            message["content"] = message["content"].strip()
            if message["role"] == "system":
                if system_message is None:
                    system_message = message["content"]
                else:
                    # If multiple system messages, concatenate them
                    system_message += "\n" + message["content"]
            else:
                other_messages.append(message)

        if not other_messages:
            raise ValueError("At least one message is required")

        # Check if inversion is needed
        needs_inversion = other_messages[0]["role"] == "assistant"
        if needs_inversion:
            other_messages = self.invert_messages(other_messages)

        anthropic_config = {
            "model": model,
            "max_tokens": config.max_tokens or max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "messages": other_messages,
        }

        if system_message:
            anthropic_config["system"] = system_message

        self.model_name = model
        cache_key_params = anthropic_config.copy()
        cache_key_params["messages"] = messages  # Keep original messages for caching

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)
        anthropic_response = self.client.messages.create(**anthropic_config)

        response_content = anthropic_response.content[0].text

        # Always strip "Human: " prefix, regardless of inversion
        if response_content.startswith("Human: "):
            response_content = response_content[7:]

        response = TextGenerationResponse(
            text=[Message(role="assistant", content=response_content)],
            logprobs=[],
            config=anthropic_config,
            usage={
                "prompt_tokens": anthropic_response.usage.input_tokens,
                "completion_tokens": anthropic_response.usage.output_tokens,
                "total_tokens": anthropic_response.usage.input_tokens
                + anthropic_response.usage.output_tokens,
            },
            response=anthropic_response,
        )

        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def invert_messages(self, messages):
        inverted = []
        for i, message in enumerate(messages):
            if i % 2 == 0:
                inverted.append({"role": "user", "content": message["content"]})
            else:
                inverted.append({"role": "assistant", "content": message["content"]})
        return inverted
    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
