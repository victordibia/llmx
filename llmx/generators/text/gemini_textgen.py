from typing import Union, List, Dict
import os
from google import genai
from google.genai import types
from dataclasses import asdict

from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages


class GeminiTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = None,
        provider: str = "gemini",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        api_key = api_key or os.environ.get(
            "GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", None)
        )
        if api_key is None:
            raise ValueError(
                "Gemini API key is not set. Please set the GEMINI_API_KEY environment variable."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "gemini-1.5-flash"

    def format_messages(self, messages):
        system_message = None
        formatted_messages = []
        for message in messages:
            content = message["content"].strip()
            if message["role"] == "system":
                system_message = content if system_message is None else system_message + "\n" + content
            else:
                role = "model" if message["role"] == "assistant" else "user"
                formatted_messages.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=content)])
                )
        return system_message, formatted_messages

    def generate(
        self,
        messages: Union[List[Dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name
        self.model_name = model

        system_message, formatted_messages = self.format_messages(messages)
        if not formatted_messages:
            raise ValueError("At least one message is required")

        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            self.model_max_token_dict.get(model, 8192) - prompt_tokens - 10, 200
        )

        stop_sequences = config.stop if isinstance(config.stop, list) else (
            [config.stop] if config.stop else None
        )
        max_output_tokens = config.max_tokens or max_tokens

        cache_key_params = {
            "model": model,
            "messages": messages,
            "system_message": system_message,
            "generation_config": {
                "candidate_count": config.n,
                "max_output_tokens": max_output_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "stop_sequences": stop_sequences,
            },
        }

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        generation_config = types.GenerateContentConfig(
            system_instruction=system_message,
            candidate_count=config.n,
            max_output_tokens=max_output_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            stop_sequences=stop_sequences,
        )

        gemini_response = self.client.models.generate_content(
            model=model, contents=formatted_messages, config=generation_config
        )

        response_text = [
            Message(role="assistant", content=candidate.content.parts[0].text)
            for candidate in gemini_response.candidates
        ]

        usage = {}
        if gemini_response.usage_metadata:
            usage = {
                "prompt_tokens": gemini_response.usage_metadata.prompt_token_count,
                "completion_tokens": gemini_response.usage_metadata.candidates_token_count,
                "total_tokens": gemini_response.usage_metadata.total_token_count,
            }

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=cache_key_params["generation_config"],
            usage=usage,
            response=gemini_response,
        )

        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
