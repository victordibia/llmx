from dataclasses import asdict
import os
import logging
from typing import Dict, Union
from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse, Message
from ...utils import (
    cache_request,
    gcp_request,
    gcp_genai_request,
    get_models_maxtoken_dict,
    num_tokens_from_messages,
    get_gcp_credentials,
)

logger = logging.getLogger("llmx")


class GeminiTextGenerator(TextGenerator):
    def __init__(
        self,
        api_key: str = os.environ.get("GEMINI_API_KEY", None),
        gemini_key_file: str = os.environ.get("GEMINI_SERVICE_ACCOUNT_KEY_FILE", None),
        project_id: str = os.environ.get("GEMINI_PROJECT_ID", None),
        project_location=os.environ.get("GEMINI_PROJECT_LOCATION", "us-central1"),
        provider: str = "gemini",
        model: str = None,
        models: Dict = None,
    ):
        super().__init__(provider=provider)

        if api_key is None and gemini_key_file is None:
            raise ValueError(
                "GEMINI_API_KEY or GEMINI_SERVICE_ACCOUNT_KEY_FILE  must be set."
            )
        if api_key:
            self.api_key = api_key
            self.credentials = None
            self.project_id = None
            self.project_location = None
        else:
            self.project_id = project_id
            self.project_location = project_location
            self.api_key = None
            self.credentials = get_gcp_credentials(gemini_key_file) if gemini_key_file else None

        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.model_name = model or "gemini-1.5-flash"

    def format_messages(self, messages):
        gemini_messages = []
        system_messages = ""
        for message in messages:
            if message["role"] == "system":
                system_messages += message["content"] + "\n"
            else:
                if not gemini_messages or (gemini_messages[-1] and gemini_messages[-1]["role"] != message["role"]):
                    gemini_message = {
                        "role": message["role"],
                        "parts": message["content"],
                    }
                    gemini_messages.append(gemini_message)
                else:
                    gemini_messages[-1]["content"] += "\n" + message["content"]

        if len(gemini_messages) > 2 and len(gemini_messages) % 2 == 0:
            print(len(gemini_messages))
            merged_content = (
                gemini_messages[-2]["content"] + "\n" + gemini_messages[-1]["content"]
            )
            gemini_messages[-2]["content"] = merged_content
            gemini_messages.pop()

        if len(gemini_messages) == 0:
            logger.info("No messages to send to GEMINI")

        return system_messages, gemini_messages

    def generate(
        self,
        messages: Union[list[dict], str],
        config: TextGenerationConfig = TextGenerationConfig(),
        **kwargs,
    ) -> TextGenerationResponse:
        use_cache = config.use_cache
        model = config.model or self.model_name

        system_messages, messages = self.format_messages(messages)
        self.model_name = model

        max_tokens = self.model_max_token_dict[model] if model in self.model_max_token_dict else 1024
        gemini_config = {
            "temperature": config.temperature,
            "max_output_tokens": config.max_tokens or max_tokens,
            "candidate_count": config.n,
            "top_p": config.top_p,
            "top_k": config.top_k,
        }

        api_url = ""
        if self.api_key:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateMessage?key={self.api_key}"

            gemini_payload = {
                "contents": messages,
                "parameters": gemini_config,
                "system_messages": system_messages,
            }

        else:
            api_url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.project_location}/publishers/google/models/{model}:predict"

            gemini_payload = {
                "contents": {"parts":messages["content"], "role":messages["author"]},
                "parameters": gemini_config,
                "system_messages": system_messages,
            }

        cache_key_params = {**gemini_payload, "model": model, "api_url": api_url}

        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)

        gemini_response = gcp_genai_request(
            url=api_url, body=gemini_payload, method="POST", credentials=self.credentials, api_key=self.api_key, model=self.model_name
        )

        candidates = gemini_response

        response_text = []
        for x in candidates:
            content = x.content.parts[0].text
            response_text.append(
              Message(
                  role="assistant" if x.content.role == "model" else x.content.role,
                  content=content.strip(),
              )
            )

        response = TextGenerationResponse(
            text=response_text,
            logprobs=[],
            config=gemini_config,
            usage={
                "total_tokens": num_tokens_from_messages(
                    response_text, model=self.model_name
                )
            },
            # Not passing gemini response due to parts in response structure.
            # This causes TextGenerationResponse.__post_init__() asdict() to fail.
            response=[],
        )

        cache_request(
            cache=self.cache, params=(cache_key_params), values=asdict(response)
        )
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)
