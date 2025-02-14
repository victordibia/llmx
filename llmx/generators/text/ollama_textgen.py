import os
import requests
from dataclasses import asdict
from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages


class OllamaTextGenerator(TextGenerator):
    def __init__(
            self, 
            model: str = None, 
            base_url: str = None, 
            models: Dict = None, 
            **kwargs
        ):
        super().__init__(provider="ollama")
        self.model_name = model or "gemma2"
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
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
            self.model_max_token_dict.get(model, 4096) - prompt_tokens - 10, 200
        )

        ollama_payload = {
            "model": model,
            "prompt": messages if isinstance(messages, str) else messages[-1]["content"],  # Assume last message
            "temperature": config.temperature,
            "max_tokens": max_tokens,
            "top_p": config.top_p,
            "stream": False,
        }

        self.model_name = model
        cache_key_params = (ollama_payload) | {"messages": messages}
        
        if use_cache:
            response = cache_request(cache=self.cache, params=cache_key_params)
            if response:
                return TextGenerationResponse(**response)
            
        url = f"{self.base_url}/api/generate"
        response = requests.post(url, json=ollama_payload)
        oai_response = response.json()
        
        if response.status_code != 200:
            raise ValueError(f"Ollama API error: {response.status_code}, {response.text}")

        total_tokens = oai_response.get("eval_count", 0) 
        prompt_tokens = oai_response.get("prompt_eval_count", 0) 

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens,
        }

        response_obj = TextGenerationResponse(
            text=[Message(content= oai_response["response"], role="assistant")],
            logprobs=[],
            config=ollama_payload,
            usage=usage,
        )

        cache_request(
            cache=self.cache, params=cache_key_params, values=asdict(response_obj)
        )

        return response_obj

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)