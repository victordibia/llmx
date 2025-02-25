import ollama
import os
from typing import Union, List, Dict
from .base_textgen import TextGenerator
from ...datamodel import Message, TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict, num_tokens_from_messages
from dataclasses import asdict


class OllamaTextGenerator(TextGenerator):
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "llama3.1",
        models: Dict = None,
    ):
        super().__init__(provider=provider)
        self.model_name = model
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

        
        if isinstance(messages, list):
            prompt = "\n".join([msg["content"] for msg in messages])
        else:
            prompt = messages

        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            generated_text = response.message.content  

            response_obj = TextGenerationResponse(
                text=[Message(role="assistant", content=generated_text)],
                logprobs=[],
                config={"model": model, "max_tokens": max_tokens},
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(generated_text.split()),
                    "total_tokens": prompt_tokens + len(generated_text.split()),
                },
            )

            if use_cache:
                cache_request(cache=self.cache, params=(prompt, model), values=asdict(response_obj))

            return response_obj

        except Exception as e:
            return TextGenerationResponse(text=[Message(role="error", content=f"⚠️ Ollama Error: {str(e)}")])

    def count_tokens(self, text) -> int:
        return len(text.split())  #Approximate token count for simplicity