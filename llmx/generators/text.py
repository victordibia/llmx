
from dataclasses import asdict, dataclass
import os
from typing import Dict, List
import openai


from diskcache import Cache
from ..utils import cache_request, num_tokens_from_messages
from ..datamodel import TextGenerationConfig, TextGenerationResponse


class TextGenerator(object):
    def __init__(self, model_type: str = "openai", **kwargs):
        self.model_type = model_type
        self.model_name = kwargs.get("model_name", "gpt-3.5-turbo")
        self.cache_dir = kwargs.get(
            "cache_dir", os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)), 'cache'))
        self.cache = Cache(self.cache_dir)

    def generate(self, config: TextGenerationConfig, use_cache=True,
                 **kwargs) -> TextGenerationResponse:
        pass

    def count_tokens(self, text) -> int:
        pass


class OpenAITextGenerator(TextGenerator):
    def __init__(self, api_key: str = os.environ.get("OPENAI_API_KEY", None),
                 model_type: str = "openai",
                 organization: str = None,
                 ):
        super().__init__(model_type=model_type)
        if api_key is None:
            raise ValueError("OpenAI API key is not set")
        openai.api_key = api_key
        if organization:
            openai.organization = organization
        self.api_key = api_key
        self.max_context_tokens = 4024

    def generate(self, config: TextGenerationConfig, use_cache=True,
                 **kwargs) -> TextGenerationResponse:
        print("using cache: ", use_cache)
        self.model_name = config.model
        if use_cache:
            response = cache_request(cache=self.cache, params=asdict(config))
            if response:
                return TextGenerationResponse(**response)

        prompt_tokens = num_tokens_from_messages(config.messages)
        max_tokens = max(self.max_context_tokens - prompt_tokens, 200)

        oai_response = openai.ChatCompletion.create(
            model=config.model,
            messages=config.messages,
            n=config.n,
            temperature=config.temperature,
            max_tokens=max_tokens,
            top_p=config.top_p,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
        )

        response = TextGenerationResponse(
            text=[dict(x.message) for x in oai_response.choices],
            logprobs=[],
            config=config,
            usage=dict(oai_response.usage)
        )
        # if use_cache:
        cache_request(cache=self.cache, params=asdict(config), values=asdict(response))
        return response

    def count_tokens(self, text) -> int:
        return num_tokens_from_messages(text)


@dataclass
class DialogueTemplate():
    system: str = None
    model_type: str = "default"
    messages: List[Dict[str, str]] = None
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"

    def get_inference_prompt(self) -> str:
        if self.model_type == "default":
            prompt = ""
            system_prompt = self.system_token + "\n" + self.system + self.end_token + "\n" if self.system else ""
            if self.messages is None:
                raise ValueError("Dialogue template must have at least one message.")
            for message in self.messages:
                if message["role"] == "system":
                    system_prompt += self.system_token + "\n" + \
                        message["content"] + self.end_token + "\n"
                elif message["role"] == "user":
                    prompt += self.user_token + "\n" + message["content"] + self.end_token + "\n"
                else:
                    prompt += self.assistant_token + "\n" + \
                        message["content"] + self.end_token + "\n"
            prompt += self.assistant_token
            if system_prompt:
                prompt = system_prompt + prompt
            return prompt
        elif self.model_type == "alpaca":
            prompt = self.user_token + "\n" + (self.system + "\n" if self.system else "")
            for message in self.messages:
                prompt += message["content"] + "\n"
            prompt = prompt + " " + self.assistant_token + "\n"
            # print(instruction)
            return prompt


class HFTextGenerator(TextGenerator):


    def __init__(self, model_type: str = "default", device_map=None, **kwargs):
        
                # Check if transformers package is installed
        try:
            import transformers
            from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
        except ImportError:
            raise ImportError("Please install the `transformers` package to use the HFTextGenerator class.")
        
        # Check if torch package is installed
        try:
            import torch
        except ImportError:
            raise ImportError("Please install the `torch` package to use the HFTextGenerator class.")
        
        super().__init__(model_type=model_type)

        self.model_name = kwargs.get("model_name", "HuggingFaceH4/starchat-alpha")
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.device = kwargs.get("device", "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=device_map, load_in_8bit=self.load_in_8bit)
        if not device_map:
            self.model.to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = kwargs.get("max_length", 1024)
        self.max_context_tokens = kwargs.get(
            "max_context_tokens",
            self.model.config.max_position_embeddings)

        if self.model_type == "alpaca":
            self.dialogue_template = DialogueTemplate(
                model_type="alpaca",
                end_token=self.tokenizer.eos_token,
                user_token="### Instruction:",
                assistant_token="### Response:"
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
        else:
            self.dialogue_template = DialogueTemplate()

    def post_process_response(self, response):
        response = response.split(
            self.dialogue_template.assistant_token)[-1].replace(self.dialogue_template.end_token, "").strip()
        response = {'role': 'assistant', 'content': response}
        return response

    def messages_to_instruction(self, messages):
        instruction = "### Instruction: "
        for message in messages:
            instruction += message["content"] + "\n"
        instruction = instruction + "### Response: "
        # print(instruction)
        return instruction

    def generate(self, config: TextGenerationConfig, use_cache=True,
                 **kwargs) -> TextGenerationResponse:
        config.model = self.model_name
        config_kwargs = {**asdict(config), **kwargs}
        if use_cache:
            response = cache_request(cache=self.cache, params=(config_kwargs))
            if response:
                return TextGenerationResponse(**response)

        self.dialogue_template.messages = config.messages
        prompt = self.dialogue_template.get_inference_prompt()
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False).to(
            self.model.device)
        input_ids = batch["input_ids"]

        max_new_tokens = kwargs.get("max_new_tokens", self.max_context_tokens - input_ids.shape[-1])
        print("Prompt tokens: ", input_ids.shape[-1], " | Max new tokens: ", max_new_tokens)

        top_k = kwargs.get("top_k", config.top_k)
        min_new_tokens = kwargs.get("min_new_tokens", 32)
        repetition_penalty = kwargs.get("repetition_penalty", 1.0)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=max(config.temperature, 0.01),
            top_p=config.top_p,
            top_k=top_k,
            num_return_sequences=config.n,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
        )
        with torch.no_grad():
            generated_ids = self.model.generate(**batch, generation_config=gen_config)

        text_response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)

        # print(text_response)
        prompt_tokens = len(batch["input_ids"][0])
        total_tokens = 0
        for row in generated_ids:
            total_tokens += len(row)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens}

        response = TextGenerationResponse(
            text=[self.post_process_response(x) for x in text_response],
            logprobs=[],
            config=config,
            usage=usage
        )
        # if use_cache:
        cache_request(cache=self.cache, params=(config_kwargs), values=asdict(response))
        return response

    def count_tokens(self, text: str):
        return len(self.tokenizer(text)["input_ids"])