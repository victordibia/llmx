from typing import Dict, Union
from dataclasses import asdict, dataclass
from transformers import (AutoTokenizer, AutoModelForCausalLM, GenerationConfig)
import torch


from .base_textgen import TextGenerator
from ...datamodel import TextGenerationConfig, TextGenerationResponse
from ...utils import cache_request, get_models_maxtoken_dict


@dataclass
class DialogueTemplate:
    system: str = None
    dialogue_type: str = "default"
    messages: list[dict[str, str]] = None
    system_token: str = "<|system|>"
    user_token: str = "<|user|>"
    assistant_token: str = "<|assistant|>"
    end_token: str = "<|end|>"

    def get_inference_prompt(self) -> str:
        if self.dialogue_type == "default":
            prompt = ""
            system_prompt = (
                self.system_token + "\n" + self.system + self.end_token + "\n"
                if self.system
                else ""
            )
            if self.messages is None:
                raise ValueError("Dialogue template must have at least one message.")
            for message in self.messages:
                if message["role"] == "system":
                    system_prompt += (
                        self.system_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
                elif message["role"] == "user":
                    prompt += (
                        self.user_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
                else:
                    prompt += (
                        self.assistant_token
                        + "\n"
                        + message["content"]
                        + self.end_token
                        + "\n"
                    )
            prompt += self.assistant_token
            if system_prompt:
                prompt = system_prompt + prompt
            return prompt
        elif self.dialogue_type == "alpaca":
            prompt = (
                self.user_token + "\n" + (self.system + "\n" if self.system else "")
            )
            for message in self.messages:
                prompt += message["content"] + "\n"
            prompt = prompt + " " + self.assistant_token + "\n"
            # print(instruction)
            return prompt
        elif self.dialogue_type == "llama2":
            prompt = "[INST]"
            system_prompt = ""
            other_prompt = ""

            for message in self.messages:
                if message["role"] == "system":
                    system_prompt += message["content"] + "\n"
                elif message["role"] == "assistant":
                    other_prompt += message["content"] + "  \n"
                else:
                    other_prompt += "[INST] " + message["content"] + "[/INST]\n"

            prompt = (
                prompt
                + f" <<SYS>> {system_prompt} <</SYS>> \n"
                + other_prompt
                + "[/INST]"
            )


class HFTextGenerator(TextGenerator):
    def __init__(self,
                 provider: str = "huggingface",
                 models: Dict = None,
                 device_map=None, **kwargs):

        super().__init__(provider=provider)

        self.dialogue_type = kwargs.get("dialogue_type", "alpaca")

        self.model_name = kwargs.get("model", "uukuguy/speechless-llama2-hermes-orca-platypus-13b")
        self.load_in_8bit = kwargs.get("load_in_8bit", False)
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        self.device = kwargs.get("device", self.get_default_device())

        # load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map=device_map, load_in_8bit=self.load_in_8bit,
            trust_remote_code=self.trust_remote_code)
        if not device_map:
            self.model.to(self.device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.max_length = kwargs.get("max_length", 1024)

        self.model_max_token_dict = get_models_maxtoken_dict(models)
        self.max_context_tokens = kwargs.get(
            "max_context_tokens", self.model.config.max_position_embeddings
        ) or self.model_max_token_dict[self.model_name]

        if self.dialogue_type == "alpaca":
            self.dialogue_template = DialogueTemplate(
                dialogue_type="alpaca",
                end_token=self.tokenizer.eos_token,
                user_token="### Instruction:",
                assistant_token="### Response:",
            )
            self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            self.model.config.bos_token_id = 1
            self.model.config.eos_token_id = 2
        else:
            self.dialogue_template = DialogueTemplate(end_token=self.tokenizer.eos_token)

    def get_default_device(self):
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def post_process_response(self, response):
        response = (
            response.split(self.dialogue_template.assistant_token)[-1]
            .replace(self.dialogue_template.end_token, "")
            .strip()
        )
        response = {"role": "assistant", "content": response}
        return response

    def messages_to_instruction(self, messages):
        instruction = "### Instruction: "
        for message in messages:
            instruction += message["content"] + "\n"
        instruction = instruction + "### Response: "
        # print(instruction)
        return instruction

    def generate(
            self, messages: Union[list[dict],
                                  str],
            config: TextGenerationConfig = TextGenerationConfig(),
            **kwargs) -> TextGenerationResponse:
        use_cache = config.use_cache
        config.model = self.model_name
        cache_key_params = {
            **asdict(config),
            **kwargs,
            "messages": messages,
            "dialogue_type": self.dialogue_type}
        if use_cache:
            response = cache_request(cache=self.cache, params=(cache_key_params))
            if response:
                return TextGenerationResponse(**response)

        self.dialogue_template.messages = messages
        prompt = self.dialogue_template.get_inference_prompt()
        batch = self.tokenizer(
            prompt, return_tensors="pt", return_token_type_ids=False
        ).to(self.model.device)
        input_ids = batch["input_ids"]

        max_new_tokens = kwargs.get(
            "max_new_tokens", self.max_context_tokens - input_ids.shape[-1]
        )
        # print(
        #     "***********Prompt tokens: ",
        #     input_ids.shape[-1],
        #     " | Max new tokens: ",
        #     max_new_tokens,
        # )

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

        text_response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        # print(text_response, "*************")
        prompt_tokens = len(batch["input_ids"][0])
        total_tokens = 0
        for row in generated_ids:
            total_tokens += len(row)

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": total_tokens - prompt_tokens,
            "total_tokens": total_tokens,
        }

        response = TextGenerationResponse(
            text=[self.post_process_response(x) for x in text_response],
            logprobs=[],
            config=config,
            usage=usage,
        )
        # if use_cache:
        cache_request(cache=self.cache, params=(cache_key_params), values=asdict(response))
        return response

    def count_tokens(self, text: str):
        return len(self.tokenizer(text)["input_ids"])
