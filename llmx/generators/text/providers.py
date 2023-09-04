# This file contains the list of providers and models that are available supported by LLMX.

providers = {
    "openai": {
        "name": "OpenAI",
        "description": "OpenAI's GPT-3 and GPT-4 models.",
        "models": [
            {"name": "gpt-4", "max_tokens": 8192},
            {"name": "gpt-4-0314", "max_tokens": 8192},
            {"name": "gpt-4-0613", "max_tokens": 8192},
            {"name": "gpt-4-32k", "max_tokens": 32768},
            {"name": "gpt-4-32k-0613", "max_tokens": 32768},
            {"name": "gpt-3.5-turbo", "max_tokens": 4096},
            {"name": "gpt-3.5-turbo-0301", "max_tokens": 4096},
            {"name": "gpt-3.5-turbo-16k", "max_tokens": 16384},
            {"name": "gpt-3.5-turbo-0613", "max_tokens": 4096},
            {"name": "gpt-3.5-turbo-16k-0613", "max_tokens": 16384},
        ],
    },
    "google": {
        "name": "Google",
        "description": "Google's LLM models.",
        "models": [
            {"name": "codechat-bison", "max_tokens": 1024},
            {"name": "chat-bison", "max_tokens": 1024},
            {"name": "codechat-bison-32k", "max_tokens": 32768},
            {"name": "chat-bison-32k", "max_tokens": 32768},
        ],
    },
    "cohere": {
        "name": "Cohere",
        "description": "Cohere's LLM models.",
        "models": [
            {"name": "command", "max_tokens": 4096},
            {"name": "command-nightly", "max_tokens": 4096},
        ],
    },
    "huggingface": {
        "name": "HuggingFace",
        "description": "HuggingFace's LLM models.",
        "models": [
            {"name": "TheBloke/Llama-2-7b-chat-fp16", "max_tokens": 4096},
            {"name": "TheBloke/Llama-2-13B-fp16", "max_tokens": 4096},
            {"name": "TheBloke/gpt4-x-vicuna-13B-HF", "max_tokens": 2040},
        ],
    },
}

# providers["palm"] = providers["google"]
