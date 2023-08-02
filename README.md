# LLMX - An API for Language Models

[![PyPI version](https://badge.fury.io/py/llmx.svg)](https://badge.fury.io/py/llmx)

A simple python package that provides a unified interface to several LLM providers [ OpenAI (default), PaLM, Cohere and local HuggingFace Models ].

There is nothing special about this library, but some of the requirements I needed when I startec building this (that other libraries did not have):

- **Unified Model Interface**: Single interface to create LLM text generators with support for **multiple LLM providers**.

```python
from llmx import  text_generator as generator

openai_generator = generator(provider="openai")
palm_generator = generator(provider="google") # or palm
cohere_generator = generator(provider="cohere") # or palm
hf_generator = generator(provider="huggingface") # run locally
```

- **Unified Messaging Interface**. Standardizes on the OpenAI ChatML format. For example, the standard prompt sent a model is formatted as an array of objects, where each object has a role (`system`, `user`, or `assistant`) and content of the form. A single request is list one only one message (e.g., write code to plot a cosine wave signal). A conversation is a list of messages e.g. write code for x, update the axis to y, etc. For all models.

```python
messages = [
    {"role": "user", "content": "You are a helpful assistant that can explain concepts clearly to a 6 year old child."},
    {"role": "user", "content": "What is  gravity?"}
]
```

- **Good Utils (e.g., Caching etc)**: E.g. being able to use caching for faster responses. General policy is that cache is used if config (including messages) is the same. If you want to force a new response, set `use_cache=False` in the `generate` call.

```python
response = gen.generate(messages=messages, use_cache=True)

# TextGenerationResponse(text=[Message(role='assistant', content="Gravity is like a magical force that pulls things towards each other. It's what keeps us on the ground and stops us from floating away into space. ... ")], config=TextGenerationConfig(n=1, temperature=0.1, max_tokens=8147, top_p=1.0, top_k=50, frequency_penalty=0.0, presence_penalty=0.0, model_type='openai', model='gpt-4', stop=None), logprobs=[], usage={'prompt_tokens': 34, 'completion_tokens': 69, 'total_tokens': 103})
```

Are there other libraries that do things like this really well? Yes! I'd recommend looking at [guidance](https://github.com/microsoft/guidance) which does a lot more. Interested in optimized inference? Try somthing like [vllm](https://github.com/vllm-project/vllm).

## Installation

Install from pypi. Please use python3.9 or higher.

```bash
pip install llmx
```

Install in development mode

```bash
git clone
cd llmx
pip install -e .
```

Note that you may want to use the latest version of pip to install this package.
`python3 -m pip install --upgrade pip`

## Usage

Set your api keys first

```bash
export OPENAI_API_KEY=<your key>
export PALM_API_KEY=<your key>
export COHERE_API_KEY=<your key>
```

```python
from llmx import  text_generator as generator
from llmx.datamodel import TextGenerationConfig

messages =  messages = [
    {"role": "system", "content": "You are a helpful assistant that can explain concepts clearly to a 6 year old child."},
    {"role": "user", "content": "What is  gravity?"}
]

openai_gen = generator(provider="openai")
openai_config = TextGenerationConfig(model="gpt-4", max_tokens=50)
openai_response = openai_gen.generate(messages, config=openai_config, use_cache=True)
print(openai_response.text[0].content)

```

See the [tutorial](/notebooks/tutorial.ipynb) for more examples.

## Current Work

- Supported models
  - [x] OpenAI
  - [x] PaLM
  - [x] Cohere
  - [x] HuggingFace (local)

## Caveats

- **Prompting**. llmx makes some assumptions around how prompts are constructed e.g., how the chat message interface is assembled into a prompt for each model type. If your application or use case requires more control over the prompt, you may want to use a different library (ideally query the LLM models directly).
- **Inference Optimization**. This library is not really designed for speed, but more for rapid experimentation using multiple models. If you are looking for a library that is optimized for inference, I'd recommend looking at [vllm](https://github.com/vllm-project/vllm) or [tgi](https://github.com/huggingface/text-generation-inference)
