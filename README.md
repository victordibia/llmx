# LLMX - An API for Language Models

[![PyPI version](https://badge.fury.io/py/llmx.svg)](https://badge.fury.io/py/llmx)

A simple python package that provides a unified interface to several LLM providers [ OpenAI (default), PaLM, Cohere and local HuggingFace Models ].

There is nothing special about this library, but some of the requirements I needed when I startec building this (that other libraries did not have):

- **Unified Model Interface**: Single interface to create LLM text generators with support for **multiple LLM providers**.

```python

openai_generator = TextGenerator(provider="openai")
palm_generator = TextGenerator(provider="google") # or palm
cohere_generator = TextGenerator(provider="cohere") # or palm
hf_generator = TextGenerator(provider="huggingface") # run locally
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
response = gen.generate(config=config, use_cache=True)
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
from llmx.generators.text.textgen import TextGenerator
from llmx.datamodel import TextGenerationConfig

gen = TextGenerator(provider="openai")
config = TextGenerationConfig(messages=[
        {"role": "user", "content": "What is the height of the Eiffel Tower?"},
    ])
response = gen.generate(config=config, use_cache=False)
print(response.text)
# [{'role': 'assistant', 'content': 'The height of the Eiffel Tower is 324 meters (1,063 feet).'}]
```

## Current Work

- Supported models
  - [x] OpenAI
  - [x] PaLM
  - [x] Cohere
  - [ ] HuggingFace (local)
