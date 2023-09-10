# LLMX - An API for Chat Fine-Tuned Language Models

[![PyPI version](https://badge.fury.io/py/llmx.svg)](https://badge.fury.io/py/llmx)

A simple python package that provides a unified interface to several LLM providers of chat fine-tuned models [OpenAI, AzureOpenAI, PaLM, Cohere and local HuggingFace Models].

> **Note**
> llmx wraps multiple api providers and its interface _may_ change as the providers as well as the general field of LLMs evolve.

There is nothing particularly special about this library, but some of the requirements I needed when I started building this (that other libraries did not have):

- **Unified Model Interface**: Single interface to create LLM text generators with support for **multiple LLM providers**.

```python
from llmx import  llm

gen = llm(provider="openai") # support azureopenai models too.
gen = llm(provider="palm") # or google
gen = llm(provider="cohere") # or palm
gen = llm(provider="hf", model="uukuguy/speechless-llama2-hermes-orca-platypus-13b", device_map="auto") # run huggingface model locally
```

- **Unified Messaging Interface**. Standardizes on the OpenAI ChatML message format and is designed for _chat finetuned_ models. For example, the standard prompt sent a model is formatted as an array of objects, where each object has a role (`system`, `user`, or `assistant`) and content (see below). A single request is list of only one message (e.g., write code to plot a cosine wave signal). A conversation is a list of messages e.g. write code for x, update the axis to y, etc. Same format for all models.

```python
messages = [
    {"role": "user", "content": "You are a helpful assistant that can explain concepts clearly to a 6 year old child."},
    {"role": "user", "content": "What is  gravity?"}
]
```

- **Good Utils (e.g., Caching etc)**: E.g. being able to use caching for faster responses. General policy is that cache is used if config (including messages) is the same. If you want to force a new response, set `use_cache=False` in the `generate` call.

```python
response = gen.generate(messages=messages, config=TextGeneratorConfig(n=1, use_cache=True))
```

Output looks like

```bash

TextGenerationResponse(
  text=[Message(role='assistant', content="Gravity is like a magical force that pulls things towards each other. It's what keeps us on the ground and stops us from floating away into space. ... ")],
  config=TextGenerationConfig(n=1, temperature=0.1, max_tokens=8147, top_p=1.0, top_k=50, frequency_penalty=0.0, presence_penalty=0.0, provider='openai', model='gpt-4', stop=None),
  logprobs=[], usage={'prompt_tokens': 34, 'completion_tokens': 69, 'total_tokens': 103})

```

Are there other libraries that do things like this really well? Yes! I'd recommend looking at [guidance](https://github.com/microsoft/guidance) which does a lot more. Interested in optimized inference? Try somthing like [vllm](https://github.com/vllm-project/vllm).

## Installation

Install from pypi. Please use **python3.10** or higher.

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

Set your api keys first for each service.

```bash
# for openai and cohere
export OPENAI_API_KEY=<your key>
export COHERE_API_KEY=<your key>

# for PaLM (Vertex AI), setup a gcp project, and get a service account key file
export PALM_SERVICE_ACCOUNT_KEY_FILE= <path to your service account key file>
export PALM_PROJECT_ID=<your gcp project id>
export PALM_PROJECT_LOCATION=<your project location>
```

```python
from llmx import llm
from llmx.datamodel import TextGenerationConfig

messages = [
    {"role": "system", "content": "You are a helpful assistant that can explain concepts clearly to a 6 year old child."},
    {"role": "user", "content": "What is  gravity?"}
]

openai_gen = llm(provider="openai")
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
- **Inference Optimization**. This library is not really designed for speed, but more for rapid experimentation using multiple models. If you are looking for a library that is optimized for inference (tensor parrelization, distributed inference etc), I'd recommend looking at [vllm](https://github.com/vllm-project/vllm) or [tgi](https://github.com/huggingface/text-generation-inference)

## Citation

If you use this library in your work, please cite:

```bibtex
@software{victordibiallmx,
author = {Victor Dibia},
license = {MIT},
month =  {10},
title = {LLMX - An API for Chat Fine-Tuned Language Models},
url = {https://github.com/victordibia/llmx},
year = {2023}
}
```
