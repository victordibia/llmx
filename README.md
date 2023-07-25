# LLMX

A simple python package that provides a unified interface to several LLM providers - [OpenAI](https://platform.openai.com/docs/api-reference/authentication) (default), Palm, and local HuggingFace Models.

There is nothing special about this library, but some of the requirements I needed when I startec building this (that other libraries did not have):

- Uses typed datamodels for model configuration: This makes it easier to build web apis (fast api) on top of this library. For example, the text generation config is a pydantic model.

```python
config = TextGenerationConfig(
    model="gpt-3.5-turbo-0301",
    n=1,
    temperature=0.5,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
```

- Unified Interface: Switch between LLM providers with a single line of code. Standardizes on the OpenAI ChatML format. For example, the standard prompt sent a model is formatted as an array of objects, where each object has a role (`system`, `user`, or `assistant`) and content of the form.

```python

messages = [
    {"role": "user", "content": "You are a helpful assistant that can explain concepts clearly to a 6 year old child."},
    {"role": "user", "content": "What is  gravity?"}
]
```

Are there other libraries that do things like this really well? Yes! I'd recommend looking at guidance.

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
```

```python
from llmx import OpenAITextGenerator, TextGenerationConfig

gen = OpenAITextGenerator()
config = TextGenerationConfig(
    model="gpt-3.5-turbo-0301",
    messages=[
        {"role": "user", "content": "What is the height of the Eiffel Tower?"},
    ],
    n=1,
    temperature=0.5,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)
response = gen.generate(config=config, use_cache=False)
print(response.text)
# [{'role': 'assistant', 'content': 'The height of the Eiffel Tower is 324 meters (1,063 feet).'}]
```
