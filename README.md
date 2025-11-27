# LLMX with Ollama extension 
This repository is a fork of [llmx](https://github.com/victordibia/llmx) with added support for running Ollama models locally.
It extends llmx by integrating locally hosted Ollama models and their execution features.
You can install this fork directly from the GitHub repository using pip.

Use this version if you want seamless integration of Ollama models within the llmx workflow.
Contributions and feedback are welcome to further improve Ollama compatibility.



## Prerequisite-Ollama local setup
Prerequisite: A working local Ollama setup must be installed and running on your machine before using this fork

Go to the official Ollama website (https://ollama.com) and download the installer.
After installation , verify the installation by running the below command from command line.
<pre>
ollama -v
</pre>

To list available models:
<pre>
ollama list
</pre>

To download and run a model i.e. llama3.2:3b
<pre>
ollama run llama3.2:3b
</pre>

## Testing llmx-ollama extension
<pre>
python .\tests\test_generators.py
</pre>

## Usage
```python
from llmx import llm

# Define your messages and config as needed
messages = [
    {"role": "user", "content": "What is the capital city of Germany?"}
]

config = TextGenerationConfig(
    temperature=0.4,
    use_cache=False
)

ollama_gen = llm(provider="ollama", model="llama3.2:3b")
response = ollama_gen.generate(messages, config=config)
answer = response.text[0].content

print("Summary:", answer)

```
