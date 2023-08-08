import pytest
import os
from llmx import llm
from llmx.datamodel import TextGenerationConfig


config = TextGenerationConfig(
    n=2,
    temperature=0.4,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    use_cache=False
)

messages = [
    {"role": "user",
     "content": "What is the capital of France? Only respond with the exact answer"}]


def test_openai():
    openai_gen = llm(provider="openai")
    openai_response = openai_gen.generate(messages, config=config)
    answer = openai_response.text[0].content
    print(openai_response.text[0].content)

    assert ("paris" in answer.lower())
    assert len(openai_response.text) == 2


def test_google():
    google_gen = llm(provider="google")
    config.model = "chat-bison@001"
    google_response = google_gen.generate(messages, config=config)
    answer = google_response.text[0].content
    print(google_response.text[0].content)

    assert ("paris" in answer.lower())
    # assert len(google_response.text) == 2 palm may chose to return 1 or 2 responses


def test_cohere():
    cohere_gen = llm(provider="cohere")
    config.model = "command"
    cohere_response = cohere_gen.generate(messages, config=config)
    answer = cohere_response.text[0].content
    print(cohere_response.text[0].content)

    assert ("paris" in answer.lower())
    assert len(cohere_response.text) == 2


@pytest.mark.skipif(os.environ.get("LLMX_RUNALL", None) is None
                    or os.environ.get("LLMX_RUNALL", None) == "False", reason="takes too long")
def test_hf_local():
    hf_local_gen = llm(
        provider="hf",
        model="TheBloke/Llama-2-7b-chat-fp16",
        device_map="auto")
    hf_local_response = hf_local_gen.generate(messages, config=config)
    answer = hf_local_response.text[0].content
    print(hf_local_response.text[0].content)

    assert ("paris" in answer.lower())
    assert len(hf_local_response.text) == 2
