from llmx.generators.text.textgen import TextGenerator
from llmx.datamodel import TextGenerationConfig


config = TextGenerationConfig(
    n=2,
    temperature=0.01,
    max_tokens=100,
    top_p=1.0,
    top_k=50,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    messages=[
        {"role": "user", "content": "What is the height of the Eiffel Tower?"},
    ],
)


def test_openai():
    openai_gen = TextGenerator(provider="openai")
    openai_response = openai_gen.generate(config=config, use_cache=False)
    answer = openai_response.text[0].content
    print(openai_response.text[0].content)

    assert (
        "324" in answer or "1,063" in answer or "1,063 ft" in answer or "1063" in answer
    )
    assert len(openai_response.text) == 2


def test_google():
    google_gen = TextGenerator(provider="google")
    config.model = "models/chat-bison-001"
    google_response = google_gen.generate(config=config, use_cache=False)
    answer = google_response.text[0].content
    print(google_response.text[0].content)

    assert (
        "324" in answer or "1,063" in answer or "1,063 ft" in answer or "1063" in answer
    )
    assert len(google_response.text) == 2


def test_cohere():
    cohere_gen = TextGenerator(provider="cohere")
    config.model = "command"
    cohere_response = cohere_gen.generate(config=config, use_cache=False)
    answer = cohere_response.text[0].content
    print(cohere_response.text[0].content)

    assert (
        "324" in answer or "1,063" in answer or "1,063 ft" in answer or "1063" in answer
    )
    assert len(cohere_response.text) == 2
