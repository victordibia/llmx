import typer
import os
from .generators.text.providers import providers

app = typer.Typer()


@app.command()
def models():
    print("Available models:")
    for provider in providers.items():
        print(f"Provider: {provider[1]['name']}")
        for model in provider[1]["models"]:
            print(f"  - {model['name']}")


@app.command()
def list():
    print("list")


def run():
    app()


if __name__ == "__main__":
    app()
