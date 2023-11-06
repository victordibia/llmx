# This file contains the list of providers and models that are available supported by LLMX.


from llmx.utils import load_config


config = load_config()
providers = providers = config["providers"] if "providers" in config else None

providers = config["providers"]
