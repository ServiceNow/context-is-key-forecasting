"""
Configuration variables for the benchmark

"""

import os

from pathlib import Path

# Model weight storage
MODEL_STORAGE_PATH = Path(os.environ.get("STARCASTER_MODEL_STORE", "./models"))
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Evaluation configuration
DEFAULT_N_SAMPLES = 50
RESULT_CACHE_PATH = Path(os.environ.get("STARCASTER_RESULT_CACHE", "./inference_cache"))
METRIC_SCALING_CACHE_PATH = os.environ.get(
    "STARCASTER_METRIC_SCALING_CACHE", "./metric_scaling_cache"
)

# OpenAI configuration
OPENAI_USE_AZURE = (
    os.environ.get("STARCASTER_OPENAI_USE_AZURE", "False").lower() == "true"
)
OPENAI_API_KEY = os.environ.get("STARCASTER_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("STARCASTER_OPENAI_API_VERSION", None)
OPENAI_AZURE_ENDPOINT = os.environ.get("STARCASTER_OPENAI_AZURE_ENDPOINT", None)


DATA_STORAGE_PATH = Path(os.environ.get("STARCASTER_DATA_STORE", "benchmark/data"))
if not DATA_STORAGE_PATH.exists():
    DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
