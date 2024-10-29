"""
Configuration variables for the benchmark

"""

import os

from pathlib import Path

# Storage configuration
# CIK_MODEL_STORE
# Store the model weights of the baselines in this folder.
# Default to "./models"
# CIK_DATA_STORE
# Store the downloaded datasets in this folder.
# Default to "benchmark/data"
# CIK_DOMINICK_STORE
# Store the downloaded dataset for the tasks using the Dominick dataset in this folder.
# Default to CIK_DATA_STORE + "/dominicks"
# CIK_TRAFFIC_DATA_STORE
# Store the downloaded dataset for the tasks using the Traffic dataset in this folder.
# Default to CIK_DATA_STORE + "/traffic_data"
# HF_HOME
# Location of the cache when downloading some datasets from Hugging Face.
# Default to CIK_DATA_STORE + "/hf_cache"
MODEL_STORAGE_PATH = Path(os.environ.get("CIK_MODEL_STORE", "./models"))
if not os.path.exists(MODEL_STORAGE_PATH):
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
DATA_STORAGE_PATH = Path(os.environ.get("CIK_DATA_STORE", "./data"))
if not DATA_STORAGE_PATH.exists():
    DATA_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
DOMINICK_STORAGE_PATH = os.environ.get(
    "CIK_DOMINICK_STORE", os.path.join(DATA_STORAGE_PATH, "dominicks")
)
TRAFFIC_STORAGE_PATH = os.environ.get(
    "CIK_TRAFFIC_DATA_STORE", os.path.join(DATA_STORAGE_PATH, "traffic_data")
)
HF_CACHE_DIR = os.environ.get("HF_HOME", os.path.join(DATA_STORAGE_PATH, "hf_cache"))

# Evaluation configuration
# CIK_RESULT_CACHE
# Store the output of the baselines in this folder, to avoid recomputing them.
# Default to "./inference_cache"
# CIK_METRIC_SCALING_CACHE
# Store the scaling factors for each task in this folder, to avoid recomputing them.
# Default to "./metric_scaling_cache"
# CIK_METRIC_SCALING_CACHE
# If set, the metric computation will also compute an estimate of the variance of the metric.
# By default, only compute the metric itself.
DEFAULT_N_SAMPLES = 50
RESULT_CACHE_PATH = Path(os.environ.get("CIK_RESULT_CACHE", "./inference_cache"))
METRIC_SCALING_CACHE_PATH = os.environ.get(
    "CIK_METRIC_SCALING_CACHE", "./metric_scaling_cache"
)
COMPUTE_METRIC_VARIANCE = bool(
    os.environ.get("CIK_METRIC_COMPUTE_VARIANCE", "")
)  # bool("") == False

# OpenAI configuration
# CIK_OPENAI_USE_AZURE
# If set to "True", then the baselines using OpenAI models will use the Azure client, instead of the OpenAI client.
# Default to "False"
# CIK_OPENAI_API_KEY
# Must be set to use baselines using OpenAI models to your API key (either Azure or OpenAI depending on the value of CIK_OPENAI_USE_AZURE).
# CIK_OPENAI_API_VERSION
# If set, select the chosen API version when calling OpenAI model using the Azure client.
# CIK_OPENAI_AZURE_ENDPOINT
# Select the Azure endpoint to use when calling OpenAI models.
OPENAI_USE_AZURE = os.environ.get("CIK_OPENAI_USE_AZURE", "False").lower() == "true"
OPENAI_API_KEY = os.environ.get("CIK_OPENAI_API_KEY", "")
OPENAI_API_VERSION = os.environ.get("CIK_OPENAI_API_VERSION", None)
OPENAI_AZURE_ENDPOINT = os.environ.get("CIK_OPENAI_AZURE_ENDPOINT", None)

# Llama-405b configuration
# CIK_LLAMA31_405B_URL
# Must be set to the API URL for the Llama-3.1-405b baseline. We expect a vLLM server.
# CIK_LLAMA31_405B_API_KEY
# Must be set to the API key for your Llama-3.1-405b API
LLAMA31_405B_URL = os.environ.get("CIK_LLAMA31_405B_URL", None)
LLAMA31_405B_API_KEY = os.environ.get("CIK_LLAMA31_405B_API_KEY", None)

# Nixtla configuration
# CIK_NIXTLA_BASE_URL
# Must be set to the Azure API URL for the Nixtla TimeGEN baseline.
# CIK_NIXTLA_API_KEY
# Must be set to your Azure API key for the Nixtla TimeGEN baseline.
NIXTLA_BASE_URL = os.environ.get("CIK_NIXTLA_BASE_URL", None)
NIXTLA_API_KEY = os.environ.get("CIK_NIXTLA_API_KEY", None)
