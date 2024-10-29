"""
Direct prompt method

"""

import logging
import numpy as np
import os
import torch
import requests
from functools import partial
import time

from types import SimpleNamespace

from .base import Baseline
from ..config import (
    LLAMA31_405B_URL,
    LLAMA31_405B_API_KEY,
    OPENAI_API_KEY,
    OPENAI_API_VERSION,
    OPENAI_AZURE_ENDPOINT,
    OPENAI_USE_AZURE,
)
from .utils import extract_html_tags

from .hf_utils.dp_hf_api import LLM_MAP, get_model_and_tokenizer, hf_generate

# For OpenRouter
from openai import OpenAI
from os import getenv

logger = logging.getLogger("DirectPrompt")

# As of 28 Sep 2024
OPENROUTER_COSTS = {
    "openrouter-llama-3-8b-instruct-DeepInfra": {"input": 0.000055, "output": 0.000055},
    "openrouter-llama-3-8b-instruct-NovitaAI": {"input": 0.000063, "output": 0.000063},
    "openrouter-llama-3-8b-instruct-Together": {"input": 0.00007, "output": 0.00007},
    "openrouter-llama-3-8b-instruct-Lepton": {"input": 0.000162, "output": 0.000162},
    "openrouter-llama-3-8b-instruct-Mancer": {"input": 0.0001875, "output": 0.001125},
    "openrouter-llama-3-8b-instruct-Fireworks": {"input": 0.0002, "output": 0.0002},
    "openrouter-llama-3-8b-instruct-Mancer (private)": {
        "input": 0.00025,
        "output": 0.0015,
    },
    "openrouter-llama-3-70b-instruct-DeepInfra": {"input": 0.00035, "output": 0.0004},
    "openrouter-llama-3-70b-instruct-NovitaAI": {"input": 0.00051, "output": 0.00074},
    "openrouter-llama-3-70b-instruct-Together": {"input": 0.000792, "output": 0.000792},
    "openrouter-llama-3-70b-instruct-Lepton": {"input": 0.0008, "output": 0.0008},
    "openrouter-llama-3-70b-instruct-Fireworks": {"input": 0.0009, "output": 0.0009},
    "openrouter-mixtral-8x7b-instruct-DeepInfra": {"input": 0.00024, "output": 0.00024},
    "openrouter-mixtral-8x7b-instruct-Fireworks": {"input": 0.0005, "output": 0.0005},
    "openrouter-mixtral-8x7b-instruct-Lepton": {"input": 0.0005, "output": 0.0005},
    "openrouter-mixtral-8x7b-instruct-Together": {"input": 0.00054, "output": 0.00054},
}


def dict_to_obj(data):
    if isinstance(data, dict):
        # Recursively convert dictionary values
        return SimpleNamespace(
            **{key: dict_to_obj(value) for key, value in data.items()}
        )
    elif isinstance(data, list):
        # Recursively convert each item in the list
        return [dict_to_obj(item) for item in data]
    else:
        # Return the data if it's neither a dict nor a list
        return data


@torch.inference_mode()
def huggingface_model_client(
    llm, tokenizer, model, messages, n=1, max_tokens=10000, temperature=1.0, **kwargs
):
    # Process messages into a prompt
    prompt = ""
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            prompt += f"{content}\n"
        elif role == "user":
            prompt += f"{content}\n"

    responses = hf_generate(
        llm,
        tokenizer,
        prompt,
        batch_size=n,
        temp=temperature,
        top_p=0.9,
        max_new_tokens=max_tokens,
    )

    # Now extract the assistant's reply
    choices = []
    for response_text in responses:
        # Create a message object
        message = SimpleNamespace(content=response_text)
        # Create a choice object
        choice = SimpleNamespace(message=message)
        choices.append(choice)

    # Create a usage object (we can estimate tokens)
    usage = SimpleNamespace(
        prompt_tokens=0,  # batch['input_ids'].shape[-1],
        completion_tokens=0,  # output.shape[-1] - batch['input_ids'].shape[-1],
    )

    # Create a response object
    response = SimpleNamespace(choices=choices, usage=usage)

    return response


def openrouter_client(model, messages, n=1, max_tokens=10000, temperature=1.0):
    """
    Client for OpenRouter chat models
    """
    # gets API Key from environment variable OPENAI_API_KEY
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=getenv("OPENROUTER_API_KEY"),
    )
    model_from = "meta-llama" if model.startswith("llama") else "mistralai"
    completion = client.chat.completions.create(
        model=f"{model_from}/{model[11:]}",  # exclude "openrouter-" from the model
        messages=messages,
        n=n,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return completion


def llama_3_1_405b_instruct_client(
    model, messages, n=1, max_tokens=10000, temperature=1.0
):
    """
    Request completions from the Llama 3.1 405B Instruct model hosted on Toolkit

    Parameters:
    -----------
    messages: list
        The list of messages to send to the model (same format as OpenAI API)
    max_tokens: int, default=10000
        The maximum number of tokens to use in the completion
    temperature: float, default=0.7
        The temperature to use in the completion
    n: int, default=1
        The number of completions to generate

    """

    headers = {
        "Authorization": f"Bearer {LLAMA31_405B_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
    }

    response = requests.post(
        LLAMA31_405B_URL,
        headers=headers,
        json=payload,
        verify=False,
        timeout=600,
    )
    response.raise_for_status()
    status = response.status_code
    if status != 200:
        raise Exception(
            f"API returned non-200 status code: {status}.", f"Response: {response.text}"
        )

    return dict_to_obj(response.json())


class DirectPrompt(Baseline):
    """
    A simple baseline that uses any instruction-tuned LLM to produce forecastss

    Parameters:
    -----------
    model: str
        The name of the model to use for forecasting
    use_context: bool, default=True
        If True, use context in the prompt, otherwise ignore it
    fail_on_invalid: bool, default=True
        If True, raise an exception if an invalid sample is encountered
        in the forecast. Otherwise, print a warning and skip the sample.
    n_retries: int, default=3
        The number of retries to use in rejection sampling
    max_batch_size: int, default=None
        If not None, the maximum batch size on the attemps (before the retries)
    batch_size_on_retry: int, default=5
        The batch size to use on retries
    token_cost: dict, default=None
            The cost of tokens used in the API call. If provided, the cost of the API call will be estimated.
            Expected keys are "input" and "output" for the price of input and output tokens, respectively.

    """

    __version__ = "0.0.5"  # Modification will trigger re-caching

    def __init__(
        self,
        model,
        use_context=True,
        fail_on_invalid=True,
        n_retries=3,
        batch_size_on_retry=5,
        batch_size=None,
        token_cost: dict = None,
        temperature: float = 1.0,
        dry_run: bool = False,
    ) -> None:

        self.model = model
        self.use_context = use_context
        self.fail_on_invalid = fail_on_invalid
        if model == "llama-3.1-405b-instruct":
            self.n_retries = 10
        elif model.startswith("openrouter-"):
            self.n_retries = 15
        else:
            self.n_retries = 3  # Default for GPT-4o
        self.batch_size = batch_size
        self.batch_size_on_retry = batch_size_on_retry
        self.token_cost = token_cost
        self.total_input_cost = 0  # For OpenRouter
        self.total_output_cost = 0  # For OpenRouter
        self.total_cost = 0  # Accumulator for monetary value of queries
        self.temperature = temperature
        self.dry_run = dry_run

        if not dry_run and self.model in LLM_MAP.keys():
            self.llm, self.tokenizer = get_model_and_tokenizer(
                llm_path=None, llm_type=self.model
            )
        else:
            self.llm, self.tokenizer = None, None
        self.client = self.get_client()

    def get_client(self):
        """
        Setup the OpenAI client based on configuration preferences

        """
        if self.model.startswith("gpt"):
            if OPENAI_USE_AZURE:
                logger.info("Using Azure OpenAI client.")
                from openai import AzureOpenAI

                client = AzureOpenAI(
                    api_key=OPENAI_API_KEY,
                    api_version=OPENAI_API_VERSION,
                    azure_endpoint=OPENAI_AZURE_ENDPOINT,
                ).chat.completions.create
            else:
                logger.info("Using standard OpenAI client.")
                from openai import OpenAI

                client = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create

        elif self.model == "llama-3.1-405b-instruct":
            return partial(llama_3_1_405b_instruct_client, temperature=self.temperature)

        elif self.model.startswith("openrouter-"):
            return partial(
                openrouter_client,
                temperature=self.temperature,
            )

        elif self.model in LLM_MAP.keys():
            return partial(
                huggingface_model_client,
                llm=self.llm,
                tokenizer=self.tokenizer,
                temperature=self.temperature,
            )

        else:
            raise NotImplementedError(f"Model {self.model} not supported.")

        return client

    def make_prompt(self, task_instance, max_digits=6):
        """
        Generate the prompt for the model

        Notes:
        - Assumes a uni-variate time series

        """
        logger.info("Building prompt for model.")

        # Extract time series data
        hist_time = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        hist_value = task_instance.past_time.values[:, -1]
        pred_time = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").values
        # "g" Print up to max_digits digits, although it switch to scientific notation when y >= 1e6,
        # so switch to "f" without any digits after the dot if y is too large.
        history = "\n".join(
            f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
            for x, y in zip(hist_time, hist_value)
        )

        # Extract context
        context = ""
        if self.use_context:
            if task_instance.background:
                context += f"Background: {task_instance.background}\n"
            if task_instance.constraints:
                context += f"Constraints: {task_instance.constraints}\n"
            if task_instance.scenario:
                context += f"Scenario: {task_instance.scenario}\n"

        prompt = f"""
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
{context}
</context>

Here is a historical time series in (timestamp, value) format:
<history>
{history}
</history>

Now please predict the value at the following timestamps: {pred_time}.

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""
        return prompt

    def __call__(self, task_instance, n_samples):
        """
        Infer forecasts from the model

        Parameters:
        -----------
        task_instance: TimeSeriesTask
            The task instance to forecast
        n_samples: int
            The number of samples to generate
        n_retries: int
            The number of rejection sampling steps
        batch_size_on_retry: int
            The batch size to use on retries. This is useful to avoid asking for way too many samples
            from the API.

        Returns:
        --------
        samples: np.ndarray, shape [n_samples, time dimension, number of variables]
            The forecast samples. Note: only univariate is supported at the moment (number of variables = 1)
        extra_info: dict
            A dictionary containing informations pertaining to the cost of running this model
        """

        default_batch_size = n_samples if not self.batch_size else self.batch_size
        if self.batch_size:
            assert (
                self.batch_size * self.n_retries >= n_samples
            ), f"Not enough iterations to cover {n_samples} samples"
        assert (
            self.batch_size_on_retry <= default_batch_size
        ), f"Batch size on retry should be equal to or less than {default_batch_size}"

        starting_time = time.time()
        total_client_time = 0.0

        prompt = self.make_prompt(task_instance)
        messages = [
            {
                "role": "system",
                "content": "You are a useful forecasting assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        # Get forecast samples via rejection sampling until we have the desired number of samples
        # or until we run out of retries

        total_tokens = {"input": 0, "output": 0}
        valid_forecasts = []

        max_batch_size = task_instance.max_directprompt_batch_size
        if max_batch_size is not None:
            batch_size = min(default_batch_size, max_batch_size)
            n_retries = self.n_retries + default_batch_size // batch_size
        else:
            batch_size = default_batch_size
            n_retries = self.n_retries

        llm_outputs = []

        while len(valid_forecasts) < n_samples and n_retries > 0:
            logger.info(f"Requesting forecast of {batch_size} samples from the model.")
            client_start_time = time.time()
            chat_completion = self.client(
                model=self.model, n=batch_size, messages=messages
            )
            total_client_time += time.time() - client_start_time
            total_tokens["input"] += chat_completion.usage.prompt_tokens
            total_tokens["output"] += chat_completion.usage.completion_tokens

            logger.info("Parsing forecasts from completion.")
            for choice in chat_completion.choices:
                llm_outputs.append(choice.message.content)
                try:
                    # Extract forecast from completion
                    forecast = extract_html_tags(choice.message.content, ["forecast"])[
                        "forecast"
                    ][0]
                    forecast = forecast.replace("(", "").replace(")", "")
                    forecast = forecast.split("\n")
                    forecast = {
                        x.split(",")[0]
                        .replace("'", "")
                        .replace('"', ""): float(x.split(",")[1])
                        for x in forecast
                    }

                    # Get forecasted values at expected timestamps (will fail if model hallucinated timestamps, which is ok)
                    forecast = [
                        forecast[t]
                        for t in task_instance.future_time.index.strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                    ]

                    # Append forecast to list of valid forecasts
                    valid_forecasts.append(forecast)

                    # If OpenRouter, compute costs here as costs differ per call
                    if self.model.startswith("openrouter-"):
                        # Get provider
                        provider = chat_completion.provider
                        # Make string
                        model_name = self.model + "-" + provider
                        # Compute costs with Openrouter cost dict
                        input_cost = (
                            total_tokens["input"]
                            / 1000
                            * OPENROUTER_COSTS[model_name]["input"]
                        )
                        output_cost = (
                            total_tokens["output"]
                            / 1000
                            * OPENROUTER_COSTS[model_name]["input"]
                        )
                        current_cost = round(input_cost + output_cost, 2)
                        logger.info(f"Forecast cost: {current_cost}$")

                        self.total_input_cost += input_cost
                        self.total_output_cost += output_cost
                        self.total_cost += current_cost
                        total_tokens = {
                            "input": 0,
                            "output": 0,
                        }  # As cost for current tokens has been accounted for, reset total_tokens
                except Exception as e:
                    logger.info("Sample rejected due to invalid format.")
                    logger.debug(f"Rejection details: {e}")
                    logger.debug(f"Choice: {choice.message.content}")

            n_retries -= 1
            if max_batch_size is not None:
                # Do not go down to self.batch_size_on_retry until we are almost done
                remaining_samples = n_samples - len(valid_forecasts)
                batch_size = max(remaining_samples, self.batch_size_on_retry)
                batch_size = min(batch_size, max_batch_size)
            else:
                batch_size = self.batch_size_on_retry

            valid_forecasts = valid_forecasts[:n_samples]
            logger.info(f"Got {len(valid_forecasts)}/{n_samples} valid forecasts.")
            if len(valid_forecasts) < n_samples:
                logger.info(f"Remaining retries: {n_retries}.")

        # If we couldn't collect enough forecasts, raise exception if desired
        if self.fail_on_invalid and len(valid_forecasts) < n_samples:
            raise RuntimeError(
                f"Failed to get {n_samples} valid forecasts. Got {len(valid_forecasts)} instead."
            )

        extra_info = {
            "total_input_tokens": total_tokens["input"],
            "total_output_tokens": total_tokens["output"],
            "llm_outputs": llm_outputs,
        }

        # Estimate cost of API calls
        logger.info(f"Total tokens used: {total_tokens}")
        if self.model.startswith("openrouter-"):
            extra_info["input_token_cost"] = self.total_input_cost
            extra_info["output_token_cost"] = self.total_output_cost
            extra_info["total_token_cost"] = self.total_cost

        elif self.token_cost is not None:
            input_cost = total_tokens["input"] / 1000 * self.token_cost["input"]
            output_cost = total_tokens["output"] / 1000 * self.token_cost["output"]
            current_cost = round(input_cost + output_cost, 2)
            logger.info(f"Forecast cost: {current_cost}$")
            self.total_cost += current_cost

            extra_info["input_token_cost"] = self.token_cost["input"]
            extra_info["output_token_cost"] = self.token_cost["output"]
            extra_info["total_token_cost"] = current_cost

        # Convert the list of valid forecasts to a numpy array
        samples = np.array(valid_forecasts)[:, :, None]

        extra_info["total_time"] = time.time() - starting_time
        extra_info["total_client_time"] = total_client_time

        return samples, extra_info

    @property
    def cache_name(self):
        args_to_include = [
            "model",
            "use_context",
            "fail_on_invalid",
            "n_retries",
        ]
        if not self.model.startswith("gpt"):
            args_to_include.append("temperature")
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )
