import pandas as pd
import numpy as np
from .base import Baseline
from ..base import BaseTask
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import set_seed
import os
import sys
from typing import Optional, Union, Dict, Callable, Iterable

import warnings
import logging
import json
import sys

from huggingface_hub import hf_hub_download

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the 'models' directory inside 'Time-LLM'
time_llm_models_path = os.path.join(script_dir, "Time-LLM", "models")

# Add 'models' directory to sys.path
sys.path.append(time_llm_models_path)

from timellm.models.TimeLLM import Model as TimeLLMModel

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def truncate_mse_loss(future_time, future_pred):
    # Assumes future_time.shape == (B, T1) and future_pred.shape == (B, T2)
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.mse_loss(future_time[..., :min_length], future_pred[..., :min_length])


def truncate_mae_loss(future_time, future_pred):
    # Assumes future_time.shape == (B, T1) and future_pred.shape == (B, T2)
    min_length = min(future_time.shape[-1], future_pred.shape[-1])
    return F.l1_loss(future_time[..., :min_length], future_pred[..., :min_length])


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def find_pred_len_from_path(path: str) -> int:
    if "pl_96" or "pl96" in path:
        pred_len = 96
    elif "pl_192" or "pl192" in path:
        pred_len = 192
    elif "pl_336" or "pl336" in path:
        pred_len = 336
    elif "pl720" or "pl720" in path:
        pred_lent = 720
    else:
        raise ValueError(
            f"Could not determine prediction length of model from path {path}. Expected path to contain a substring of the form 'pl_{{pred_len}}' or 'pl{{pred_len}}'."
        )

    return pred_len


def find_model_name_from_path(path: str) -> str:
    path = path.lower()
    if "time-llm" in path or "timellm" in path:
        model_name = "time-llm"
    elif "unitime" in path:
        model_name = "unitime"
    else:
        raise ValueError(
            f"Could not determine model name from path {path}. Expected path to contain either 'time-llm', 'timellm', or 'unitime'."
        )

    return model_name


TIME_LLM_CONFIGS = DotDict(
    {
        "task_name": "long_term_forecast",
        "seq_len": 512,
        "enc_in": 7,
        "d_model": 32,
        "d_ff": 128,
        "llm_layers": 32,
        "llm_dim": 4096,
        "patch_len": 16,
        "stride": 8,
        "llm_model": "LLAMA",
        "llm_layers": 32,
        "prompt_domain": 1,
        "content": None,
        "dropout": 0.1,
        "d_model": 32,
        "n_heads": 8,
        "enc_in": 7,
    }
)


class TimeLLMWrapper(nn.Module):

    def __init__(self, time_llm_model):
        super().__init__()

        assert isinstance(
            time_llm_model, TimeLLMModel
        ), f"TimeLLMWrapper can only wrap a model of class TimeLLM.Model but got {type(time_llm_model)}"
        self.base_model = time_llm_model

    def forward(self, past_time, context):
        self.base_model.description = context
        return self.base_model(
            x_enc=past_time.unsqueeze(-1), x_mark_enc=None, x_dec=None, x_mark_dec=None
        ).squeeze(-1)


class WrappedBaseline(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.base_model = model
        if isinstance(self.base_model, TimeLLMModel):
            self.wrapped_model = TimeLLMWrapper(self.base_model)
        else:
            raise ValueError(
                f"WrappedBaseline can only wrap a model of class TimeLLM.Model but got {type(model)}"
            )

    def forward(self, past_time, context):
        return self.wrapped_model(past_time, context)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict, assign)


class EvaluationPipeline:

    def __init__(
        self,
        model: TimeLLMModel,
        metrics: Optional[Union[Callable, Dict[str, Callable]]] = None,
    ):
        self.metrics = (
            metrics if metrics is not None else {"mse_loss": truncate_mse_loss}
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            warnings.warn(
                "Warning: No CUDA device detected, proceeding with EvaluationPipeline on CPU ....."
            )

        self.model = WrappedBaseline(model).to(self.device)

    # TODO: This method needs to be replaced to handle actual CiK benchmark
    def get_evaluation_loader(self) -> Iterable:
        samples = []
        for sample in self.dataset.values():
            past_time = (
                torch.from_numpy(sample["past_time"].to_numpy().T)
                .float()
                .to(self.device)
            )
            future_time = (
                torch.from_numpy(sample["future_time"].to_numpy().T)
                .float()
                .to(self.device)
            )
            context = sample["context"]

            samples.append([past_time, future_time, context])

        return samples

    def compute_loss(self, future_time, future_pred):
        return {
            m_name: m(future_time, future_pred) for m_name, m in self.metrics.items()
        }

    def evaluation_step(self, past_time, future_time, context):
        with torch.no_grad():
            future_pred = self.model(past_time, context)
            loss = self.compute_loss(future_time, future_pred)
        return loss, future_pred

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        infer_dataloader = self.get_evaluation_loader()
        losses, predictions = {m_name: [] for m_name in self.metrics.keys()}, []
        for past_time, future_time, context in infer_dataloader:
            loss_dict, preds = self.evaluation_step(past_time, future_time, context)

            for m_name, loss in loss_dict.items():
                losses[m_name].append(loss)
            predictions.append(preds)

        self.model.train()
        return losses, predictions


class TimeLLMForecaster(Baseline):

    __version__ = "0.0.8"  # Modification will trigger re-caching

    def __init__(
        self,
        use_context,
        dataset="ETTh2",
        pred_len=96,
        seed: int = 42,
        hf_repo="transfer-starcaster/time-llm-starcaster",
    ):

        self.use_context = use_context
        self.dataset = dataset
        self.pred_len = pred_len
        self.seed = seed

        set_seed(self.seed)
        ckpt_filename = f"TimeLLM-{dataset}-pl_{pred_len}-ckpt.pth"

        # Get the directory of the current script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the time-llm checkpoints directory
        ckpt_dir = os.path.join(script_dir, "Time-LLM", "checkpoints")

        # Create the time-llm checkpoints directory if it doesn't exist
        os.makedirs(ckpt_dir, exist_ok=True)

        # Path to the local checkpoint file
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

        # Check if the checkpoint exists locally, otherwise download it
        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_filename)

        args = DotDict(dict())

        args.pred_len = 96
        args.model_name = "time-llm"  # "unitime"
        args.seed = seed
        self.model_name = args.model_name

        if args.model_name == "time-llm":
            args.update(TIME_LLM_CONFIGS)

        print(f"Initializing model from config:\n{args} .....")

        if args.model_name == "time-llm":
            self.model = TimeLLMModel(args).to(torch_device)
            self.backbone = TIME_LLM_CONFIGS.llm_model

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(
                ckpt["module"]
            )  # TODO: Change this to not be specific to the Time-LLM checkpoint        else:
        super().__init__()

    def __call__(
        self,
        task_instance: BaseTask,
        n_samples: Optional[int] = 1,
    ):
        set_seed(self.seed)
        self.model.pred_len = task_instance.future_time.shape[0]
        pipeline = EvaluationPipeline(
            self.model,
            metrics={"mse_loss": truncate_mse_loss, "mae_loss": truncate_mae_loss},
        )

        if self.use_context:
            context = self._make_prompt(task_instance)
        else:
            context = ""
        past_time = (
            torch.tensor(
                task_instance.past_time[[task_instance.past_time.columns[-1]]]
                .to_numpy()
                .transpose(),  # (1, len(past_time))
                dtype=torch.float32,
            )
            .expand(n_samples, -1)
            .to(torch_device)
        )
        future_time = (
            torch.tensor(
                task_instance.future_time[[task_instance.future_time.columns[-1]]]
                .to_numpy()
                .transpose(),  # (1, len(future_time))
                dtype=torch.float32,
            )
            .expand(n_samples, -1)
            .to(torch_device)
        )
        # non-determinism inherent to the model/GPU
        # We get samples from the model itself
        batch_size = 25
        subgroup_size = 5
        predictions = []

        for i in range(0, batch_size, subgroup_size):
            past_time_subgroup = past_time[i : i + subgroup_size]
            future_time_subgroup = future_time[i : i + subgroup_size]
            context_subgroup = context

            _, preds_subgroup = pipeline.evaluation_step(
                past_time_subgroup,
                future_time_subgroup,
                context_subgroup,
            )
            predictions.append(preds_subgroup)

        prediction_tensor = torch.cat(predictions, dim=0)
        if prediction_tensor.shape[-1] < future_time.shape[-1]:
            last_value = prediction_tensor[:, -1].unsqueeze(-1)
            repeat_count = future_time.shape[-1] - prediction_tensor.shape[-1]
            prediction_tensor = torch.cat(
                [prediction_tensor, last_value.repeat(1, repeat_count)], dim=-1
            )

        prediction_tensor = prediction_tensor.unsqueeze(-1)

        return prediction_tensor.cpu().numpy()

    def _make_prompt(self, task_instance):
        """
        Formats the prompt and adds it to the LLMP arguments

        """
        prompt = f"""
        Forecast the future values of this time series, while considering the following
        background knowledge, scenario, and constraints.

        Background knowledge:
        {task_instance.background}

        Scenario:
        {task_instance.scenario}

        Constraints:
        {task_instance.constraints}

        """

        return prompt

    @property
    def cache_name(self) -> str:
        args_to_include = [
            "model_name",
            "backbone",
            "use_context",
            "dataset",
            "pred_len",
        ]
        return f"{self.__class__.__name__}_" + "_".join(
            [f"{k}={getattr(self, k)}" for k in args_to_include]
        )


# if __name__ == "__main__":

#     class DummyTask:
#         def __init__(self):
#             self.past_time = pd.Series(
#                 np.random.randn(100), index=pd.date_range("20210101", periods=100)
#             ).to_frame()
#             self.future_time = pd.Series(
#                 np.random.randn(10), index=pd.date_range("20210501", periods=10)
#             ).to_frame()
#             self.background = "The background is this"
#             self.scenario = "The scenario is this"
#             self.constraints = "The constraints are this"

#     task_instance = DummyTask()

#     dataset = "ETTh1"
#     pred_len = 96
#     forecaster = TimeLLMForecaster(dataset, pred_len, seed=42)
#     predictions = forecaster(task_instance)
#     print(predictions)
