import pandas as pd
import numpy as np
from .base import Baseline
from ..base import BaseTask
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from transformers import set_seed
import os
import sys
from typing import Optional, Union, Dict, Callable, Iterable

import warnings
import logging
import json
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path to the 'models' directory inside 'Time-LLM'
unitime_models_path = os.path.join(script_dir, "UniTime", "models")
unitime_path = os.path.join(script_dir, "UniTime")

# Add 'models' directory to sys.path
sys.path.append(unitime_models_path)
sys.path.append(unitime_path)

from unitime.models.unitime import UniTime as UniTimeModel

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


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
UNITIME_CONFIGS = DotDict(
    {
        "max_token_num": 17,
        "mask_rate": 0.5,
        "patch_len": 16,
        "max_backcast_len": 96,
        "max_forecast_len": 720,
        "logger": logger,
        "model_path": "gpt2",
        "lm_layer_num": 6,
        "lm_ft_type": "freeze",
        "ts_embed_dropout": 0.3,
        "dec_trans_layer_num": 2,
        "dec_head_dropout": 0.1,
    }
)


class UniTimeWrapper(nn.Module):

    def __init__(self, unitime_model):
        super().__init__()

        assert isinstance(
            unitime_model, UniTimeModel
        ), f"UniTimeWrapper can only wrap a model of class UniTimeModel but got {type(unitime_model)}"
        self.base_model = unitime_model

    def forward(self, past_time, context):
        past_time = past_time.unsqueeze(-1)
        mask = torch.ones_like(past_time)
        data_id = -1
        seq_len = past_time.shape[1]
        stride = 16

        info = (data_id, seq_len, stride, context)
        return self.base_model(info=info, x_inp=past_time, mask=mask).squeeze(-1)


class UniTimeBaseline(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.base_model = model
        if isinstance(self.base_model, UniTimeModel):
            self.wrapped_model = UniTimeWrapper(self.base_model)
        else:
            raise ValueError(
                f"UniTime can only wrap a model of class UniTimeModel but got {type(model)}"
            )

    def forward(self, past_time, context):
        return self.wrapped_model(past_time, context)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return self.base_model.load_state_dict(state_dict, strict, assign)


class EvaluationPipeline:

    def __init__(
        self,
        model: UniTimeModel,
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

        self.model = UniTimeBaseline(model).to(self.device)

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


class UniTimeForecaster(Baseline):

    __version__ = "0.0.3"  # Modification will trigger re-caching

    def __init__(
        self,
        use_context,
        dataset="etth1",
        pred_len=96,
        per_dataset_checkpoint=False,
        hf_repo="transfer-starcaster/unitime-starcaster",
        seed: int = 42,
    ):
        self.use_context = use_context
        self.dataset = dataset
        self.pred_len = pred_len
        self.per_dataset_checkpoint = per_dataset_checkpoint
        self.seed = seed

        if per_dataset_checkpoint:
            ckpt_filename = f"UniTime-{dataset}-pl_{pred_len}-ckpt.pth"
        else:
            ckpt_filename = "UniTime-unified-max_pl_720-max_context_128-ckpt.pth"
            max_token_num_str = ckpt_filename.split("context_")[1].split("-")[0]
            self.max_token_num = int(max_token_num_str)
            UNITIME_CONFIGS.max_token_num = self.max_token_num
            self.dataset = "unified"

        # Get the directory of the current script file
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the UniTime checkpoints directory
        ckpt_dir = os.path.join(script_dir, "UniTime", "checkpoints")

        # Create the UniTime checkpoints directory if it doesn't exist
        os.makedirs(ckpt_dir, exist_ok=True)

        # Path to the local checkpoint file
        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

        # Check if the checkpoint exists locally, otherwise download it
        if not os.path.exists(ckpt_path):
            ckpt_path = hf_hub_download(repo_id=hf_repo, filename=ckpt_filename)

        set_seed(self.seed)
        args = DotDict(dict())

        args.pred_len = 96
        args.model_name = "unitime"
        args.seed = seed
        self.model_name = args.model_name

        args.update(UNITIME_CONFIGS)

        print(f"Initializing model from config:\n{args} .....")

        self.model = UniTimeModel(args).to(torch_device)
        self.backbone = UNITIME_CONFIGS.model_path

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            self.model.load_state_dict(
                ckpt
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
        _, predictions = pipeline.evaluation_step(
            past_time,
            future_time,
            context,
        )

        predictions = predictions.unsqueeze(-1)

        predictions = predictions[
            :,
            UNITIME_CONFIGS.max_backcast_len : UNITIME_CONFIGS.max_backcast_len
            + len(task_instance.future_time),
            :,
        ]

        return predictions.cpu().numpy()

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
            "max_token_num",
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

#     dataset = "etth1"
#     pred_len = 96
#     forecaster = UniTimeForecaster(dataset, pred_len, seed=42)
#     predictions = forecaster(task_instance)
#     print(predictions)
