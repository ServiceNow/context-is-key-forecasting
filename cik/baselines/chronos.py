import pandas as pd
import numpy as np
import time
from .base import Baseline
from ..base import BaseTask

from chronos import ChronosPipeline
import torch

from transformers import set_seed


class ChronosForecaster(Baseline):

    __version__ = "0.1.0"  # Modification will trigger re-caching

    def __init__(self, model_size, seed=42):
        """
        Get predictions from a Chronos model.

        Notes:
        ------
        This model requires a seasonal periodicity, which it currently gets from a
        hard coded association from the data index frequency (hourly -> 24 hours periods).
        """
        self.seed = seed
        self.model_size = model_size
        super().__init__()

    def __call__(self, task_instance: BaseTask, n_samples: int) -> np.ndarray:
        starting_time = time.time()
        samples, extra_info = self.forecast(
            task_instance,
            n_samples=n_samples,
        )
        extra_info["total_time"] = time.time() - starting_time

        return samples, extra_info

    def forecast(
        self,
        task_instance,
        n_samples: int,
    ) -> np.ndarray:
        """
        This method allows a forecast to be done without requiring a complete BaseTask instance.
        This is primarily meant to be called inside a BaseTask constructor when doing rejection sampling or similar approaches.
        """
        # If there is no period, then disable the seasonal component of the model (seasonal_periods will be ignored)
        set_seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = ChronosPipeline.from_pretrained(
            f"amazon/chronos-t5-{self.model_size}",
            device_map=device,  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )

        hist_values = torch.tensor(
            task_instance.past_time[[task_instance.past_time.columns[-1]]].values,
            dtype=torch.bfloat16,
        ).flatten()

        start_inference = time.time()
        # num_series, num_samples, num_timesteps
        model_preds = pipeline.predict(
            context=hist_values,
            prediction_length=len(task_instance.future_time),
            num_samples=n_samples,
            limit_prediction_length=False,
        )
        end_inference = time.time()

        # (1, num_samples, num_timesteps, num_series)
        model_preds = model_preds.permute(1, 2, 0)

        return model_preds.cpu().numpy(), {
            "inference_time": end_inference - start_inference
        }

    @property
    def cache_name(self) -> str:
        return f"{self.__class__.__name__}_{self.model_size}"
