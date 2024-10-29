import logging
import numpy as np
import os
import pandas as pd
import torch
import time

from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from lag_llama.gluon.estimator import LagLlamaEstimator

from .utils import torch_default_device
from ..config import MODEL_STORAGE_PATH

LAG_LLAMA_WEIGHTS_PATH = f"{MODEL_STORAGE_PATH}/lag-llama.ckpt"
if not os.path.exists(LAG_LLAMA_WEIGHTS_PATH):
    logging.info("Downloading Lag-Llama weights...")
    os.system(
        f"huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir {MODEL_STORAGE_PATH}"
    )
    logging.info("Lag-Llama weights downloaded.")


def get_lag_llama_predictions(
    dataset,
    prediction_length,
    device,
    batch_size=1,
    context_length=32,
    use_rope_scaling=False,
    num_samples=100,
    num_parallel_samples=100,
    seed=42,
):
    """
    Generates forecasts using the Lag-Llama model.

    Parameters:
    -----------
    dataset (Dataset): The dataset to generate predictions for.
    prediction_length (int): The number of timesteps to predict.
    device (str): The device to run the model on (e.g., 'cpu' or 'cuda').
    batch_size (int, optional): The batch size to use. Default is 1.
    context_length (int, optional): The context length for the model. Default is 32.
    use_rope_scaling (bool, optional): Whether to use ROPE scaling. Default is False.
    num_samples (int, optional): The number of samples to generate for each timestep. Default is 100.
    num_parallel_samples (int, optional): The number of parallel samples to generate for each timestep. This should be equal to or less than num_samples. Default is 100.

    Returns:
    tuple: A tuple containing:
        - forecasts (list): A list of forecast objects. Each forecast is of shape (num_samples, prediction_length).
        - tss (list): A list of time series objects with the ground truth corresponding to the forecasts. Each time series is of shape (prediction length,).
        - extra_info (dict): A dictionary containing timing information.

    """
    logging.info("Generating forecasts using Lag-Llama...")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ckpt = torch.load(
        LAG_LLAMA_WEIGHTS_PATH,
        map_location=device,
    )  # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(
            1.0, (context_length + prediction_length) / estimator_args["context_length"]
        ),
    }

    estimator = LagLlamaEstimator(
        ckpt_path=LAG_LLAMA_WEIGHTS_PATH,
        prediction_length=prediction_length,
        context_length=context_length,  # Lag-Llama was trained with a context length of 32, but can work with any context length
        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,
        batch_size=batch_size,
        num_parallel_samples=num_parallel_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    start_inference = time.time()
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor, num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)
    end_inference = time.time()

    return forecasts, tss, {"inference_time": end_inference - start_inference}


def prepare_dataset(history, forecast):
    """
    Packages the dataset in the format expected by the Lag-Llama model.

    Parameters:
    -----------
    history: pd.DataFrame
        The historical time series.
    forecast: pd.DataFrame
        The future time series.

    Returns:
    --------
    PandasDataset: The dataset in the format expected by the Lag-Llama model.

    """
    logging.info("Preparing dataset for Lag-Llama...")
    # Making sure that both inputs have the same columns name, since otherwise the concat would fail
    assert (history.columns == forecast.columns).all()

    history = history.astype("float32")
    forecast = forecast.astype("float32")
    df = pd.concat((history, forecast), axis="index")
    # Create a PandasDataset
    ds = PandasDataset(dict(df))

    return ds


def lag_llama(task_instance, n_samples, batch_size=1, device=None):
    """
    Get Lag-Llama predictions for a given task instance.

    Parameters:
    -----------
    task_instance: Task
        The task instance to generate predictions for.
    n_samples: int
        The number of samples to generate for each prediction.
    batch_size: int, optional
        The batch size to use for inference. Default is 1.
    device: str, optional
        The device to run the model on (e.g., 'cpu' or 'cuda'). Default is None.

    Returns:
    --------
    np.ndarray: The generated predictions, shape=(n_samples, prediction_length, 1).

    """

    if device is None:
        device = torch_default_device()

    starting_time = time.time()

    # Package the dataset in the format expected by the Lag-Llama model
    dataset = prepare_dataset(
        task_instance.past_time[[task_instance.past_time.columns[-1]]],
        task_instance.future_time[[task_instance.future_time.columns[-1]]],
    )

    # Generate forecasts using the Lag-Llama model
    forecasts, _, extra_info = get_lag_llama_predictions(
        dataset=dataset,
        prediction_length=task_instance.future_time.shape[0],
        device=device,
        num_samples=n_samples,
        batch_size=batch_size,
    )
    dtype = task_instance.past_time.dtypes.iloc[0]
    samples = format_llama_predictions(forecasts, dtype)
    extra_info["total_time"] = time.time() - starting_time

    return samples, extra_info


lag_llama.__version__ = "0.0.2"  # Modification will trigger re-caching


def format_llama_predictions(forecasts, dtype):
    return np.stack([f.samples for f in forecasts], axis=-1).astype(dtype)
