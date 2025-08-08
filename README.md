# Context is Key: A Benchmark for Forecasting with Essential Textual Information

📄 [Paper (ICML 2025)](https://arxiv.org/abs/2410.18959) -
🌐 [Website](https://servicenow.github.io/context-is-key-forecasting) -
✉️ [Contact](mailto:arjun.ashok@servicenow.com,andrew.williams1@servicenow.com,alexandre.drouin@servicenow.com) -
🌟 [Contributors](#contributors) -
📝 [Citation](#citing-this-work) -
📦 [ICML 2025 Release](https://github.com/ServiceNow/context-is-key-forecasting/tree/v1.0.0) -
📊 [Official Results](#official-results) -
🤗 [Hugging Face Dataset](https://huggingface.co/datasets/ServiceNow/context-is-key)

![poster](https://github.com/user-attachments/assets/523ae60a-6c3d-42bf-80b8-83e23d8e7ab0)

## Overview of code

* **Baselines:** All baseline code can be found [here](./cik_benchmark/baselines).
* **Tasks:** All baseline code can be found [here](./cik_benchmark/tasks).
* **Metrics:** All metric-related code can be found [here](./cik_benchmark/metrics).
* **Experiments:** Code used to run the experiments can be found [here](./experiments).


## Official results

Here are the updated aggregated benchmark results (equivalent to Table 1 in the paper).
The full experimental results can also be found [here](./results/results_complete.csv).

This table holds for the version of the benchmark as of July 11th, 2025.
See the [CHANGELOG](./CHANGELOG) for how the benchmark was updated since the ICML 2025 release.

|                                         | Average RCPRS   | Intemporal Information   | Historical Information   | Future Information   | Covariate Information   | Causal Information   |
|:----------------------------------------|:----------------|:-------------------------|:-------------------------|:---------------------|:------------------------|:---------------------|
| Direct Prompt - Llama-3.1-405B-Instruct | 0.143 ± 0.006   | 0.174 ± 0.010            | 0.146 ± 0.001            | 0.075 ± 0.005        | 0.144 ± 0.008           | 0.303 ± 0.037        |
| Direct Prompt - Llama-3-70B-Instruct    | 0.286 ± 0.004   | 0.336 ± 0.006            | 0.180 ± 0.003            | 0.193 ± 0.006        | 0.228 ± 0.004           | 0.629 ± 0.019        |
| Direct Prompt - Mixtral-8x7B-Instruct   | 0.519 ± 0.009   | 0.723 ± 0.014            | 0.236 ± 0.002            | 0.241 ± 0.001        | 0.353 ± 0.004           | 0.850 ± 0.012        |
| Direct Prompt - Qwen-2.5-7B-Instruct    | 0.290 ± 0.003   | 0.290 ± 0.004            | 0.176 ± 0.003            | 0.287 ± 0.007        | 0.240 ± 0.002           | 0.524 ± 0.003        |
| Direct Prompt - Qwen-2.5-0.5B-Instruct  | 0.463 ± 0.018   | 0.609 ± 0.028            | 0.165 ± 0.004            | 0.218 ± 0.012        | 0.476 ± 0.022           | 0.428 ± 0.006        |
| Direct Prompt - GPT-4o                  | 0.191 ± 0.004   | 0.218 ± 0.007            | 0.118 ± 0.001            | 0.121 ± 0.001        | 0.143 ± 0.003           | 0.363 ± 0.011        |
| Direct Prompt - GPT-4o-mini             | 0.354 ± 0.004   | 0.475 ± 0.007            | 0.139 ± 0.002            | 0.143 ± 0.002        | 0.341 ± 0.004           | 0.645 ± 0.011        |
| LLMP - Llama3-70B-Instruct              | 0.540 ± 0.013   | 0.438 ± 0.018            | 0.516 ± 0.029            | 0.847 ± 0.024        | 0.547 ± 0.016           | 0.396 ± 0.028        |
| LLMP - Llama3-70B                       | 0.237 ± 0.006   | 0.212 ± 0.005            | 0.121 ± 0.008            | 0.299 ± 0.017        | 0.194 ± 0.004           | 0.365 ± 0.011        |
| LLMP - Mixtral-8x7B-Instruct            | 0.265 ± 0.004   | 0.242 ± 0.007            | 0.173 ± 0.004            | 0.324 ± 0.005        | 0.219 ± 0.005           | 0.440 ± 0.007        |
| LLMP - Mixtral-8x7B                     | 0.264 ± 0.008   | 0.250 ± 0.008            | 0.119 ± 0.003            | 0.310 ± 0.019        | 0.231 ± 0.006           | 0.466 ± 0.011        |
| LLMP - Qwen-2.5-7B-Instruct             | 1.974 ± 0.019   | 2.509 ± 0.030            | 2.857 ± 0.056            | 1.653 ± 0.008        | 1.702 ± 0.023           | 1.333 ± 0.080        |
| LLMP - Qwen-2.5-7B                      | 0.910 ± 0.034   | 1.149 ± 0.043            | 1.002 ± 0.053            | 0.601 ± 0.071        | 0.640 ± 0.044           | 0.929 ± 0.016        |
| LLMP - Qwen-2.5-0.5B-Instruct           | 1.937 ± 0.011   | 2.444 ± 0.017            | 1.960 ± 0.063            | 1.443 ± 0.010        | 1.805 ± 0.012           | 1.201 ± 0.017        |
| LLMP - Qwen-2.5-0.5B                    | 1.995 ± 0.011   | 2.546 ± 0.018            | 2.082 ± 0.052            | 1.579 ± 0.015        | 1.821 ± 0.013           | 1.225 ± 0.013        |
| UniTime                                 | 0.370 ± 0.001   | 0.457 ± 0.002            | 0.155 ± 0.000            | 0.194 ± 0.003        | 0.395 ± 0.001           | 0.423 ± 0.001        |
| Time-LLM (ETTh1)                        | 0.475 ± 0.001   | 0.517 ± 0.002            | 0.183 ± 0.000            | 0.403 ± 0.002        | 0.441 ± 0.001           | 0.481 ± 0.002        |
| Lag-Llama                               | 0.327 ± 0.004   | 0.330 ± 0.005            | 0.167 ± 0.005            | 0.293 ± 0.008        | 0.294 ± 0.004           | 0.494 ± 0.014        |
| Chronos-Large                           | 0.326 ± 0.001   | 0.314 ± 0.002            | 0.179 ± 0.003            | 0.379 ± 0.003        | 0.255 ± 0.002           | 0.460 ± 0.004        |
| TimeGEN                                 | 0.353 ± 0.000   | 0.332 ± 0.000            | 0.177 ± 0.000            | 0.405 ± 0.000        | 0.292 ± 0.000           | 0.474 ± 0.000        |
| Moirai-Large                            | 0.520 ± 0.006   | 0.596 ± 0.009            | 0.140 ± 0.001            | 0.431 ± 0.002        | 0.499 ± 0.007           | 0.438 ± 0.011        |
| ARIMA                                   | 0.475 ± 0.006   | 0.557 ± 0.009            | 0.200 ± 0.007            | 0.350 ± 0.003        | 0.375 ± 0.006           | 0.440 ± 0.011        |
| ETS                                     | 0.530 ± 0.009   | 0.639 ± 0.014            | 0.362 ± 0.014            | 0.315 ± 0.006        | 0.402 ± 0.010           | 0.507 ± 0.017        |
| Exponential Smoothing                   | 0.605 ± 0.013   | 0.703 ± 0.020            | 0.493 ± 0.016            | 0.397 ± 0.006        | 0.480 ± 0.015           | 0.828 ± 0.060        |

## Setting environment variables

Here is a list of all environment variables which the Context-is-Key benchmark will access:

| Variable Name               | Description                                                                                     | Default Value                        |
|-----------------------------|-------------------------------------------------------------------------------------------------|--------------------------------------|
| **CIK_MODEL_STORE**         | Folder to store model weights for the baselines.                                                | `./models`                           |
| **CIK_DATA_STORE**          | Folder to store downloaded datasets.                                                            | `./data`                     |
| **CIK_DOMINICK_STORE**      | Folder to store the Dominick dataset for specific tasks.                                        | `CIK_DATA_STORE + /dominicks`        |
| **CIK_TRAFFIC_DATA_STORE**  | Folder to store the Traffic dataset for specific tasks.                                         | `CIK_DATA_STORE + /traffic_data`     |
| **HF_HOME**                 | Cache location for downloading datasets from Hugging Face.                                      | `CIK_DATA_STORE + /hf_cache`         |
| **CIK_RESULT_CACHE**        | Folder to store the output of baselines to avoid recomputation.                                 | `./inference_cache`                  |
| **CIK_METRIC_SCALING_CACHE**| Folder to store scaling factors for each task to avoid recomputation.                           | `./metric_scaling_cache`             |
| **CIK_METRIC_COMPUTE_VARIANCE** | If set, computes an estimate of the variance of the metric.                              | Only compute metric itself by default|
| **CIK_OPENAI_USE_AZURE**    | If set to "True", use Azure client instead of OpenAI client for baselines using OpenAI models.  | `False`                              |
| **CIK_OPENAI_API_KEY**      | API key for accessing OpenAI models.                                    | None (Required for baseline)         |
| **CIK_OPENAI_API_VERSION**  | API version for OpenAI models when using the Azure client.                                     | None                                 |
| **CIK_OPENAI_AZURE_ENDPOINT** | Azure endpoint for calling OpenAI models.                                                    | None                                 |
| **CIK_LLAMA31_405B_URL**    | API URL for the Llama-3.1-405b baseline.                                | None (Required for baseline)         |
| **CIK_LLAMA31_405B_API_KEY**| API key for the Llama-3.1-405b API.                                     | None (Required for baseline)         |
| **CIK_NIXTLA_BASE_URL**     | Azure API URL for the Nixtla TimeGEN baseline.                             | None (Required for baseline)         |
| **CIK_NIXTLA_API_KEY**          | Azure API key for the Nixtla TimeGEN baseline.                          | None (Required for baseline)         |

## 🌟 Contributors

[![CiK contributors](https://contrib.rocks/image?repo=ServiceNow/context-is-key-forecasting&max=2000)](https://github.com/ServiceNow/context-is-key-forecasting/graphs/contributors)

## Citing this work

Please cite the following paper:
```
@inproceedings{
williams2025context,
title={Context is Key: A Benchmark for Forecasting with Essential Textual Information},
author={Andrew Robert Williams and Arjun Ashok and {\'E}tienne Marcotte and Valentina Zantedeschi and Jithendaraa Subramanian and Roland Riachi and James Requeima and Alexandre Lacoste and Irina Rish and Nicolas Chapados and Alexandre Drouin},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=ih2WuBT1Fn}
}
```
