# Context is Key: A Benchmark for Forecasting with Essential Textual Information

📝 [[Paper]](https://arxiv.org/abs/2410.18959) 🌐 [[Website]](https://servicenow.github.io/context-is-key-forecasting) ✉️ [[Contact]](mailto:arjun.ashok@servicenow.com,andrew.williams1@servicenow.com,alexandre.drouin@servicenow.com) ⭐ [[Citation]](#citing-this-work)

![banner](https://github.com/user-attachments/assets/ca15c5f6-a1cc-4db0-891b-767dad4f241e)


## Overview of code

* **Baselines:** All baseline code can be found [here](./benchmark/baselines).
* **Tasks:** All baseline code can be found [here](./benchmark/tasks).
* **Metrics:** All metric-related code can be found [here](./benchmark/metrics).
* **Experiments:** Code used to run the experiments can be found [here](./experiments).


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

## Citing this work

Please cite the following paper:
```
@misc{williams2024contextkeybenchmarkforecasting,
      title={Context is Key: A Benchmark for Forecasting with Essential Textual Information}, 
      author={Andrew Robert Williams and Arjun Ashok and Étienne Marcotte and Valentina Zantedeschi and Jithendaraa Subramanian and Roland Riachi and James Requeima and Alexandre Lacoste and Irina Rish and Nicolas Chapados and Alexandre Drouin},
      year={2024},
      eprint={2410.18959},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.18959}, 
}
```
