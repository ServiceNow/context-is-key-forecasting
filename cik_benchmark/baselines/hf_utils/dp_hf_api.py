from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizerFast,
    MistralForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    Mamba2Config,
    Mamba2ForCausalLM,
)
import torch
import gc

LLM_MAP = {
    "llama-2-7B": "meta-llama/Llama-2-7b",
    "llama-2-70B": "meta-llama/Llama-2-70b",
    "llama-3-8B": "meta-llama/Meta-Llama-3-8B",
    "llama-3-8B-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70B": "meta-llama/Meta-Llama-3-70B",
    "llama-3-70B-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "mixtral-8x7B": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7B-instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "phi-3-mini-128k-instruct": "microsoft/Phi-3-mini-128k-instruct",
    "gemma-2-9B": "google/gemma-2-9b",
    "gemma-2-9B-instruct": "google/gemma-2-9b-it",
    "gemma-2-27B": "google/gemma-2-27b",
    "gemma-2-27B-instruct": "google/gemma-2-27b-it",
    # New models for the rebuttal
    "qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-32B-Instruct": "Qwen/Qwen2.5-32B-Instruct",
    "qwen2.5-72B-Instruct": "Qwen/Qwen2.5-72B-Instruct",
    "Mistral-Small-Instruct-2409": "mistralai/Mistral-Small-Instruct-2409",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
    "falcon-40b-instruct": "tiiuae/falcon-40b-instruct",
    # SSMs
    # "Hymba-1.5B-Base": "nvidia/Hymba-1.5B-Base", # needs FlexAttention installation
    # "Hymba-1.5B-Instruct": "nvidia/Hymba-1.5B-Instruct", # needs FlexAttention installation
    # "mamba-2.8B": "state-spaces/mamba-2.8b-hf",
    # "mamba-1.4B": "state-spaces/mamba-1.4b-hf",
    # "mamba-790m": "state-spaces/mamba-790m-hf",
    # "mamba-370m": "state-spaces/mamba-370m-hf",
    # "mamba-130m": "state-spaces/mamba-130m-hf",
    # "mamba2-2.7B": "state-spaces/mamba2-2.7b",
    # "mamba2-1.3B": "state-spaces/mamba2-1.3b",
    # "mamba2-780m": "state-spaces/mamba2-780m",
    # "mamba2-370m": "state-spaces/mamba2-370m",
    # "mamba2-130m": "state-spaces/mamba2-130m",
    "Zamba-7B-v1": "Zyphra/Zamba-7B-v1",
    # "Zamba2-7B": "Zyphra/Zamba2-7B",
    # "Zamba2-2.7B": "Zyphra/Zamba2-2.7B",
    # "Zamba2-1.2B": "Zyphra/Zamba2-1.2B",
    # "Zamba2-7B-Instruct": "Zyphra/Zamba2-7B-Instruct",
    # "Zamba2-2.7B-Instruct": "Zyphra/Zamba2-2.7B-Instruct",
    # "Zamba2-1.2B-Instruct": "Zyphra/Zamba2-1.2B-Instruct",
    # "Falcon3-Mamba-7B-Base": "tiiuae/Falcon3-Mamba-7B-Base",
    # "Falcon3-Mamba-7B-Instruct": "tiiuae/Falcon3-Mamba-7B-Instruct",
    # "Bamba-9B": "ibm-fms/Bamba-9B",
}


DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_tokenizer(llm_path, llm_type):
    if llm_path is None:
        llm_path = LLM_MAP[llm_type]
    if "llama-2" in llm_type:
        tokenizer = LlamaTokenizer.from_pretrained(
            llm_path, use_fast=False, padding_side="left"
        )
    elif "llama-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path, padding_side="left", legacy=False
        )
    elif "phi-3" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
        )
    elif "mixtral" in llm_type or "gemma" in llm_type or "qwen" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
    elif "Mistral-Small" in llm_type:
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "mistralai/Mistral-Small-Instruct-2409"
        )
        tokenizer.pad_token = tokenizer.eos_token
    elif "Mistral-7B" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
    elif "falcon" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
    elif "mamba-" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
    elif "Zamba2-" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
    elif "Zamba-" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(llm_path)
    else:
        assert False

    special_tokens_dict = dict()
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model_and_tokenizer(llm_path, llm_type):
    if llm_path is None:
        llm_path = LLM_MAP[llm_type]
    tokenizer = get_tokenizer(llm_path, llm_type)
    if "llama-2" in llm_type:
        model = LlamaForCausalLM.from_pretrained(
            llm_path, device_map="auto", torch_dtype=torch.float16
        )
    elif "llama-3" in llm_type:
        model = LlamaForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    elif "phi-3" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
    elif "mixtral" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    elif "gemma" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    elif "qwen" in llm_type or "Mistral-7B" in llm_type or "falcon" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif "Mistral-Small" in llm_type:
        model = MistralForCausalLM.from_pretrained(
            llm_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif "mamba-" in llm_type:
        model = MambaForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif "mamba2-" in llm_type:
        model = Mamba2ForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    elif "Zamba2-" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    elif "Zamba-" in llm_type:
        model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        assert False

    model.eval()
    return model, tokenizer


# This assumes that there is only a single prompt and it gets replicated batch_size times
@torch.inference_mode()
def hf_generate(model, tokenizer, input_str, batch_size, temp, top_p, max_new_tokens):
    batch = tokenizer([input_str], return_tensors="pt")
    batch = {k: v.repeat(batch_size, 1) for k, v in batch.items()}
    batch = {k: v.cuda() for k, v in batch.items()}
    num_input_ids = batch["input_ids"].shape[1]

    generate_ids = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        renormalize_logits=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Clean up to free memory
    del batch
    del generate_ids
    gc.collect()
    torch.cuda.empty_cache()

    return gen_strs


# this assumes a batch of different prompts that may be different lengths
@torch.inference_mode()
def hf_generate_batch(model, tokenizer, prompts, temp, top_p, max_new_tokens):
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    num_input_ids = batch["input_ids"].shape[1]

    generate_ids = model.generate(
        **batch,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=temp,
        top_p=top_p,
        renormalize_logits=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_strs = tokenizer.batch_decode(
        generate_ids[:, num_input_ids:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Clean up to free memory
    del batch
    del generate_ids
    gc.collect()
    torch.cuda.empty_cache()

    return gen_strs
