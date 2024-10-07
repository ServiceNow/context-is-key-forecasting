from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
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
    elif "mixtral" in llm_type or "gemma" in llm_type:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
        )
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
