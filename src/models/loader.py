"""Load the configured local causal language model.

Set ICE_MODEL_NAME to choose a compatible causal LM. Set ICE_LOCAL_FILES_ONLY=1
to prohibit Hugging Face Hub access and require the model/tokenizer to be cached.
"""
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def load_qwen():
    model_name = os.getenv("ICE_MODEL_NAME", DEFAULT_MODEL_NAME)
    local_only = os.getenv("ICE_LOCAL_FILES_ONLY", "0").strip().lower() in {"1", "true", "yes"}
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        local_files_only=local_only,
    )
    model.eval()
    return model, tokenizer
