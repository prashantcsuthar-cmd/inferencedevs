"""Load the configured causal language model for ICE.

Defaults to CPU for predictable local execution. Set ICE_DEVICE=auto to use the
model loader's default device placement, or ICE_DEVICE=cuda when CUDA is wanted.
Set ICE_LOCAL_FILES_ONLY=1 to require cached model/tokenizer files.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def format_chat_prompt(tokenizer, prompt: str) -> str:
    """Format a user prompt using the model's chat template when available."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template) and getattr(tokenizer, "chat_template", None):
        return apply_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_qwen():
    model_name = os.getenv("ICE_MODEL_NAME", DEFAULT_MODEL_NAME)
    local_only = os.getenv("ICE_LOCAL_FILES_ONLY", "0").strip().lower() in {
        "1", "true", "yes"
    }
    device_choice = os.getenv("ICE_DEVICE", "cpu").strip().lower()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        local_files_only=local_only,
    )

    if device_choice == "cpu":
        model.to("cpu")
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("ICE_DEVICE=cuda was set, but CUDA is unavailable.")
        model.to("cuda")
    elif device_choice != "auto":
        raise ValueError("ICE_DEVICE must be one of: cpu, cuda, auto.")

    model.eval()
    return model, tokenizer
