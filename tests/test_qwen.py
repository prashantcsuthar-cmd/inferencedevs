"""Opt-in real-model smoke test.

Run with ICE_RUN_MODEL_TESTS=1 after installing dependencies and downloading
(or allowing download of) the configured model.
"""
import os
import pytest

if os.getenv("ICE_RUN_MODEL_TESTS") != "1":
    pytest.skip("Set ICE_RUN_MODEL_TESTS=1 to run the real-model test.", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from src.models.loader import load_qwen, format_chat_prompt


def test_qwen_loads_and_generates():
    model, tokenizer = load_qwen()
    prompt = format_chat_prompt(tokenizer, "Return a JSON object containing a name.")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
    assert outputs.shape[-1] > inputs["input_ids"].shape[-1]
