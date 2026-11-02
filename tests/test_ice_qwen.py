"""Opt-in model/LogitsProcessor integration smoke test.

This test checks the Hugging Face LogitsProcessor interface only. It uses a
fake compatibility engine and is not evidence that the grammar engine is
correct or that outputs satisfy a schema.
"""
import os
import pytest

if os.getenv("ICE_RUN_MODEL_TESTS") != "1":
    pytest.skip("Set ICE_RUN_MODEL_TESTS=1 to run the real-model test.", allow_module_level=True)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import LogitsProcessorList
from src.models.loader import load_qwen, format_chat_prompt
from src.decoding.logits_processor import ICELogitsProcessor


class FakeCompatibilityEngine:
    def get_valid_tokens(self, state):
        return [10, 20, 30, 40]


def test_logits_processor_with_qwen():
    model, tokenizer = load_qwen()
    processor = ICELogitsProcessor(FakeCompatibilityEngine())
    processor.set_state("TEST_STATE")
    prompt = format_chat_prompt(tokenizer, "Return a JSON object.")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits[:, -1, :]
    masked = processor(inputs["input_ids"], logits)

    allowed = {10, 20, 30, 40}
    assert all(torch.isfinite(masked[0, i]) == torch.isfinite(logits[0, i]) for i in allowed)
    assert all(torch.isneginf(masked[0, i]) for i in range(masked.shape[-1]) if i not in allowed)

    generated = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        logits_processor=LogitsProcessorList([processor]),
    )
    assert generated[0, -1].item() in allowed
