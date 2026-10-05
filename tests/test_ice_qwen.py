import torch
from transformers import LogitsProcessorList
from src.models.loader import load_qwen
from src.decoding.logits_processor import ICELogitsProcessor


class FakeCompatibilityEngine:
    """
    Temporary compatibility engine for integration testing.

    Later this will be replaced by the real
    grammar-based compatibility engine.
    """

    def get_valid_tokens(self, state):
        # Allow only a small set of token IDs
        return [10, 20, 30, 40]


def test_ice_with_qwen():

    print("\nLoading Qwen model...")

    model, tokenizer = load_qwen()

    print("Qwen loaded.")

    # --------------------------------------------------
    # 1. Create compatibility engine
    # --------------------------------------------------

    compatibility_engine = FakeCompatibilityEngine()

    # --------------------------------------------------
    # 2. Create ICE processor
    # --------------------------------------------------

    processor = ICELogitsProcessor(
        compatibility_engine
    )

    processor.set_state("TEST_STATE")

    # --------------------------------------------------
    # 3. Tokenize prompt
    # --------------------------------------------------

    prompt = "Return a JSON object."

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # --------------------------------------------------
    # 4. Run model once
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    logits = outputs.logits[:, -1, :]

    print("\nOriginal logits shape:")
    print(logits.shape)

    # --------------------------------------------------
    # 5. Apply ICE
    # --------------------------------------------------

    masked_logits = processor(
        inputs["input_ids"],
        logits
    )

    print("\nMasked logits shape:")
    print(masked_logits.shape)

    # --------------------------------------------------
    # 6. Verify masking
    # --------------------------------------------------

    print("\nChecking ICE masking...")

    for token_id in range(masked_logits.shape[-1]):

        if token_id in [10, 20, 30, 40]:

            assert masked_logits[0, token_id] == logits[0, token_id]

        else:

            assert masked_logits[0, token_id] == float("-inf")

    print("ICE successfully masked Qwen logits!")

        # ------------------------------------------------------------
    # 7. Test ICE inside the actual model.generate() loop
    # ------------------------------------------------------------
    print("\nTesting ICE with model.generate()...")

    generation_processors = LogitsProcessorList([
        processor
    ])

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=generation_processors,
        )

    print("Generated token IDs:", generated)

    # The newly generated token should be one of the
    # tokens allowed by our FakeCompatibilityEngine.
    generated_token_id = generated[0, -1].item()

    assert generated_token_id in [10, 20, 30, 40]

    print(
        "Generated token is allowed by ICE:",
        generated_token_id
    )

    print("model.generate() integration PASSED!")

    print("\nIntegration test PASSED!")