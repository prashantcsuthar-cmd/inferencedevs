import torch

from src.decoding.logits_processor import ICELogitsProcessor


class FakeCompatibilityEngine:
    """
    Small fake compatibility engine used only for testing.

    Suppose our grammar says only token IDs
    1 and 3 are legal.
    """

    def get_valid_tokens(self, state):
        return [1, 3]


def test_illegal_tokens_are_masked():

    # Create fake compatibility engine
    compatibility_engine = FakeCompatibilityEngine()

    # Create ICE processor
    processor = ICELogitsProcessor(
        compatibility_engine
    )

    # We don't need a real grammar state for this test
    processor.set_state("TEST_STATE")

    # Fake model logits for 5 vocabulary tokens
    scores = torch.tensor([
        [1.0, 2.0, 3.0, 4.0, 5.0]
    ])

    # Apply ICE
    masked_scores = processor(
        input_ids=torch.tensor([[10]]),
        scores=scores
    )

    print("Original logits:")
    print(scores)

    print("\nMasked logits:")
    print(masked_scores)

    # Token IDs 1 and 3 are legal
    assert masked_scores[0, 1] == 2.0
    assert masked_scores[0, 3] == 4.0

    # Token IDs 0, 2 and 4 are illegal
    assert masked_scores[0, 0] == float("-inf")
    assert masked_scores[0, 2] == float("-inf")
    assert masked_scores[0, 4] == float("-inf")

    print("\nICE masking test PASSED!")