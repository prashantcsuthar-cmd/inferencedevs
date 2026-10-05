import torch

from src.schema.states import State, GrammarState
from src.decoding.logits_processor import ICELogitsProcessor


class MockCompatibilityEngine:
    """Mock engine that cleanly implements signature compliance."""

    def get_valid_tokens(self, state):
        if state.grammar_state == GrammarState.EXPECT_KEY_QUOTE:
            return [1, 3]
        return [0]


def test_illegal_tokens_are_masked():
    mock_engine = MockCompatibilityEngine()
    processor = ICELogitsProcessor(mock_engine)

    active_state = State(GrammarState.EXPECT_KEY_QUOTE)
    processor.set_state(active_state)

    scores = torch.tensor([[1.0, 2.0, 1.5, 4.0, 0.5]], dtype=torch.float32)
    input_ids = torch.tensor([[10]], dtype=torch.long)

    masked_scores = processor(input_ids, scores)

    # Valid tokens retained
    assert masked_scores[0, 1].item() == 2.0
    assert masked_scores[0, 3].item() == 4.0

    # Invalid tokens masked to -inf
    assert masked_scores[0, 0].item() == float("-inf")
    assert masked_scores[0, 2].item() == float("-inf")
    assert masked_scores[0, 4].item() == float("-inf")