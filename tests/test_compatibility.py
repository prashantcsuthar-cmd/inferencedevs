from src.tokenizer.compatibility import CompatibilityEngine
from src.schema.states import State, GrammarState


class FakeTokenizer:
    """
    Tiny tokenizer used only for testing CompatibilityEngine.
    """

    def __init__(self):
        self.tokens = {
            0: "{",
            1: '"',
            2: "name",
            3: ":",
            4: '"',
            5: "John",
            6: "}",
            7: "INVALID",
        }
        self.vocab_size = len(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def convert_ids_to_tokens(self, token_id: int):
        return self.tokens.get(token_id, None)

    def convert_tokens_to_string(self, tokens: list):
        return "".join(tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.tokens.get(tid, "") for tid in token_ids)


def test_compatibility_engine():

    tokenizer = FakeTokenizer()
    engine = CompatibilityEngine(tokenizer)

    initial_state = State(GrammarState.EXPECT_OBJECT_START)

    valid_tokens = engine.get_valid_tokens(initial_state)

    print("\nValid token IDs:", valid_tokens)

    # At the beginning of the grammar, the object must start with "{"
    assert 0 in valid_tokens

    # These should not be legal immediately
    assert 3 not in valid_tokens
    assert 6 not in valid_tokens
    assert 7 not in valid_tokens

    print("\nCompatibility test PASSED!")