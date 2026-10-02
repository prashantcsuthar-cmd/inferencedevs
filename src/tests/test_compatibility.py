from src.tokenizer.compatibility import CompatibilityEngine
from src.schema.states import State, GrammarState


class FakeTokenizer:
    """
    Tiny tokenizer used only for testing.

    This avoids loading Qwen while testing
    the compatibility engine.
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

    def __len__(self):
        return len(self.tokens)

    def decode(
        self,
        token_ids,
        skip_special_tokens=False
    ):

        return "".join(
            self.tokens[token_id]
            for token_id in token_ids
        )


def test_compatibility_engine():

    tokenizer = FakeTokenizer()

    engine = CompatibilityEngine(
        tokenizer
    )

    initial_state = State(
        GrammarState.EXPECT_OBJECT_START
    )

    valid_tokens = engine.get_valid_tokens(
        initial_state
    )

    print("\nValid token IDs:")
    print(valid_tokens)

    print("\nValid token text:")

    for token_id in valid_tokens:

        print(
            token_id,
            repr(tokenizer.decode([token_id]))
        )

    # At the beginning of the grammar,
    # the object must start with "{"
    assert 0 in valid_tokens

    # These should not be legal immediately.
    assert 3 not in valid_tokens
    assert 6 not in valid_tokens
    assert 7 not in valid_tokens

    print("\nCompatibility test PASSED!")