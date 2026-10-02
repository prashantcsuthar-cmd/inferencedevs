from typing import List

from src.schema.grammar import transition
from src.schema.states import State


class CompatibilityEngine:
    """
    Determines which tokenizer tokens are legal for
    the current grammar state.

    The engine tests each token by passing its decoded
    characters through the grammar transition function.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Cache results so the same token does not need
        # to be decoded repeatedly.
        self.token_text_cache = {}

    def decode_token(self, token_id: int) -> str:
        """
        Convert a token ID into the text represented by
        that token.
        """

        if token_id not in self.token_text_cache:

            text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False
            )

            self.token_text_cache[token_id] = text

        return self.token_text_cache[token_id]

    def is_token_valid(
        self,
        state: State,
        token_id: int
    ) -> bool:
        """
        Check whether a single token can legally follow
        the current grammar state.
        """

        token_text = self.decode_token(token_id)

        # Empty tokens cannot advance the grammar.
        if token_text == "":
            return False

        current_state = state

        # Test every character contained in the token.
        for char in token_text:

            current_state = transition(
                current_state,
                char
            )

            # DEAD_END means this token violates
            # the grammar.
            if current_state.grammar_state.name == "DEAD_END":
                return False

        return True

    def get_valid_tokens(
        self,
        state: State
    ) -> List[int]:
        """
        Return all tokenizer token IDs that are legal
        from the current grammar state.
        """

        valid_tokens = []

        vocabulary_size = len(self.tokenizer)

        for token_id in range(vocabulary_size):

            if self.is_token_valid(
                state,
                token_id
            ):
                valid_tokens.append(token_id)

        return valid_tokens