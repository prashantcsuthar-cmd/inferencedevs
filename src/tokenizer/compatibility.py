from typing import List

from src.schema.grammar import transition
from src.schema.states import State
from src.tokenizer.trie import TokenTrie
from src.tokenizer.vocabulary import Vocabulary


class CompatibilityEngine:
    """
    Compute the admissible token set A(S).

    The tokenizer vocabulary is indexed through the existing
    prefix trie so shared token prefixes are checked only once.
    """

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        self.token_text_cache = {}

        self.vocabulary = Vocabulary(
            tokenizer
        )

        self.trie = TokenTrie()

        self.trie.build(
            self.vocabulary
        )

        # Cache admissible tokens for repeated states.
        self._state_cache = {}

    def decode_token(
        self,
        token_id: int
    ) -> str:

        if token_id not in self.token_text_cache:

            self.token_text_cache[token_id] = (
                self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False
                )
            )

        return self.token_text_cache[
            token_id
        ]

    def is_token_valid(
        self,
        state: State,
        token_id: int
    ) -> bool:

        token_text = self.decode_token(
            token_id
        )

        if token_text == "":
            return False

        current_state = state

        for char in token_text:

            current_state = transition(
                current_state,
                char
            )

            if (
                current_state.grammar_state.name
                == "DEAD_END"
            ):
                return False

        return True

    def get_valid_tokens(
        self,
        state: State
    ) -> List[int]:

        cache_key = (
            state.grammar_state,
            state.buffer
        )

        cached = self._state_cache.get(
            cache_key
        )

        if cached is not None:
            return cached

        valid_tokens = []

        def visit(
            node,
            current_state
        ):

            # Every token ending here is legal because
            # the entire prefix reached this node legally.
            if node.token_ids:

                valid_tokens.extend(
                    node.token_ids
                )

            for (
                char,
                child
            ) in node.children.items():

                next_state = transition(
                    current_state,
                    char
                )

                if (
                    next_state.grammar_state.name
                    == "DEAD_END"
                ):
                    continue

                visit(
                    child,
                    next_state
                )

        visit(
            self.trie.get_root(),
            state
        )

        self._state_cache[
            cache_key
        ] = valid_tokens

        return valid_tokens