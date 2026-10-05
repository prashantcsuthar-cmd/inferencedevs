from dataclasses import astuple
from typing import List, Dict, Tuple
from src.schema.grammar import transition
from src.schema.states import State, GrammarState
from src.tokenizer.trie import TokenTrie
from src.tokenizer.vocabulary import Vocabulary


class CompatibilityEngine:
    """
    Iterative Admissibility Engine computing A(S) via Trie traversal
    without recursion or state-hashing traps.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = Vocabulary(tokenizer)
        self.trie = TokenTrie()
        self.trie.build(self.vocabulary)
        self._state_cache: Dict[Tuple, List[int]] = {}

    def _get_state_cache_key(self, state: State) -> Tuple:
        """Dynamically hash all state dataclass fields cleanly."""
        raw_values = astuple(state)
        hashable_values = []
        for val in raw_values:
            if isinstance(val, (list, set)):
                hashable_values.append(tuple(sorted(val)))
            else:
                hashable_values.append(val)
        return tuple(hashable_values)

    def decode_token(self, token_id: int) -> str:
        return self.vocabulary.get_text(token_id)

    def is_token_valid(self, state: State, token_id: int) -> bool:
        token_text = self.decode_token(token_id)
        if not token_text:
            return False

        current_state = state
        for char in token_text:
            current_state = transition(current_state, char)
            if current_state.grammar_state == GrammarState.DEAD_END:
                return False

        return True

    def get_valid_tokens(self, state: State) -> List[int]:
        cache_key = self._get_state_cache_key(state)
        cached = self._state_cache.get(cache_key)
        if cached is not None:
            return cached

        valid_tokens: List[int] = []

        # Iterative DFS stack: stores (node, state)
        stack = [(self.trie.get_root(), state)]

        while stack:
            node, current_state = stack.pop()

            if node.token_ids:
                valid_tokens.extend(node.token_ids)

            for char, child in node.children.items():
                next_state = transition(current_state, char)

                if next_state.grammar_state == GrammarState.DEAD_END:
                    continue

                stack.append((child, next_state))

        # Dynamic Initial Whitespace Clipping: handled natively at the token layer
        if state.grammar_state == GrammarState.EXPECT_OBJECT_START:
            filtered_tokens = []
            for tid in valid_tokens:
                text = self.vocabulary.get_text(tid)
                # Suppress tokens containing ONLY whitespace formatting characters
                if text.strip() or "{" in text:
                    filtered_tokens.append(tid)
            if filtered_tokens:
                valid_tokens = filtered_tokens

        self._state_cache[cache_key] = valid_tokens
        return valid_tokens