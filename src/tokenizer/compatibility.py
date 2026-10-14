
from typing import Dict, List, Tuple

from src.schema.grammar import transition
from src.schema.states import State, GrammarState
from src.tokenizer.trie import TokenTrie
from src.tokenizer.vocabulary import Vocabulary


class CompatibilityEngine:
    """
    Computes grammar-admissible token IDs using iterative trie traversal.

    The grammar is advanced one character at a time through each indexed
    token fragment. A token is accepted only if its entire indexed fragment
    avoids DEAD_END.

    Limitation:
    Correctness depends on Vocabulary's text fragments matching the
    tokenizer's actual emitted text. Byte-level and context-dependent
    tokenizers may require a byte-aware incremental decoder instead.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = Vocabulary(tokenizer)

        self.trie = TokenTrie()
        self.trie.build(self.vocabulary)

        # State is a frozen dataclass in this project. Using the state
        # directly avoids manually flattening nested schema structures.
        # Cached results are immutable tuples to prevent caller mutation.
        self._state_cache: Dict[State, Tuple[int, ...]] = {}

        self._special_token_ids = set(
            self.vocabulary.special_token_ids
        )

        self._eos_token_ids = self._get_eos_token_ids()

    def _get_eos_token_ids(self) -> set:
        """Collect configured end-of-sequence token IDs."""
        eos = getattr(self.tokenizer, "eos_token_id", None)

        if eos is None:
            return set()

        if isinstance(eos, (list, tuple, set)):
            return {token_id for token_id in eos if isinstance(token_id, int)}

        return {eos} if isinstance(eos, int) else set()

    def decode_token(self, token_id: int) -> str:
        return self.vocabulary.get_text(token_id)

    def is_token_valid(self, state: State, token_id: int) -> bool:
        """
        Check one token against the grammar.

        Special tokens are not ordinary JSON characters. EOS is accepted
        only after the grammar reaches DONE.
        """
        if token_id < 0 or token_id >= len(self.vocabulary):
            return False

        if token_id in self._special_token_ids:
            return (
                state.grammar_state == GrammarState.DONE
                and token_id in self._eos_token_ids
            )

        if self.vocabulary.is_unsafe_token(token_id):
            return False

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
        """
        Return admissible token IDs for the supplied grammar state.

        The returned list is a copy; callers cannot mutate the cache.
        """
        try:
            cached = self._state_cache.get(state)
        except TypeError as exc:
            raise TypeError(
                "ICE cache requires a hashable State. Ensure State and all "
                "nested schema/context fields are immutable and hashable."
            ) from exc

        if cached is not None:
            return list(cached)

        valid_token_ids: List[int] = []

        # Each stack entry carries the grammar state after consuming
        # the character prefix represented by that trie node.
        stack = [(self.trie.get_root(), state)]

        while stack:
            node, current_state = stack.pop()

            # Token IDs terminate at this exact prefix. The state is
            # already the result of consuming the complete token text.
            if node.token_ids:
                valid_token_ids.extend(node.token_ids)

            for char, child in node.children.items():
                next_state = transition(current_state, char)

                if next_state.grammar_state == GrammarState.DEAD_END:
                    continue

                stack.append((child, next_state))

        # EOS has no ordinary text fragment and is handled separately.
        if state.grammar_state == GrammarState.DONE:
            valid_token_ids.extend(self._eos_token_ids)

        # Ensure IDs are unique and fall within the tokenizer vocabulary.
        vocab_size = len(self.vocabulary)
        valid_token_ids = sorted({
            token_id
            for token_id in valid_token_ids
            if isinstance(token_id, int)
            and not isinstance(token_id, bool)
            and 0 <= token_id < vocab_size
        })

        result = tuple(valid_token_ids)

        try:
            self._state_cache[state] = result
        except TypeError as exc:
            raise TypeError(
                "ICE could not cache this grammar state because it is "
                "not hashable. Check nested schema and context fields."
            ) from exc

        return list(result)

    def clear_cache(self) -> None:
        """Clear cached admissible-token sets."""
        self._state_cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._state_cache)
