import pytest
from src.schema.states import State, GrammarState
from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.trie import TokenTrie
from src.tokenizer.compatibility import CompatibilityEngine


class MockTokenizer:
    """Mock Hugging Face BPE tokenizer with convert_ids_to_tokens support."""
    def __init__(self):
        self.vocab = {
            0: "{",
            1: '"',
            2: "name",
            3: '":',
            4: '"John"',
            5: "}",
            6: "invalid_syntax_symbol",
        }
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.vocab)

    def convert_ids_to_tokens(self, token_id: int):
        return self.vocab.get(token_id, None)

    def convert_tokens_to_string(self, tokens: list):
        return "".join(tokens)

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.vocab.get(tid, "") for tid in token_ids)


def test_vocabulary_building():
    tok = MockTokenizer()
    vocab = Vocabulary(tok)
    
    assert len(vocab) == 7
    assert vocab.get_text(2) == "name"
    assert vocab.get_token_ids('"') == [1]


def test_trie_construction_and_lookup():
    tok = MockTokenizer()
    vocab = Vocabulary(tok)
    trie = TokenTrie()
    trie.build(vocab)

    assert trie.get_token_ids("{") == [0]
    assert trie.get_token_ids("name") == [2]
    assert trie.get_node("nam") is not None
    assert trie.get_node("invalid_prefix") is None


def test_compatibility_engine_dfs():
    tok = MockTokenizer()
    engine = CompatibilityEngine(tok)

    initial_state = State(GrammarState.START)
    valid_tokens = engine.get_valid_tokens(initial_state)

    # Token 0 ("{") MUST be valid from GrammarState.START
    assert 0 in valid_tokens
    # Token 6 ("invalid_syntax_symbol") MUST NOT be valid from GrammarState.START
    assert 6 not in valid_tokens


def test_compatibility_engine_caching():
    tok = MockTokenizer()
    engine = CompatibilityEngine(tok)

    state_a = State(GrammarState.START)
    state_b = State(GrammarState.START)

    tokens_a = engine.get_valid_tokens(state_a)
    tokens_b = engine.get_valid_tokens(state_b)

    # Output must match and cache must hit
    assert tokens_a == tokens_b
    assert len(engine._state_cache) == 1