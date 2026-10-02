from src.tokenizer.vocabulary import Vocabulary
from src.tokenizer.trie import TokenTrie


class FakeTokenizer:

    def __init__(self):

        self.tokens = {
            0: "{",
            1: '"',
            2: "name",
            3: ":",
            4: "John",
            5: "}",
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


def test_vocabulary():

    tokenizer = FakeTokenizer()

    vocabulary = Vocabulary(
        tokenizer
    )

    assert len(vocabulary) == 6

    assert vocabulary.get_text(0) == "{"

    assert vocabulary.get_text(2) == "name"

    assert vocabulary.get_token_ids(
        "{"
    ) == [0]

    print("\nVocabulary test PASSED!")


def test_trie():

    tokenizer = FakeTokenizer()

    vocabulary = Vocabulary(
        tokenizer
    )

    trie = TokenTrie()

    trie.build(vocabulary)

    root = trie.get_root()

    assert "{" in root.children

    assert '"' in root.children

    assert "n" in root.children

    print("Trie test PASSED!")