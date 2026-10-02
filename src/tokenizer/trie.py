from typing import Dict, List


class TrieNode:
    """
    A single node in the token trie.
    """

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.token_ids: List[int] = []


class TokenTrie:
    """
    Prefix tree for tokenizer tokens.

    Each character in a token becomes a path
    through the trie.

    Example:

        token = "name"

        root
          |
          n
          |
          a
          |
          m
          |
          e
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, text: str, token_id: int) -> None:
        """
        Insert one token into the trie.
        """

        # Ignore empty token text.
        if not text:
            return

        node = self.root

        # Create one trie node for each character.
        for char in text:

            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]

        # Store the token ID at the final node.
        node.token_ids.append(token_id)

    def build(self, vocabulary) -> None:
        """
        Build the trie from a Vocabulary object.
        """

        for token_id, text in vocabulary.id_to_text.items():
            self.insert(text, token_id)

    def get_root(self) -> TrieNode:
        """
        Return the root node.
        """

        return self.root

    def get_node(self, text: str):
        """
        Find the trie node corresponding to a text prefix.

        Returns:
            TrieNode if the prefix exists.
            None otherwise.
        """

        node = self.root

        for char in text:

            if char not in node.children:
                return None

            node = node.children[char]

        return node

    def get_token_ids(self, text: str) -> List[int]:
        """
        Return token IDs that exactly match the given text.
        """

        node = self.get_node(text)

        if node is None:
            return []

        return node.token_ids