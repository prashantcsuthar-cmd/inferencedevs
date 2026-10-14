
from typing import Dict, List, Optional


class TrieNode:
    """A node in a character-prefix trie."""

    __slots__ = ("children", "token_ids")

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.token_ids: List[int] = []


class TokenTrie:
    """
    Prefix trie over the text fragments supplied by Vocabulary.

    Every terminal node stores all token IDs represented by that exact
    text fragment. Special and unsafe tokens should be excluded by the
    vocabulary layer before insertion.
    """

    def __init__(self):
        self.root = TrieNode()
        self.token_count = 0
        self.node_count = 1

    def insert(self, text: str, token_id: int) -> None:
        """Insert a non-empty text fragment and its token ID."""
        if not isinstance(text, str) or not text:
            return

        node = self.root

        for char in text:
            child = node.children.get(char)

            if child is None:
                child = TrieNode()
                node.children[char] = child
                self.node_count += 1

            node = child

        if token_id not in node.token_ids:
            node.token_ids.append(token_id)
            self.token_count += 1

    def build(self, vocabulary) -> None:
        """Build the trie from a Vocabulary instance."""
        self.root = TrieNode()
        self.token_count = 0
        self.node_count = 1

        for token_id in vocabulary.get_text_token_ids():
            text = vocabulary.get_text(token_id)
            self.insert(text, token_id)

    def get_root(self) -> TrieNode:
        return self.root

    def get_node(self, text: str) -> Optional[TrieNode]:
        """Return the node for a prefix, or None if it does not exist."""
        node = self.root

        for char in text:
            node = node.children.get(char)

            if node is None:
                return None

        return node

    def get_token_ids(self, text: str) -> List[int]:
        """Return IDs ending exactly at the specified text."""
        node = self.get_node(text)

        if node is None:
            return []

        return list(node.token_ids)
