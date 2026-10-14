
from typing import Dict, List, Set


class Vocabulary:
    """
    Maps tokenizer IDs to text fragments for grammar-guided decoding.

    Important:
    - Token display strings are not universally equivalent to decoded text.
    - Special tokens are tracked separately from ordinary text tokens.
    - Empty or unsafe text fragments are not inserted into the character trie.
    - This class does not claim that singleton decoding is exact for every
      byte-level or context-dependent tokenizer.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        self.id_to_text: Dict[int, str] = {}
        self.text_to_ids: Dict[str, List[int]] = {}

        self.special_token_ids: Set[int] = set()
        self.unsafe_token_ids: Set[int] = set()

        self._build()

    def _get_vocab_size(self) -> int:
        """
        Prefer len(tokenizer), which includes added tokens in standard
        Hugging Face tokenizers. The model's logits dimension must still
        be checked separately by the caller.
        """
        try:
            return len(self.tokenizer)
        except (TypeError, AttributeError):
            return int(self.tokenizer.vocab_size)

    def _decode_single_token(self, token_id: int) -> str:
        """
        Obtain the tokenizer's best available single-token text fragment.

        This is a practical representation for character-trie grammars,
        not a universal guarantee of exact incremental decoding.
        """
        raw_token = self.tokenizer.convert_ids_to_tokens(token_id)

        if raw_token is None:
            return ""

        # Use the tokenizer's own token-to-string conversion where available.
        if hasattr(self.tokenizer, "convert_tokens_to_string"):
            text = self.tokenizer.convert_tokens_to_string([raw_token])
        else:
            text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        if text is None:
            return ""

        return text

    def _build(self) -> None:
        vocab_size = self._get_vocab_size()
        special_ids = set(
            getattr(self.tokenizer, "all_special_ids", []) or []
        )

        self.special_token_ids = special_ids

        for token_id in range(vocab_size):
            if token_id in special_ids:
                self.id_to_text[token_id] = ""
                continue

            try:
                text = self._decode_single_token(token_id)
            except Exception:
                # Do not invent a textual representation when conversion fails.
                self.id_to_text[token_id] = ""
                self.unsafe_token_ids.add(token_id)
                continue

            if not isinstance(text, str) or not text:
                self.id_to_text[token_id] = ""
                self.unsafe_token_ids.add(token_id)
                continue

            # A replacement character can indicate that an incomplete
            # byte sequence was decoded independently. Fail closed.
            if "\ufffd" in text:
                self.id_to_text[token_id] = ""
                self.unsafe_token_ids.add(token_id)
                continue

            self.id_to_text[token_id] = text
            self.text_to_ids.setdefault(text, []).append(token_id)

    def get_text(self, token_id: int) -> str:
        """Return the indexed text fragment, or an empty string."""
        return self.id_to_text.get(token_id, "")

    def get_token_ids(self, text: str) -> List[int]:
        """Return IDs whose indexed text exactly matches text."""
        return list(self.text_to_ids.get(text, []))

    def is_special_token(self, token_id: int) -> bool:
        return token_id in self.special_token_ids

    def is_unsafe_token(self, token_id: int) -> bool:
        return token_id in self.unsafe_token_ids

    def get_text_token_ids(self) -> List[int]:
        """Return IDs represented by non-empty, indexed text fragments."""
        return [
            token_id
            for token_id, text in self.id_to_text.items()
            if text and token_id not in self.special_token_ids
            and token_id not in self.unsafe_token_ids
        ]

    def __len__(self) -> int:
        return len(self.id_to_text)
