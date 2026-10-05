from typing import Dict, List


class Vocabulary:
    """
    Stores character mappings for tokenizer token IDs, preserving
    explicit byte indicators and structural whitespace across BPE formats.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.id_to_text: Dict[int, str] = {}
        self.text_to_ids: Dict[str, List[int]] = {}
        self._build()

    def _build(self):
        vocab_size = getattr(self.tokenizer, "vocab_size", len(self.tokenizer))

        for token_id in range(vocab_size):
            try:
                # Primary path: convert ID to token string representation
                raw_token = self.tokenizer.convert_ids_to_tokens(token_id)
                if raw_token is None:
                    text = ""
                elif hasattr(self.tokenizer, "convert_tokens_to_string"):
                    text = self.tokenizer.convert_tokens_to_string([raw_token])
                else:
                    text = self.tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                text = ""

            if text is None:
                text = ""

            self.id_to_text[token_id] = text

            if text not in self.text_to_ids:
                self.text_to_ids[text] = []
            self.text_to_ids[text].append(token_id)

    def get_text(self, token_id: int) -> str:
        return self.id_to_text.get(token_id, "")

    def get_token_ids(self, text: str) -> List[int]:
        return self.text_to_ids.get(text, [])

    def __len__(self):
        return len(self.id_to_text)