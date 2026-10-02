from typing import Dict


class Vocabulary:
    """
    Stores the relationship between token IDs
    and the text represented by each token.
    """

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

        self.id_to_text: Dict[int, str] = {}
        self.text_to_ids: Dict[str, list[int]] = {}

        self._build()

    def _build(self):
        """
        Build token ID -> text and text -> token IDs mappings.
        """

        print("Building vocabulary...")

        for token_id in range(len(self.tokenizer)):

            text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False
            )

            self.id_to_text[token_id] = text

            if text not in self.text_to_ids:

                self.text_to_ids[text] = []

            self.text_to_ids[text].append(token_id)

        print(
            f"Vocabulary built: "
            f"{len(self.id_to_text)} tokens"
        )

    def get_text(self, token_id: int) -> str:
        """
        Return the text represented by a token ID.
        """

        return self.id_to_text.get(
            token_id,
            ""
        )

    def get_token_ids(self, text: str) -> list[int]:
        """
        Return all token IDs representing the given text.
        """

        return self.text_to_ids.get(
            text,
            []
        )

    def __len__(self):
        return len(self.id_to_text)