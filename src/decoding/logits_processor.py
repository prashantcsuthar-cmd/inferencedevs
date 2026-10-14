
import torch
from transformers import LogitsProcessor


class ICELogitsProcessor(LogitsProcessor):
    """Apply grammar-derived token constraints to generation logits."""

    def __init__(self, compatibility_engine):
        self.compatibility_engine = compatibility_engine
        self.current_state = None

    def set_state(self, state):
        self.current_state = state

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        if self.current_state is None:
            raise ValueError(
                "ICE: grammar tracking state has not been declared."
            )

        if scores.ndim != 2:
            raise ValueError(
                "ICE expects scores shaped [batch, vocab], "
                f"received {tuple(scores.shape)}."
            )

        valid_token_ids = self.compatibility_engine.get_valid_tokens(
            self.current_state
        )

        if not valid_token_ids:
            raise RuntimeError(
                "ICE: admissible token set is empty for grammar state "
                f"{self.current_state!r}. Unrestricted fallback is prohibited."
            )

        vocab_size = scores.shape[-1]
        valid_token_ids = list(set(valid_token_ids))

        invalid_ids = [
            token_id
            for token_id in valid_token_ids
            if not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= vocab_size
        ]

        if invalid_ids:
            raise ValueError(
                "ICE: compatibility engine returned invalid token IDs: "
                f"{invalid_ids[:10]}; logits vocabulary dimension is "
                f"{vocab_size}."
            )

        indices = torch.tensor(
            valid_token_ids,
            dtype=torch.long,
            device=scores.device,
        )

        mask = torch.full_like(scores, float("-inf"))
        mask[:, indices] = 0.0

        return scores + mask
