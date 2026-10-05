import torch
from transformers import LogitsProcessor


class ICELogitsProcessor(LogitsProcessor):
    """
    ICE logits processor compliant with Hugging Face transformers signature standards.
    """

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
        # 1. Fetch valid tokens for current state
        valid_token_ids = self.compatibility_engine.get_valid_tokens(
            self.current_state
        )

        # 2. Build full mask (-inf)
        mask = torch.full_like(scores, float("-inf"))

        # 3. Unmask valid indices safely via Tensor
        if valid_token_ids:
            indices = torch.tensor(
                valid_token_ids, dtype=torch.long, device=scores.device
            )
            mask[:, indices] = 0.0

        return scores + mask