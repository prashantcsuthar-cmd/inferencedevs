import torch
from transformers import LogitsProcessor


class ICELogitsProcessor(LogitsProcessor):
    """
    ICE logits processor strictly compliant with Hugging Face transformers standards.
    Contains exactly 2 positional parameter mappings to prevent native parameter check drops.
    """

    def __init__(self, compatibility_engine):
        self.compatibility_engine = compatibility_engine
        self.current_state = None

    def set_state(self, state):
        self.current_state = state

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if self.current_state is None:
            raise ValueError(
                "ICE Execution Error: Active grammar tracking state has not been declared."
            )

        # 1. Fetch valid tokens safely from the admissibility engine
        valid_token_ids = self.compatibility_engine.get_valid_tokens(
            self.current_state
        )

        # 2. Build explicit -inf mask matrix matching scores tensor specs
        mask = torch.full_like(scores, float("-inf"))

        # 3. Unmask valid indices safely
        if valid_token_ids:
            indices = torch.tensor(
                valid_token_ids, dtype=torch.long, device=scores.device
            )
            mask[:, indices] = 0.0
        else:
            mask[:, :] = 0.0

        return scores + mask