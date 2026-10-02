import torch
from transformers import LogitsProcessor


class ICELogitsProcessor(LogitsProcessor):
    """
    ICE (Inferencedevs Constrained Engine) logits processor.

    This processor prevents the language model from selecting
    tokens that are illegal according to the current grammar state.

    Valid tokens:
        mask = 0

    Invalid tokens:
        mask = -inf

    After adding the mask to the model's logits, invalid tokens
    receive zero probability after softmax.
    """

    def __init__(self, compatibility_engine):
        """
        Parameters
        ----------
        compatibility_engine:
            Object responsible for determining which token IDs
            are valid for the current grammar state.
        """

        self.compatibility_engine = compatibility_engine

        # Current grammar state.
        # The actual state will be updated by the decoding engine.
        self.current_state = None

    def set_state(self, state):
        """
        Update the grammar state used by ICE.
        """

        self.current_state = state

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the ICE token mask to the model's logits.

        Parameters
        ----------
        input_ids:
            Tokens generated so far.

        scores:
            Raw logits produced by the language model.

        Returns
        -------
        torch.FloatTensor
            Masked logits.
        """

        # ---------------------------------------------------------
        # 1. Get valid token IDs from the compatibility engine
        # ---------------------------------------------------------

        valid_token_ids = (
            self.compatibility_engine
            .get_valid_tokens(self.current_state)
        )

        # ---------------------------------------------------------
        # 2. Create a mask where EVERY token is initially illegal
        # ---------------------------------------------------------

        mask = torch.full_like(
            scores,
            float("-inf")
        )

        # ---------------------------------------------------------
        # 3. Mark valid tokens as legal
        # ---------------------------------------------------------

        mask[:, valid_token_ids] = 0.0

        # ---------------------------------------------------------
        # 4. Add the mask to the model's logits
        # ---------------------------------------------------------

        masked_scores = scores + mask

        return masked_scores