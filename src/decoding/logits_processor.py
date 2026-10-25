"""Fail-closed logits masking for ICE.

A processor instance stores one active grammar state. Create one instance per
request/generation; do not share a mutable instance between concurrent requests.
"""
import torch
from transformers import LogitsProcessor


class ICELogitsProcessor(LogitsProcessor):
    def __init__(self, compatibility_engine):
        self.compatibility_engine = compatibility_engine
        self.current_state = None

    def set_state(self, state):
        self.current_state = state

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.current_state is None:
            raise ValueError("ICE grammar state has not been set.")
        if scores.ndim != 2:
            raise ValueError(f"Expected scores shaped [batch, vocab], got {tuple(scores.shape)}")
        valid = self.compatibility_engine.get_valid_tokens(self.current_state)
        vocab_size = scores.shape[-1]
        invalid = [i for i in valid if not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= vocab_size]
        if invalid:
            raise ValueError(f"Compatibility engine returned out-of-range token IDs: {invalid[:10]}; logits vocab size is {vocab_size}.")
        valid = sorted(set(valid))
        if not valid:
            raise RuntimeError(f"ICE has no admissible tokens for state {self.current_state!r}; unrestricted fallback is prohibited.")
        mask = torch.full_like(scores, float("-inf"))
        indices = torch.tensor(valid, dtype=torch.long, device=scores.device)
        mask[:, indices] = 0.0
        masked = scores + mask
        if not torch.isfinite(masked).any(dim=-1).all():
            raise RuntimeError("ICE masked every logit; refusing unconstrained fallback.")
        return masked
