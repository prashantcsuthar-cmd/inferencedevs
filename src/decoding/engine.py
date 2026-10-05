"""Optimized Incremental ICE decoding engine with unified telemetry tracking."""

from __future__ import annotations

import time
from typing import Any

import torch

from src.schema.grammar import transition
from src.schema.states import GrammarState, State
from src.tokenizer.compatibility import CompatibilityEngine
from src.decoding.logits_processor import ICELogitsProcessor


class ICEDecodingEngine:
    """Hardware-aware ICE decoder with optimized single-pass token masking."""

    def __init__(self, model, tokenizer, compatibility_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.compatibility_engine = (
            compatibility_engine
            if compatibility_engine is not None
            else CompatibilityEngine(tokenizer)
        )
        self.processor = ICELogitsProcessor(self.compatibility_engine)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Start from the JSON schema grammar start state
        state = State(GrammarState.EXPECT_OBJECT_START)
        generated_ids = []

        vocabulary_size = getattr(self.tokenizer, "vocab_size", len(self.tokenizer))
        valid_token_counts = []
        masked_token_counts = []
        masking_ratios = []

        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past_key_values = getattr(outputs, "past_key_values", None)

            for _ in range(max_new_tokens):
                self.processor.set_state(state)

                # Single-pass extraction for telemetry and logit processing
                valid_token_ids = self.compatibility_engine.get_valid_tokens(state)
                valid_count = len(valid_token_ids)
                masked_count = vocabulary_size - valid_count
                masking_ratio = masked_count / vocabulary_size if vocabulary_size else 0.0

                valid_token_counts.append(valid_count)
                masked_token_counts.append(masked_count)
                masking_ratios.append(masking_ratio)

                # Pass pre-computed valid IDs into processor
                masked_logits = self.processor(
                    input_ids, outputs.logits[:, -1, :], valid_token_ids=valid_token_ids
                )

                next_token = torch.argmax(masked_logits, dim=-1)
                token_id = int(next_token.item())

                # Safe token string decoding via CompatibilityEngine
                token_text = self.compatibility_engine.decode_token(token_id)

                # Advance grammar state character by character
                next_state = state
                for char in token_text:
                    next_state = transition(next_state, char)
                    if next_state.grammar_state == GrammarState.DEAD_END:
                        raise RuntimeError(
                            f"ICE Boundary Enforcement Failure: Chosen token {token_id} ({token_text!r}) violates schema syntax."
                        )

                generated_ids.append(token_id)
                state = next_state

                if state.grammar_state == GrammarState.DONE:
                    break

                input_ids = next_token.unsqueeze(0)

                if attention_mask is not None:
                    new_mask_bit = torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, new_mask_bit], dim=1)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = getattr(outputs, "past_key_values", None)

        elapsed = time.perf_counter() - start_time
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = len(generated_ids)

        average_valid_tokens = (
            sum(valid_token_counts) / len(valid_token_counts)
            if valid_token_counts
            else 0.0
        )
        average_masked_tokens = (
            sum(masked_token_counts) / len(masked_token_counts)
            if masked_token_counts
            else 0.0
        )
        average_masking_ratio = (
            sum(masking_ratios) / len(masking_ratios)
            if masking_ratios
            else 0.0
        )

        return {
            "text": generated_text,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "ms_per_token": (
                (elapsed / generated_tokens * 1000) if generated_tokens else 0.0
            ),
            "final_state": state,
            "vocabulary_size": vocabulary_size,
            "valid_token_counts": valid_token_counts,
            "masked_token_counts": masked_token_counts,
            "masking_ratios": masking_ratios,
            "average_valid_tokens": average_valid_tokens,
            "average_masked_tokens": average_masked_tokens,
            "average_masking_ratio": average_masking_ratio,
            "minimum_valid_tokens": (
                min(valid_token_counts) if valid_token_counts else 0
            ),
            "maximum_valid_tokens": (
                max(valid_token_counts) if valid_token_counts else 0
            ),
        }