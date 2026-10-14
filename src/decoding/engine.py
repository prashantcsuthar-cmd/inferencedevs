
"""Incremental ICE decoding engine with schema initialization and telemetry."""

from __future__ import annotations

import time
from typing import Any

import torch

from src.decoding.logits_processor import ICELogitsProcessor
from src.schema.grammar import transition
from src.schema.parser import compile_schema_to_state
from src.schema.states import GrammarState, State
from src.tokenizer.compatibility import CompatibilityEngine


class ICEDecodingEngine:
    """Generate JSON using ICE grammar-guided token masking."""

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
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Initialize the grammar from the supplied schema when available.
        if schema is not None:
            state = compile_schema_to_state(schema)
        else:
            state = State(GrammarState.EXPECT_OBJECT_START)

        generated_ids: list[int] = []

        vocabulary_size = outputs_vocab_size = getattr(
            self.tokenizer, "vocab_size", len(self.tokenizer)
        )

        valid_token_counts: list[int] = []
        masked_token_counts: list[int] = []
        masking_ratios: list[float] = []

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

                # Compute admissible tokens for telemetry.
                valid_token_ids = self.compatibility_engine.get_valid_tokens(
                    state
                )
                valid_count = len(set(valid_token_ids))
                masked_count = max(0, vocabulary_size - valid_count)
                masking_ratio = (
                    masked_count / vocabulary_size
                    if vocabulary_size
                    else 0.0
                )

                valid_token_counts.append(valid_count)
                masked_token_counts.append(masked_count)
                masking_ratios.append(masking_ratio)

                # Keep the Hugging Face-compatible processor signature:
                # __call__(input_ids, scores).
                masked_logits = self.processor(
                    input_ids,
                    outputs.logits[:, -1, :],
                )

                if not torch.isfinite(masked_logits).any(dim=-1).all():
                    raise RuntimeError(
                        "ICE produced no finite logits for the current "
                        f"grammar state: {state!r}"
                    )

                next_token = torch.argmax(masked_logits, dim=-1)
                token_id = int(next_token.item())
                token_text = self.compatibility_engine.decode_token(token_id)

                # Advance the grammar using the compatibility engine's
                # token text representation.
                next_state = state
                for char in token_text:
                    next_state = transition(next_state, char)

                    if next_state.grammar_state == GrammarState.DEAD_END:
                        raise RuntimeError(
                            "ICE boundary enforcement failure: selected "
                            f"token {token_id} ({token_text!r}) entered "
                            "a dead-end grammar state."
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
                    attention_mask = torch.cat(
                        [attention_mask, new_mask_bit],
                        dim=1,
                    )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = getattr(
                    outputs, "past_key_values", None
                )

        elapsed = time.perf_counter() - start_time
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
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
                elapsed / generated_tokens * 1000
                if generated_tokens
                else 0.0
            ),
            "final_state": state,
            "vocabulary_size": outputs_vocab_size,
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
