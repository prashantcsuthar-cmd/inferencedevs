"""Incremental Phase 3 decoding engine."""

from __future__ import annotations

import time
from typing import Any

import torch

from src.schema.grammar import transition
from src.schema.states import GrammarState, State
from src.tokenizer.compatibility import CompatibilityEngine
from src.decoding.logits_processor import ICELogitsProcessor


class ICEDecodingEngine:
    """CPU-first ICE decoder with explicit grammar-state management."""

    def __init__(self, model, tokenizer, compatibility_engine=None):
        self.model = model
        self.tokenizer = tokenizer
        self.compatibility_engine = (
            compatibility_engine
            if compatibility_engine is not None
            else CompatibilityEngine(tokenizer)
        )
        self.processor = ICELogitsProcessor(
            self.compatibility_engine
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> dict[str, Any]:

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        # Phase 3 starts from the beginning of the JSON object.
        state = State(
            GrammarState.EXPECT_OBJECT_START
        )

        generated_ids = []

        start_time = time.perf_counter()

        with torch.no_grad():

            # Initial forward pass.
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

            past_key_values = outputs.past_key_values

            for _ in range(max_new_tokens):

                # Give ICE the current grammar state.
                self.processor.set_state(state)

                # Apply the actual ICE logit mask.
                masked_logits = self.processor(
                    input_ids,
                    outputs.logits[:, -1, :],
                )

                # Deterministic decoding, matching the baseline.
                next_token = torch.argmax(
                    masked_logits,
                    dim=-1
                )

                token_id = int(
                    next_token.item()
                )

                token_text = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                )

                if not token_text:
                    raise RuntimeError(
                        f"Token {token_id} decoded to empty text."
                    )

                # Advance grammar state using the actual token text.
                next_state = state

                for char in token_text:

                    next_state = transition(
                        next_state,
                        char
                    )

                    if (
                        next_state.grammar_state
                        == GrammarState.DEAD_END
                    ):
                        raise RuntimeError(
                            f"ICE selected illegal token "
                            f"{token_id}: {token_text!r}"
                        )

                generated_ids.append(
                    token_id
                )

                state = next_state

                # The JSON object is complete.
                if (
                    state.grammar_state
                    == GrammarState.DONE
                ):
                    break

                # Feed only the newly generated token on
                # subsequent passes, using the KV cache.
                input_ids = next_token.unsqueeze(0)

                if attention_mask is not None:

                    attention_mask = torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (
                                    attention_mask.shape[0],
                                    1
                                ),
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = (
                    outputs.past_key_values
                )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        generated_tokens = len(
            generated_ids
        )

        return {
            "text": generated_text,
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "ms_per_token": (
                elapsed
                / generated_tokens
                * 1000
                if generated_tokens
                else 0.0
            ),
            "final_state": state,
        }