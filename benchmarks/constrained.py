import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.schema.states import State, GrammarState
from src.schema.grammar import transition
from src.tokenizer.compatibility import CompatibilityEngine


def generate_constrained(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
):
    """
    Phase 3 constrained generation.

    The grammar state is explicitly maintained by the decoding loop.
    This avoids relying on transformers.generate() to manage ICE state.
    """

    # ---------------------------------------------------------
    # 1. Create compatibility engine
    # ---------------------------------------------------------

    compatibility_engine = CompatibilityEngine(tokenizer)

    # ---------------------------------------------------------
    # 2. Start grammar
    # ---------------------------------------------------------

    state = State(
        grammar_state=GrammarState.EXPECT_OBJECT_START,
        buffer=""
    )

    # ---------------------------------------------------------
    # 3. Tokenize prompt
    # ---------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    input_ids = inputs["input_ids"]

    # Keep attention mask if tokenizer provides one.
    attention_mask = inputs.get("attention_mask")

    # ---------------------------------------------------------
    # 4. Explicit constrained decoding loop
    # ---------------------------------------------------------

    model.eval()

    with torch.no_grad():

        for step in range(max_new_tokens):

            # Run the model.
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Logits for the next token.
            logits = outputs.logits[:, -1, :]

            # -------------------------------------------------
            # Get grammar-valid tokens
            # -------------------------------------------------

            valid_token_ids = (
                compatibility_engine
                .get_valid_tokens(state)
            )

            if not valid_token_ids:
                raise RuntimeError(
                    "Grammar produced no valid tokens at "
                    f"generation step {step}. "
                    f"Current state: {state}"
                )

            # -------------------------------------------------
            # Mask invalid tokens
            # -------------------------------------------------

            masked_logits = torch.full_like(
                logits,
                float("-inf")
            )

            masked_logits[:, valid_token_ids] = (
                logits[:, valid_token_ids]
            )

            # -------------------------------------------------
            # Select highest-probability valid token
            # -------------------------------------------------

            next_token = torch.argmax(
                masked_logits,
                dim=-1
            )

            token_id = next_token.item()

            # -------------------------------------------------
            # Decode selected token
            # -------------------------------------------------

            token_text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False
            )

            if token_text == "":
                raise RuntimeError(
                    f"Selected token {token_id} decoded to "
                    "empty text."
                )

            # -------------------------------------------------
            # Advance grammar state
            # -------------------------------------------------

            for char in token_text:
                state = transition(
                    state,
                    char
                )

                if state.grammar_state == GrammarState.DEAD_END:
                    raise RuntimeError(
                        "Grammar entered DEAD_END after "
                        f"token {token_id!r} "
                        f"({token_text!r})."
                    )

            # -------------------------------------------------
            # Append token to sequence
            # -------------------------------------------------

            next_token = next_token.unsqueeze(-1)

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=-1
            )

            if attention_mask is not None:
                new_mask = torch.ones(
                    (attention_mask.shape[0], 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                attention_mask = torch.cat(
                    [attention_mask, new_mask],
                    dim=-1
                )

            # -------------------------------------------------
            # Stop when grammar is complete
            # -------------------------------------------------

            if state.grammar_state == GrammarState.DONE:
                break

    # ---------------------------------------------------------
    # 5. Decode final sequence
    # ---------------------------------------------------------

    return tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True
    )


if __name__ == "__main__":

    print("=" * 60)
    print("ICE PHASE 3 - CONSTRAINED BENCHMARK")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )

    prompt = """Extract the following information as JSON:
Customer Rahul Sharma placed order ORD12345.
Issue: Package was delivered late.

Required fields:
customer_name, order_id, issue.
"""

    print("\nGenerating constrained output...\n")

    output = generate_constrained(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=128,
    )

    print("Generated output:")
    print(output)

    print("\n" + "=" * 60)
    print("CONSTRAINED BENCHMARK COMPLETE")
    print("=" * 60)