"""
Phase 3 - Baseline Benchmark

Runs Qwen2.5-0.5B-Instruct WITHOUT ICE constraints.

This gives us the baseline measurements that we will
later compare against the ICE constrained engine.
"""

import json
import time
import sys
from pathlib import Path

import torch


# ---------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.models.loader import load_qwen


# ---------------------------------------------------------
# Baseline generation
# ---------------------------------------------------------

def generate_baseline(
    model,
    tokenizer,
    prompt,
    max_new_tokens=128,
):
    """
    Generate text using Qwen normally.

    IMPORTANT:
    No ICE grammar or logits processor is used here.
    This is our baseline.
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    # Move inputs to the same device as the model
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    # Measure generation time
    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    end_time = time.perf_counter()

    elapsed = end_time - start_time

    # Remove the original prompt tokens
    prompt_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[:, prompt_length:]

    generated_text = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    # Number of generated tokens
    generated_tokens = generated_ids.shape[1]

    # Avoid division by zero
    if generated_tokens > 0:
        ms_per_token = (
            elapsed / generated_tokens
        ) * 1000
    else:
        ms_per_token = 0.0

    return {
        "text": generated_text,
        "generated_tokens": generated_tokens,
        "elapsed_seconds": elapsed,
        "ms_per_token": ms_per_token,
    }


# ---------------------------------------------------------
# JSON validation
# ---------------------------------------------------------

def check_json_validity(text):
    """
    Check whether the generated text is valid JSON.

    Returns:
        True  -> valid JSON
        False -> invalid JSON
    """

    try:
        json.loads(text)
        return True

    except (json.JSONDecodeError, TypeError):
        return False


# ---------------------------------------------------------
# Simple benchmark
# ---------------------------------------------------------

def run_baseline_benchmark():
    """
    Run a small baseline experiment.

    We intentionally start with a small number of examples.
    Later Phase 3 can scale this to 100-500 samples.
    """

    print("=" * 60)
    print("ICE PHASE 3 - BASELINE BENCHMARK")
    print("=" * 60)

    print("\nLoading Qwen...")

    model, tokenizer = load_qwen()

    # -----------------------------------------------------
    # Benchmark prompts
    # -----------------------------------------------------

    prompts = [
        """
Extract the following information and return JSON only.

Customer: Rahul Sharma
Order ID: ORD12345
Issue: Package was delivered late.

Required JSON fields:
customer_name
order_id
issue
""",

        """
Extract the following information and return JSON only.

Customer: Priya Kumar
Order ID: ORD98765
Issue: Product arrived damaged.

Required JSON fields:
customer_name
order_id
issue
""",

        """
Extract the following information and return JSON only.

Customer: Arjun Mehta
Order ID: ORD54321
Issue: Wrong item was delivered.

Required JSON fields:
customer_name
order_id
issue
""",
    ]

    results = []

    # -----------------------------------------------------
    # Run each example
    # -----------------------------------------------------

    for index, prompt in enumerate(prompts, start=1):

        print(f"\n--- Example {index} ---")

        result = generate_baseline(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
        )

        valid_json = check_json_validity(
            result["text"]
        )

        result["json_valid"] = valid_json

        results.append(result)

        print("\nGenerated output:")
        print(result["text"])

        print(
            f"\nGenerated tokens: "
            f"{result['generated_tokens']}"
        )

        print(
            f"Time: "
            f"{result['elapsed_seconds']:.4f} seconds"
        )

        print(
            f"Latency: "
            f"{result['ms_per_token']:.2f} ms/token"
        )

        print(
            f"Valid JSON: "
            f"{valid_json}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    valid_count = sum(
        result["json_valid"]
        for result in results
    )

    total_count = len(results)

    validity_percentage = (
        valid_count / total_count
    ) * 100

    average_latency = sum(
        result["ms_per_token"]
        for result in results
    ) / total_count

    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)

    print(
        f"JSON validity: "
        f"{validity_percentage:.2f}%"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms/token"
    )

    print("=" * 60)


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    run_baseline_benchmark()