"""Phase 3 - ICE constrained benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


from src.decoding.engine import (
    ICEDecodingEngine
)

from src.models.loader import (
    load_qwen
)

from src.schema.states import (
    GrammarState
)


def check_json_validity(
    text: str
) -> bool:

    try:

        json.loads(text)

        return True

    except (
        json.JSONDecodeError,
        TypeError
    ):

        return False


def run_constrained_benchmark():

    print("=" * 60)
    print("ICE PHASE 3 - CONSTRAINED BENCHMARK")
    print("=" * 60)

    print("\nLoading Qwen...")

    model, tokenizer = load_qwen()

    print("\nCreating ICE decoding engine...")

    engine = ICEDecodingEngine(
        model,
        tokenizer
    )

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

    for index, prompt in enumerate(
        prompts,
        start=1
    ):

        print(
            f"\n--- Example {index} ---"
        )

        result = engine.generate(
            prompt,
            max_new_tokens=128
        )

        result["json_valid"] = (
            check_json_validity(
                result["text"]
            )
        )

        results.append(
            result
        )

        print(
            "\nGenerated output:"
        )

        print(
            result["text"]
        )

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
            f"{result['json_valid']}"
        )

        print(
            f"Final grammar state: "
            f"{result['final_state'].grammar_state.name}"
        )

    valid_count = sum(
        result["json_valid"]
        for result in results
    )

    done_count = sum(
        result["final_state"].grammar_state
        == GrammarState.DONE
        for result in results
    )

    average_latency = (
        sum(
            result["ms_per_token"]
            for result in results
        )
        / len(results)
    )

    average_valid_tokens = (
    sum(
        result["average_valid_tokens"]
        for result in results
    )
    / len(results)
    )

    average_masked_tokens = (
    sum(
        result["average_masked_tokens"]
        for result in results
    )
    / len(results)
    )

    average_masking_ratio = (
    sum(
        result["average_masking_ratio"]
        for result in results
    )
    / len(results)
    )

    minimum_valid_tokens = min(
    result["minimum_valid_tokens"]
    for result in results
    )

    maximum_valid_tokens = max(
    result["maximum_valid_tokens"]
    for result in results
    )

    vocabulary_size = results[0][
    "vocabulary_size"
    ]

    print(
        "\n" + "=" * 60
    )

    print(
        "CONSTRAINED SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        f"JSON validity: "
        f"{valid_count / len(results) * 100:.2f}%"
    )

    print(
        f"Grammar DONE rate: "
        f"{done_count / len(results) * 100:.2f}%"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms/token"
    )

    print(
    f"Vocabulary size: "
    f"{vocabulary_size:,}"
    )

    print(
    f"Average valid tokens: "
    f"{average_valid_tokens:.2f}"
    )

    print(
    f"Average masked tokens: "
    f"{average_masked_tokens:.2f}"
    )

    print(
    f"Average masking ratio: "
    f"{average_masking_ratio * 100:.4f}%"
    )

    print(
    f"Minimum valid tokens: "
        f"{minimum_valid_tokens}"
    )

    print(
    f"Maximum valid tokens: "
    f"{maximum_valid_tokens}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    run_constrained_benchmark()