"""Phase 4.3 - Larger ICE benchmark."""

from __future__ import annotations

import csv
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


RESULTS_DIR = (
    PROJECT_ROOT
    / "benchmarks"
    / "results"
)

RESULTS_FILE = (
    RESULTS_DIR
    / "phase4_results.csv"
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


PROMPTS = [

    (
        "Customer: Rahul Sharma. "
        "Order ID: ORD12345. "
        "Issue: Package was delivered late."
    ),

    (
        "Customer: Priya Kumar. "
        "Order ID: ORD98765. "
        "Issue: Product arrived damaged."
    ),

    (
        "Customer: Arjun Mehta. "
        "Order ID: ORD54321. "
        "Issue: Wrong item was delivered."
    ),

    (
        "Customer: Neha Singh. "
        "Order ID: ORD10001. "
        "Issue: Package has not arrived."
    ),

    (
        "Customer: Rohan Gupta. "
        "Order ID: ORD10002. "
        "Issue: Customer received the wrong size."
    ),

    (
        "Customer: Ananya Rao. "
        "Order ID: ORD10003. "
        "Issue: Customer wants a refund."
    ),

    (
        "Customer: Vikram Patel. "
        "Order ID: ORD10004. "
        "Issue: Payment was charged twice."
    ),

    (
        "Customer: Sneha Iyer. "
        "Order ID: ORD10005. "
        "Issue: Tracking information is unavailable."
    ),

    (
        "Customer: Karan Shah. "
        "Order ID: ORD10006. "
        "Issue: Package arrived with missing items."
    ),

    (
        "Customer: Meera Joshi. "
        "Order ID: ORD10007. "
        "Issue: Delivery address needs to be changed."
    ),

    (
        "Customer: Aditya Verma. "
        "Order ID: ORD10008. "
        "Issue: Order was cancelled unexpectedly."
    ),

    (
        "Customer: Pooja Nair. "
        "Order ID: ORD10009. "
        "Issue: Customer received an empty package."
    ),

    (
        "Customer: Rohit Kapoor. "
        "Order ID: ORD10010. "
        "Issue: Product stopped working."
    ),

    (
        "Customer: Kavya Menon. "
        "Order ID: ORD10011. "
        "Issue: Customer received a duplicate order."
    ),

    (
        "Customer: Sameer Khan. "
        "Order ID: ORD10012. "
        "Issue: Delivery was attempted at the wrong address."
    ),

    (
        "Customer: Divya Reddy. "
        "Order ID: ORD10013. "
        "Issue: Customer cannot track the shipment."
    ),

    (
        "Customer: Nikhil Das. "
        "Order ID: ORD10014. "
        "Issue: Product arrived with a broken part."
    ),

    (
        "Customer: Aisha Thomas. "
        "Order ID: ORD10015. "
        "Issue: Customer was charged an incorrect amount."
    ),

    (
        "Customer: Manish Sethi. "
        "Order ID: ORD10016. "
        "Issue: Customer wants to change the order."
    ),

    (
        "Customer: Isha Kulkarni. "
        "Order ID: ORD10017. "
        "Issue: Package was delivered to a neighbor."
    ),
]


def build_prompt(
    information: str
) -> str:

    return f"""
Extract the following information as JSON.

{information}

Required JSON fields:
customer_name
order_id
issue
"""


def run_large_benchmark():

    print("=" * 60)
    print("ICE PHASE 4.3 - LARGE BENCHMARK")
    print("=" * 60)

    print("\nLoading Qwen...")

    model, tokenizer = load_qwen()

    print("\nCreating ICE decoding engine...")

    engine = ICEDecodingEngine(
        model,
        tokenizer
    )

    results = []

    print(
        f"\nRunning {len(PROMPTS)} benchmark cases..."
    )

    for index, information in enumerate(
        PROMPTS,
        start=1
    ):

        print(
            f"\n--- Case {index}/{len(PROMPTS)} ---"
        )

        prompt = build_prompt(
            information
        )

        result = engine.generate(
            prompt,
            max_new_tokens=128
        )

        json_valid = check_json_validity(
            result["text"]
        )

        grammar_done = (
            result["final_state"].grammar_state
            == GrammarState.DONE
        )

        row = {
            "case": index,
            "json_valid": json_valid,
            "grammar_done": grammar_done,
            "generated_tokens": (
                result["generated_tokens"]
            ),
            "elapsed_seconds": (
                result["elapsed_seconds"]
            ),
            "ms_per_token": (
                result["ms_per_token"]
            ),
            "vocabulary_size": (
                result["vocabulary_size"]
            ),
            "average_valid_tokens": (
                result["average_valid_tokens"]
            ),
            "average_masked_tokens": (
                result["average_masked_tokens"]
            ),
            "average_masking_ratio": (
                result["average_masking_ratio"]
            ),
            "minimum_valid_tokens": (
                result["minimum_valid_tokens"]
            ),
            "maximum_valid_tokens": (
                result["maximum_valid_tokens"]
            ),
        }

        results.append(row)

        print(
            f"JSON valid: {json_valid}"
        )

        print(
            f"Grammar DONE: {grammar_done}"
        )

        print(
            f"Tokens: "
            f"{row['generated_tokens']}"
        )

        print(
            f"Latency: "
            f"{row['ms_per_token']:.2f} ms/token"
        )

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------

    total_cases = len(results)

    valid_json_count = sum(
        row["json_valid"]
        for row in results
    )

    done_count = sum(
        row["grammar_done"]
        for row in results
    )

    average_tokens = (
        sum(
            row["generated_tokens"]
            for row in results
        )
        / total_cases
    )

    average_latency = (
        sum(
            row["ms_per_token"]
            for row in results
        )
        / total_cases
    )

    minimum_latency = min(
        row["ms_per_token"]
        for row in results
    )

    maximum_latency = max(
        row["ms_per_token"]
        for row in results
    )

    average_masking_ratio = (
        sum(
            row["average_masking_ratio"]
            for row in results
        )
        / total_cases
    )

    average_valid_tokens = (
        sum(
            row["average_valid_tokens"]
            for row in results
        )
        / total_cases
    )

    average_masked_tokens = (
        sum(
            row["average_masked_tokens"]
            for row in results
        )
        / total_cases
    )

    vocabulary_size = results[0][
        "vocabulary_size"
    ]

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        results[0].keys()
    )

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 4.3 SUMMARY")
    print("=" * 60)

    print(
        f"Benchmark cases: "
        f"{total_cases}"
    )

    print(
        f"JSON validity: "
        f"{valid_json_count / total_cases * 100:.2f}%"
    )

    print(
        f"Grammar DONE rate: "
        f"{done_count / total_cases * 100:.2f}%"
    )

    print(
        f"Average generated tokens: "
        f"{average_tokens:.2f}"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms/token"
    )

    print(
        f"Minimum latency: "
        f"{minimum_latency:.2f} ms/token"
    )

    print(
        f"Maximum latency: "
        f"{maximum_latency:.2f} ms/token"
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
        f"\nResults saved to:"
        f"\n{RESULTS_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":

    run_large_benchmark()