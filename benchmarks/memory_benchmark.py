"""Phase 4.4 - ICE memory and resource diagnostics."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil


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


PROCESS = psutil.Process(
    os.getpid()
)


def memory_mb() -> float:
    """Return current process RSS in MB."""
    return (
        PROCESS.memory_info().rss
        / (1024 * 1024)
    )


def json_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (
        json.JSONDecodeError,
        TypeError
    ):
        return False


def run_memory_benchmark():

    print("=" * 60)
    print("ICE PHASE 4.4 - MEMORY DIAGNOSTICS")
    print("=" * 60)

    gc.collect()

    baseline_memory = memory_mb()

    print(
        f"\nMemory before model loading: "
        f"{baseline_memory:.2f} MB"
    )

    print("\nLoading Qwen...")

    model, tokenizer = load_qwen()

    gc.collect()

    model_memory = memory_mb()

    print(
        f"Memory after model loading: "
        f"{model_memory:.2f} MB"
    )

    print("\nCreating ICE decoding engine...")

    engine_start = time.perf_counter()

    engine = ICEDecodingEngine(
        model,
        tokenizer
    )

    engine_creation_time = (
        time.perf_counter()
        - engine_start
    )

    gc.collect()

    engine_memory = memory_mb()

    print(
        f"Memory after ICE initialization: "
        f"{engine_memory:.2f} MB"
    )

    print(
        f"ICE initialization time: "
        f"{engine_creation_time:.2f} seconds"
    )

    prompt = """
Extract the following information as JSON.

Customer: Rahul Sharma
Order ID: ORD12345
Issue: Package was delivered late.

Required JSON fields:
customer_name
order_id
issue
"""

    print("\nRunning constrained generation...")

    generation_start_memory = memory_mb()

    generation_start = time.perf_counter()

    result = engine.generate(
        prompt,
        max_new_tokens=128
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    generation_end_memory = memory_mb()

    peak_memory = max(
        generation_start_memory,
        generation_end_memory
    )

    print("\nGenerated output:")
    print(result["text"])

    print("\n" + "=" * 60)
    print("PHASE 4.4 MEMORY SUMMARY")
    print("=" * 60)

    print(
        f"Initial process memory: "
        f"{baseline_memory:.2f} MB"
    )

    print(
        f"After model loading: "
        f"{model_memory:.2f} MB"
    )

    print(
        f"After ICE initialization: "
        f"{engine_memory:.2f} MB"
    )

    print(
        f"Before generation: "
        f"{generation_start_memory:.2f} MB"
    )

    print(
        f"After generation: "
        f"{generation_end_memory:.2f} MB"
    )

    print(
        f"Observed generation memory: "
        f"{peak_memory:.2f} MB"
    )

    print(
        f"Model memory increase: "
        f"{model_memory - baseline_memory:.2f} MB"
    )

    print(
        f"ICE initialization increase: "
        f"{engine_memory - model_memory:.2f} MB"
    )

    print(
        f"Generation memory increase: "
        f"{peak_memory - generation_start_memory:.2f} MB"
    )

    print(
        f"Generation time: "
        f"{generation_time:.4f} seconds"
    )

    print(
        f"Generated tokens: "
        f"{result['generated_tokens']}"
    )

    print(
        f"Latency: "
        f"{result['ms_per_token']:.2f} ms/token"
    )

    print(
        f"Valid JSON: "
        f"{json_valid(result['text'])}"
    )

    print("=" * 60)


if __name__ == "__main__":
    run_memory_benchmark()