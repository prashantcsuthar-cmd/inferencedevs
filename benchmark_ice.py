
import statistics
import time

import torch

from src.decoding.engine import ICEDecodingEngine
from src.models.loader import load_qwen
from test_real_ice import PROMPT, SCHEMA, validate_result


MAX_NEW_TOKENS = 100
REPETITIONS = 5


def baseline_generate(model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt")
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    start = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )

    elapsed = time.perf_counter() - start
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "text": text,
        "elapsed": elapsed,
        "tokens": len(new_ids),
    }


def ice_generate(engine):
    result = engine.generate(
        PROMPT,
        max_new_tokens=MAX_NEW_TOKENS,
        schema=SCHEMA,
    )

    return {
        "text": result["text"],
        "elapsed": result["elapsed_seconds"],
        "tokens": result["generated_tokens"],
    }


def summarize(name, runs):
    latencies = [run["elapsed"] for run in runs]
    valid_count = sum(
        validate_result(run["text"])[0] for run in runs
    )
    total_tokens = sum(run["tokens"] for run in runs)
    total_seconds = sum(latencies)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Valid outputs: {valid_count}/{len(runs)}")
    print(f"Validity rate: {valid_count / len(runs) * 100:.1f}%")
    print(f"Mean latency: {statistics.mean(latencies):.3f} s")
    print(f"Median latency: {statistics.median(latencies):.3f} s")
    print(
        "Aggregate generated tokens/sec: "
        f"{total_tokens / total_seconds:.2f}"
        if total_seconds else "Aggregate generated tokens/sec: N/A"
    )

    for index, run in enumerate(runs, start=1):
        valid, reason = validate_result(run["text"])
        print(
            f"Run {index}: {run['elapsed']:.3f}s, "
            f"{run['tokens']} tokens, "
            f"{'VALID' if valid else 'INVALID'}"
        )
        if not valid:
            print(f"  Reason: {reason}")
            print(f"  Output: {run['text']!r}")


def main():
    print("Loading model...")
    model, tokenizer = load_qwen()
    engine = ICEDecodingEngine(model, tokenizer)

    # Warm up both paths; exclude these runs from the measurements.
    print("Warming up baseline...")
    baseline_generate(model, tokenizer)

    print("Warming up ICE...")
    ice_generate(engine)

    baseline_runs = []
    ice_runs = []

    for index in range(REPETITIONS):
        print(f"\nMeasured run {index + 1}/{REPETITIONS}")

        # Alternate execution order to reduce systematic order bias.
        if index % 2 == 0:
            baseline_runs.append(baseline_generate(model, tokenizer))
            ice_runs.append(ice_generate(engine))
        else:
            ice_runs.append(ice_generate(engine))
            baseline_runs.append(baseline_generate(model, tokenizer))

    summarize("BASELINE — unconstrained generation", baseline_runs)
    summarize("ICE — schema-constrained generation", ice_runs)


if __name__ == "__main__":
    main()
