import gc
import os
import sys
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB on Windows/Linux."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

        pid = os.getpid()
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid
        )
        if handle:
            try:
                if ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return counters.WorkingSetSize / (1024 * 1024)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        return 0.0
    else:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    print("=" * 60)
    print("ICE PHASE 4.4 - MEMORY RESOURCE BENCHMARK")
    print("=" * 60)

    mem_baseline = get_process_memory_mb()
    print(f"Baseline RSS Memory: {mem_baseline:.2f} MB")

    # Force correct import tracking by pinning path constraints
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from src.decoding.engine import ICEDecodingEngine

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
    )

    print("\nCreating ICE Engine...")
    engine = ICEDecodingEngine(model=model, tokenizer=tokenizer)
    engine_loaded_mem = get_process_memory_mb()
    print(f"Engine Loaded RSS Memory: {engine_loaded_mem:.2f} MB")
    print(f"Model Runtime Weight Footprint: {engine_loaded_mem - mem_baseline:.2f} MB")

    test_prompt = "Extract user information into schema structure: Rahul Sharma, ORD12345, Late Delivery."

    print("\nRunning test generation run...")
    mem_before_gen = get_process_memory_mb()
    start_time = time.time()

    # Call engine matching its actual method signature
    result = engine.generate(test_prompt, max_new_tokens=64)

    elapsed = time.time() - start_time
    mem_after_gen = get_process_memory_mb()

    # Dynamic result parsing safely handles structural return types
    if isinstance(result, dict):
        generated_text = result.get("text", "")
        ms_per_token = result.get("ms_per_token", 0.0)
        tokens = result.get("generated_tokens", 0)
    else:
        generated_text = str(result)
        tokens = len(tokenizer.encode(generated_text))
        ms_per_token = (elapsed / tokens * 1000) if tokens > 0 else 0.0

    print("\n" + "-" * 40)
    print("INFERENCE PROFILE RESULTS")
    print("-" * 40)
    print(f"Generated text: {generated_text!r}")
    print(f"Generated tokens: {tokens}")
    print(f"Total time: {elapsed:.4f}s | Latency: {ms_per_token:.2f} ms/token")
    print(f"Peak RSS Memory: {mem_after_gen:.2f} MB")
    print(f"Active Inference Transient Memory Overhead: {mem_after_gen - mem_before_gen:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()