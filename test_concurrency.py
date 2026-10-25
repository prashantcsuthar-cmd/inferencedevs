
import json
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "http://127.0.0.1:8000/stream"

SCHEMAS = {
    "person": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
        "required": ["name", "age"],
        "additionalProperties": False,
    },
    "device": {
        "type": "object",
        "properties": {
            "device": {"type": "string"},
            "enabled": {"type": "boolean"},
        },
        "required": ["device", "enabled"],
        "additionalProperties": False,
    },
}


def run_request(kind):
    prompts = {
        "person": (
            "Return a JSON person object with name Alex and age 25."
        ),
        "device": (
            "Return a JSON device object with device sensor and "
            "enabled true."
        ),
    }

    response = requests.get(
        URL,
        params={
            "prompt": prompts[kind],
            "schema": json.dumps(SCHEMAS[kind]),
            "max_steps": 128,
        },
        stream=True,
        timeout=300,
    )
    response.raise_for_status()

    terminal = None
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        event = json.loads(line[6:])
        if event.get("runner_type") == "terminal":
            terminal = event
            break

    response.close()

    if terminal is None:
        return kind, False, "No terminal event"

    result = terminal["ice"]
    return kind, (
        result["generation_completed"]
        and result["json_valid"]
        and result["schema_valid"] is True
    ), result


if __name__ == "__main__":
    # Repeat requests to increase the chance of exposing state leakage.
    jobs = ["person", "device"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_request, jobs))

    for kind, passed, details in results:
        print(f"{kind}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(details)

    if not all(passed for _, passed, _ in results):
        raise SystemExit("Concurrency test failed")

    print("All concurrent-request checks passed.")
