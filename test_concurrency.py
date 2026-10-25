
import json
from concurrent.futures import ThreadPoolExecutor

import requests
from jsonschema import Draft202012Validator

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
        "person": "Return a JSON person object with name Alex and age 25.",
        "device": (
            "Return a JSON device object with device sensor "
            "and enabled true."
        ),
    }

    try:
        with requests.get(
            URL,
            params={
                "prompt": prompts[kind],
                "schema": json.dumps(SCHEMAS[kind]),
                "max_steps": 128,
            },
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get(
                "content-type", ""
            ).lower()

            if "text/event-stream" not in content_type:
                return kind, False, (
                    f"Unexpected Content-Type: {content_type}"
                )

            seen_runners = set()
            terminal = None

            for line in response.iter_lines(
                decode_unicode=True
            ):
                if not line or not line.startswith("data: "):
                    continue

                event = json.loads(line[6:])
                runner = event.get("runner_type")

                if runner in {"ice", "baseline"}:
                    seen_runners.add(runner)

                if runner == "terminal":
                    terminal = event
                    break

        if terminal is None:
            return kind, False, "No terminal event"

        if not {"ice", "baseline"}.issubset(seen_runners):
            return kind, False, (
                f"Missing stream events: {seen_runners}"
            )

        if "ice" not in terminal or "baseline" not in terminal:
            return kind, False, (
                "Terminal event does not separate ICE and baseline"
            )

        ice = terminal["ice"]
        baseline = terminal["baseline"]

        if not ice.get("generation_completed"):
            return kind, False, {
                "failure": "ICE generation did not complete",
                "ice": ice,
            }

        if not ice.get("json_valid"):
            return kind, False, {
                "failure": "Server reports invalid ICE JSON",
                "ice": ice,
            }

        if ice.get("schema_valid") is not True:
            return kind, False, {
                "failure": "Server reports invalid ICE schema",
                "ice": ice,
            }

        # Independently validate the actual returned ICE text.
        parsed = json.loads(ice["text"])
        Draft202012Validator(SCHEMAS[kind]).validate(parsed)

        # Confirm that the baseline has its own report and text.
        if "text" not in baseline or "status" not in baseline:
            return kind, False, {
                "failure": "Incomplete baseline report",
                "baseline": baseline,
            }

        return kind, True, {
            "ice_output": parsed,
            "baseline_status": baseline["status"],
            "baseline_json_valid": baseline["json_valid"],
            "baseline_schema_valid": baseline["schema_valid"],
        }

    except Exception as exc:
        return kind, False, f"{type(exc).__name__}: {exc}"


if __name__ == "__main__":
    jobs = ["person", "device"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run_request, jobs))

    for kind, passed, details in results:
        print(f"{kind}: {'PASS' if passed else 'FAIL'}")
        print(details)

    if not all(passed for _, passed, _ in results):
        raise SystemExit("Concurrency test failed")

    print("All concurrent-request checks passed.")
