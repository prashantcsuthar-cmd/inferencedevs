
import json
import requests

URL = "http://127.0.0.1:8000/stream"

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}

params = {
    "prompt": (
        "Return a JSON person object with name Alex and age 25. "
        "Output only the JSON object."
    ),
    "schema": json.dumps(schema),
    "max_steps": 128,
}

try:
    with requests.get(
        URL,
        params=params,
        stream=True,
        timeout=(5, 180),
    ) as response:
        print("HTTP status:", response.status_code)

        if not response.ok:
            print("API error response:")
            print(response.text)
            raise SystemExit(1)

        print("Reading SSE stream...")

        terminal_found = False

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode("utf-8")

            if not line.startswith("data: "):
                continue

            event = json.loads(line[6:])

            if event.get("runner_type") == "terminal":
                terminal_found = True
                print("\nTerminal result:")
                print(json.dumps(event, indent=2))
                break

        if not terminal_found:
            raise RuntimeError(
                "Stream ended without a terminal event."
            )

except requests.RequestException as exc:
    print("Request failed:", repr(exc))
    raise SystemExit(1)
