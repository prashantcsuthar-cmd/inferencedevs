
import json
import requests

schema = {
    "type": "object",
    "properties": {
        "device": {"type": "string"},
        "enabled": {"type": "boolean"},
    },
    "required": ["device", "enabled"],
    "additionalProperties": False,
}

params = {
    "prompt": (
        'Return a JSON object with device "sensor" and enabled true. '
        "Output only the JSON object."
    ),
    "schema": json.dumps(schema),
    "max_steps": 128,
}

with requests.get(
    "http://127.0.0.1:8000/stream",
    params=params,
    stream=True,
    timeout=(5, 180),
) as response:
    print("HTTP:", response.status_code)
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue

        if isinstance(line, bytes):
            line = line.decode("utf-8")

        if line.startswith("data: "):
            event = json.loads(line[6:])

            if event.get("runner_type") == "terminal":
                print(json.dumps(event, indent=2))
                break
