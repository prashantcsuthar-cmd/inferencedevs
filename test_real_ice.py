
import json

from src.decoding.engine import ICEDecodingEngine
from src.models.loader import load_qwen


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age", "active"],
    "additionalProperties": False,
}

PROMPT = """
Return exactly one JSON object with these fields:
- name: a string
- age: a number
- active: a boolean

Use these values: name is "Alex", age is 25, active is true.
Do not include Markdown fences or explanatory text.
"""


def validate_result(text: str) -> tuple[bool, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"

    if not isinstance(data, dict):
        return False, "Output is not a JSON object."

    expected_keys = {"name", "age", "active"}
    if set(data.keys()) != expected_keys:
        return False, f"Unexpected keys: {set(data.keys())}"

    if not isinstance(data["name"], str):
        return False, "'name' must be a string."

    # bool is a subclass of int in Python, so explicitly exclude it.
    if isinstance(data["age"], bool) or not isinstance(
        data["age"], (int, float)
    ):
        return False, "'age' must be a number."

    if not isinstance(data["active"], bool):
        return False, "'active' must be a boolean."

    return True, "Valid JSON object; required keys and types match."


def main():
    print("Loading Qwen model and tokenizer...")
    model, tokenizer = load_qwen()

    print("Initializing ICE...")
    engine = ICEDecodingEngine(model, tokenizer)

    print("Running schema-constrained generation...")
    result = engine.generate(
        PROMPT,
        max_new_tokens=100,
        schema=SCHEMA,
    )

    generated_text = result["text"]
    valid, reason = validate_result(generated_text)

    print("\n--- GENERATED OUTPUT ---")
    print(generated_text)

    print("\n--- VALIDATION ---")
    print("Schema check:", "PASS" if valid else "FAIL")
    print("Reason:", reason)
    print("Generated tokens:", result["generated_tokens"])
    print(f"Elapsed: {result['elapsed_seconds']:.3f} seconds")
    print(f"Final grammar state: {result['final_state']}")

    if not valid:
        raise SystemExit(
            "REAL-MODEL VERIFICATION FAILED. Do not claim schema "
            "conformance based on this run."
        )

    print("\nREAL-MODEL VERIFICATION PASSED")


if __name__ == "__main__":
    main()
