from typing import Dict, Any, List, Tuple
from .states import State, GrammarState


def compile_schema_to_state(schema: Dict[str, Any]) -> State:
    """
    Compiles a JSON Schema dictionary into an initial Grammar State machine.
    Extracts required keys from the schema root to enforce precise structural generation.
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a valid dictionary.")

    # Extract object property keys if specified in JSON Schema format
    properties = schema.get("properties", {})
    required_keys: List[str] = schema.get("required", [])

    # If required list is not explicitly provided, default to all property keys
    if not required_keys and properties:
        keys_tuple: Tuple[str, ...] = tuple(properties.keys())
    else:
        keys_tuple = tuple(required_keys)

    return State(
        grammar_state=GrammarState.START,
        buffer="",
        expected_keys=keys_tuple
    )