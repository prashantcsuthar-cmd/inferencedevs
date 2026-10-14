
from typing import Any, Dict

from .states import GrammarState, SchemaNode, State


SUPPORTED_TYPES = {"object", "string", "number", "boolean", "null"}


def _compile_node(schema: Dict[str, Any]) -> SchemaNode:
    if not isinstance(schema, dict):
        raise ValueError("Each schema definition must be a dictionary.")

    schema_type = schema.get("type")
    if schema_type is not None and schema_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported schema type: {schema_type!r}. "
            f"Supported types: {sorted(SUPPORTED_TYPES)}"
        )

    raw_properties = schema.get("properties", {})
    raw_required = schema.get("required", [])
    additional = schema.get("additionalProperties", True)

    if not isinstance(raw_properties, dict):
        raise ValueError("'properties' must be a dictionary.")

    if not isinstance(raw_required, list) or not all(
        isinstance(key, str) for key in raw_required
    ):
        raise ValueError("'required' must be a list of strings.")

    if len(raw_required) != len(set(raw_required)):
        raise ValueError("'required' must not contain duplicate keys.")

    if not isinstance(additional, bool):
        raise ValueError("'additionalProperties' must be a boolean.")

    if schema_type not in (None, "object") and (
        raw_properties or raw_required or "additionalProperties" in schema
    ):
        raise ValueError(
            "'properties', 'required', and 'additionalProperties' "
            "are only supported for object schemas."
        )

    unknown_required = set(raw_required) - set(raw_properties)
    if unknown_required:
        raise ValueError(
            "'required' contains keys not declared in 'properties': "
            f"{sorted(unknown_required)}"
        )

    compiled_properties = tuple(
        (key, _compile_node(child_schema))
        for key, child_schema in raw_properties.items()
    )

    return SchemaNode(
        type=schema_type,
        properties=compiled_properties,
        required=tuple(raw_required),
        additional_properties=additional,
    )


def compile_schema_to_state(schema: Dict[str, Any]) -> State:
    root_node = _compile_node(schema)

    if root_node.type not in (None, "object"):
        raise ValueError("The root schema must describe a JSON object.")

    return State(
        grammar_state=GrammarState.START,
        expected_keys=tuple(key for key, _ in root_node.properties),
        schema_root=root_node,
        object_stack=(),
        value_schema=root_node,
    )
