"""Compiler for ICE's deliberately limited JSON Schema subset.

Enforced keywords: type, properties, required, additionalProperties, title,
description. Annotation keywords are accepted but do not affect validation.
Arrays, numeric bounds, enums, patterns, unions, references, and other schema
features are not implemented and are rejected rather than silently ignored.
"""
from typing import Any, Dict
from .states import GrammarState, SchemaNode, State

SUPPORTED_TYPES = {"object", "string", "number", "boolean", "null"}
SUPPORTED_KEYWORDS = {"type", "properties", "required", "additionalProperties", "title", "description"}


def _compile_node(schema: Dict[str, Any], path: str = "$", *, root: bool = False) -> SchemaNode:
    if not isinstance(schema, dict):
        raise ValueError(f"Schema at {path} must be an object.")

    unsupported = set(schema) - SUPPORTED_KEYWORDS
    if unsupported:
        names = ", ".join(sorted(str(k) for k in unsupported))
        raise ValueError(f"Unsupported schema keyword(s) at {path}: {names}")

    schema_type = schema.get("type")
    if schema_type is None and any(k in schema for k in ("properties", "required", "additionalProperties")):
        schema_type = "object"
    if schema_type is not None and schema_type not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported schema type at {path}: {schema_type!r}; supported types: {sorted(SUPPORTED_TYPES)}")
    if root and schema_type not in (None, "object"):
        raise ValueError("The root schema must describe a JSON object.")

    raw_properties = schema.get("properties", {})
    raw_required = schema.get("required", [])
    additional = schema.get("additionalProperties", False)
    if not isinstance(raw_properties, dict) or not all(isinstance(k, str) for k in raw_properties):
        raise ValueError(f"'properties' at {path} must be an object with string keys.")
    if not isinstance(raw_required, list) or not all(isinstance(k, str) for k in raw_required):
        raise ValueError(f"'required' at {path} must be a list of strings.")
    if len(raw_required) != len(set(raw_required)):
        raise ValueError(f"'required' at {path} must not contain duplicate keys.")
    if not isinstance(additional, bool):
        raise ValueError(f"'additionalProperties' at {path} must be boolean.")
    if schema_type not in (None, "object") and (raw_properties or raw_required or "additionalProperties" in schema):
        raise ValueError(f"Object keywords are only supported for object schemas at {path}.")
    unknown_required = set(raw_required) - set(raw_properties)
    if unknown_required:
        raise ValueError(f"'required' at {path} contains undeclared keys: {', '.join(sorted(unknown_required))}")

    properties = tuple((key, _compile_node(child, f"{path}.properties.{key}")) for key, child in raw_properties.items())
    return SchemaNode(type=schema_type, properties=properties, required=tuple(raw_required), additional_properties=additional)


def compile_schema_to_state(schema: Dict[str, Any]) -> State:
    root_node = _compile_node(schema, root=True)
    if root_node.type is None:
        root_node = SchemaNode(type="object", properties=root_node.properties, required=root_node.required, additional_properties=root_node.additional_properties)
    return State(grammar_state=GrammarState.START, expected_keys=tuple(key for key, _ in root_node.properties), schema_root=root_node, object_stack=(), value_schema=root_node)
