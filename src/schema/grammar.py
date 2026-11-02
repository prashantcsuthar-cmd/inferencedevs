
import re
from typing import Optional

from .states import GrammarState, State, ObjectContext, SchemaNode


WHITESPACE = {" ", "\t", "\n", "\r"}
HEX_DIGITS = set("0123456789abcdefABCDEF")

JSON_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
)

JSON_NUMBER_PREFIX_PATTERN = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?"
)


def _is_number_complete(num_str: str) -> bool:
    return JSON_NUMBER_PATTERN.fullmatch(num_str) is not None


def _dead(state: State) -> State:
    return State(
        GrammarState.DEAD_END,
        schema_root=state.schema_root,
        object_stack=state.object_stack,
    )


def _check_required_keys_present(ctx: ObjectContext) -> bool:
    return all(key in ctx.seen_keys for key in ctx.schema.required)


def _active_value_schema(state: State) -> Optional[SchemaNode]:
    return state.value_schema


def _replace_top(
    stack: tuple,
    context: ObjectContext,
) -> tuple:
    return stack[:-1] + (context,)


def _finish_value(state: State) -> State:
    """Move to the delimiter state after a complete scalar value."""
    return State(
        GrammarState.EXPECT_COMMA_OR_END,
        expected_keys=state.expected_keys,
        current_key=state.current_key,
        nesting_depth=state.nesting_depth,
        schema_root=state.schema_root,
        object_stack=state.object_stack,
        value_schema=None,
    )


def _close_object(state: State) -> State:
    """Close the innermost object after validating its required keys."""
    if not state.object_stack:
        return _dead(state)

    context = state.object_stack[-1]

    if not _check_required_keys_present(context):
        return _dead(state)

    remaining_stack = state.object_stack[:-1]

    if not remaining_stack:
        return State(
            GrammarState.DONE,
            schema_root=state.schema_root,
            object_stack=(),
            nesting_depth=0,
        )

    # The nested object's value has completed. Resume the parent object.
    return State(
        GrammarState.EXPECT_COMMA_OR_END,
        expected_keys=state.expected_keys,
        current_key=state.current_key,
        nesting_depth=max(0, state.nesting_depth - 1),
        schema_root=state.schema_root,
        object_stack=remaining_stack,
        value_schema=None,
    )


def transition(state: State, char: str) -> State:
    current = state.grammar_state

    if current == GrammarState.DEAD_END:
        return state

    if current == GrammarState.DONE:
        # A completed root object cannot be followed by more JSON content.
        return _dead(state)

    # Numbers need a delimiter to terminate their token.
    if char in WHITESPACE:
        if current == GrammarState.READING_NUM:
            if not _is_number_complete(state.buffer):
                return _dead(state)

            return _finish_value(state)

        # Whitespace is ignored outside strings and keys.
        if current not in {
            GrammarState.EXPECT_KEY_NAME,
            GrammarState.READING_STRING,
            GrammarState.READING_STRING_ESCAPE,
            GrammarState.READING_UNICODE_1,
            GrammarState.READING_UNICODE_2,
            GrammarState.READING_UNICODE_3,
            GrammarState.READING_UNICODE_4,
        }:
            return state

    # 1. Root object start.
    if current in (
        GrammarState.START,
        GrammarState.EXPECT_OBJECT_START,
    ):
        if char != "{":
            return _dead(state)

        root_schema = state.schema_root or SchemaNode(type="object")

        if root_schema.type not in (None, "object"):
            return _dead(state)

        return State(
            GrammarState.EXPECT_KEY_QUOTE,
            expected_keys=state.expected_keys,
            schema_root=root_schema,
            object_stack=(ObjectContext(schema=root_schema),),
            nesting_depth=0,
        )

    # 2. Beginning an object's key, or closing an empty object.
    if current == GrammarState.EXPECT_KEY_QUOTE:
        if char == '"':
            return State(
                GrammarState.EXPECT_KEY_NAME,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                nesting_depth=state.nesting_depth,
            )

        if char == "}":
            return _close_object(state)

        return _dead(state)

    # A comma requires another key; therefore '}' is invalid here.
    if current == GrammarState.EXPECT_KEY_QUOTE_AFTER_COMMA:
        if char != '"':
            return _dead(state)

        return State(
            GrammarState.EXPECT_KEY_NAME,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            nesting_depth=state.nesting_depth,
        )

    # 3. Key characters.
    if current == GrammarState.EXPECT_KEY_NAME:
        if not state.object_stack:
            return _dead(state)

        context = state.object_stack[-1]
        schema = context.schema

        if char == '"':
            key = state.buffer

            # Reject duplicate keys.
            if key in context.seen_keys:
                return _dead(state)

            property_schema = schema.get_property(key)

            # Reject undeclared keys when additional properties are disabled.
            if (
                property_schema is None
                and not schema.additional_properties
            ):
                return _dead(state)

            updated_context = ObjectContext(
                schema=schema,
                seen_keys=context.seen_keys + (key,),
                current_property_schema=property_schema,
            )

            return State(
                GrammarState.EXPECT_COLON,
                current_key=key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=_replace_top(
                    state.object_stack,
                    updated_context,
                ),
                value_schema=property_schema,
                nesting_depth=state.nesting_depth,
            )

        # Property names cannot contain unescaped control characters.
        # Escaped property names are not supported by this implementation.
        if ord(char) < 0x20 or char == "\\":
            return _dead(state)

        candidate = state.buffer + char

        # When additionalProperties is false, every key prefix must be
        # compatible with an unused declared property.
        if not schema.additional_properties:
            possible = any(
                key.startswith(candidate)
                and key not in context.seen_keys
                for key, _ in schema.properties
            )

            if not possible:
                return _dead(state)

        return State(
            GrammarState.EXPECT_KEY_NAME,
            buffer=candidate,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            nesting_depth=state.nesting_depth,
        )

    # 4. Colon.
    if current == GrammarState.EXPECT_COLON:
        if char != ":":
            return _dead(state)

        return State(
            GrammarState.EXPECT_VALUE,
            current_key=state.current_key,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            value_schema=state.value_schema,
            nesting_depth=state.nesting_depth,
        )

    # 5. Route a value using its own schema.
    if current == GrammarState.EXPECT_VALUE:
        active_schema = _active_value_schema(state)
        expected_type = active_schema.type if active_schema else None

        if char == "{":
            if expected_type not in (None, "object"):
                return _dead(state)

            child_schema = active_schema or SchemaNode(type="object")
            child_context = ObjectContext(schema=child_schema)

            return State(
                GrammarState.EXPECT_KEY_QUOTE,
                expected_keys=state.expected_keys,
                current_key=state.current_key,
                schema_root=state.schema_root,
                object_stack=state.object_stack + (child_context,),
                value_schema=None,
                nesting_depth=state.nesting_depth + 1,
            )

        if char == '"':
            if expected_type not in (None, "string"):
                return _dead(state)

            return State(
                GrammarState.READING_STRING,
                buffer="",
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=active_schema,
                nesting_depth=state.nesting_depth,
            )

        if char in "0123456789-":
            if expected_type not in (None, "number"):
                return _dead(state)

            return State(
                GrammarState.READING_NUM,
                buffer=char,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=active_schema,
                nesting_depth=state.nesting_depth,
            )

        if char in ("t", "f"):
            if expected_type not in (None, "boolean"):
                return _dead(state)

            literal = "true" if char == "t" else "false"

            return State(
                GrammarState.READING_LITERAL,
                buffer=char,
                target_literal=literal,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=active_schema,
                nesting_depth=state.nesting_depth,
            )

        if char == "n":
            if expected_type not in (None, "null"):
                return _dead(state)

            return State(
                GrammarState.READING_LITERAL,
                buffer="n",
                target_literal="null",
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=active_schema,
                nesting_depth=state.nesting_depth,
            )

        return _dead(state)

    # 6. String value.
    if current == GrammarState.READING_STRING:
        if char == "\\":
            return State(
                GrammarState.READING_STRING_ESCAPE,
                buffer=state.buffer,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=state.value_schema,
                nesting_depth=state.nesting_depth,
            )

        if char == '"':
            return _finish_value(state)

        if ord(char) < 0x20:
            return _dead(state)

        return State(
            GrammarState.READING_STRING,
            buffer=state.buffer + char,
            current_key=state.current_key,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            value_schema=state.value_schema,
            nesting_depth=state.nesting_depth,
        )

    # 7. String escapes.
    if current == GrammarState.READING_STRING_ESCAPE:
        if char in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
            return State(
                GrammarState.READING_STRING,
                buffer=state.buffer,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=state.value_schema,
                nesting_depth=state.nesting_depth,
            )

        if char == "u":
            return State(
                GrammarState.READING_UNICODE_1,
                buffer=state.buffer,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=state.value_schema,
                nesting_depth=state.nesting_depth,
            )

        return _dead(state)

    unicode_states = {
        GrammarState.READING_UNICODE_1: GrammarState.READING_UNICODE_2,
        GrammarState.READING_UNICODE_2: GrammarState.READING_UNICODE_3,
        GrammarState.READING_UNICODE_3: GrammarState.READING_UNICODE_4,
    }

    if current in unicode_states:
        if char not in HEX_DIGITS:
            return _dead(state)

        return State(
            unicode_states[current],
            buffer=state.buffer,
            current_key=state.current_key,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            value_schema=state.value_schema,
            nesting_depth=state.nesting_depth,
        )

    if current == GrammarState.READING_UNICODE_4:
        if char not in HEX_DIGITS:
            return _dead(state)

        return State(
            GrammarState.READING_STRING,
            buffer=state.buffer,
            current_key=state.current_key,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            value_schema=state.value_schema,
            nesting_depth=state.nesting_depth,
        )

    # 8. Number.
    if current == GrammarState.READING_NUM:
        if char in "0123456789.eE+-":
            candidate = state.buffer + char

            # Permit only prefixes that could still become valid numbers.
            if not JSON_NUMBER_PREFIX_PATTERN.fullmatch(candidate):
                return _dead(state)

            return State(
                GrammarState.READING_NUM,
                buffer=candidate,
                current_key=state.current_key,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                value_schema=state.value_schema,
                nesting_depth=state.nesting_depth,
            )

        if not _is_number_complete(state.buffer):
            return _dead(state)

        # A delimiter terminates the number. Process it in the completed
        # value state so commas, closing braces, and whitespace are handled
        # consistently.
        completed = _finish_value(state)
        return transition(completed, char)

    # 9. Boolean and null literals.
    if current == GrammarState.READING_LITERAL:
        candidate = state.buffer + char

        if not state.target_literal.startswith(candidate):
            return _dead(state)

        if candidate == state.target_literal:
            return _finish_value(state)

        return State(
            GrammarState.READING_LITERAL,
            buffer=candidate,
            target_literal=state.target_literal,
            current_key=state.current_key,
            expected_keys=state.expected_keys,
            schema_root=state.schema_root,
            object_stack=state.object_stack,
            value_schema=state.value_schema,
            nesting_depth=state.nesting_depth,
        )

    # 10. Comma or object close.
    if current == GrammarState.EXPECT_COMMA_OR_END:
        if char == ",":
            if not state.object_stack:
                return _dead(state)

            context = state.object_stack[-1]
            schema = context.schema

            # If extra properties are prohibited, a comma is only legal
            # when at least one declared property remains unused.
            has_unused_declared_property = any(
                key not in context.seen_keys
                for key, _ in schema.properties
            )

            if (
                not schema.additional_properties
                and not has_unused_declared_property
            ):
                return _dead(state)

            return State(
                GrammarState.EXPECT_KEY_QUOTE_AFTER_COMMA,
                expected_keys=state.expected_keys,
                schema_root=state.schema_root,
                object_stack=state.object_stack,
                nesting_depth=state.nesting_depth,
            )

        if char == "}":
            return _close_object(state)

        return _dead(state)

    return _dead(state)


def delta_star(initial_state: State, text: str) -> State:
    state = initial_state

    for char in text:
        state = transition(state, char)

        if state.grammar_state == GrammarState.DEAD_END:
            return state

    # An unfinished number is a viable prefix only if it could still
    # become a valid JSON number. It is not yet a completed value.
    if (
        state.grammar_state == GrammarState.READING_NUM
        and not JSON_NUMBER_PREFIX_PATTERN.fullmatch(state.buffer)
    ):
        return _dead(state)

    # A Unicode escape must contain exactly four hexadecimal digits.
    # If generation ends during the escape, the string is incomplete.
    if state.grammar_state in {
        GrammarState.READING_UNICODE_1,
        GrammarState.READING_UNICODE_2,
        GrammarState.READING_UNICODE_3,
        GrammarState.READING_UNICODE_4,
    }:
        return _dead(state)

    return state
