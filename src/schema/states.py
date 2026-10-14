
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class GrammarState(Enum):
    START = 1
    EXPECT_OBJECT_START = 2
    EXPECT_KEY_QUOTE = 3
    EXPECT_KEY_NAME = 4
    EXPECT_COLON = 5
    EXPECT_VALUE = 6
    READING_STRING = 7
    READING_STRING_ESCAPE = 8
    READING_NUM = 9
    READING_LITERAL = 10
    EXPECT_COMMA_OR_END = 11
    DONE = 12
    DEAD_END = 13
    EXPECT_KEY_QUOTE_AFTER_COMMA = 14
    READING_UNICODE_1 = 15
    READING_UNICODE_2 = 16
    READING_UNICODE_3 = 17
    READING_UNICODE_4 = 18


@dataclass(frozen=True)
class SchemaNode:
    """Immutable representation of the supported JSON Schema subset."""
    type: Optional[str] = None
    properties: Tuple[Tuple[str, "SchemaNode"], ...] = ()
    required: Tuple[str, ...] = ()
    additional_properties: bool = True

    def get_property(self, name: str) -> Optional["SchemaNode"]:
        for property_name, node in self.properties:
            if property_name == name:
                return node
        return None


@dataclass(frozen=True)
class ObjectContext:
    """Schema and key tracking for one currently open JSON object."""
    schema: SchemaNode
    seen_keys: Tuple[str, ...] = ()
    current_property_schema: Optional[SchemaNode] = None


@dataclass(frozen=True)
class State:
    grammar_state: GrammarState
    buffer: str = ""
    target_literal: str = ""
    expected_keys: Tuple[str, ...] = ()
    current_key: Optional[str] = None
    nesting_depth: int = 0
    schema_root: Optional[SchemaNode] = None
    object_stack: Tuple[ObjectContext, ...] = ()
    # Schema of the value currently being parsed, independent of the
    # schema of the object containing that value.
    value_schema: Optional[SchemaNode] = None
