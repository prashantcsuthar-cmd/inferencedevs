from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


class GrammarState(Enum):
    START = auto()
    EXPECT_OBJECT_START = auto()
    EXPECT_KEY_QUOTE = auto()
    EXPECT_KEY_NAME = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    READING_STRING = auto()
    READING_STRING_ESCAPE = auto()
    READING_NUM = auto()
    READING_LITERAL = auto()
    EXPECT_COMMA_OR_END = auto()
    DONE = auto()
    DEAD_END = auto()


@dataclass(frozen=True)
class State:
    grammar_state: GrammarState
    buffer: str = ""
    target_literal: str = ""
    expected_keys: Tuple[str, ...] = field(default_factory=tuple)
    current_key: Optional[str] = None