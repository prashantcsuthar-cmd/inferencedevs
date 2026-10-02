from enum import Enum, auto
from dataclasses import dataclass


class GrammarState(Enum):
    EXPECT_OBJECT_START = auto()
    EXPECT_KEY_QUOTE = auto()
    EXPECT_KEY_NAME = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE = auto()
    EXPECT_COMMA_OR_END = auto()
    DONE = auto()
    DEAD_END = auto()


@dataclass
class State:
    grammar_state: GrammarState
    buffer: str = ""