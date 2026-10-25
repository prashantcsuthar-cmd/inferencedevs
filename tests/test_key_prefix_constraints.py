
import pytest

from src.schema.grammar import delta_star
from src.schema.parser import compile_schema_to_state
from src.schema.states import GrammarState


@pytest.fixture
def device_state():
    return compile_schema_to_state({
        "type": "object",
        "properties": {
            "device": {"type": "string"},
            "enabled": {"type": "boolean"},
        },
        "required": ["device", "enabled"],
        "additionalProperties": False,
    })


def test_valid_device_key_prefix_is_allowed(device_state):
    state = delta_star(device_state, '{"dev')

    assert state.grammar_state == GrammarState.EXPECT_KEY_NAME
    assert state.buffer == "dev"


def test_impossible_device_key_prefix_is_rejected(device_state):
    state = delta_star(device_state, '{"id')

    assert state.grammar_state == GrammarState.DEAD_END


def test_undeclared_key_is_rejected(device_state):
    state = delta_star(device_state, '{"id_device"')

    assert state.grammar_state == GrammarState.DEAD_END


def test_declared_key_can_be_completed(device_state):
    state = delta_star(device_state, '{"device"')

    assert state.grammar_state == GrammarState.EXPECT_COLON


def test_already_seen_key_cannot_be_repeated(device_state):
    state = delta_star(
        device_state,
        '{"device":"sensor","dev',
    )

    assert state.grammar_state == GrammarState.DEAD_END
