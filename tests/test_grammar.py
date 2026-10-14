import pytest
from src.schema.states import GrammarState, State
from src.schema.grammar import delta_star, transition, _is_number_complete
from src.schema.parser import compile_schema_to_state


def test_object_start_transition():
    state = State(GrammarState.START)
    result = transition(state, "{")
    assert result.grammar_state == GrammarState.EXPECT_KEY_QUOTE


def test_key_quote_transition():
    state = State(GrammarState.EXPECT_KEY_QUOTE)
    result = transition(state, '"')
    assert result.grammar_state == GrammarState.EXPECT_KEY_NAME


def test_key_name_transition():
    state = State(GrammarState.EXPECT_KEY_NAME)
    result = transition(state, "a")
    assert result.grammar_state == GrammarState.EXPECT_KEY_NAME
    assert result.buffer == "a"


def test_colon_transition():
    state = State(GrammarState.EXPECT_COLON)
    result = transition(state, ":")
    assert result.grammar_state == GrammarState.EXPECT_VALUE


def test_string_value_routing_and_whitespace():
    # Verify transition from EXPECT_VALUE into READING_STRING
    state = State(GrammarState.EXPECT_VALUE)
    res1 = transition(state, '"')
    assert res1.grammar_state == GrammarState.READING_STRING
    assert res1.buffer == ""

    # Verify interior whitespace preservation inside string value
    res2 = transition(res1, "A")
    res3 = transition(res2, " ")
    res4 = transition(res3, "B")
    assert res4.grammar_state == GrammarState.READING_STRING
    assert res4.buffer == "A B"


def test_invalid_character_causes_dead_end():
    state = State(GrammarState.EXPECT_OBJECT_START)
    result = transition(state, "[")
    assert result.grammar_state == GrammarState.DEAD_END


def test_delta_star_valid_object():
    initial_state = State(GrammarState.START)
    result = delta_star(initial_state, '{"name":"John"}')
    assert result.grammar_state == GrammarState.DONE


def test_delta_star_spaces_in_string_value():
    initial_state = State(GrammarState.START)
    result = delta_star(initial_state, '{"name": "Rahul Sharma"}')
    assert result.grammar_state == GrammarState.DONE


def test_delta_star_numbers_and_booleans():
    initial_state = State(GrammarState.START)
    res_num = delta_star(initial_state, '{"age": 25}')
    assert res_num.grammar_state == GrammarState.DONE

    res_bool = delta_star(initial_state, '{"active": true}')
    assert res_bool.grammar_state == GrammarState.DONE


def test_delta_star_nested_object():
    """Verify that recursive object nesting (e.g. flight path telemetry) parses to DONE successfully."""
    initial_state = State(GrammarState.EXPECT_OBJECT_START)
    nested_payload = '{"flight_path": {"start": {"lat": 37.77}, "end": {"lat": 34.05}}}'
    result = delta_star(initial_state, nested_payload)
    assert result.grammar_state == GrammarState.DONE


def test_schema_enforcement_rejection():
    # Compiling schema that requires specific keys
    schema = {"properties": {"name": {}, "age": {}}, "required": ["name"]}
    initial_state = compile_schema_to_state(schema)

    # Valid key should pass
    valid_res = delta_star(initial_state, '{"name": "Alice"}')
    assert valid_res.grammar_state == GrammarState.DONE

    # Key not present in schema required/properties list must hit DEAD_END
    invalid_res = delta_star(initial_state, '{"unauthorized_key": "data"}')
    assert invalid_res.grammar_state == GrammarState.DEAD_END


@pytest.mark.parametrize(
    "value",
    ["0", "-0", "12", "-12", "3.14", "2e5", "-1.2E-3"],
)
def test_valid_json_numbers(value):
    assert _is_number_complete(value)


@pytest.mark.parametrize(
    "value",
    ["", "-", "+1", "01", "1.", ".5", "1e", "1e+", "--1"],
)
def test_invalid_json_numbers(value):
    assert not _is_number_complete(value)


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,}',
        '{"a":"hello",}',
        '{"a":true,}',
    ],
)
def test_trailing_commas_are_rejected(payload):
    result = delta_star(State(GrammarState.START), payload)
    assert result.grammar_state == GrammarState.DEAD_END


@pytest.mark.parametrize(
    "payload",
    [
        r'{"text":"\u12G4"}',
        r'{"text":"\u123"}',
        r'{"text":"\uZZZZ"}',
    ],
)
def test_invalid_unicode_escapes_are_rejected(payload):
    result = delta_star(State(GrammarState.START), payload)
    assert result.grammar_state == GrammarState.DEAD_END


@pytest.mark.parametrize(
    "payload",
    [
        r'{"text":"\u1234"}',
        r'{"text":"hello\nworld"}',
        r'{"text":"quote: \"ok\""}',
    ],
)
def test_valid_string_escapes_are_accepted(payload):
    result = delta_star(State(GrammarState.START), payload)
    assert result.grammar_state == GrammarState.DONE


def test_nested_schema_rejects_unauthorized_child_key():
    schema = {
        "type": "object",
        "properties": {
            "flight_path": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            }
        },
        "required": ["flight_path"],
        "additionalProperties": False,
    }
    initial_state = compile_schema_to_state(schema)
    result = delta_star(
        initial_state,
        '{"flight_path":{"start":37,"end":34,"unauthorized":1}}',
    )
    assert result.grammar_state == GrammarState.DEAD_END


def test_nested_schema_rejects_missing_required_child_key():
    schema = {
        "type": "object",
        "properties": {
            "flight_path": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            }
        },
        "required": ["flight_path"],
        "additionalProperties": False,
    }
    initial_state = compile_schema_to_state(schema)
    result = delta_star(
        initial_state,
        '{"flight_path":{"start":37}}',
    )
    assert result.grammar_state == GrammarState.DEAD_END


def test_nested_schema_accepts_valid_child_keys():
    schema = {
        "type": "object",
        "properties": {
            "flight_path": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            }
        },
        "required": ["flight_path"],
        "additionalProperties": False,
    }
    initial_state = compile_schema_to_state(schema)
    result = delta_star(
        initial_state,
        '{"flight_path":{"start":37,"end":34}}',
    )
    assert result.grammar_state == GrammarState.DONE


def test_schema_allows_optional_declared_property():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    initial_state = compile_schema_to_state(schema)
    result = delta_star(
        initial_state,
        '{"name":"Alice","age":25}',
    )
    assert result.grammar_state == GrammarState.DONE