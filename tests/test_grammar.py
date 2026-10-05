import pytest
from src.schema.states import GrammarState, State
from src.schema.grammar import delta_star, transition
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