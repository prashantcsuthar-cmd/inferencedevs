from src.schema.states import GrammarState, State
from src.schema.grammar import delta_star, transition


def test_object_start_transition():
    state = State(GrammarState.EXPECT_OBJECT_START)

    result = transition(state, "{")

    assert result.grammar_state == GrammarState.EXPECT_KEY_QUOTE


def test_key_quote_transition():
    state = State(GrammarState.EXPECT_KEY_QUOTE)

    result = transition(state, '"')

    assert result.grammar_state == GrammarState.EXPECT_KEY_NAME


def test_key_name_transition():
    state = State(GrammarState.EXPECT_KEY_NAME)

    result = transition(state, "n")

    assert result.grammar_state == GrammarState.EXPECT_KEY_NAME
    assert result.buffer == "n"


def test_colon_transition():
    state = State(GrammarState.EXPECT_COLON)

    result = transition(state, ":")

    assert result.grammar_state == GrammarState.EXPECT_VALUE


def test_invalid_character_causes_dead_end():
    state = State(GrammarState.EXPECT_OBJECT_START)

    result = transition(state, "[")

    assert result.grammar_state == GrammarState.DEAD_END


def test_delta_star_valid_object():
    state = State(GrammarState.EXPECT_OBJECT_START)

    result = delta_star(state, '{"name":"John"}')

    assert result.grammar_state == GrammarState.DONE


def test_delta_star_detects_dead_end():
    state = State(GrammarState.EXPECT_OBJECT_START)

    result = delta_star(state, '[invalid]')

    assert result.grammar_state == GrammarState.DEAD_END


def test_dead_end_stops_processing():
    state = State(GrammarState.EXPECT_OBJECT_START)

    result = delta_star(state, '[this should never be accepted]')

    assert result.grammar_state == GrammarState.DEAD_END