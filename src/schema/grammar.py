from .states import GrammarState, State


WHITESPACE = {" ", "\t", "\n", "\r"}


def transition(state: State, char: str) -> State:
    """
    Apply one character transition to the grammar state.

    This is the basic transition function δ(S, character).
    """

    current = state.grammar_state

    # Ignore whitespace where appropriate
    if char in WHITESPACE:
        return state

    # -----------------------------------------
    # 1. Expect the beginning of an object
    # -----------------------------------------
    if current == GrammarState.EXPECT_OBJECT_START:
        if char == "{":
            return State(GrammarState.EXPECT_KEY_QUOTE)
        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 2. Expect opening quote for a key
    # -----------------------------------------
    if current == GrammarState.EXPECT_KEY_QUOTE:
        if char == '"':
            return State(GrammarState.EXPECT_KEY_NAME)
        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 3. Reading the key name
    # -----------------------------------------
    if current == GrammarState.EXPECT_KEY_NAME:
        if char == '"':
            return State(GrammarState.EXPECT_COLON)

        # A key cannot contain a control character
        if ord(char) < 32:
            return State(GrammarState.DEAD_END)

        return State(
            GrammarState.EXPECT_KEY_NAME,
            state.buffer + char
        )

    # -----------------------------------------
    # 4. Expect colon after key
    # -----------------------------------------
    if current == GrammarState.EXPECT_COLON:
        if char == ":":
            return State(GrammarState.EXPECT_VALUE)
        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 5. Expect a value
    # -----------------------------------------
    if current == GrammarState.EXPECT_VALUE:

        # String value
        if char == '"':
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                '"'
            )

        # Simple non-string value
        if char in "0123456789-tfn":
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                char
            )

        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 6. Expect comma or closing object
    # -----------------------------------------
    if current == GrammarState.EXPECT_COMMA_OR_END:

        if char == ",":
            return State(GrammarState.EXPECT_KEY_QUOTE)

        if char == "}":
            return State(GrammarState.DONE)

        # Continue reading a value
        if state.buffer == '"':
            if char == '"':
                return State(
                    GrammarState.EXPECT_COMMA_OR_END,
                    ""
                )

            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                state.buffer + char
            )

        if char not in {",", "}"}:
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                state.buffer + char
            )

        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 7. DONE is terminal
    # -----------------------------------------
    if current == GrammarState.DONE:
        return State(GrammarState.DEAD_END)

    # -----------------------------------------
    # 8. DEAD_END is terminal
    # -----------------------------------------
    if current == GrammarState.DEAD_END:
        return state

    return State(GrammarState.DEAD_END)


def delta_star(initial_state: State, text: str) -> State:
    """
    Compute δ*(S, text).

    The text is processed character-by-character until:
      - the input is consumed, or
      - DEAD_END is reached.
    """

    state = initial_state

    for char in text:
        state = transition(state, char)

        if state.grammar_state == GrammarState.DEAD_END:
            return state

    return state