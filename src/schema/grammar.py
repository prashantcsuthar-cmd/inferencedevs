from .states import GrammarState, State

WHITESPACE = {" ", "\t", "\n", "\r"}


def transition(state: State, char: str) -> State:
    current = state.grammar_state

    if current in (GrammarState.DEAD_END, GrammarState.DONE):
        return State(GrammarState.DEAD_END) if current == GrammarState.DONE else state

    # Preserve interior whitespace inside string keys and values
    if char in WHITESPACE and current not in {
        GrammarState.EXPECT_KEY_NAME,
        GrammarState.READING_STRING,
        GrammarState.READING_STRING_ESCAPE,
    }:
        return state

    # 1. Object Start
    if current in (GrammarState.START, GrammarState.EXPECT_OBJECT_START):
        if char == "{":
            return State(
                GrammarState.EXPECT_KEY_QUOTE,
                expected_keys=state.expected_keys
            )
        return State(GrammarState.DEAD_END)

    # 2. Key Opening Quote
    if current == GrammarState.EXPECT_KEY_QUOTE:
        if char == '"':
            return State(
                GrammarState.EXPECT_KEY_NAME,
                buffer="",
                expected_keys=state.expected_keys
            )
        if char == "}" and not state.expected_keys:
            return State(GrammarState.DONE)
        return State(GrammarState.DEAD_END)

    # 3. Reading Key Name
    if current == GrammarState.EXPECT_KEY_NAME:
        if char == '"':
            key_found = state.buffer
            if state.expected_keys and key_found not in state.expected_keys:
                return State(GrammarState.DEAD_END)
            return State(
                GrammarState.EXPECT_COLON,
                current_key=key_found,
                expected_keys=state.expected_keys
            )
        if ord(char) < 32:
            return State(GrammarState.DEAD_END)
        return State(
            GrammarState.EXPECT_KEY_NAME,
            buffer=state.buffer + char,
            expected_keys=state.expected_keys
        )

    # 4. Colon Separator
    if current == GrammarState.EXPECT_COLON:
        if char == ":":
            return State(
                GrammarState.EXPECT_VALUE,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        return State(GrammarState.DEAD_END)

    # 5. Value Routing
    if current == GrammarState.EXPECT_VALUE:
        if char == '"':
            return State(
                GrammarState.READING_STRING,
                buffer="",
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char in "0123456789-":
            return State(
                GrammarState.READING_NUM,
                buffer=char,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char == "t":
            return State(
                GrammarState.READING_LITERAL,
                buffer="t",
                target_literal="true",
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char == "f":
            return State(
                GrammarState.READING_LITERAL,
                buffer="f",
                target_literal="false",
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char == "n":
            return State(
                GrammarState.READING_LITERAL,
                buffer="n",
                target_literal="null",
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        return State(GrammarState.DEAD_END)

    # 6. Reading String Values
    if current == GrammarState.READING_STRING:
        if char == "\\":
            return State(
                GrammarState.READING_STRING_ESCAPE,
                buffer=state.buffer,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char == '"':
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if ord(char) < 32:
            return State(GrammarState.DEAD_END)
        return State(
            GrammarState.READING_STRING,
            buffer=state.buffer + char,
            current_key=state.current_key,
            expected_keys=state.expected_keys
        )

    # 7. Escape Sequences
    if current == GrammarState.READING_STRING_ESCAPE:
        if char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
            return State(
                GrammarState.READING_STRING,
                buffer=state.buffer + char,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        return State(GrammarState.DEAD_END)

    # 8. Numeric Literals
    if current == GrammarState.READING_NUM:
        if char in "0123456789.eE+-":
            return State(
                GrammarState.READING_NUM,
                buffer=state.buffer + char,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        if char == ",":
            remaining = tuple(k for k in state.expected_keys if k != state.current_key)
            return State(GrammarState.EXPECT_KEY_QUOTE, expected_keys=remaining)
        if char == "}":
            return State(GrammarState.DONE)
        if char in WHITESPACE:
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        return State(GrammarState.DEAD_END)

    # 9. Boolean / Null Literals
    if current == GrammarState.READING_LITERAL:
        new_buffer = state.buffer + char
        if not state.target_literal.startswith(new_buffer):
            return State(GrammarState.DEAD_END)
        if new_buffer == state.target_literal:
            return State(
                GrammarState.EXPECT_COMMA_OR_END,
                current_key=state.current_key,
                expected_keys=state.expected_keys
            )
        return State(
            GrammarState.READING_LITERAL,
            buffer=new_buffer,
            target_literal=state.target_literal,
            current_key=state.current_key,
            expected_keys=state.expected_keys
        )

    # 10. Delimiters
    if current == GrammarState.EXPECT_COMMA_OR_END:
        if char == ",":
            remaining = tuple(k for k in state.expected_keys if k != state.current_key)
            return State(GrammarState.EXPECT_KEY_QUOTE, expected_keys=remaining)
        if char == "}":
            return State(GrammarState.DONE)
        return State(GrammarState.DEAD_END)

    return State(GrammarState.DEAD_END)


def delta_star(initial_state: State, text: str) -> State:
    state = initial_state
    for char in text:
        state = transition(state, char)
        if state.grammar_state == GrammarState.DEAD_END:
            return state
    return state