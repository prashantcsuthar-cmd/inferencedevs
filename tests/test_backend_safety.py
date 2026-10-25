import pytest
import torch
from src.schema.parser import compile_schema_to_state
from src.schema.grammar import delta_star
from src.schema.states import State, GrammarState
from src.decoding.logits_processor import ICELogitsProcessor


def test_unsupported_schema_keyword_is_rejected():
    with pytest.raises(ValueError, match="Unsupported schema keyword"):
        compile_schema_to_state({"type": "object", "properties": {}, "patternProperties": {}})


def test_schema_type_mismatch_is_rejected_by_grammar():
    state = compile_schema_to_state({
        "type": "object",
        "properties": {"age": {"type": "number"}},
        "required": ["age"],
        "additionalProperties": False,
    })
    assert delta_star(state, '{"age":"old"}').grammar_state == GrammarState.DEAD_END


def test_schema_accepts_valid_required_property():
    state = compile_schema_to_state({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    })
    assert delta_star(state, '{"name":"Ada"}').grammar_state == GrammarState.DONE


def test_schema_rejects_missing_required_property():
    state = compile_schema_to_state({
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    })
    assert delta_star(state, '{}').grammar_state == GrammarState.DEAD_END


class _Compat:
    def get_valid_tokens(self, state):
        return [1, 3]


def test_logits_processor_masks_without_mutating_input_scores():
    processor = ICELogitsProcessor(_Compat())
    processor.set_state(State(GrammarState.START))
    scores = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    masked = processor(torch.tensor([[9]]), scores)
    assert torch.equal(scores, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.isneginf(masked[0, 0]) and torch.isneginf(masked[0, 2])
    assert masked[0, 1].item() == 2.0 and masked[0, 3].item() == 4.0


def test_empty_allowed_set_fails_closed():
    class EmptyCompat:
        def get_valid_tokens(self, state): return []
    processor = ICELogitsProcessor(EmptyCompat())
    processor.set_state(State(GrammarState.START))
    with pytest.raises(RuntimeError, match="unrestricted fallback is prohibited"):
        processor(torch.tensor([[9]]), torch.tensor([[1.0, 2.0]]))
