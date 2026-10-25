
"""ICE local dual-stream SSE server. Generation completion and validation are reported separately."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.models.loader import load_qwen
from src.decoding.engine import ICEDecodingEngine
from src.schema.states import GrammarState, State
from src.schema.grammar import transition
from src.schema.parser import compile_schema_to_state
from src.decoding.logits_processor import ICELogitsProcessor


app = FastAPI(title="ICE Engine Streaming Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MODEL = TOKENIZER = ENGINE = None


@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER, ENGINE

    MODEL, TOKENIZER = load_qwen()
    ENGINE = ICEDecodingEngine(model=MODEL, tokenizer=TOKENIZER)

    print("ICE local inference engine initialized.", flush=True)


@app.get("/status")
def get_status():
    if MODEL is None or TOKENIZER is None or ENGINE is None:
        return {
            "status": "STARTING",
            "device": "unknown",
            "engine": "ICE",
            "vocabulary_capacity": 0,
        }

    return {
        "status": "ONLINE",
        "device": str(getattr(MODEL, "device", "CPU")),
        "engine": "ICE — grammar-guided constrained decoding",
        "vocabulary_capacity": len(TOKENIZER),
    }


def validate_against_schema(data: Any, node) -> tuple[bool, str]:
    """Independently validate output against the supported schema subset."""
    typ = node.type

    if typ is None:
        return True, ""

    if typ == "object":
        if not isinstance(data, dict):
            return False, f"Expected object, got {type(data).__name__}"

        for key in node.required:
            if key not in data:
                return False, f"Missing required property: {key!r}"

        for key, value in data.items():
            child = node.get_property(key)

            if child is None:
                if not node.additional_properties:
                    return False, f"Additional property {key!r} is not allowed"
            else:
                ok, error = validate_against_schema(value, child)
                if not ok:
                    return False, f"Property {key!r}: {error}"

    elif typ == "string":
        if not isinstance(data, str):
            return False, f"Expected string, got {type(data).__name__}"

    elif typ == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            return False, f"Expected number, got {type(data).__name__}"

    elif typ == "boolean":
        if not isinstance(data, bool):
            return False, f"Expected boolean, got {type(data).__name__}"

    elif typ == "null":
        if data is not None:
            return False, f"Expected null, got {type(data).__name__}"

    return True, ""


def validate_output(text: str, status: str, schema_root) -> dict:
    result = {
        "status": status,
        "generation_completed": status == "COMPLETED",
        "validation_performed": False,
        "json_valid": False,
        "schema_valid": False,
        "error": "",
    }

    if status != "COMPLETED":
        result["error"] = (
            f"Generation ended with status {status}; "
            "output was not validated as complete."
        )
        return result

    try:
        parsed = json.loads(text)
        result["json_valid"] = True
    except (json.JSONDecodeError, TypeError) as exc:
        result["validation_performed"] = True
        result["error"] = f"Invalid JSON: {exc}"
        return result

    result["validation_performed"] = True

    if schema_root is None:
        # No schema was supplied; JSON syntax was checked only.
        result["schema_valid"] = None
    else:
        ok, error = validate_against_schema(parsed, schema_root)
        result["schema_valid"] = ok
        result["error"] = error

    return result


def _sse(payload: dict) -> str:
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.get("/stream")
async def stream_generation(
    prompt: str,
    request: Request,
    schema: str | None = None,
    max_steps: int = 64,
):
    if not prompt.strip():
        raise HTTPException(
            status_code=422,
            detail="prompt must not be empty",
        )

    if not 1 <= max_steps <= 512:
        raise HTTPException(
            status_code=422,
            detail="max_steps must be between 1 and 512",
        )

    if schema is not None:
        try:
            schema_dict = json.loads(schema)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"schema is not valid JSON: {exc.msg}",
            ) from exc
    else:
        # Without a supplied schema, constrain JSON-object syntax
        # without inventing property names.
        schema_dict = None

    if schema_dict is None:
        initial_state = State(GrammarState.START)
        schema_root = None
    else:
        try:
            initial_state = compile_schema_to_state(schema_dict)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid or unsupported schema: {exc}",
            ) from exc

        schema_root = initial_state.schema_root

    async def event_generator():
        if ENGINE is None or MODEL is None or TOKENIZER is None:
            yield _sse({"error": "Engine components are not loaded."})
            return

        try:
            inputs = TOKENIZER(
                prompt,
                return_tensors="pt",
            ).to(MODEL.device)

            attention = inputs.get("attention_mask")

            ice_ids = inputs["input_ids"].clone()
            base_ids = inputs["input_ids"].clone()

            ice_mask = attention.clone() if attention is not None else None
            base_mask = attention.clone() if attention is not None else None

            with torch.inference_mode():
                ice_out = MODEL(
                    input_ids=ice_ids,
                    attention_mask=ice_mask,
                    use_cache=True,
                )

                base_out = MODEL(
                    input_ids=base_ids,
                    attention_mask=base_mask,
                    use_cache=True,
                )

            ice_past = getattr(ice_out, "past_key_values", None)
            base_past = getattr(base_out, "past_key_values", None)

            ice_state = initial_state

            # Each request gets its own mutable logits processor.
            request_processor = ICELogitsProcessor(
                ENGINE.compatibility_engine
            )

            ice_status = "RUNNING"
            base_status = "RUNNING"

            ice_tokens: list[int] = []
            base_tokens: list[int] = []

            vocab_size = int(ice_out.logits.shape[-1])

            for step in range(1, max_steps + 1):
                if await request.is_disconnected():
                    return

                ice_text = ""
                base_text = ""
                telemetry = {}
                allowed_sample = []
                base_has_error = False

                # ---------------- ICE constrained generation ----------------
                if ice_status == "RUNNING":
                    started = time.perf_counter()

                    valid_ids = (
                        ENGINE.compatibility_engine.get_valid_tokens(
                            ice_state
                        )
                    )

                    allowed_sample = [
                        ENGINE.compatibility_engine.decode_token(token_id)
                        for token_id in valid_ids[:6]
                    ]

                    request_processor.set_state(ice_state)

                    try:
                        with torch.inference_mode():
                            masked_logits = request_processor(
                                ice_ids,
                                ice_out.logits[:, -1, :],
                            )

                            next_ice = torch.argmax(
                                masked_logits,
                                dim=-1,
                            )

                        tid = int(next_ice.item())
                        ice_text = (
                            ENGINE.compatibility_engine.decode_token(tid)
                        )

                        # DEBUG LOG 1:
                        # Record which token was selected and the grammar
                        # state before applying its decoded text.
                        print(
                            f"[ICE DEBUG] request={id(request)} "
                            f"step={step} token_id={tid} "
                            f"token_text={ice_text!r} "
                            f"state_before={ice_state.grammar_state.name} "
                            f"valid_count={len(valid_ids)}",
                            flush=True,
                        )

                        next_state = ice_state

                        for char in ice_text:
                            next_state = transition(next_state, char)

                            if (
                                next_state.grammar_state
                                == GrammarState.DEAD_END
                            ):
                                raise RuntimeError(
                                    "Selected token violated grammar transition."
                                )

                        # DEBUG LOG 2:
                        # Use next_state because ice_state has not yet
                        # been updated at this point.
                        print(
                            f"[ICE DEBUG] request={id(request)} "
                            f"step={step} "
                            f"state_after={next_state.grammar_state.name}",
                            flush=True,
                        )

                        ice_tokens.append(tid)
                        ice_state = next_state

                        if ice_state.grammar_state == GrammarState.DONE:
                            ice_status = "COMPLETED"

                        elif tid == getattr(TOKENIZER, "eos_token_id", None):
                            # EOS before grammar completion is not success.
                            ice_status = "ERROR"

                        elif step == max_steps:
                            ice_status = "MAX_TOKENS"

                        telemetry = {
                            "total_vocab": vocab_size,
                            "valid_count": len(set(valid_ids)),
                            "masked_count": max(
                                0,
                                vocab_size - len(set(valid_ids)),
                            ),
                            "ms_per_token": (
                                time.perf_counter() - started
                            ) * 1000,
                        }

                        if ice_status == "RUNNING":
                            ice_ids = next_ice.unsqueeze(0)

                            if ice_mask is not None:
                                ice_mask = torch.cat(
                                    [
                                        ice_mask,
                                        torch.ones(
                                            (ice_mask.shape[0], 1),
                                            dtype=ice_mask.dtype,
                                            device=ice_mask.device,
                                        ),
                                    ],
                                    dim=1,
                                )

                            with torch.inference_mode():
                                ice_out = MODEL(
                                    input_ids=ice_ids,
                                    attention_mask=ice_mask,
                                    past_key_values=ice_past,
                                    use_cache=True,
                                )

                            ice_past = getattr(
                                ice_out,
                                "past_key_values",
                                None,
                            )

                    except Exception as exc:
                        ice_status = "ERROR"
                        telemetry["error"] = str(exc)

                # ---------------- Unconstrained baseline ----------------
                if base_status == "RUNNING":
                    with torch.inference_mode():
                        next_base = torch.argmax(
                            base_out.logits[:, -1, :],
                            dim=-1,
                        )

                    btid = int(next_base.item())
                    base_text = (
                        ENGINE.compatibility_engine.decode_token(btid)
                    )
                    base_tokens.append(btid)

                    if btid == getattr(TOKENIZER, "eos_token_id", None):
                        base_status = "COMPLETED"

                    elif step == max_steps:
                        base_status = "MAX_TOKENS"

                    decoded = TOKENIZER.decode(
                        base_tokens,
                        skip_special_tokens=True,
                    )

                    try:
                        json.loads(decoded)
                    except json.JSONDecodeError:
                        base_has_error = True

                    if base_status == "RUNNING":
                        base_ids = next_base.unsqueeze(0)

                        if base_mask is not None:
                            base_mask = torch.cat(
                                [
                                    base_mask,
                                    torch.ones(
                                        (base_mask.shape[0], 1),
                                        dtype=base_mask.dtype,
                                        device=base_mask.device,
                                    ),
                                ],
                                dim=1,
                            )

                        with torch.inference_mode():
                            base_out = MODEL(
                                input_ids=base_ids,
                                attention_mask=base_mask,
                                past_key_values=base_past,
                                use_cache=True,
                            )

                        base_past = getattr(
                            base_out,
                            "past_key_values",
                            None,
                        )

                if telemetry and telemetry.get("total_vocab"):
                    telemetry["masking_ratio"] = (
                        telemetry["masked_count"]
                        / telemetry["total_vocab"]
                    )

                yield _sse(
                    {
                        "step": step,
                        "runner_type": "ice",
                        "token_text": ice_text,
                        "active_grammar_state": ice_state.grammar_state.name,
                        "telemetry": telemetry,
                        "status": ice_status,
                        "sample_allowed_tokens": allowed_sample,
                    }
                )

                yield _sse(
                    {
                        "step": step,
                        "runner_type": "baseline",
                        "token_text": base_text,
                        "has_error": base_has_error,
                        "status": base_status,
                    }
                )

                if ice_status != "RUNNING" and base_status != "RUNNING":
                    break

                await asyncio.sleep(0.01)

            ice_final = TOKENIZER.decode(
                ice_tokens,
                skip_special_tokens=True,
            )

            base_final = TOKENIZER.decode(
                base_tokens,
                skip_special_tokens=True,
            )

            ice_report = validate_output(
                ice_final,
                ice_status,
                schema_root,
            )

            base_report = validate_output(
                base_final,
                base_status,
                schema_root,
            )

            yield _sse(
                {
                    "runner_type": "terminal",
                    "ice": {
                        **ice_report,
                        "text": ice_final,
                    },
                    "baseline": {
                        **base_report,
                        "text": base_final,
                    },
                }
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            yield _sse(
                {
                    "error": (
                        f"Inference failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                }
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


frontend_dir = Path(__file__).resolve().parent / "frontend"

if frontend_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=frontend_dir, html=True),
        name="frontend",
    )
