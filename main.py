
"""ICE local dual-stream SSE server.

Generation completion and output validation are reported separately.
ICE and baseline generation maintain independent request-local state.
"""

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
from jsonschema import Draft202012Validator, SchemaError

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
MODEL_ERROR: str | None = None


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER, ENGINE, MODEL_ERROR

    try:
        MODEL, TOKENIZER = load_qwen()
        ENGINE = ICEDecodingEngine(model=MODEL, tokenizer=TOKENIZER)
        MODEL_ERROR = None
        print("ICE local inference engine initialized.", flush=True)
    except Exception as exc:
        # Keep the API and static frontend available so /status can explain
        # missing model files or an invalid model configuration.
        MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        print(f"ICE model initialization failed: {MODEL_ERROR}", flush=True)


# ---------------------------------------------------------------------------
# Health/status endpoint
# ---------------------------------------------------------------------------

@app.get("/status")
def get_status():
    if MODEL is None or TOKENIZER is None or ENGINE is None:
        return {
            "status": "ERROR" if MODEL_ERROR else "STARTING",
            "device": "unknown",
            "engine": "ICE",
            "vocabulary_capacity": 0,
            "error": MODEL_ERROR,
        }

    return {
        "status": "ONLINE",
        "device": str(getattr(MODEL, "device", "CPU")),
        "engine": "ICE — grammar-guided constrained decoding",
        "vocabulary_capacity": len(TOKENIZER),
    }


# ---------------------------------------------------------------------------
# Independent output validation
# ---------------------------------------------------------------------------

def validate_output(
    text: str,
    status: str,
    schema_dict: dict | None,
) -> dict:
    """Validate a completed generation independently of ICE's grammar."""

    result = {
        "status": status,
        "generation_completed": status == "COMPLETED",
        "validation_performed": False,
        "json_valid": False,
        "schema_valid": None,
        "error": "",
    }

    # Incomplete output must not be reported as a successful generation.
    if status != "COMPLETED":
        result["error"] = (
            f"Generation ended with status {status}; "
            "output was not validated as complete."
        )
        return result

    # First validate JSON syntax.
    try:
        parsed = json.loads(text)
        result["json_valid"] = True

    except (json.JSONDecodeError, TypeError) as exc:
        result["validation_performed"] = True
        result["error"] = f"Invalid JSON: {exc}"
        return result

    result["validation_performed"] = True

    # JSON syntax can be validated without a schema.
    if schema_dict is None:
        result["error"] = (
            "No schema supplied; JSON syntax was checked only."
        )
        return result

    # Validate the parsed object against the supplied JSON Schema.
    try:
        validator = Draft202012Validator(schema_dict)

        errors = sorted(
            validator.iter_errors(parsed),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )

        if errors:
            result["schema_valid"] = False
            result["error"] = "; ".join(
                f"{list(error.absolute_path)}: {error.message}"
                for error in errors[:5]
            )
        else:
            result["schema_valid"] = True

    except SchemaError as exc:
        # Normally caught before generation, but retain defensive handling.
        result["schema_valid"] = False
        result["error"] = f"Invalid JSON Schema: {exc.message}"

    return result


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(payload: dict) -> str:
    """Serialize one Server-Sent Event."""
    return "data: " + json.dumps(
        payload,
        ensure_ascii=False,
    ) + "\n\n"


# ---------------------------------------------------------------------------
# Streaming generation endpoint
# ---------------------------------------------------------------------------

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

    # Parse and validate the supplied JSON Schema before generation.
    if schema is not None:
        try:
            schema_dict = json.loads(schema)

        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"schema is not valid JSON: {exc.msg}",
            ) from exc

        if not isinstance(schema_dict, dict):
            raise HTTPException(
                status_code=422,
                detail="schema must be a JSON object",
            )

        try:
            Draft202012Validator.check_schema(schema_dict)

        except SchemaError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON Schema: {exc.message}",
            ) from exc

    else:
        schema_dict = None

    # Compile the schema into ICE's supported grammar representation.
    if schema_dict is None:
        # Without a schema, use the existing generic grammar state.
        initial_state = State(GrammarState.START)

    else:
        try:
            initial_state = compile_schema_to_state(schema_dict)

        except (TypeError, ValueError, NotImplementedError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid or unsupported schema: {exc}",
            ) from exc


    async def event_generator():
        if ENGINE is None or MODEL is None or TOKENIZER is None:
            yield _sse({
                "error": "Engine components are not loaded."
            })
            return

        try:
            # ---------------------------------------------------------------
            # Tokenize the prompt and initialize independent generation paths.
            # ---------------------------------------------------------------
            inputs = TOKENIZER(
                prompt,
                return_tensors="pt",
            ).to(MODEL.device)

            attention = inputs.get("attention_mask")

            ice_ids = inputs["input_ids"].clone()
            base_ids = inputs["input_ids"].clone()

            ice_mask = (
                attention.clone()
                if attention is not None
                else None
            )
            base_mask = (
                attention.clone()
                if attention is not None
                else None
            )

            # Prefill both paths using the same prompt.
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

            ice_past = getattr(
                ice_out,
                "past_key_values",
                None,
            )
            base_past = getattr(
                base_out,
                "past_key_values",
                None,
            )

            # Request-local grammar and processor.
            ice_state = initial_state

            request_processor = ICELogitsProcessor(
                ENGINE.compatibility_engine
            )

            ice_status = "RUNNING"
            base_status = "RUNNING"

            ice_tokens: list[int] = []
            base_tokens: list[int] = []

            vocab_size = int(ice_out.logits.shape[-1])

            # ---------------------------------------------------------------
            # Generate one token per active path per iteration.
            # ---------------------------------------------------------------
            for step in range(1, max_steps + 1):
                if await request.is_disconnected():
                    return

                ice_text = ""
                base_text = ""
                telemetry = {}
                allowed_sample = []
                base_has_error = False

                # ===========================================================
                # ICE constrained generation
                # ===========================================================
                if ice_status == "RUNNING":
                    started = time.perf_counter()

                    try:
                        valid_ids = (
                            ENGINE.compatibility_engine.get_valid_tokens(
                                ice_state
                            )
                        )

                        allowed_sample = [
                            ENGINE.compatibility_engine.decode_token(
                                token_id
                            )
                            for token_id in valid_ids[:6]
                        ]

                        request_processor.set_state(ice_state)

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

                        print(
                            f"[ICE DEBUG] request={id(request)} "
                            f"step={step} token_id={tid} "
                            f"token_text={ice_text!r} "
                            f"state_before="
                            f"{ice_state.grammar_state.name} "
                            f"valid_count={len(valid_ids)}",
                            flush=True,
                        )

                        # Advance the grammar state character by character.
                        next_state = ice_state

                        for char in ice_text:
                            next_state = transition(
                                next_state,
                                char,
                            )

                            if (
                                next_state.grammar_state
                                == GrammarState.DEAD_END
                            ):
                                raise RuntimeError(
                                    "Selected token violated grammar "
                                    "transition."
                                )

                        print(
                            f"[ICE DEBUG] request={id(request)} "
                            f"step={step} "
                            f"state_after="
                            f"{next_state.grammar_state.name}",
                            flush=True,
                        )

                        ice_tokens.append(tid)
                        ice_state = next_state

                        if (
                            ice_state.grammar_state
                            == GrammarState.DONE
                        ):
                            ice_status = "COMPLETED"

                        elif tid == getattr(
                            TOKENIZER,
                            "eos_token_id",
                            None,
                        ):
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

                        # Compute the next logits only if generation continues.
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
                        telemetry["error"] = (
                            f"{type(exc).__name__}: {exc}"
                        )

                # ===========================================================
                # Unconstrained baseline generation
                # ===========================================================
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

                    if btid == getattr(
                        TOKENIZER,
                        "eos_token_id",
                        None,
                    ):
                        base_status = "COMPLETED"

                    elif step == max_steps:
                        base_status = "MAX_TOKENS"

                    # This is a running diagnostic, not the final validator.
                    decoded = TOKENIZER.decode(
                        base_tokens,
                        skip_special_tokens=True,
                    )

                    try:
                        json.loads(decoded)
                    except (json.JSONDecodeError, TypeError):
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

                # ===========================================================
                # Telemetry and SSE events
                # ===========================================================
                if telemetry.get("total_vocab"):
                    telemetry["masking_ratio"] = (
                        telemetry["masked_count"]
                        / telemetry["total_vocab"]
                    )

                yield _sse({
                    "step": step,
                    "runner_type": "ice",
                    "token_text": ice_text,
                    "active_grammar_state": (
                        ice_state.grammar_state.name
                    ),
                    "telemetry": telemetry,
                    "status": ice_status,
                    "sample_allowed_tokens": allowed_sample,
                })

                yield _sse({
                    "step": step,
                    "runner_type": "baseline",
                    "token_text": base_text,
                    "has_error": base_has_error,
                    "status": base_status,
                })

                if (
                    ice_status != "RUNNING"
                    and base_status != "RUNNING"
                ):
                    break

                await asyncio.sleep(0.01)

            # ---------------------------------------------------------------
            # Independently validate both final outputs.
            # ---------------------------------------------------------------
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
                schema_dict,
            )

            base_report = validate_output(
                base_final,
                base_status,
                schema_dict,
            )

            yield _sse({
                "runner_type": "terminal",
                "ice": {
                    **ice_report,
                    "text": ice_final,
                },
                "baseline": {
                    **base_report,
                    "text": base_final,
                },
            })

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            yield _sse({
                "error": (
                    f"Inference failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Optional frontend static-file mounting
# ---------------------------------------------------------------------------

frontend_dir = Path(__file__).resolve().parent / "frontend"

if frontend_dir.is_dir():
    app.mount(
        "/",
        StaticFiles(
            directory=frontend_dir,
            html=True,
        ),
        name="frontend",
    )
