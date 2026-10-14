"""
ICE Engine - Production-Grade Local FastAPI Server
Exposes high-speed, dual-stream SSE capabilities with robust completion tracking and post-run validation.
"""

import json
import asyncio
import os
import sys
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# Enforce project path bounds cleanly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.models.loader import load_qwen
from src.decoding.engine import ICEDecodingEngine
from src.schema.states import State, GrammarState
from src.schema.grammar import transition

app = FastAPI(title="ICE Engine Streaming Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
TOKENIZER = None
ENGINE = None

@app.on_event("startup")
def startup_event():
    global MODEL, TOKENIZER, ENGINE
    print("=" * 60)
    print("INITIALISING ICE ENGINE OFFLINE INFERENCE SERVER")
    print("=" * 60)
    
    MODEL, TOKENIZER = load_qwen()
    ENGINE = ICEDecodingEngine(model=MODEL, tokenizer=TOKENIZER)
    
    print("Server components mapped successfully to local hardware memory space.")

@app.get("/status")
def get_status():
    return {
        "status": "ONLINE",
        "device": str(getattr(MODEL, "device", "CPU")),
        "engine": "Inferencedevs Constrained Engine (ICE) V1.0",
        "vocabulary_capacity": len(TOKENIZER) if TOKENIZER else 0
    }

@app.get("/stream")
async def stream_generation(prompt: str, request: Request):
    """
    Asynchronous SSE channel streaming paired tokens from both 
    the ICE constrained decoding pipeline and a genuine unconstrained baseline run 
    with robust independent status tracking and validation.
    """
    async def sse_event_generator():
        if ENGINE is None or MODEL is None:
            yield "data: " + json.dumps({"error": "Engine components are not loaded."}) + "\n\n"
            return

        print(f"\nProcessing real-time dual-stream request for prompt: {prompt!r}")
        
        inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
        import time
        
        # --- ICE Stream State Setup ---
        ice_state = State(GrammarState.EXPECT_OBJECT_START)
        vocabulary_size = len(TOKENIZER)
        
        ice_input_ids = inputs["input_ids"].clone()
        ice_attention_mask = inputs.get("attention_mask")
        if ice_attention_mask is not None:
            ice_attention_mask = ice_attention_mask.clone()

        # --- Baseline Stream State Setup ---
        base_input_ids = inputs["input_ids"].clone()
        base_attention_mask = inputs.get("attention_mask")
        if base_attention_mask is not None:
            base_attention_mask = base_attention_mask.clone()

        with torch.no_grad():
            ice_outputs = MODEL(input_ids=ice_input_ids, attention_mask=ice_attention_mask, use_cache=True)
            ice_past_kv = ice_outputs.past_key_values

            base_outputs = MODEL(input_ids=base_input_ids, attention_mask=base_attention_mask, use_cache=True)
            base_past_kv = base_outputs.past_key_values

        step_index = 0
        max_steps = 64
        
        ice_status = "RUNNING"  # RUNNING, COMPLETED, MAX_TOKENS, ERROR
        base_status = "RUNNING"

        ice_accumulated_tokens = []
        base_accumulated_tokens = []

        while step_index < max_steps and (ice_status == "RUNNING" or base_status == "RUNNING"):
            if await request.is_disconnected():
                print("Inference lifecycle terminated safely by client disconnection.")
                break

            step_index += 1
            t_start = time.perf_counter()

            # ==========================================
            # 1. ICE CONSTRAINED STEP
            # ==========================================
            ice_token_text = ""
            ice_metrics = {}
            valid_token_ids = []

            if ice_status == "RUNNING":
                ENGINE.processor.set_state(ice_state)
                valid_token_ids = ENGINE.compatibility_engine.get_valid_tokens(ice_state)
                valid_count = len(valid_token_ids)
                masked_count = vocabulary_size - valid_count
                masking_ratio = masked_count / vocabulary_size if vocabulary_size else 0.0

                with torch.inference_mode():
                    masked_logits = ENGINE.processor(
                        ice_input_ids, ice_outputs.logits[:, -1, :], valid_token_ids=valid_token_ids
                    )
                    next_ice_token = torch.argmax(masked_logits, dim=-1)
                    ice_token_id = int(next_ice_token.item())

                ice_token_text = ENGINE.compatibility_engine.decode_token(ice_token_id)
                ice_accumulated_tokens.append(ice_token_id)

                # Real grammar state transition validation
                next_state = ice_state
                for char in ice_token_text:
                    next_state = transition(next_state, char)
                    if next_state.grammar_state == GrammarState.DEAD_END:
                        ice_status = "ERROR"
                        break

                ice_state = next_state
                t_end = time.perf_counter()

                ice_metrics = {
                    "total_vocab": vocabulary_size,
                    "valid_count": valid_count,
                    "masked_count": masked_count,
                    "masking_ratio": float(masking_ratio),
                    "ms_per_token": float((t_end - t_start) * 1000)
                }

                if ice_state.grammar_state == GrammarState.DONE:
                    ice_status = "COMPLETED"
                elif ice_status != "ERROR" and step_index >= max_steps:
                    ice_status = "MAX_TOKENS"

                ice_input_ids = next_ice_token.unsqueeze(0)
                if ice_attention_mask is not None:
                    new_bit = torch.ones((ice_attention_mask.shape[0], 1), dtype=ice_attention_mask.dtype, device=ice_attention_mask.device)
                    ice_attention_mask = torch.cat([ice_attention_mask, new_bit], dim=1)

                with torch.no_grad():
                    ice_outputs = MODEL(
                        input_ids=ice_input_ids,
                        attention_mask=ice_attention_mask,
                        past_key_values=ice_past_kv,
                        use_cache=True
                    )
                    ice_past_kv = getattr(ice_outputs, "past_key_values", None)

            # ==========================================
            # 2. GENUINE UNCONSTRAINED BASELINE STEP
            # ==========================================
            base_token_text = ""
            base_has_error = False

            if base_status == "RUNNING":
                with torch.inference_mode():
                    base_logits = base_outputs.logits[:, -1, :]
                    next_base_token = torch.argmax(base_logits, dim=-1)
                    base_token_id = int(next_base_token.item())

                base_token_text = ENGINE.compatibility_engine.decode_token(base_token_id)
                base_accumulated_tokens.append(base_token_id)

                if base_token_id == TOKENIZER.eos_token_id:
                    base_status = "COMPLETED"
                elif step_index >= max_steps:
                    base_status = "MAX_TOKENS"

                # Independent JSON syntax check on accumulated baseline output
                full_base_text = TOKENIZER.decode(base_accumulated_tokens, skip_special_tokens=True)
                try:
                    json.loads(full_base_text)
                    base_has_error = False
                except json.JSONDecodeError:
                    base_has_error = True

                base_input_ids = next_base_token.unsqueeze(0)
                if base_attention_mask is not None:
                    new_bit = torch.ones((base_attention_mask.shape[0], 1), dtype=base_attention_mask.dtype, device=base_attention_mask.device)
                    base_attention_mask = torch.cat([base_attention_mask, new_bit], dim=1)

                with torch.no_grad():
                    base_outputs = MODEL(
                        input_ids=base_input_ids,
                        attention_mask=base_attention_mask,
                        past_key_values=base_past_kv,
                        use_cache=True
                    )
                    base_past_kv = getattr(base_outputs, "past_key_values", None)

            # Yield ICE Payload
            ice_payload = {
                "step": step_index,
                "runner_type": "ice",
                "token_text": ice_token_text,
                "active_grammar_state": ice_state.grammar_state.name,
                "telemetry": ice_metrics,
                "status": ice_status,
                "sample_allowed_tokens": [TOKENIZER.decode([tid]) for tid in valid_token_ids[:6]] if valid_token_ids else []
            }
            yield f"data: {json.dumps(ice_payload)}\n\n"

            # Yield Baseline Payload
            base_payload = {
                "step": step_index,
                "runner_type": "baseline",
                "token_text": base_token_text,
                "has_error": base_has_error,
                "status": base_status
            }
            yield f"data: {json.dumps(base_payload)}\n\n"

            if ice_status != "RUNNING" and base_status != "RUNNING":
                break

            await asyncio.sleep(0.03)

        # --- Final Post-Run Validation Report ---
        final_ice_text = TOKENIZER.decode(ice_accumulated_tokens, skip_special_tokens=True)
        print(f"\n[Validation Report] ICE Final Text: {final_ice_text!r}")
        print(f"[Validation Report] ICE Final Status: {ice_status}")
        try:
            parsed_json = json.loads(final_ice_text)
            print(f"[Validation Report] JSON Syntax Check: SUCCESS -> {parsed_json}")
        except json.JSONDecodeError as e:
            print(f"[Validation Report] JSON Syntax Check: FAILED -> {e}")

        print(f"Dual-stream inference completed successfully in {step_index} steps.")

    return StreamingResponse(sse_event_generator(), media_type="text/event-stream")