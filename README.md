# 🧊 ICE: Inferencedevs Constrained Engine

> **Offline-First, Grammar-Guided Constrained Decoding for Local SLMs**  
> *Developed by Team Inferencedevs for Tech Eximius 2.0*

---

## 🎯 Overview & Core Pitch

Small Local Language Models (SLMs) running on edge hardware frequently struggle with strict JSON syntax and schema compliance, leading to downstream API crashes and system failures. Standard fixes rely on prompt engineering or retry loops—doubling latency and compute costs.

**Inferencedevs Constrained Engine (ICE)** eliminates these failures by sitting directly inside the PyTorch generation loop at every single token step. Using a mathematical finite state machine and subword prefix indexing, ICE calculates which vocabulary tokens are structurally legal according to a target JSON schema and applies an additive logit mask ($-\infty$) to block illegal choices. 

**ICE forces small local models (like Qwen2.5-0.5B on CPU) to generate 100% syntactically valid structured data on the first pass, with zero external network dependencies.**

---

## 📐 Mathematical Formulation

Given an autoregressive token history $t_{<i} = (t_1, t_2, \dots, t_{i-1})$ and current grammar state $S_{i-1}$:

1. **State Transition:**
   $$S_i = \delta^*(S_{i-1}, D(t_{i-1}))$$
   *(where $D(v)$ returns exact un-truncated token byte sequence semantics)*

2. **Admissibility Set Construction:**
   $$A(S_i) = \{ v \in V \mid \delta^*(S_i, D(v)) \neq \bot \}$$

3. **Additive Logit Interception:**
   $$M(S_i)_v = \begin{cases} 0 & \text{if } v \in A(S_i) \\ -\infty & \text{if } v \notin A(S_i) \end{cases}$$
   $$\mathbf{z}'_i = \mathbf{z}_i + \mathbf{M}(S_i)$$

4. **Constrained Softmax Sampling:**
   $$P_{\text{ICE}}(t_i = v \mid t_{<i}, S_i) = \frac{\exp(z'_{i, v})}{\sum_{k \in A(S_i)} \exp(z'_{i, k})}$$

---

## 📁 System & File Structure

```text
inferencedevs-ice/
├── src/
│   ├── schema/
│   │   ├── states.py          # GrammarState Enum & TransitionResult containers
│   │   ├── grammar.py         # Dynamic transition function delta*(S, text)
│   │   └── parser.py          # JSON schema compiler converting JSON -> state graph
│   ├── tokenizer/
│   │   ├── vocabulary.py      # Raw byte/canonical token metadata extractor
│   │   ├── trie.py            # Subword Prefix Trie for O(k) sequence indexing
│   │   └── compatibility.py   # Admissibility engine computing valid token set A(S)
│   ├── decoding/
│   │   ├── logits_processor.py# PyTorch custom LogitsProcessor interceptor
│   │   └── engine.py          # Incremental generation loop runner
│   ├── models/
│   │   └── loader.py          # Hardware-aware CPU loader for Qwen2.5-0.5B
│   ├── diagnostics/
│   │   ├── metrics.py         # Stats tracker (mask ratios, latency overhead)
│   │   └── tracing.py         # Detailed state debugger & tracer
│   └── validation/
│       └── validator.py       # Independent schema & syntax evaluation suite
├── benchmarks/
│   ├── schemas/               # Test schemas (Levels L1 through L5)
│   ├── baseline.py            # Unconstrained generation runner
│   ├── constrained.py         # ICE constrained generation runner
│   └── reference.py          # Brute-force reference engine for validation
├── tests/
│   ├── test_grammar.py        # Transition rules & dead-end handling tests
│   ├── test_tokens.py         # Exact byte-matching unit tests
│   ├── test_masking.py        # PyTorch LogitsProcessor tensor operation tests
│   └── test_equivalence.py    # Trie vs. Brute-force reference equivalence tests
├── demo/
│   └── app.py                 # Streamlit side-by-side diagnostic dashboard
├── main.py                    # Terminal execution entrypoint
├── requirements.txt
└── README.md


## 🗺️ Project Phases & Implementation Roadmap

### Phase 1: Pure Grammar & State Machine *(Days 1–2)*
* **Objective:** Build state representation and dynamic character-by-character transition logic without loading LLM weights.
* **Target Files:** `src/schema/states.py`, `src/schema/grammar.py`, `src/schema/parser.py`, `tests/test_grammar.py`.
* **Expectations:**
  * Dynamic schema parser (no hardcoded field names).
  * `transition(state, text)` returns `DEAD_END` ($\bot$) on any syntax violation.
  * Unit tests passing for valid, invalid, and partial subword text sequences.

### Phase 2: Token Compatibility & Prefix Trie *(Days 2–3)*
* **Objective:** Efficiently pre-index vocabulary byte sequences to accelerate admissible set computation $A(S)$.
* **Target Files:** `src/tokenizer/vocabulary.py`, `src/tokenizer/trie.py`, `src/tokenizer/compatibility.py`, `tests/test_tokens.py`.
* **Expectations:**
  * Byte-level token extraction preserving raw byte representations without truncation.
  * Prefix Trie structure grouping subwords to avoid full $O(\vert{}V\vert{})$ vocabulary scans.
  * `get_valid_tokens(state)` filtering legal token IDs accurately.

### Phase 3: PyTorch Logits Interception & SLM Integration *(Days 3–4)*
* **Objective:** Connect Qwen2.5-0.5B-Instruct and apply dynamic state masking directly to PyTorch logit tensors during autoregressive generation.
* **Target Files:** `src/models/loader.py`, `src/decoding/logits_processor.py`, `src/decoding/engine.py`, `tests/test_masking.py`.
* **Expectations:**
  * Custom `LogitsProcessor` intercepting output scores and setting invalid indices to $-\infty$.
  * Incremental state updates ($S_{i+1} = \delta^*(S_i, D(t_i))$) after each sampled token.
  * Successful CPU execution of Qwen2.5-0.5B with single-pass valid JSON generation.

### Phase 4: Benchmarks, Diagnostics & Streamlit Demo *(Days 4–5)*
* **Objective:** Quantify performance metrics against baseline unconstrained SLM and present real-time execution.
* **Target Files:** `benchmarks/`, `src/diagnostics/`, `src/validation/`, `demo/app.py`.
* **Expectations:**
  * Experimental proof of 100% syntactic validity for supported schema grammars.
  * Benchmark report measuring Per-Token Overhead (ms/tok), Masking Ratio (%), and Peak RAM.
  * Interactive Streamlit dashboard showing real-time decoding, token masking, and state updates.

---

## 🛠️ Developer Workflow & How to Push Code

To keep the repository organized and avoid merge conflicts, team members pushing code should follow this workflow in VS Code terminal:

### 1. Initial Setup (First Time Only)
```bash
# Clone the repository
git clone [https://github.com/prashantcsuthar-cmd/inferencedevs.git](https://github.com/prashantcsuthar-cmd/inferencedevs.git)
cd inferencedevs

# Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt



### 2. Daily Workflow & Pushing Code
Always pull the latest changes before starting work or pushing new features:

```bash
# Get the latest changes from main
git pull origin main

# Check modified files
git status

# Stage all updated files
git add .

# Commit your changes with a clear message
git commit -m "feat: implement Phase X <module_name>"

# Push to repository
git push origin main
