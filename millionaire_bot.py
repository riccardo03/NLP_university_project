# ─────────────────────────────────────────────────────────────────────────────
# Section 1 · Imports and constants
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from transformers import logging as transformers_logging

from rag_entertainment import rag_entertainment
from rag_history      import rag_history
from rag_science      import rag_science
from rag_maths        import rag_maths
from typing import Optional

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

COMP_ENTERTAINMENT      = 0
COMP_HISTORY_POLITICS   = 1
COMP_SCIENCE_NATURE     = 2
COMP_MATHS              = 3

COMP_NAMES = {
    COMP_ENTERTAINMENT:    "Entertainment",
    COMP_HISTORY_POLITICS: "Ancient History & Politics",
    COMP_SCIENCE_NATURE:   "Science & Nature",
    COMP_MATHS:            "Maths",
}

_MAX_TOKENS = {
    COMP_ENTERTAINMENT:    30,
    COMP_HISTORY_POLITICS: 30,
    COMP_SCIENCE_NATURE:   30,
    COMP_MATHS:            30,
}


# (Science RAG moved back to rag_science.py)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 · Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    global _model, _tokenizer, _pipe
    print(f"Loading model: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    _model.config.max_length = None
    _model.generation_config = GenerationConfig(
         pad_token_id=_tokenizer.pad_token_id or _tokenizer.eos_token_id,
         eos_token_id=_tokenizer.eos_token_id,
    )
    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )

    print("The model is ready to answer.")
    warmup_models()
    # Initialize science RAG now that the model and environment are ready.
    try:
        import rag_science
        rag_science.setup_science_rag()
    except Exception as e:
        print(f"Warning: science RAG setup failed: {e}")


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 30, **kwargs) -> str:
    if _pipe is None:
        raise RuntimeError("You must call load_model() first.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    do_sample   = True
    temperature = 0.1
    outputs = _pipe(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
    )

    result = outputs[0]["generated_text"]
    if isinstance(result, str):
        return result.strip()
    return result[-1]["content"].strip()


def warmup_models() -> None:
    """No-op: cross-encoder removed, no models require pre-loading."""
    print("  [Warmup] All models ready (no pre-loading required).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 · System prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    COMP_ENTERTAINMENT: (
    "You are an Entertainment quiz expert. Pick exactly one option (0, 1, 2, or 3).\n\n"

    "CONTEXT FORMAT (when provided):\n"
    "  - 'WIKIPEDIA (key passages)': authoritative passages — read first.\n"
    "  - '[i] <option> (score X.X)': evidence retrieved specifically for option i.\n"
    "  - '★ STRONGEST EVIDENCE': RAG's top-ranked option (strong hint, NOT infallible).\n"
    "  - '(no specific evidence)': no snippet was found for that option.\n\n"

    "DECISION HIERARCHY:\n"
    "  1. CONTEXT FIRST: if a Wikipedia passage directly answers the question, "
       "trust it even when it contradicts your prior.\n"
    "  2. INTERNAL KNOWLEDGE: if context is missing, irrelevant, or silent on the "
       "specific fact asked, fall back on your own knowledge.\n"
    "  3. SILENCE != FALSE: the context not mentioning a fact never refutes it.\n\n"

    "ANTI-HALLUCINATION (strict):\n"
    "  - Do NOT write 'as stated in the context', 'according to the passage', or any "
       "similar attribution unless you can quote the exact phrase. Inventing a "
       "citation is the worst error you can make.\n"
    "  - When relying on your own knowledge, prefix your reasoning with "
       "'From general knowledge:' — never disguise a guess as a citation.\n"
    "  - Treat the ★ marker as a strong hint, but override it if the Wikipedia "
       "passages clearly point elsewhere or if the marked snippet is irrelevant.\n\n"

    "STRATEGY:\n"
    "  - Reason internally to eliminate wrong options; keep the visible output short.\n"
    "  - For NOT/EXCEPT questions, pick the option WITHOUT supporting evidence.\n"
    "  - If multiple options remain plausible, prefer the most specific, widely "
       "recognized fact in entertainment history.\n\n"

    "OUTPUT (strict, exactly two lines):\n"
    "  Line 1: ANSWER: <digit>\n"
    "  Line 2: ONE sentence. Either paraphrase the supporting passage, or start "
       "with 'From general knowledge:' followed by the fact you relied on."
),

    COMP_HISTORY_POLITICS: (
        "You are a history and politics expert. "
        "Given context (if any), a question, and four numbered options, "
        "the VERY FIRST LINE of your response must be exactly: ANSWER: <digit> (where digit is 0, 1, 2, or 3). "
        "Then provide a 1-sentence explanation of why that answer is correct."
    ),

    COMP_SCIENCE_NATURE: (
        "You are a careful science tutor. Use the provided context to answer "
        "multiple-choice science questions. Reason briefly (2-4 sentences), "
        "then end with EXACTLY one line: 'Answer: [N]' where N is 0, 1, 2, or 3."
    ),

    COMP_MATHS: (
        "You are a Mathematical Reasoning Engine. Your goal is to solve the problem "
        "step-by-step and select the correct option (0, 1, 2, or 3). "
        "HIERARCHY OF TRUTH: "
        "1. PROVIDED CONTEXT: If the context contains formulas, values, or definitions, "
        "you MUST use them, even if they differ from your internal knowledge. "
        "2. INTERNAL KNOWLEDGE: Use your mathematical expertise only if the context is "
        "missing or irrelevant to the specific calculation asked. "
        "OPERATIONAL PROTOCOL: "
        "1. IDENTIFY: Determine the specific mathematical domain (Algebra, Geometry, "
        "Calculus, Statistics, Probability, or Logic). "
        "2. EXTRACT: Isolate all numerical values, variables, units, and constraints. "
        "3. SOLVE: Execute the calculation step-by-step. Apply any formula from the context strictly. "
        "Show intermediate work for complex calculations. "
        "4. VERIFY: Test each option (0, 1, 2, 3) by substituting it back into the problem. "
        "Record which option satisfies the equation or logical condition. "
        "PRECISION & VALIDATION RULES: "
        "- DIMENSIONAL ANALYSIS: Verify the final answer has correct units (meters, kg, "
        "probability ∈ [0,1], degrees, etc.). Unit mismatch = wrong answer. "
        "- ROUNDING: If the context provides approximate values (π ≈ 3.14), use that precision. "
        "Match the precision shown in the options (e.g., 2 decimal places). "
        "- GEOMETRY: Distinguish between radius/diameter, area/volume, perimeter/circumference. "
        "Confirm the question explicitly states which measurement is requested. "
        "- PROBABILITY: Verify the sample space is complete (e.g., probabilities sum to 1). "
        "- COMBINATORICS: Beware of off-by-one errors (n vs n-1), and distinguish between "
        "combinations (order irrelevant) and permutations (order matters). "
        "- LOGIC: If confused, test the contrapositive or use truth tables. "
        "COMMON PITFALLS TO AVOID: "
        "- Sign errors (forgetting negatives in algebraic solutions). "
        "- Confusing operators (sum vs product, derivative vs integral). "
        "- Premature rounding (keep full precision during intermediate steps). "
        "OUTPUT FORMAT: "
        "The VERY FIRST LINE of your response must be exactly: ANSWER: <digit>. "
        "Then provide a concise 1-2 sentence derivation showing the key calculation or reasoning."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 · RAG dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_context(comp_id: int, question_text: str, option_texts: list = None) -> str:
    """
    Select the correct RAG pipeline based on competition.
    """
    if comp_id == COMP_ENTERTAINMENT:
        return rag_entertainment(question_text, generate_answer_fn=generate_answer,
                                 option_texts=option_texts or [])
    elif comp_id == COMP_HISTORY_POLITICS:
        return rag_history(question_text)
    elif comp_id == COMP_SCIENCE_NATURE:
        return rag_science(question_text, option_texts or [])
    elif comp_id == COMP_MATHS:
        return rag_maths(question_text)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 · Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}


def extract_answer_id(text: str, num_options: int = 4) -> int:
    """
    Robust extraction of a digit answer from model output.
    """
    # Priority 0: explicit structured tag "ANSWER: X"
    tag_match = re.search(r"\bANSWER\s*:\s*([0-3])\b", text, re.I)
    if tag_match:
        idx = int(tag_match.group(1))
        if idx < num_options:
            return idx

    # Priority 1: standalone digit within valid range
    digit_matches = re.findall(r"\b([0-3])\b", text)
    for m in digit_matches:
        idx = int(m)
        if idx < num_options:
            return idx

    # Priority 2: A/B/C/D letter mapping
    letter_matches = re.findall(r"\b([A-Da-d])\b", text)
    for m in letter_matches:
        idx = _LETTER_MAP.get(m.lower(), -1)
        if 0 <= idx < num_options:
            return idx

    # default answer 0
    print("Defaulting to 0")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 · Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(question_text: str, options: list, context: str) -> str:
    """
    Assemble the user-facing prompt with context, question, and options.
    """
    options_str = "\n".join(f"  [{opt.id}] {opt.text}" for opt in options)
    ctx_block = f"Context:\n{context}\n\n" if context.strip() else ""
    return (
        f"{ctx_block}"
        f"Question: {question_text}\n\n"
        f"Options:\n{options_str}\n\n"
        f"Output FIRST on its own line:\n"
        f"ANSWER: X\n"
        f"(where X is 0, 1, 2, or 3)\n"
        f"Then explain briefly in 1 sentence."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 · Game loop
# ─────────────────────────────────────────────────────────────────────────────

def play_game(game, comp_id: int) -> dict:
    """
    Play a full game session, one question at a time.
    Returns a structured evaluation log.
    """
    comp_name = COMP_NAMES.get(comp_id, f"Competition {comp_id}")
    system_prompt = SYSTEM_PROMPTS[comp_id]

    log = {
        "competition": comp_id,
        "competition_name": comp_name,
        "level_reached": 0,
        "earnings": 0.0,
        "questions": [],
    }

    print(f"\n{'='*60}")
    print(f"  Starting: {comp_name}")
    print(f"{'='*60}")

    while game.in_progress:
        question = game.current_question
        if not question:
            print("No question available — ended, the game has.")
            break

        level = game.current_level
        time_left = game.time_remaining or 30.0

        print(f"\n--- Level {level} | Time: {time_left:.1f}s ---")
        print(f"Q: {question.text}")
        for opt in question.options:
            print(f"  [{opt.id}] {opt.text}")

        option_texts = [opt.text for opt in question.options]

        # Retrieve context from the appropriate RAG tool
        print("  [RAG] Searching for context...")
        t0 = time.time()
        context = get_context(comp_id, question.text, option_texts)
        rag_elapsed = time.time() - t0

        snippet = context[:120].replace("\n", " ") if context else "(none)"
        print(f"  [RAG] Done in {rag_elapsed:.1f}s. Context: {snippet}...")

        # Build prompt and generate answer
        user_prompt = build_user_prompt(question.text, question.options, context)
        print("  [LLM] Thinking...")
        t1 = time.time()
        max_tokens = _MAX_TOKENS[comp_id]

        raw_output = generate_answer(system_prompt, user_prompt, max_new_tokens=max_tokens)
        answer_id = extract_answer_id(raw_output, num_options=len(question.options))

        # Self-consistency disabled — model is too slow (~2 tok/s) for 3 LLM calls in 30s

        llm_elapsed = time.time() - t1
        print(f"  [LLM] Output: '{raw_output}' → Answer ID: {answer_id} (in {llm_elapsed:.1f}s)")

        # Record question before submitting
        q_record = {
            "level": level,
            "question": question.text,
            "options": [{"id": o.id, "text": o.text} for o in question.options],
            "model_answer": answer_id,
            "correct": None,
            "timed_out": False,
        }

        # Submit the answer
        result = game.answer(answer_id)

        q_record["correct"]   = result.correct
        q_record["timed_out"] = result.timed_out
        log["questions"].append(q_record)

        if result.timed_out:
            print("  ⏰ TIMED OUT! We could not move on.")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            break
        elif result.correct:
            print(f"  ✓ CORRECT! Earned so far: ${result.earned_amount:,.2f}")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            if result.game_over:
                print(f"\n  🏆 GAME COMPLETE! All questions answered!")
        else:
            print(f"  ✗ WRONG! Game over. Earned: ${result.earned_amount:,.2f}")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            break

    print(f"\n{'='*60}")
    print(f"  {comp_name} — Level reached: {log['level_reached']} | Earnings: ${log['earnings']:,.2f}")
    print(f"{'='*60}\n")

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 · Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation(log: dict) -> None:
    """
    Print a clear summary of a completed game.
    """
    comp_name = log.get("competition_name", f"Competition {log['competition']}")
    questions = log.get("questions", [])

    total     = len(questions)
    correct   = sum(1 for q in questions if q.get("correct"))
    timed_out = sum(1 for q in questions if q.get("timed_out"))
    accuracy  = correct / total if total > 0 else 0.0

    print(f"\n{'─'*50}")
    print(f"  EVALUATION — {comp_name}")
    print(f"{'─'*50}")
    print(f"  Level reached : {log['level_reached']}")
    print(f"  Earnings      : ${log['earnings']:,.2f}")
    print(f"  Questions     : {total}")
    print(f"  Correct       : {correct}")
    print(f"  Timed out     : {timed_out}")
    print(f"  Accuracy      : {accuracy:.1%}")
    print(f"{'─'*50}")

    # Per-question breakdown
    for i, q in enumerate(questions, 1):
        status = "✓" if q.get("correct") else ("⏰" if q.get("timed_out") else "✗")
        ans_id = q.get("model_answer", "?")
        chosen = next(
            (o["text"] for o in q.get("options", []) if o["id"] == ans_id),
            str(ans_id),
        )
        print(f"  [{status}] L{q['level']}: {q['question'][:60]}... → [{ans_id}] {chosen[:30]}")
    print()


def print_all_evaluations(logs: list) -> None:
    """
    Summarize all games across all competitions.
    """
    print("\n" + "═" * 60)
    print("  OVERALL SUMMARY — PoliMillionaire Bot")
    print("═" * 60)

    total_correct = 0
    total_questions = 0

    for log in logs:
        questions = log.get("questions", [])
        correct   = sum(1 for q in questions if q.get("correct"))
        total     = len(questions)
        total_correct   += correct
        total_questions += total
        accuracy = correct / total if total > 0 else 0.0
        name = log.get("competition_name", f"Comp {log['competition']}")
        print(
            f"  {name:<35} | "
            f"Lvl {log['level_reached']:>2} | "
            f"${log['earnings']:>10,.2f} | "
            f"Acc {accuracy:.0%}"
        )

    overall = total_correct / total_questions if total_questions > 0 else 0.0
    print(f"{'─'*60}")
    print(f"  Overall accuracy: {overall:.1%}  ({total_correct}/{total_questions} correct)")
    print("═" * 60 + "\n")
