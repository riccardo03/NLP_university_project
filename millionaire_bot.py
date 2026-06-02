# ─────────────────────────────────────────────────────────────────────────────
# Section 1 · Imports and constants
# ─────────────────────────────────────────────────────────────────────────────
import os
import random
import re
import time
import warnings
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)
from transformers import logging as transformers_logging
from rag_entertainment import rag_entertainment
from rag_history       import rag_history
from rag_science       import rag_science
from rag_math                 import rag_maths
from rag_news                  import rag_news
from rag_philosophy_psychology import rag_philosophy_psychology
from typing import Optional

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

COMP_ENTERTAINMENT             = 0
COMP_HISTORY_POLITICS          = 1
COMP_SCIENCE_NATURE            = 2
COMP_MATHS                     = 3
COMP_PHILOSOPHY_AND_PSYCHOLOGY = 4
COMP_NEWS                      = 5

COMP_NAMES = {
    COMP_ENTERTAINMENT:             "Entertainment",
    COMP_HISTORY_POLITICS:          "Ancient History & Politics",
    COMP_SCIENCE_NATURE:            "Science & Nature",
    COMP_MATHS:                     "Maths",
    COMP_PHILOSOPHY_AND_PSYCHOLOGY: "Philosophy & Psychology",
    COMP_NEWS:                      "News",
}

_MAX_TOKENS = {
    COMP_ENTERTAINMENT:             60,
    COMP_HISTORY_POLITICS:          60,
    COMP_SCIENCE_NATURE:            60,
    COMP_MATHS:                     300,   # chain-of-thought reasoning needs more room
    COMP_PHILOSOPHY_AND_PSYCHOLOGY: 60,
    COMP_NEWS:                      60,
}

_model     = None
_tokenizer = None
_pipe      = None

# ─────────────────────────────────────────────────────────────────────────────
# Section 2 · Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _pipe_call(messages: list, max_new_tokens: int) -> str:
    """
    Central pipeline wrapper.
    Handles the 'System role not supported' fallback for models that only
    accept a single user turn (merges system + user into one message).
    """
    try:
        outputs = _pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
    except Exception as e:
        if "System role not supported" in str(e):
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
            merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
            outputs = _pipe(merged, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            raise
    result = outputs[0]["generated_text"]
    return result.strip() if isinstance(result, str) else result[-1]["content"].strip()


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    global _model, _tokenizer, _pipe
    print(f"Loading model: {model_name}")

    # 4-bit quantization — reduces VRAM usage with negligible accuracy loss
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
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

    # Science RAG — no extra dependencies
    try:
        import rag_science
        rag_science.setup_science_rag()
    except Exception as e:
        print(f"Warning: science RAG setup failed: {e}")

    # Maths RAG — pass LLM callback + Wolfram key so it can call the solver
    try:
        import rag_math
        rag_math.setup_maths_rag(
            llm_callback=_pipe_call,
            tokenizer=_tokenizer,
            model=_model,
            wolfram_app_id=os.environ.get("WOLFRAM_APP_ID"),
        )
    except Exception as e:
        print(f"Warning: maths RAG setup failed: {e}")


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 40, **kwargs) -> str:
    if _pipe is None:
        raise RuntimeError("You must call load_model() first.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return _pipe_call(messages, max_new_tokens)


def warmup_models() -> None:
    """No-op: cross-encoder removed, no models require pre-loading."""
    print("  [Warmup] All models ready (no pre-loading required).")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 · System prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    # ── Entertainment: unchanged from official ──────────────────────────────
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
    # ── History: unchanged from official ────────────────────────────────────
    COMP_HISTORY_POLITICS: (
        "You are a history and politics expert. "
        "Given context (if any), a question, and four numbered options, "
        "the VERY FIRST LINE of your response must be exactly: ANSWER: <digit> (where digit is 0, 1, 2, or 3). "
        "Then provide a 1-sentence explanation of why that answer is correct."
    ),
    # ── Science: unchanged from official ────────────────────────────────────
    COMP_SCIENCE_NATURE: (
        "You are a careful science tutor. Use the provided context to answer "
        "multiple-choice science questions. Reason briefly (2-4 sentences), "
        "then end with EXACTLY one line: 'Answer: [N]' where N is 0, 1, 2, or 3."
    ),
    # ── Maths: updated from dev version ─────────────────────────────────────
    COMP_MATHS: (
        "You are an expert mathematician solving a multiple-choice problem.\n"
        "A Context block may be provided — it can contain:\n"
        "  • 'DIRECT_ANSWER: X' — a verified computed result; output 'ANSWER: X' immediately, no reasoning needed.\n"
        "  • 'PAL computation result' — a number; find which option matches it and output that index.\n"
        "  • 'Wolfram result' — an evaluated expression; find which option matches and output that index.\n"
        "  • 'Statistics/Probability Reference' — use it to reason about the correct option.\n"
        "  • 'Mathematical Theory Context' — use it to evaluate True/False statements.\n\n"
        "PROCESS (when no DIRECT_ANSWER is given):\n"
        "1. Read the Context carefully.\n"
        "2. If Context gives a computed value, identify which option it corresponds to — "
           "remember X is the INDEX (0,1,2,3), not the value itself.\n"
        "3. If Context is absent or irrelevant, reason step-by-step in 2-3 short sentences.\n"
        "4. For True/False statement pairs: evaluate each statement independently.\n"
        "5. For statistics questions: recall the precise definition before choosing.\n\n"
        "CRITICAL: ANSWER: X means option INDEX, not the option's text value.\n"
        "Example: computed answer is -49/12 ≈ -4.08. Option [3] says '-49/12'. Output: ANSWER: 3\n\n"
        "Your VERY LAST LINE must be exactly: ANSWER: X"
    ),
    # ── News: unchanged from dev ─────────────────────────────────────────────
    COMP_NEWS: (
        "You are a news analyst answering multiple-choice questions about recent events. "
        "An article is provided as the sole source of truth — read it carefully before answering. "
        "ALWAYS prioritize the article over your own knowledge. "
        "CRITICAL: reply with the INDEX of the correct option (0, 1, 2, or 3), "
        "NOT the value of the answer itself. For example, if the answer is '3' "
        "and option [0] is '3', reply ANSWER: 0. "
        "The VERY FIRST LINE must be exactly: ANSWER: <digit> (0, 1, 2, or 3). "
        "Then one sentence explaining why, quoting the article."
    ),
    # ── Philosophy & Psychology: unchanged from dev ──────────────────────────
    COMP_PHILOSOPHY_AND_PSYCHOLOGY: (
        "You are a philosophy and psychology expert answering multiple-choice questions. "
        "An article may be provided as context — read it carefully before answering. "
        "ALWAYS prioritize the article over your own knowledge when it is relevant. "
        "The VERY FIRST LINE must be exactly: ANSWER: <digit> (0, 1, 2, or 3). "
        "Then one sentence explaining why."
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
        # time_budget gives the solver room to run PAL/Wolfram before the 30s deadline
        return rag_maths(question_text, option_texts or [], time_budget=25.0)
    elif comp_id == COMP_NEWS:
        return rag_news(question_text, option_texts=option_texts or [])
    elif comp_id == COMP_PHILOSOPHY_AND_PSYCHOLOGY:
        return rag_philosophy_psychology(question_text, option_texts=option_texts or [])
    return ""

# ─────────────────────────────────────────────────────────────────────────────
# Section 5 · Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}

def extract_answer_id(text: str, num_options: int = 4) -> int:
    """
    Robust extraction of a digit answer from model output.
    Falls back to a random choice (instead of always 0) when no answer is found,
    preserving the expected 25 % accuracy floor without introducing bias.
    """
    # Priority 0: explicit structured tag "ANSWER: X"
    tag_match = re.search(r"\bANSWER\s*:\s*([0-3])\b", text, re.I)
    if tag_match:
        idx = int(tag_match.group(1))
        if idx < num_options:
            return idx

    # Priority 1: standalone digit within valid range
    for m in re.findall(r"\b([0-3])\b", text):
        idx = int(m)
        if idx < num_options:
            return idx

    # Priority 2: A/B/C/D letter mapping
    for m in re.findall(r"\b([A-Da-d])\b", text):
        idx = _LETTER_MAP.get(m.lower(), -1)
        if 0 <= idx < num_options:
            return idx

    # Fallback: random — at 25 % expected accuracy, never worse than always-0
    fallback = random.randint(0, num_options - 1)
    print(f"  [extract_answer_id] No answer found — random fallback → {fallback}")
    return fallback

# ─────────────────────────────────────────────────────────────────────────────
# Section 6 · Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(question_text: str, options: list, context: str) -> str:
    """
    Assemble the user-facing prompt with context, question, and options.
    """
    options_str = "\n".join(f"  [{opt.id}] {opt.text}" for opt in options)
    ctx_block   = f"Context:\n{context}\n\n" if context.strip() else ""
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
    comp_name     = COMP_NAMES.get(comp_id, f"Competition {comp_id}")
    system_prompt = SYSTEM_PROMPTS[comp_id]
    log = {
        "competition":      comp_id,
        "competition_name": comp_name,
        "level_reached":    0,
        "earnings":         0.0,
        "questions":        [],
    }

    print(f"\n{'='*60}")
    print(f"  Starting: {comp_name}")
    print(f"{'='*60}")

    while game.in_progress:
        question = game.current_question
        if not question:
            print("No question available — ended, the game has.")
            break

        level     = game.current_level
        time_left = game.time_remaining or 30.0
        print(f"\n--- Level {level} | Time: {time_left:.1f}s ---")
        print(f"Q: {question.text}")
        for opt in question.options:
            print(f"  [{opt.id}] {opt.text}")

        option_texts = [opt.text for opt in question.options]

        # Retrieve context from the appropriate RAG tool
        print("  [RAG] Searching for context...")
        t0      = time.time()
        context = get_context(comp_id, question.text, option_texts)
        rag_elapsed = time.time() - t0
        snippet = context[:120].replace("\n", " ") if context else "(none)"
        print(f"  [RAG] Done in {rag_elapsed:.1f}s. Context: {snippet}...")

        # ── DIRECT_ANSWER fast-exit (Maths only) ─────────────────────────
        # When the maths RAG has a verified computation result, trust it
        # directly and skip the main LLM call — which tends to override
        # correct numeric answers.
        if comp_id == COMP_MATHS:
            da_m = re.search(r'DIRECT_ANSWER\s*:\s*([0-3])', context)
            if da_m:
                answer_id = int(da_m.group(1))
                print(f"  [DIRECT] Skipping LLM — using RAG answer [{answer_id}]")
                q_record = {
                    "level":        level,
                    "question":     question.text,
                    "options":      [{"id": o.id, "text": o.text} for o in question.options],
                    "model_answer": answer_id,
                    "correct":      None,
                    "timed_out":    False,
                }
                result = game.answer(answer_id)
                q_record["correct"]   = result.correct
                q_record["timed_out"] = result.timed_out
                log["questions"].append(q_record)
                if result.timed_out:
                    print("  ⏰ TIMED OUT!")
                    log["level_reached"] = level
                    log["earnings"]      = result.earned_amount
                    break
                elif result.correct:
                    print(f"  ✓ CORRECT! Earned so far: ${result.earned_amount:,.2f}")
                    log["level_reached"] = level
                    log["earnings"]      = result.earned_amount
                    if result.game_over:
                        print("  🏆 GAME COMPLETE!")
                else:
                    print(f"  ✗ WRONG! Earned: ${result.earned_amount:,.2f}")
                    log["level_reached"] = level
                    log["earnings"]      = result.earned_amount
                    break
                continue  # next question

        # Build prompt and generate answer
        user_prompt = build_user_prompt(question.text, question.options, context)
        print("  [LLM] Thinking...")
        t1         = time.time()
        max_tokens = _MAX_TOKENS[comp_id]
        raw_output = generate_answer(system_prompt, user_prompt, max_new_tokens=max_tokens)
        answer_id  = extract_answer_id(raw_output, num_options=len(question.options))
        llm_elapsed = time.time() - t1
        print(f"  [LLM] Output: '{raw_output}' → Answer ID: {answer_id} (in {llm_elapsed:.1f}s)")

        # Record question before submitting
        q_record = {
            "level":        level,
            "question":     question.text,
            "options":      [{"id": o.id, "text": o.text} for o in question.options],
            "model_answer": answer_id,
            "correct":      None,
            "timed_out":    False,
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
    total_correct   = 0
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