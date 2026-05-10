# ─────────────────────────────────────────────────────────────────────────────
# Section 1 · Imports and constants
# ─────────────────────────────────────────────────────────────────────────────

import re
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline

from rag_entertainment import rag_entertainment
from rag_history      import rag_history
from rag_science      import rag_science, warmup_reranker
from rag_maths        import rag_maths

warnings.filterwarnings("ignore")

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


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 10, **kwargs) -> str:
    if _pipe is None:
        raise RuntimeError("You must call load_model() first.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    do_sample   = kwargs.pop("do_sample", False)
    temperature = kwargs.pop("temperature", 1.0)
    gen_kwargs  = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        return_full_text=False,
        **kwargs,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
 
    outputs = _pipe(messages, **gen_kwargs)
   
    result = outputs[0]["generated_text"]
    if isinstance(result, str):
        return result.strip()
    return result[-1]["content"].strip()


def warmup_models() -> None:
    """
    Force-load all lazily-initialized models before the game timer starts.
    Warm the cross-encoder now,  we avoid cold timeouts later.
    """
    print("  [Warmup] Loading cross-encoder...")
    warmup_reranker()
    print("  [Warmup] All models ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 · System prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    COMP_ENTERTAINMENT: (
        "You are an expert in entertainment, pop culture, movies, music, television, sports, and celebrity culture. "
        "You have encyclopedic knowledge of release dates, awards, chart positions, box office records, and cultural trivia across all decades and regions. "
        "Given a multiple-choice question and optional context, output ONLY the single digit (0, 1, 2, or 3) of the correct answer. "
        "No punctuation, no explanation, no reasoning — just the digit."
    ),

    COMP_HISTORY_POLITICS: (
        "You are a historian and political scientist with expertise spanning ancient civilizations, medieval history, modern geopolitics, wars, revolutions, constitutions, and world leaders. "
        "When context is provided, prioritize it over your own knowledge. "
        "Output ONLY the single digit (0, 1, 2, or 3) of the correct answer. "
        "No punctuation, no explanation, no reasoning — just the digit."
    ),

    COMP_SCIENCE_NATURE: (
        "You are a scientist with deep expertise in biology, chemistry, physics, earth sciences, astronomy, and ecology. "
        "When context is provided, extract the answer directly from it — do not rely on prior knowledge if the context is sufficient. "
        "If the question contains phrases like 'according to the article' or 'according to the text', treat the provided context as the authoritative source. "
        "Respond with exactly this format and nothing else:\n"
        "ANSWER: <digit>\n"
        "where <digit> is 0, 1, 2, or 3."
    ),

    COMP_MATHS: (
        "You are a precise mathematician. "
        "Solve the problem step by step, showing all intermediate calculations clearly. "
        "If numerical values are provided in the context (percentages, pre-computed results, factorials, etc.), use them directly. "
        "Your final line must be exactly:\n"
        "ANSWER: <digit>\n"
        "where <digit> is 0, 1, 2, or 3 — the index of the correct option. "
        "Do not write anything after the ANSWER line."
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
        return rag_science(question_text, option_texts or [], generate_answer_fn=generate_answer)
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
        "Reply with ONLY the option number (0, 1, 2, or 3)."
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
        tokens = 200 

        raw_output = generate_answer(system_prompt, user_prompt, max_new_tokens=tokens)
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
