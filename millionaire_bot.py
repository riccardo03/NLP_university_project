"""
PoliMillionaire chatbot — NLP university assignment.
Wise, this module is. Answer questions, it shall.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Section 1 · Imports and constants
# ─────────────────────────────────────────────────────────────────────────────

import re
import math
import time
import json
import warnings
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore")

# Competition IDs, clear they must be
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

# Loaded once, the model will be — heavy it is
_model     = None
_tokenizer = None
_pipe      = None


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    """Load the LLM into memory. Called once, it should be."""
    global _model, _tokenizer, _pipe

    print(f"Loading model, patience you must have: {model_name}")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    _pipe = pipeline(
        "text-generation",
        model=_model,
        tokenizer=_tokenizer,
    )
    print("Ready to answer, the model is.")


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    """
    Generate a fast answer — 10 new tokens, greedy decoding.
    Speed requires this, the 30-second timer does.
    """
    if _pipe is None:
        raise RuntimeError("Load the model first, you must. Call load_model().")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    outputs = _pipe(
        messages,
        max_new_tokens=10,
        do_sample=False,
        return_full_text=False,
    )
    # String or message list, both we handle
    result = outputs[0]["generated_text"]
    if isinstance(result, str):
        return result.strip()
    return result[-1]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 · System prompt templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    COMP_ENTERTAINMENT: (
        "An expert in entertainment, movies, music, and pop culture you are. "
        "Given a multiple-choice question and context, reply with ONLY the digit "
        "(0, 1, 2, or 3) of the best answer. No explanation needed."
    ),
    COMP_HISTORY_POLITICS: (
        "A scholar of ancient history, classical civilizations, and political systems you are. "
        "Given a multiple-choice question and context, reply with ONLY the digit "
        "(0, 1, 2, or 3) of the best answer. No explanation needed."
    ),
    COMP_SCIENCE_NATURE: (
        "A scientist with deep knowledge of biology, chemistry, physics, and natural phenomena you are. "
        "Given a multiple-choice question and context, reply with ONLY the digit "
        "(0, 1, 2, or 3) of the best answer. No explanation needed."
    ),
    COMP_MATHS: (
        "A mathematician you are. "
        "Think step by step through the calculation, then reply with ONLY the digit "
        "(0, 1, 2, or 3) that matches the correct numerical answer among the options. "
        "Show your chain-of-thought before the final digit."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 · RAG pipelines — one per competition
# ─────────────────────────────────────────────────────────────────────────────

# --- 4a. Entertainment → DuckDuckGo search ---

def rag_entertainment(query: str, num_results: int = 3) -> str:
    """
    Search DuckDuckGo for entertainment context.
    The web, our knowledge base it is.
    """
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                snippet = r.get("body", "")
                if snippet:
                    results.append(snippet)

        # Combined context, we build
        context = " ".join(results)
        return context[:1500] if context else ""
    except Exception as exc:
        # Silent failure — still play, we must
        print(f"  [RAG-Entertainment] Failed, it has: {exc}")
        return ""


# --- 4b. Ancient History & Politics → Wikipedia ---

def rag_history(query: str, sentences: int = 5) -> str:
    """
    Fetch Wikipedia summary for historical context.
    The encyclopedia of all knowledge, Wikipedia is.
    """
    try:
        import wikipedia

        wikipedia.set_lang("en")
        try:
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
            return summary
        except wikipedia.exceptions.DisambiguationError as e:
            # Among the options, the first we choose
            try:
                summary = wikipedia.summary(e.options[0], sentences=sentences)
                return summary
            except Exception:
                return ""
        except wikipedia.exceptions.PageError:
            # Not found, empty context we return
            return ""
    except Exception as exc:
        print(f"  [RAG-History] Failed, it has: {exc}")
        return ""


# --- 4c. Science & Nature → Wikipedia with Wikidata fallback ---

def _wikidata_sparql(query: str) -> str:
    """Query Wikidata SPARQL endpoint for a scientific entity description."""
    import urllib.request
    import urllib.parse

    # A label-based search on Wikidata, we perform
    sparql = f"""
    SELECT ?item ?itemLabel ?itemDescription WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                        wikibase:api "EntitySearch";
                        mwapi:search "{query}";
                        mwapi:language "en".
        ?item wikibase:apiOutputItem mwapi:item.
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    params = urllib.parse.urlencode({"query": sparql, "format": "json"})
    full_url = f"{url}?{params}"
    try:
        req = urllib.request.Request(
            full_url,
            headers={"User-Agent": "PoliMillionaireBot/1.0 (NLP assignment)"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            desc = bindings[0].get("itemDescription", {}).get("value", "")
            label = bindings[0].get("itemLabel", {}).get("value", "")
            return f"{label}: {desc}" if desc else label
    except Exception:
        pass
    return ""


def rag_science(query: str) -> str:
    """
    Wikipedia first, Wikidata fallback — science RAG, this is.
    Broad the knowledge of science is; multiple sources, we need.
    """
    context = rag_history(query, sentences=5)  # Wikipedia reused, efficient we are
    if context:
        return context
    # Fallback to Wikidata when Wikipedia fails
    print("  [RAG-Science] Wikipedia failed. Wikidata, we try.")
    return _wikidata_sparql(query)


# --- 4d. Maths → calculator tool ---

def _eval_expr(expr: str) -> Optional[float]:
    """
    Safely evaluate a mathematical expression.
    Only math functions allowed — dangerous code, execute we do not.
    """
    safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_names["abs"] = abs
    try:
        result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
        return float(result)
    except Exception:
        return None


def rag_maths(question_text: str) -> str:
    """
    Extract and compute mathematical expressions from the question.
    No internet needed — calculate, we shall!
    """
    results = []

    # Percentage patterns: "X% of Y", "X percent of Y"
    pct_match = re.findall(r"(\d+\.?\d*)\s*%\s*of\s*(\d+\.?\d*)", question_text, re.I)
    for pct, total in pct_match:
        val = float(pct) / 100 * float(total)
        results.append(f"{pct}% of {total} = {val}")

    # Square root patterns: "sqrt(X)", "square root of X", "√X"
    sqrt_match = re.findall(r"(?:sqrt\(|square root of|√)\s*(\d+\.?\d*)\)?", question_text, re.I)
    for n in sqrt_match:
        val = math.sqrt(float(n))
        results.append(f"sqrt({n}) = {val:.6f}")

    # Power patterns: "X^Y", "X**Y", "X to the power of Y"
    pow_match = re.findall(r"(\d+\.?\d*)\s*(?:\^|\*\*|to the power of)\s*(\d+\.?\d*)", question_text, re.I)
    for base, exp in pow_match:
        val = float(base) ** float(exp)
        results.append(f"{base}^{exp} = {val}")

    # Generic arithmetic expressions: extract numbers with operators
    arith_match = re.findall(r"(\d+\.?\d*\s*[\+\-\*\/]\s*\d+\.?\d*(?:\s*[\+\-\*\/]\s*\d+\.?\d*)*)", question_text)
    for expr in arith_match:
        val = _eval_expr(expr.replace(" ", ""))
        if val is not None:
            results.append(f"{expr.strip()} = {val}")

    # A hint we return, even if no pattern found
    if not results:
        return "No direct computation extracted. Reason carefully, you must."
    return "Computed results: " + "; ".join(results)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 · Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

_LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}


def extract_answer_id(text: str, num_options: int = 4) -> int:
    """
    Robust extraction of a digit answer from model output.
    Find the answer, we must — default to 0 if lost we are.
    """
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

    # Lost we are — default answer 0, the safe choice it is
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 · Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_user_prompt(question_text: str, options: list, context: str) -> str:
    """
    Assemble the user-facing prompt with context, question, and options.
    Clear and concise, the prompt must be.
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
# Section 7 · RAG dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def get_context(comp_id: int, question_text: str) -> str:
    """
    Select the correct RAG pipeline based on competition.
    Know which tool to use, a wise bot must.
    """
    if comp_id == COMP_ENTERTAINMENT:
        return rag_entertainment(question_text)
    elif comp_id == COMP_HISTORY_POLITICS:
        return rag_history(question_text)
    elif comp_id == COMP_SCIENCE_NATURE:
        return rag_science(question_text)
    elif comp_id == COMP_MATHS:
        return rag_maths(question_text)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Section 8 · Game loop
# ─────────────────────────────────────────────────────────────────────────────

def play_game(game, comp_id: int) -> dict:
    """
    Play a full game session, one question at a time.
    Guided by RAG and LLM, the bot is.

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

        # Retrieve context from the appropriate RAG tool
        print("  [RAG] Searching for context, we are...")
        t0 = time.time()
        context = get_context(comp_id, question.text)
        rag_elapsed = time.time() - t0

        snippet = context[:120].replace("\n", " ") if context else "(none)"
        print(f"  [RAG] Done in {rag_elapsed:.1f}s. Context: {snippet}...")

        # Build prompt and generate answer
        user_prompt = build_user_prompt(question.text, question.options, context)
        print("  [LLM] Thinking, the model is...")
        t1 = time.time()
        raw_output = generate_answer(system_prompt, user_prompt)
        llm_elapsed = time.time() - t1

        answer_id = extract_answer_id(raw_output, num_options=len(question.options))
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
            print("  ⏰ TIMED OUT! Move on, we could not.")
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
# Section 9 · Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def print_evaluation(log: dict) -> None:
    """
    Print a clear summary of a completed game.
    Judge the bot's performance, we shall.
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
    The grand picture, we reveal.
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
