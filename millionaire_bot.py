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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    # Replace model-card generation_config to eliminate max_length/temperature conflicts
    from transformers import GenerationConfig
    _model.generation_config = GenerationConfig(
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=(
            _tokenizer.pad_token_id
            if _tokenizer.pad_token_id is not None
            else _tokenizer.eos_token_id
        ),
    )
    print("Ready to answer, the model is.")
    warmup_models()


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 10, **kwargs) -> str:
    """
    Generate an answer with greedy decoding.
    Speed requires few tokens for most tasks; maths needs more for CoT.
    """
    if _pipe is None:
        raise RuntimeError("Load the model first, you must. Call load_model().")

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
    # String or message list, both we handle
    result = outputs[0]["generated_text"]
    if isinstance(result, str):
        return result.strip()
    return result[-1]["content"].strip()


def warmup_models() -> None:
    """
    Force-load all lazily-initialized models before the game timer starts.
    Warm the cross-encoder now, cold timeouts later we avoid.
    """
    print("  [Warmup] Loading cross-encoder, before game starts...")
    _rerank("warmup", ["warmup"])
    print("  [Warmup] All models ready, they are.")


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
        "Read the context, then reply with exactly: ANSWER: <digit> "
        "where <digit> is 0, 1, 2, or 3 — the index of the correct option. No other text."
    ),
    COMP_MATHS: (
        "You are a precise mathematician. "
        "Work through the problem step by step, showing all intermediate calculations. "
        "After your working, output the answer on the very last line in exactly this format:\n"
        "ANSWER: <digit>\n"
        "where <digit> is 0, 1, 2, or 3 — the index of the correct option. No other text after that line."
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
        from ddgs import DDGS

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


# --- 4c. Science & Nature — multi-stage RAG pipeline ---

# Cross-encoder loaded lazily on first Science question, heavy it need not be at import
_reranker = None


def _get_reranker():
    """Load (and cache) the cross-encoder model. Once loaded, remember it we do."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("  [RAG-Science] Loading cross-encoder, patience you must have...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  [RAG-Science] Cross-encoder ready, it is.")
    return _reranker


# Stage 1 — Query Generation (rule-based, no LLM call — fast inside the timer it must be)
def _generate_search_queries(question_text: str, option_texts: list) -> list:
    """
    Build search queries from question text and options without an LLM call.
    Instant this is; 20 seconds wasted on generation, we avoid.
    """
    queries = [question_text]
    # Add question+option combos for the first two options as complementary queries
    for opt in option_texts[:2]:
        words = opt.split()
        if len(words) >= 2:
            queries.append(f"{question_text} {opt}")
    return queries[:3]


# Stage 2 — Parallel Multi-Source Retrieval
def _retrieve_parallel(queries: list) -> list:
    """
    Run Wikipedia and DuckDuckGo in parallel for every query.
    Fast retrieval across sources, ThreadPoolExecutor provides.
    """
    tasks = []
    for q in queries:
        tasks.append(("wiki", q))
        tasks.append(("ddg",  q))

    snippets = []
    seen = set()

    def _fetch(source, query):
        if source == "wiki":
            result = rag_history(query, sentences=4)
            # Wikipedia parse failures return ""; fall through to DuckDuckGo
            if not result:
                result = rag_entertainment(query, num_results=2)
            return result
        else:
            return rag_entertainment(query, num_results=2)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, src, q): (src, q) for src, q in tasks}
        for fut in as_completed(futures, timeout=15):
            try:
                text = fut.result()
            except Exception:
                text = ""
            if text and text not in seen:
                seen.add(text)
                snippets.append(text)

    return snippets


# Stage 3 — Cross-Encoder Re-ranking
def _rerank(question_text: str, snippets: list, top_k: int = 3) -> str:
    """
    Score each snippet against the question; keep the top-k.
    Relevant context rises to the top, irrelevant falls away.
    """
    if not snippets:
        return ""
    reranker = _get_reranker()
    pairs = [(question_text, s) for s in snippets]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, snippets), reverse=True)
    top_snippets = [s for _, s in ranked[:top_k]]
    return "\n\n".join(top_snippets)


def rag_science(question_text: str, option_texts: list = None) -> str:
    """
    Multi-stage science RAG: query generation → parallel retrieval → re-ranking.
    Wikidata dropped; cross-encoder reranking added, better coverage achieved.
    """
    if option_texts is None:
        option_texts = []

    t_start = time.time()

    # Stage 1: generate diverse queries (LLM call — kept to 60 tokens for speed)
    queries = _generate_search_queries(question_text, option_texts)
    print(f"  [RAG-Science] Queries generated in {time.time()-t_start:.1f}s: {queries}")

    # Stage 2: parallel multi-source retrieval
    snippets = _retrieve_parallel(queries)
    print(f"  [RAG-Science] {len(snippets)} snippets in {time.time()-t_start:.1f}s total")

    if not snippets:
        return ""

    # Stage 3: re-rank and return top context
    context = _rerank(question_text, snippets, top_k=3)
    print(f"  [RAG-Science] RAG complete in {time.time()-t_start:.1f}s")
    return context


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


_WORD_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "half": 0.5, "quarter": 0.25, "third": 1/3, "eighth": 0.125,
}


def _word_to_num(text: str) -> str:
    """Replace English word-numbers with digits."""
    for word, val in _WORD_NUMBERS.items():
        text = re.sub(rf"\b{word}\b", str(val), text, flags=re.I)
    return text


def rag_maths(question_text: str) -> str:
    """
    Extract and compute mathematical expressions from the question.
    Handles percentages, roots, powers, factorials, combinations, word numbers,
    and falls back to sympy for general symbolic evaluation.
    """
    results = []
    q = _word_to_num(question_text)

    # Percentage: "X% of Y" or "X percent of Y"
    for pct, total in re.findall(r"(\d+\.?\d*)\s*(?:%|percent)\s*of\s*(\d+\.?\d*)", q, re.I):
        val = float(pct) / 100 * float(total)
        results.append(f"{pct}% of {total} = {val}")

    # Square root: "sqrt(X)", "square root of X", "√X"
    for n in re.findall(r"(?:sqrt\s*\(|square\s+root\s+of|√)\s*(\d+\.?\d*)\)?", q, re.I):
        results.append(f"sqrt({n}) = {math.sqrt(float(n)):.6f}")

    # Power: "X^Y", "X**Y", "X to the power of Y"
    for base, exp in re.findall(r"(\d+\.?\d*)\s*(?:\^|\*\*|to\s+the\s+power\s+of)\s*(\d+\.?\d*)", q, re.I):
        results.append(f"{base}^{exp} = {float(base) ** float(exp)}")

    # Factorial: "X!" or "factorial of X" or "X factorial"
    for n in re.findall(r"(\d+)\s*!", q):
        val = math.factorial(int(n))
        results.append(f"{n}! = {val}")
    for n in re.findall(r"factorial\s+of\s+(\d+)", q, re.I):
        results.append(f"{n}! = {math.factorial(int(n))}")

    # Combinations nCr: "n choose r", "C(n,r)", "nCr"
    for n, r in re.findall(r"(\d+)\s*[Cc](?:hoose|r)?\s*(\d+)", q):
        val = math.comb(int(n), int(r))
        results.append(f"C({n},{r}) = {val}")
    for n, r in re.findall(r"[Cc]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", q):
        val = math.comb(int(n), int(r))
        results.append(f"C({n},{r}) = {val}")

    # Inline arithmetic: e.g. "3 + 4 * 2"
    for expr in re.findall(r"(\d+\.?\d*(?:\s*[\+\-\*\/]\s*\d+\.?\d*)+)", q):
        val = _eval_expr(expr.replace(" ", ""))
        if val is not None:
            results.append(f"{expr.strip()} = {val}")

    # sympy fallback for general symbolic expressions
    if not results:
        try:
            import sympy
            # Strip non-math text; attempt to parse as expression
            cleaned = re.sub(r"[^0-9\+\-\*\/\^\(\)\.\s]", " ", q).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if re.search(r"\d", cleaned):
                val = sympy.sympify(cleaned.replace("^", "**"))
                results.append(f"sympy({cleaned}) = {float(val):.6f}")
        except Exception:
            pass

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

def get_context(comp_id: int, question_text: str, option_texts: list = None) -> str:
    """
    Select the correct RAG pipeline based on competition.
    Know which tool to use, a wise bot must.
    """
    if comp_id == COMP_ENTERTAINMENT:
        return rag_entertainment(question_text)
    elif comp_id == COMP_HISTORY_POLITICS:
        return rag_history(question_text)
    elif comp_id == COMP_SCIENCE_NATURE:
        return rag_science(question_text, option_texts or [])
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

        option_texts = [opt.text for opt in question.options]

        # Retrieve context from the appropriate RAG tool
        print("  [RAG] Searching for context, we are...")
        t0 = time.time()
        context = get_context(comp_id, question.text, option_texts)
        rag_elapsed = time.time() - t0

        snippet = context[:120].replace("\n", " ") if context else "(none)"
        print(f"  [RAG] Done in {rag_elapsed:.1f}s. Context: {snippet}...")

        # Build prompt and generate answer
        user_prompt = build_user_prompt(question.text, question.options, context)
        print("  [LLM] Thinking, the model is...")
        t1 = time.time()
        if comp_id == COMP_MATHS:
            tokens = 200
        else:
            tokens = 30

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
