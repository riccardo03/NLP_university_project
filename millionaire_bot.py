import re
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig
from transformers import logging as transformers_logging

from rag_entertainment         import rag_entertainment
from rag_history               import rag_history
from rag_science               import rag_science
from rag_maths                 import rag_maths
from rag_news                  import rag_news
from rag_philosophy_psychology import rag_philosophy_psychology

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

COMP_ENTERTAINMENT      = 0
COMP_HISTORY_POLITICS   = 1
COMP_SCIENCE_NATURE     = 2
COMP_MATHS              = 3
COMP_PHILOSOPHY_AND_PSYCHOLOGY = 4
COMP_NEWS               = 5

COMP_NAMES = {
    COMP_ENTERTAINMENT:    "Entertainment",
    COMP_HISTORY_POLITICS: "Ancient History & Politics",
    COMP_SCIENCE_NATURE:   "Science & Nature",
    COMP_MATHS:            "Maths",
    COMP_PHILOSOPHY_AND_PSYCHOLOGY: "Philosophy & Psychology",
    COMP_NEWS:             "News",
}

_MAX_TOKENS = {
    COMP_ENTERTAINMENT:    100,
    COMP_HISTORY_POLITICS: 100,
    COMP_SCIENCE_NATURE:   100,
    COMP_MATHS:            100,
    COMP_PHILOSOPHY_AND_PSYCHOLOGY: 100,
    COMP_NEWS:             100,
}

_model     = None
_tokenizer = None
_pipe      = None


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> None:
    global _model, _tokenizer, _pipe
    print(f"Loading model: {model_name}")

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
        trust_remote_code=True,  # required for Qwen
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

    try:
        import rag_entertainment as _rag_ent
        _rag_ent.setup_entertainment_rag()
    except Exception as e:
        print(f"Warning: entertainment RAG setup failed: {e}")

    try:
        import rag_science as _rag_sci
        _rag_sci.setup_science_rag()
    except Exception as e:
        print(f"Warning: science RAG setup failed: {e}")

    """try:
        import rag_maths as _rag_mth
        _rag_mth.setup_maths_rag()
    except Exception as e:
        print(f"Warning: maths RAG setup failed: {e}")"""


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 40, **kwargs) -> str:
    if _pipe is None:
        raise RuntimeError("You must call load_model() first.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    try:
        outputs = _pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    except Exception as e:
        if "System role not supported" in str(e):
            merged = [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]
            outputs = _pipe(
                merged,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        else:
            raise

    result = outputs[0]["generated_text"]
    if isinstance(result, str):
        return result.strip()
    return result[-1]["content"].strip()


SYSTEM_PROMPTS = {
    COMP_ENTERTAINMENT: (
        """You are a quiz expert answering multiple-choice entertainment questions.

        You will receive:
        - WIKIPEDIA: factual context about the subject
        - WEB CONTEXT: additional snippets from the web
        - QUESTION: the question to answer
        - OPTIONS: the 4 choices, some tagged with [retrieval: strong] or [retrieval: weak]

        Rules:
        1. Read WIKIPEDIA and WEB CONTEXT carefully before answering.
        2. [retrieval: strong] means that option has the most supporting evidence — prefer it unless Wikipedia contradicts it.
        3. [retrieval: weak] means little or no evidence was found for that option.
        4. If evidence is absent or unclear, use your own knowledge.
        5. If multiple options seem equally supported, trust WIKIPEDIA over retrieval scores.
        6. For NOT/EXCEPT questions, pick the option with the LEAST supporting evidence.
        7. Reply with ONLY a single digit: 0, 1, 2 or 3. No explanation. No text."""
    ),

    COMP_HISTORY_POLITICS: (
        "You are a history and politics expert answering a multiple choice question. "
        "ALWAYS prioritize the provided context over your own knowledge. "
        "If the context contains the answer, use it — do not override it with general reasoning. "
        "For questions asking about 'primary', 'main', or 'direct' cause/reason, choose the most proximate cause, not the most famous one. "
        "The VERY FIRST LINE must be exactly: ANSWER: <digit> (0, 1, 2, or 3). "
        "Then one sentence explaining why, referencing the context."
    ),

    COMP_SCIENCE_NATURE: (
        "You are an expert science tutor answering multiple-choice questions.\n"
        "You are given a Context that may or may not be relevant.\n"
        "Rules:\n"
        "1. If the Context directly answers the question, use it.\n"
        "2. If the Context is only loosely related, prefer your own knowledge.\n"
        "3. If the Context contradicts basic scientific facts, ignore it.\n"
        "The VERY FIRST LINE must be: ANSWER: X (where X is 0, 1, 2, or 3).\n"
        "Then one sentence of justification."
    ),

    COMP_MATHS: (
        "You are a math expert. Given the context and options, output ONLY 'Answer: [N]' where N is 0, 1, 2, or 3. No explanation."
    ),

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

    COMP_PHILOSOPHY_AND_PSYCHOLOGY: (
        "You are a philosophy and psychology expert answering multiple-choice questions. "
        "An article may be provided as context — read it carefully before answering. "
        "ALWAYS prioritize the article over your own knowledge when it is relevant. "
        "The VERY FIRST LINE must be exactly: ANSWER: <digit> (0, 1, 2, or 3). "
        "Then one sentence explaining why."
    ),
}


def get_context(comp_id: int, question_text: str, option_texts: list[str] | None = None) -> str:
    if comp_id == COMP_ENTERTAINMENT:
        return rag_entertainment(question_text, option_texts=option_texts or [])
    elif comp_id == COMP_HISTORY_POLITICS:
        return rag_history(question_text)
    elif comp_id == COMP_SCIENCE_NATURE:
        return rag_science(question_text, option_texts or [])
    elif comp_id == COMP_MATHS:
        return rag_maths(question_text, option_texts or [])
    elif comp_id == COMP_NEWS:
        return rag_news(question_text, option_texts=option_texts or [])
    elif comp_id == COMP_PHILOSOPHY_AND_PSYCHOLOGY:
        return rag_philosophy_psychology(question_text, option_texts=option_texts or [])
    return ""


_LETTER_MAP = {"a": 0, "b": 1, "c": 2, "d": 3}


def extract_answer_id(text: str, num_options: int = 4) -> int:
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

    print("Defaulting to 0")
    return 0


def build_user_prompt(question_text: str, options: list, context: str) -> str:
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


def play_game(game, comp_id: int) -> dict:
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
            print("No question available — game ended.")
            break

        level     = game.current_level
        time_left = game.time_remaining or 30.0

        print(f"\n--- Level {level} | Time: {time_left:.1f}s ---")
        print(f"Q: {question.text}")
        for opt in question.options:
            print(f"  [{opt.id}] {opt.text}")

        option_texts = [opt.text for opt in question.options]

        print("  [RAG] Searching for context...")
        t0      = time.time()
        context = get_context(comp_id, question.text, option_texts)
        print(f"  [RAG] Done in {time.time() - t0:.1f}s. Context: {context[:120].replace(chr(10), ' ')}...")

        user_prompt = build_user_prompt(question.text, question.options, context)
        print("  [LLM] Thinking...")
        t1         = time.time()
        raw_output = generate_answer(system_prompt, user_prompt, max_new_tokens=_MAX_TOKENS[comp_id])
        answer_id  = extract_answer_id(raw_output, num_options=len(question.options))
        print(f"  [LLM] Output: '{raw_output}' -> Answer ID: {answer_id} (in {time.time() - t1:.1f}s)")

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
            print("  TIMED OUT!")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            break
        elif result.correct:
            print(f"  CORRECT! Earned so far: ${result.earned_amount:,.2f}")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            if result.game_over:
                print("  GAME COMPLETE! All questions answered!")
        else:
            print(f"  WRONG! Game over. Earned: ${result.earned_amount:,.2f}")
            log["level_reached"] = level
            log["earnings"]      = result.earned_amount
            break

    print(f"\n{'='*60}")
    print(f"  {comp_name} — Level reached: {log['level_reached']} | Earnings: ${log['earnings']:,.2f}")
    print(f"{'='*60}\n")

    return log


def print_evaluation(log: dict) -> None:
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
        status = "OK" if q.get("correct") else ("TO" if q.get("timed_out") else "XX")
        ans_id = q.get("model_answer", "?")
        chosen = next(
            (o["text"] for o in q.get("options", []) if o["id"] == ans_id),
            str(ans_id),
        )
        print(f"  [{status}] L{q['level']}: {q['question'][:60]}... -> [{ans_id}] {chosen[:30]}")
    print()


def print_all_evaluations(logs: list) -> None:
    print("\n" + "=" * 60)
    print("  OVERALL SUMMARY — PoliMillionaire Bot")
    print("=" * 60)

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
    print("=" * 60 + "\n")
