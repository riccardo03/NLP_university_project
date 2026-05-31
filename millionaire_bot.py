import json
import os
import re
import tempfile
import time
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline, BitsAndBytesConfig
from transformers import logging as transformers_logging

from rag_entertainment         import rag_entertainment
from rag_history               import rag_history
from rag_science               import rag_science
from rag_math                import rag_maths
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

_model         = None
_tokenizer     = None
_pipe          = None
_whisper_model = None

def _pipe_call(messages, max_new_tokens: int) -> str:
    try:
        outputs = _pipe(messages, max_new_tokens=max_new_tokens, do_sample=False)
    except Exception as e:
        if "System role not supported" in str(e):
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user   = next((m["content"] for m in messages if m["role"] == "user"), "")
            merged = [{"role": "user", "content": f"{system}\n\n{user}"}]
            outputs = _pipe(merged, max_new_tokens=max_new_tokens, do_sample=False)
        else:
            raise
    result = outputs[0]["generated_text"]
    return result.strip() if isinstance(result, str) else result[-1]["content"].strip()



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

    try:
        import rag_science as _rag_sci
        _rag_sci.setup_science_rag()
    except Exception as e:
        print(f"Warning: science RAG setup failed: {e}")

    try:
        import rag_entertainment as _rag_ent
        _rag_ent.set_entertainment_rag()
    except Exception as e:
        print(f"Warning: entertainment RAG setup failed: {e}")

def clean_option_text(text: str) -> str:
    # remove "option A/B/C/D", "A)", "B.", etc.
    text = re.sub(r"(?i)\boption\s*[a-d]\b[:\.\)]?", "", text)
    text = re.sub(r"^[A-Da-d][\)\.\-:]\\s*", "", text)

    return text.strip()

def load_speech_model(model_name: str = "large-v3") -> None:
    global _whisper_model

    from faster_whisper import WhisperModel

    print(f"Loading faster-whisper '{model_name}'...")

    _whisper_model = WhisperModel(
        model_name,
        device="cuda",
        compute_type="int8"  # <-- THIS is your "quantization"
    )

    print("Whisper ready.")

MCQ_PROMPT = """
You are transcribing a multiple choice quiz.

STRICT FORMAT RULES:
- Keep each question on a new line.
- Keep answer choices separate whenever they are spoken separately.
- Do not add labels such as A), B), C), D), Option A, Option B, etc.
- Do not merge answer choices into a sentence.
- Do not paraphrase or reword anything.
- Preserve punctuation, numbers, names, and dates as accurately as possible.
"""

MCQ_PARSE_PROMPT = """
You are an expert quiz parser and transcript normalizer.

TASKS:
1. Extract the question and answer choices.
2. Correct obvious speech-recognition or transcription errors when the intended text is reasonably clear from context.
3. Normalize spelling, capitalization, punctuation, and formatting.
4. Normalize dates to ISO 8601 format whenever possible:
   - YYYY-MM-DD for complete dates
   - YYYY-MM for year/month only
   - YYYY for year only
5. Preserve the meaning of the original transcript.
6. Do not invent information that is not supported by the transcript.
7. If multiple interpretations are plausible, keep the original wording.
8. Use context from the question and answer choices to resolve obvious transcription mistakes when confidence is high.

OUTPUT FORMAT (strict JSON):

{
  "questions": [
    {
      "question": "...",
      "A": "...",
      "B": "...",
      "C": "...",
      "D": "..."
    }
  ]
}

RULES:
- Return valid JSON only.
- No explanations.
- No markdown.
- No additional fields.
- If an option is missing, use "".
"""


def transcribe_audio(audio_path: str, prompt: str = "") -> str:
    global _whisper_model

    segments, _ = _whisper_model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        best_of=5,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=True,
        initial_prompt=prompt or MCQ_PROMPT,
    )

    return "\n".join(seg.text.strip() for seg in segments)


def transcribe_bytes(audio_bytes: bytes, prompt: str = "") -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        return transcribe_audio(tmp_path, prompt=prompt)
    finally:
        os.unlink(tmp_path)


def parse_mcq_with_qwen(raw_text: str):
    messages = [
        {"role": "system", "content": MCQ_PARSE_PROMPT},
        {"role": "user", "content": raw_text},
    ]
    result = _pipe_call(messages, max_new_tokens=512)
    # Strip markdown code fences the model may add
    result = re.sub(r"^```(?:json)?\s*", "", result.strip())
    result = re.sub(r"\s*```$", "", result.strip())
    return json.loads(result)

def transcribe_and_parse_mcq(audio_path: str):
    raw_text = transcribe_audio(audio_path)
    return parse_mcq_with_qwen(raw_text)



def play_speech_game(client, comp_id: int) -> dict:
    """Play a competition in speech mode using Whisper ASR.

    Identical pipeline to play_game() but reads question and option text
    from WAV audio (server TTS) transcribed by Whisper instead of the
    server text field.  Falls back to the server text field if audio
    transcription fails.  The returned log is compatible with
    print_evaluation().
    """
    from millionaire_client.exceptions import GameError

    game      = client.game.start(competition_id=comp_id, mode="speech")
    comp_name = COMP_NAMES.get(comp_id, f"Competition {comp_id}")

    log = {
        "competition":      comp_id,
        "competition_name": comp_name,
        "level_reached":    0,
        "earnings":         0.0,
        "questions":        [],
    }

    print(f"\n{'='*60}")
    print(f"  Starting (SPEECH): {comp_name}")
    print(f"{'='*60}")

    while game.in_progress:
        question = game.current_question
        if not question:
            print("No question available — game ended.")
            break

        level     = game.current_level
        time_left = game.time_remaining or 30.0

        print(f"\n--- Level {level} | Time: {time_left:.1f}s ---")

        # Transcribe question audio; fall back to server text on failure
        try:
            q_audio = game.fetch_audio_question()
            q_raw   = transcribe_bytes(q_audio)
        except Exception as e:
            print(f"  [Whisper] Q audio failed: {e}. Using server text.")
            q_raw = question.text

        # Transcribe each option audio sequentially (A → B → C → D)
        real_opts = question.options
        opt_raws  = []
        for real_opt in real_opts:
            try:
                opt_audio = game.fetch_audio_option_next()
                opt_raws.append(transcribe_bytes(opt_audio))
            except GameError:
                opt_raws.append(real_opt.text)

        # Use Qwen to clean and structure the raw Whisper transcriptions
        try:
            combined  = (
                f"Question: {q_raw}\n"
                + "\n".join(f"{chr(65+i)}) {t}" for i, t in enumerate(opt_raws))
            )
            parsed    = parse_mcq_with_qwen(combined)
            mcq       = parsed["questions"][0]
            q_text    = mcq.get("question") or q_raw
            opt_texts = [
                mcq.get(letter) or opt_raws[i]
                for i, letter in enumerate("ABCD")
                if i < len(opt_raws)
            ]
        except Exception as e:
            print(f"  [Qwen MCQ] Parse failed: {e}. Using raw transcriptions.")
            q_text    = q_raw
            opt_texts = opt_raws

        print(f"  [Whisper+Qwen] Q: {q_text}")
        for i, t in enumerate(opt_texts):
            print(f"    [{real_opts[i].id}] {t}")

        # Build lightweight option objects with real server IDs + transcribed texts
        class _SpeechOpt:
            def __init__(self, id_, text): self.id, self.text = id_, text

        speech_opts = [_SpeechOpt(real_opts[i].id, opt_texts[i])
                       for i in range(len(real_opts))]

        # RAG retrieval + LLM inference (same pipeline as text mode)
        print("  [RAG] Searching for context...")
        t0      = time.time()
        context = get_context(comp_id, q_text, opt_texts)
        print(f"  [RAG] Done in {time.time() - t0:.1f}s. Context: {context[:120].replace(chr(10), ' ')}...")

        user_prompt = build_user_prompt(q_text, speech_opts, context)
        print("  [LLM] Thinking...")
        t1         = time.time()
        raw_output = generate_answer(SYSTEM_PROMPTS[comp_id], user_prompt,
                                     max_new_tokens=_MAX_TOKENS[comp_id])
        answer_id  = extract_answer_id(raw_output, num_options=len(speech_opts))
        print(f"  [LLM] Output: '{raw_output}' -> Answer ID: {answer_id} (in {time.time() - t1:.1f}s)")

        q_record = {
            "level":        level,
            "question":     q_text,
            "options":      [{"id": o.id, "text": o.text} for o in speech_opts],
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
    print(f"  {comp_name} (SPEECH) — Level reached: {log['level_reached']} | Earnings: ${log['earnings']:,.2f}")
    print(f"{'='*60}\n")

    return log


def generate_answer(system_prompt: str, user_prompt: str, max_new_tokens: int = 40) -> str:
    if _pipe is None:
        raise RuntimeError("You must call load_model() first.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return _pipe_call(messages, max_new_tokens)


SYSTEM_PROMPTS = {
    COMP_ENTERTAINMENT: (
        "You are an expert on films, TV shows, music, actors, directors, and celebrities.\n"
        "A Context block of Wikipedia excerpts and web snippets may be provided.\n"
        "Rules:\n"
        "1. If the Context explicitly names or confirms one of the options, choose it.\n"
        "2. If the Context is off-topic, ambiguous, or absent, answer from your own knowledge.\n"
        "3. Never let the Context override a fact you know with certainty to be true.\n"
        "The VERY FIRST LINE must be exactly: ANSWER: <digit> (0, 1, 2, or 3).\n"
        "Then one sentence explaining why."
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
        "You are an expert mathematical and statistical evaluator.\n"
        "RULES:\n"
        "1. If the Context contains 'DIRECT_ANSWER: X', you MUST output 'ANSWER: X' immediately.\n"
        "2. If there is 'Mathematical Theory Context', use it to deduce the correct theoretical statement.\n"
        "3. Do not try to guess calculations mentally.\n"
        "REQUIRED: Your VERY LAST LINE must be exactly:\n"
        "ANSWER: X\n"
        "where X ∈ {0, 1, 2, 3}."
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
