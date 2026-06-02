"""
Philosophy & Psychology RAG: DDG search → article fetch → LLM prompt.

No Wikipedia. No BM25 voting.
Subject extraction reuses the GLiNER model loaded by setup_entertainment_rag().
"""

import concurrent.futures

from rag_utils import (
    STOP_WORDS_BASE, TOKEN_RE,
    search_and_fetch,
    extract_subjects_gliner, extract_subjects_regex,
)

_TIMEOUT           = 4
_MAX_DDG_RESULTS   = 3

_STOP_WORDS_PHILOSOPHY_PSYCHOLOGY: frozenset[str] = STOP_WORDS_BASE | {
    "published", "reported", "stated", "said",
}

_GLINER_LABELS_PHILOSOPHY = [
    "person", "philosopher", "psychologist",
    "theory", "concept", "school of thought",
    "book", "work",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_philosophy_psychology_rag() -> None:
    """No-op: GLiNER is loaded by setup_entertainment_rag()."""


def rag_philosophy_psychology(query: str, option_texts: list[str] | None = None) -> str:
    labeled = extract_subjects_gliner(query, _GLINER_LABELS_PHILOSOPHY)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = subjects[0]
        print(f"  [Philosophy_Psychology] GLiNER entities: {labeled}")
    else:
        subjects  = extract_subjects_regex(query, _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY)
        main_term = subjects[0] if subjects else ""
        print(f"  [Philosophy_Psychology] regex subjects: {subjects}")

    if not main_term:
        tokens    = TOKEN_RE.findall(query.lower())
        main_term = " ".join(
            t for t in tokens if len(t) >= 4 and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
        )[:60]

    subject_tokens  = {s.lower() for s in subjects}
    q_content_words = [
        t for t in TOKEN_RE.findall(query.lower())
        if len(t) >= 5
        and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
        and t not in subject_tokens
    ]
    q1 = main_term
    q2 = " ".join(filter(None, [main_term, " ".join(q_content_words[:3])]))

    queries = list(dict.fromkeys(q for q in [q1, q2] if q.strip()))
    print(f"  [Philosophy_Psychology] queries: {queries}")

    q_keywords = {
        t for t in TOKEN_RE.findall(query.lower())
        if len(t) >= 4 and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
    }

    def _search_and_fetch(q_text: str) -> list[tuple[str, str]]:
        return search_and_fetch(
            q_text, q_keywords,
            max_results=_MAX_DDG_RESULTS,
            timeout=_TIMEOUT, user_agent="PhilosophyBot/1.0",
            module_tag="Philosophy_Psychology",
        )

    candidates: list[tuple[int, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_search_and_fetch, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            for text, url in fut.result():
                score = sum(1 for kw in q_keywords if kw in text.lower())
                candidates.append((score, text, url))
                print(f"  [Philosophy_Psychology] candidate (score={score}, {len(text)} chars): {url[:80]}")

    if not candidates:
        print("  [Philosophy_Psychology] No article found — LLM fallback")
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, article_text, article_url = candidates[0]
    print(f"  [Philosophy_Psychology] selected: {article_url[:80]}")

    lines: list[str] = [
        f"ARTICLE (source: {article_url}):",
        article_text,
        "",
        f"QUESTION: {query}",
        "",
    ]
    if option_texts:
        lines.append("OPTIONS:")
        for i, opt in enumerate(option_texts[:4]):
            lines.append(f"[{i}] {opt}")
        lines.append("")
    lines.append("Answer (0/1/2/3):")

    return "\n".join(lines)
