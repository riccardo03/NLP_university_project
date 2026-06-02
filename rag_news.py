"""
News RAG: date-anchored DDG search → article fetch → LLM prompt.
No caching (news changes daily).
Subject extraction uses a news-specific GLiNER call.
"""

import concurrent.futures
import datetime
import math
import re

from rag_utils import (
    STOP_WORDS_BASE, SKIP_DOMAINS, SKIP_URL_PATTERNS,
    fetch_article, ddg_search,
    extract_subjects_gliner,
)

_TIMEOUT           = 4
_MIN_ARTICLE_CHARS = 500
_MAX_DDG_RESULTS   = 3

_STOP_WORDS_NEWS: frozenset[str] = STOP_WORDS_BASE | {
    "published", "reported", "stated", "said",
}

_GLINER_LABELS_NEWS: list[str] = [
    "person", "politician", "president", "minister", "official",
    "organization", "company", "institution", "government agency",
    "country", "city", "location", "region",
    "law", "legislation", "treaty",
    "event",
]

_DATE_ISO_RE = re.compile(r'\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')
# news needs £€$ for currency reporting
_TOKEN_RE    = re.compile(r'[£€$]?[a-zA-ZÀ-ÿ0-9]+')


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

def _extract_subjects_news(text: str) -> list[str]:
    entities = extract_subjects_gliner(text, _GLINER_LABELS_NEWS, threshold=0.4)
    return [t for t, _ in entities]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_news_rag() -> None:
    """No-op: GLiNER is loaded by setup_entertainment_rag()."""


def rag_news(query: str, option_texts: list[str] | None = None) -> str:
    # --- date ---
    m = _DATE_ISO_RE.search(query)
    raw_date = m.group(0) if m else ""
    if raw_date:
        print(f"  [News] date: {raw_date!r}")
    else:
        print("  [News] No date found in question")

    # --- subjects ---
    subjects = [s for s in _extract_subjects_news(query) if not _DATE_ISO_RE.search(s)]
    if subjects:
        print(f"  [News] GLiNER entities: {subjects}")
    else:
        seen: set[str] = set()
        subjects = []
        for t in _TOKEN_RE.findall(query):
            if len(t) >= 2 and t[0].isupper() and t.lower() not in _STOP_WORDS_NEWS and t not in seen:
                subjects.append(t)
                seen.add(t)
        print(f"  [News] fallback subjects: {subjects[:6]}")

    main_term = subjects[0] if subjects else ""
    if not main_term:
        main_term = " ".join(
            t for t in _TOKEN_RE.findall(query.lower())
            if len(t) >= 5 and not t.isdigit() and t not in _STOP_WORDS_NEWS
        )[:60]

    entity_phrase   = " ".join(subjects[:6]) if subjects else main_term
    subject_tokens  = {s.lower() for s in subjects}
    q_content_words = [
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 5 and t not in _STOP_WORDS_NEWS and t not in subject_tokens
    ]

    # --- date operators ---
    date_ops = ""
    if raw_date:
        try:
            d          = datetime.date.fromisoformat(raw_date)
            day_before = (d - datetime.timedelta(days=1)).isoformat()
            day_after  = (d + datetime.timedelta(days=1)).isoformat()
            date_ops   = f"after:{day_before} before:{day_after}"
        except ValueError:
            pass

    # --- queries ---
    q1 = " ".join(filter(None, [entity_phrase, date_ops]))
    q2 = " ".join(filter(None, [entity_phrase, " ".join(q_content_words[:4]), date_ops]))

    queries = list(dict.fromkeys(q for q in [q1, q2] if q.strip()))
    print(f"  [News] queries: {queries}")

    # --- keyword sets for scoring ---
    q_keywords = {
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 4 and t not in _STOP_WORDS_NEWS
    }
    option_kw = {
        t for opt in (option_texts or [])
        for t in _TOKEN_RE.findall(opt.lower())
        if len(t) >= 2 and t not in _STOP_WORDS_NEWS
    }
    question_kw = q_keywords - option_kw

    def _search_and_fetch(q_text: str) -> list[tuple[str, str]]:
        results = []
        for url in ddg_search(q_text, max_results=_MAX_DDG_RESULTS, timeout=_TIMEOUT):
            if "wikipedia.org" in url:
                try:
                    article_date = datetime.date.fromisoformat(raw_date) if raw_date else None
                    days_old = (datetime.date.today() - article_date).days if article_date else 0
                except ValueError:
                    days_old = 0
                if days_old < 7:
                    print(f"  [News] skipping Wikipedia (recent): {url[:80]}")
                    continue
            elif any(domain in url for domain in SKIP_DOMAINS) or \
                 any(pat in url for pat in SKIP_URL_PATTERNS):
                print(f"  [News] skipping: {url[:80]}")
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as fetch_pool:
                fetch_fut = fetch_pool.submit(
                    fetch_article, url,
                    user_agent="NewsBot/1.0", timeout=_TIMEOUT,
                )
                try:
                    text = fetch_fut.result(timeout=10)
                except Exception:
                    print(f"  [News] fetch timeout: {url[:80]}")
                    text = ""
            if text and len(text) >= _MIN_ARTICLE_CHARS and any(kw in text.lower() for kw in q_keywords):
                results.append((text, url))
        return results

    candidates: list[tuple[float, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(queries))) as pool:
        futures = {pool.submit(_search_and_fetch, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            try:
                results = fut.result(timeout=8)
            except Exception:
                continue
            for text, url in results:
                raw_score = (
                    sum(2 for kw in option_kw   if kw in text.lower()) +
                    sum(1 for kw in question_kw if kw in text.lower())
                )
                score = raw_score / math.sqrt(max(len(text), 1)) * 100
                if raw_date and raw_date in text:
                    score += 10
                if raw_date and raw_date in url:
                    score += 5
                date_slash = raw_date.replace("-", "/") if raw_date else ""
                if date_slash and date_slash in url:
                    score += 5
                candidates.append((score, text, url))
                print(f"  [News] candidate (score={score:.1f}, {len(text)} chars): {url[:80]}")

    if not candidates:
        print("  [News] No article found — LLM fallback")
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    _, article_text, article_url = candidates[0]
    print(f"  [News] selected: {article_url[:80]}")

    date_display = raw_date if raw_date else "unknown"
    lines: list[str] = [
        f"ARTICLE (source: {article_url}, date: {date_display}):",
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
