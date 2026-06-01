"""
News RAG: date-anchored DDG search → article fetch → LLM prompt.
No caching (news changes daily).
Subject extraction uses a news-specific GLiNER call.
"""

import concurrent.futures
import datetime
import html as _html_module
import math
import re

import requests

_TIMEOUT                  = 4
_ARTICLE_MAX_CHARS        = 10000
_MIN_ARTICLE_CHARS        = 500
_MAX_DDG_RESULTS          = 3

_SKIP_DOMAINS: frozenset[str] = frozenset({
    "youtube.com",
    "instagram.com",
    "tiktok.com",
    "rutube.ru",
    "linkedin.com",
    "vk.com",
    "t.me",
})

_SKIP_URL_PATTERNS: frozenset[str] = frozenset({
    "/tag/", "/tags/", "/category/", "/categories/",
    "/topic/", "/topics/", "/search/", "/archive/",
})

_STOP_WORDS_NEWS: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "article",
    "published", "reported", "stated", "said",
})

_GLINER_LABELS_NEWS: list[str] = [
    "person", "politician", "president", "minister", "official",
    "organization", "company", "institution", "government agency",
    "country", "city", "location", "region",
    "law", "legislation", "treaty",
    "event",
]

_DATE_ISO_RE = re.compile(r'\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')

_TOKEN_RE     = re.compile(r'[£€$]?[a-zA-ZÀ-ÿ0-9]+')
_BLOCK_TAG_RE = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE   = re.compile(r'<[^>]+>')
_WS_RE    = re.compile(r'\s+')
_P_TAG_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

def _extract_subjects_news(text: str) -> list[str]:
    try:
        from rag_entertainment import _gliner_model
        if _gliner_model is None:
            return []
        entities = _gliner_model.predict_entities(text, _GLINER_LABELS_NEWS, threshold=0.4)
        seen: set[str] = set()
        result: list[str] = []
        for ent in entities:
            txt = ent["text"].strip()
            if txt and txt.lower() not in seen:
                seen.add(txt.lower())
                result.append(txt)
        return result
    except Exception:
        return []


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    text = _BLOCK_TAG_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = _html_module.unescape(text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _extract_body(raw_html: str, max_chars: int = _ARTICLE_MAX_CHARS) -> str:
    paragraphs = _P_TAG_RE.findall(raw_html)
    body_parts: list[str] = []
    total = 0

    for p in paragraphs:
        clean = _WS_RE.sub(" ", _TAG_RE.sub("", _html_module.unescape(p))).strip()
        if len(clean) >= 80:
            body_parts.append(clean)
            total += len(clean) + 1
            if total >= max_chars:
                break

    if body_parts:
        return " ".join(body_parts)[:max_chars]

    return _strip_html(raw_html)[:max_chars]


# ---------------------------------------------------------------------------
# DDG search — no lru_cache (news changes daily)
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = _MAX_DDG_RESULTS) -> list[str]:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url", "")
                if url:
                    results.append(url)
        return results
    except Exception as e:
        print(f"  [RAG-News] DDG error: {e}")
        return []


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def _fetch_article(url: str) -> str:
    try:
        r = requests.get(
            url,
            timeout=(_TIMEOUT, _TIMEOUT),
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        )
        if r.status_code != 200:
            return ""
        return _extract_body(r.text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_news_rag() -> None:
    """No-op: GLiNER is loaded by setup_entertainment_rag()."""
    pass


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

    # --- keyword sets for scoring (computed once, outside the candidate loop) ---
    q_keywords = {
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 4 and t not in _STOP_WORDS_NEWS
    }
    option_kw  = {
        t for opt in (option_texts or [])
        for t in _TOKEN_RE.findall(opt.lower())
        if len(t) >= 2 and t not in _STOP_WORDS_NEWS
    }
    question_kw = q_keywords - option_kw

    def _search_and_fetch(q_text: str) -> list[tuple[str, str]]:
        results = []
        for url in _ddg_search(q_text):
            if "wikipedia.org" in url:
                try:
                    article_date = datetime.date.fromisoformat(raw_date) if raw_date else None
                    days_old = (datetime.date.today() - article_date).days if article_date else 0
                except ValueError:
                    days_old = 0
                if days_old < 7:
                    print(f"  [News] skipping Wikipedia (recent): {url[:80]}")
                    continue
            elif any(domain in url for domain in _SKIP_DOMAINS) or \
                 any(pat in url for pat in _SKIP_URL_PATTERNS):
                print(f"  [News] skipping: {url[:80]}")
                continue
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as fetch_pool:
                fetch_fut = fetch_pool.submit(_fetch_article, url)
                try:
                    text = fetch_fut.result(timeout=10)
                except Exception:
                    print(f"  [News] fetch timeout: {url[:80]}")
                    text = ""
            min_chars = _MIN_ARTICLE_CHARS
            if text and len(text) >= min_chars and any(kw in text.lower() for kw in q_keywords):
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
