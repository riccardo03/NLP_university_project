"""
News RAG: date-anchored DDG search → article fetch → LLM prompt.

No Wikipedia. No BM25 voting. No caching (news changes daily).
Subject extraction reuses the GLiNER model loaded by setup_entertainment_rag().
"""

import concurrent.futures
import html as _html_module
import re

import requests

_TIMEOUT           = 5
_ARTICLE_MAX_CHARS = 4000
_MIN_ARTICLE_CHARS = 200
_MAX_DDG_RESULTS   = 5

_SKIP_DOMAINS: frozenset[str] = frozenset({
    "wikipedia.org",
    "facebook.com",
    "youtube.com",
    "reddit.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "tiktok.com",
    "threads.com",
    "rutube.ru",
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

_MONTH_FULL: list[str] = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_TO_FULL: dict[str, str] = {
    **{m.lower(): m for m in _MONTH_FULL},
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "jun": "June",    "jul": "July",     "aug": "August",
    "sep": "September", "oct": "October", "nov": "November", "dec": "December",
}
# Longest keys first so alternation in regex never short-circuits on abbreviations
_MONTH_PAT: str = "|".join(
    sorted(_MONTH_TO_FULL.keys(), key=len, reverse=True)
)

_DATE_ISO_RE = re.compile(r'\b((?:19|20)\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b')
_DATE_MDY_RE = re.compile(
    rf'\b({_MONTH_PAT})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{{2}})\b',
    re.IGNORECASE,
)
_DATE_DMY_RE = re.compile(
    rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PAT})\s+((?:19|20)\d{{2}})\b',
    re.IGNORECASE,
)
_DATE_MY_RE = re.compile(
    rf'\b({_MONTH_PAT})\s+((?:19|20)\d{{2}})\b',
    re.IGNORECASE,
)

_TOKEN_RE     = re.compile(r'[a-zA-ZÀ-ÿ0-9]+')
# Remove script/style/nav/header/footer blocks before tag stripping
_BLOCK_TAG_RE = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE    = re.compile(r'<[^>]+>')
_WS_RE     = re.compile(r'\s+')
_P_TAG_RE  = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

def _extract_date(text: str) -> tuple[str, str]:
    """
    Returns (raw_date, alt_date).
    raw_date: exactly as it appears in the text.
    alt_date: human-readable reformatted version (e.g. "May 18 2026").
    Both empty strings when no date is found.
    """
    m = _DATE_ISO_RE.search(text)
    if m:
        year, mon = m.group(1), int(m.group(2))
        day  = int(m.group(3))
        return m.group(0), f"{_MONTH_FULL[mon - 1]} {day} {year}"

    m = _DATE_MDY_RE.search(text)
    if m:
        month_full = _MONTH_TO_FULL.get(m.group(1).lower(), m.group(1).capitalize())
        return m.group(0), f"{month_full} {m.group(2)} {m.group(3)}"

    m = _DATE_DMY_RE.search(text)
    if m:
        month_full = _MONTH_TO_FULL.get(m.group(2).lower(), m.group(2).capitalize())
        return m.group(0), f"{month_full} {m.group(1)} {m.group(3)}"

    m = _DATE_MY_RE.search(text)
    if m:
        month_full = _MONTH_TO_FULL.get(m.group(1).lower(), m.group(1).capitalize())
        return m.group(0), f"{month_full} {m.group(2)}"

    return "", ""


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
    """
    Extracts the real article body, skipping nav/header/footer noise.
    Strategy:
      1. Collect <p> tags with at least 80 chars (substantial paragraphs)
      2. Concatenate them up to max_chars
      3. Fallback to _strip_html[:max_chars] if no paragraphs found
    """
    paragraphs = _P_TAG_RE.findall(raw_html)
    body_parts: list[str] = []
    total = 0

    for p in paragraphs:
        clean = _WS_RE.sub(
            " ",
            _TAG_RE.sub("", _html_module.unescape(p))
        ).strip()
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

def _ddg_search(query: str, max_results: int = _MAX_DDG_RESULTS) -> list[dict]:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                url = r.get("href") or r.get("url", "")
                if url:
                    results.append({
                        "url":   url,
                        "title": r.get("title", ""),
                        "body":  r.get("body", ""),
                    })
        return results
    except Exception as e:
        print(f"  [RAG-News] DDG error: {e}")
        return []


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def _fetch_article(url: str) -> str:
    """Fetch URL, strip HTML, return first _ARTICLE_MAX_CHARS chars of plain text."""
    try:
        r = requests.get(
            url,
            timeout=_TIMEOUT,
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
    raw_date, alt_date = _extract_date(query)
    if raw_date:
        print(f"  [News] date: {raw_date!r}  alt: {alt_date!r}")
    else:
        print("  [News] No date found in question")

    from rag_entertainment import _extract_subjects_gliner, _extract_subjects_regex

    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = subjects[0]
        print(f"  [News] GLiNER entities: {labeled}")
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""
        print(f"  [News] regex subjects: {subjects}")

    if not main_term:
        tokens    = _TOKEN_RE.findall(query.lower())
        main_term = " ".join(
            t for t in tokens if len(t) >= 4 and t not in _STOP_WORDS_NEWS
        )[:60]

    date_anchor = raw_date or ""

    entity_phrase   = " ".join(subjects[:3]) if subjects else main_term
    subject_tokens  = {s.lower() for s in subjects}
    q_content_words = [
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 5
        and t not in _STOP_WORDS_NEWS
        and t not in subject_tokens
    ]

    option_entities: list[str] = []
    if option_texts:
        for opt in option_texts[:4]:
            for token in _TOKEN_RE.findall(opt):
                if (
                    len(token) >= 3
                    and token[0].isupper()
                    and token.lower() not in _STOP_WORDS_NEWS
                    and token not in option_entities
                ):
                    option_entities.append(token)

    # Q1 — broad: all entities + date
    q1 = " ".join(filter(None, ["news", entity_phrase, date_anchor]))
    # Q2 — specific: all entities + first 4 content words + date
    q2 = " ".join(filter(None, ["news", entity_phrase, " ".join(q_content_words[:4]), date_anchor]))
    # Q3 — alternative: option named entities + date
    q3 = " ".join(filter(None, ["news", " ".join(option_entities[:4]) if option_entities else entity_phrase, date_anchor]))
    queries = list(dict.fromkeys(q for q in [q1, q2, q3] if q.strip()))
    print(f"  [News] queries: {queries}")

    q_keywords = {
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 4 and t not in _STOP_WORDS_NEWS
    }

    def _search_and_fetch(q_text: str) -> tuple[str, str]:
        for r in _ddg_search(q_text):
            url = r["url"]
            if any(domain in url for domain in _SKIP_DOMAINS) or \
               any(pat in url for pat in _SKIP_URL_PATTERNS):
                print(f"  [News] skipping: {url[:80]}")
                continue
            text = _fetch_article(url)
            if text and len(text) >= _MIN_ARTICLE_CHARS and any(kw in text.lower() for kw in q_keywords):
                return text, url
        return "", ""

    candidates: list[tuple[int, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_search_and_fetch, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            text, url = fut.result()
            if text:
                score = sum(1 for kw in q_keywords if kw in text.lower())
                candidates.append((score, text, url))
                print(f"  [News] candidate (score={score}, {len(text)} chars): {url[:80]}")

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
