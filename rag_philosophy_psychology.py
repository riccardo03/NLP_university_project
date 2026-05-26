"""
Philosophy & Psychology RAG: DDG search → article fetch → LLM prompt.

No Wikipedia. No BM25 voting.
Subject extraction reuses the GLiNER model loaded by setup_entertainment_rag().
"""

import concurrent.futures
import html as _html_module
import re

import requests

_TIMEOUT           = 5
_ARTICLE_MAX_CHARS = 4000
_MAX_DDG_RESULTS   = 3

_STOP_WORDS_PHILOSOPHY_PSYCHOLOGY: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "article",
    "published", "reported", "stated", "said",
})


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
# DDG search — no lru_cache (philosophy_psychology changes daily)
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
        print(f"  [RAG-Philosophy_Psychology] DDG error: {e}")
        return []


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

def _fetch_article(url: str) -> str:
    try:
        r = requests.get(
            url,
            timeout=_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PhilosophyBot/1.0)"},
        )
        if r.status_code != 200:
            return ""
        return _extract_body(r.text)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_philosophy_psychology_rag() -> None:
    """No-op: GLiNER is loaded by setup_entertainment_rag()."""
    pass


def rag_philosophy_psychology(query: str, option_texts: list[str] | None = None) -> str:
    from rag_entertainment import _extract_subjects_gliner, _extract_subjects_regex

    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = subjects[0]
        print(f"  [Philosophy_Psychology] GLiNER entities: {labeled}")
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""
        print(f"  [Philosophy_Psychology] regex subjects: {subjects}")

    if not main_term:
        tokens    = _TOKEN_RE.findall(query.lower())
        main_term = " ".join(
            t for t in tokens if len(t) >= 4 and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
        )[:60]

    n_opts      = min(len(option_texts), 4) if option_texts else 0

    def _clean_option(opt: str) -> str:
        words = [
            t for t in _TOKEN_RE.findall(opt.lower())
            if len(t) >= 3 and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
        ]
        return " ".join(words[:6])

    subject_tokens = {s.lower() for s in subjects}
    q_content_words = [
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 5
        and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
        and t not in subject_tokens
    ]
    synth_query = " ".join(filter(None, [
        main_term,
        " ".join(q_content_words[:3]),
    ]))

    if n_opts:
        option_queries = [
            " ".join(filter(None, [main_term, _clean_option(option_texts[i])]))
            for i in range(n_opts)
        ]
    else:
        option_queries = [main_term]

    queries = [synth_query] + option_queries if synth_query.strip() else option_queries
    print(f"  [Philosophy_Psychology] queries: {queries}")

    q_keywords = {
        t for t in _TOKEN_RE.findall(query.lower())
        if len(t) >= 4 and t not in _STOP_WORDS_PHILOSOPHY_PSYCHOLOGY
    }

    def _search_and_fetch(q_text: str) -> tuple[str, str]:
        for r in _ddg_search(q_text):
            text = _fetch_article(r["url"])
            if text and any(kw in text.lower() for kw in q_keywords):
                return text, r["url"]
        return "", ""

    candidates: list[tuple[int, str, str]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_search_and_fetch, q): q for q in queries}
        for fut in concurrent.futures.as_completed(futures):
            text, url = fut.result()
            if text:
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
