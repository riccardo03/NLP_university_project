"""
RAG pipeline for the Entertainment competition.
Wikipedia first, DuckDuckGo to supplement.
"""

import re
import concurrent.futures

_ARTICLE_REF_RE = re.compile(
    r'\b(according to|as described in|as stated in|in his own words|based on|per) '
    r'(the article|the text|the passage|the excerpt)\b', re.I
)
_LOW_QUALITY_SIGNALS = [
    r'\d{4}\s*[·•]\s',
    r'click here',
    r'subscribe',
    r'read more',
    r'sign up',
    r'\.\.\.read',
    r'youtube\.com',
    r'goo\.gl',
    r'fandom\.com',
    r'wikia\.com',
]

 

_TRUSTED_DOMAINS = {
    "wikipedia.org",
    "britannica.com",
    "imdb.com",
    "allmusic.com",
    "rottentomatoes.com",
    "biography.com",
    "rollingstone.com",
    "billboard.com",
}

_STOP_WORDS = {
    "what", "when", "which", "where", "does", "have", "this",
    "that", "from", "with", "about", "into", "their", "there",
    "been", "were", "would", "could", "should", "according"
}

_PROPER_NOUN_RE = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')

def _is_quality_snippet(text: str, url: str = "") -> bool:
    if not text or len(text.strip()) < 80:
        return False
    text_lower = text.lower()
    for pattern in _LOW_QUALITY_SIGNALS:
        if re.search(pattern, text_lower):
            return False
    if url:
        domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain and any(t in domain.group(1) for t in _TRUSTED_DOMAINS):
            return True
    return True


def _wiki(query: str) -> str:
    """Fetch a Wikipedia summary; returns '' on any failure."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=False)
            summary = wikipedia.summary(query, sentences=3, auto_suggest=False)
            paragraphs = [
                p.strip() for p in page.content.split("\n")
                if len(p.strip()) > 100
            ]
            # Take only the first 2 substantial paragraphs beyond the summary
            extra = "\n\n".join(paragraphs[:2])
            combined = f"{summary}\n\n{extra}"
            return combined[:2000]
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=3, auto_suggest=False)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""

def _wiki_is_useful(text: str) -> bool:
    if not text or len(text.strip()) < 200:
        return False
    if "may refer to:" in text.lower():
        return False
    return True

def _extract_keywords(text: str) -> list[str]:
    keywords = []
    # 4+ letter lowercase words
    for w in re.findall(r'\b[a-z]{4,}\b', text.lower()):
        if w not in _STOP_WORDS:
            keywords.append(w)
    # 2+ letter ALL-CAPS tokens (e.g. U2, AI, ET, HBO)
    for w in re.findall(r'\b[A-Z]{2,}\b', text):
        keywords.append(w.lower())
    # 4-digit years (e.g. 1994, 2023)
    for w in re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text):
        keywords.append(w)
    return list(dict.fromkeys(keywords))  # deduplicate preserving order

def _is_relevant(snippet: str, question: str) -> bool:
    keywords = _extract_keywords(question)
    if not keywords:
        return True
    snippet_lower = snippet.lower()

    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
    for match in quoted:
        title = (match[0] or match[1]).lower()
        if len(title) > 3 and title not in snippet_lower:
            return False

    matches = sum(1 for kw in keywords if kw in snippet_lower)
    return matches >= 2


_QUOTED_TITLE_RE = re.compile(r"""['"]([^'"]{2,60}?)['"]""")
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|2[0-9]{3})\b')


def _build_query(question: str) -> str:
    """
    Pure-regex query builder — no LLM call.

    Priority 1: quoted titles  e.g. 'Thriller', "E.T."
    Priority 2: multi-word proper nouns  e.g. "James Cameron", "The Godfather"
    Priority 3: significant keyword fallback (len > 4, not stop words)

    Any 4-digit year found in the question is appended to the result.
    """
    year_match = _YEAR_RE.search(question)
    year_suffix = f" {year_match.group(1)}" if year_match else ""

    # Priority 1: quoted titles
    quoted = _QUOTED_TITLE_RE.findall(question)
    if quoted:
        title = quoted[0].strip()
        print(f"  [RAG-Entertainment] Quoted title: {title!r}")
        return f"{title}{year_suffix}"

    # Priority 2: multi-word proper nouns
    proper = _PROPER_NOUN_RE.findall(question)
    if proper:
        entity = proper[0].strip()
        print(f"  [RAG-Entertainment] Proper noun: {entity!r}")
        return f"{entity}{year_suffix}"

    # Priority 3: keyword fallback
    keywords = _extract_keywords(question)
    if keywords:
        q = " ".join(keywords[:5])
        print(f"  [RAG-Entertainment] Keyword fallback: {q!r}")
        return f"{q}{year_suffix}"

    return question


def _fetch_ddg(ddg_query: str, num_results: int) -> list[str]:
    """Fetch DuckDuckGo results; returns list of snippet strings."""
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(ddg_query, max_results=num_results, timeout=3):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")
                if _is_quality_snippet(body, url):
                    results.append(f"[{title}]{body}" if title else body)
    except Exception as exc:
        print(f"  [RAG-Entertainment] DDG failed: {exc}")
    return results


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """
    Wikipedia + DuckDuckGo RAG for entertainment quiz questions.
    `generate_answer_fn` is accepted for API compatibility but not used.
    """
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Entertainment] Article-reference question — skipping search.")
        return ""

    # ------------------------------------------------------------------ #
    # Stage 1: regex-based query (no LLM call)                           #
    # ------------------------------------------------------------------ #
    ddg_query = _build_query(query)
    print(f"  [RAG-Entertainment] Final search query: {ddg_query!r}")

    # ------------------------------------------------------------------ #
    # Stage 2: Wikipedia + DDG in parallel, 4 s timeout each             #
    # ------------------------------------------------------------------ #
    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        wiki_fut = pool.submit(_wiki, ddg_query)
        ddg_fut  = pool.submit(_fetch_ddg, ddg_query, num_results)

        try:
            wiki_result = wiki_fut.result(timeout=4)
        except concurrent.futures.TimeoutError:
            print("  [RAG-Entertainment] Wikipedia timed out.")
            wiki_result = ""
        except Exception as exc:
            print(f"  [RAG-Entertainment] Wikipedia error: {exc}")
            wiki_result = ""

        try:
            ddg_results = ddg_fut.result(timeout=4)
        except concurrent.futures.TimeoutError:
            print("  [RAG-Entertainment] DDG timed out.")
            ddg_results = []
        except Exception as exc:
            print(f"  [RAG-Entertainment] DDG error: {exc}")
            ddg_results = []

    if _wiki_is_useful(wiki_result):
        print(f"  [RAG-Entertainment] Wikipedia hit ({len(wiki_result)} chars).")
        _add(wiki_result)
    else:
        print("  [RAG-Entertainment] Wikipedia miss.")

    for snippet in ddg_results:
        _add(snippet)

    # ------------------------------------------------------------------ #
    # Stage 3: relevance filter with fail-safe                            #
    # ------------------------------------------------------------------ #
    if snippets:
        relevant = [s for s in snippets if _is_relevant(s, query)]
        if relevant:
            snippets = relevant
        else:
            print("  [RAG-Entertainment] No relevant snippets — using best-effort top-2.")
            snippets = snippets[:2]

    return "\n\n".join(snippets)[:1500] if snippets else ""
