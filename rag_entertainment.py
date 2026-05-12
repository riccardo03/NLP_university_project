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

# [A-Z][a-zA-Z]+ handles mixed-case surnames like McCartney, McDonald
_PROPER_NOUN_RE = re.compile(r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+)')

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


def _build_wiki_query(query: str) -> str:
    """
    Strip the search query down to its primary entity for Wikipedia lookup.
    Wikipedia needs a page title, not a full search phrase.
    Falls back to the full query if no proper noun is found.
    """
    proper = _PROPER_NOUN_RE.search(query)
    if proper:
        return proper.group(1).strip()
    return query


_WIKI_UA = "PoliMillionaireBot/1.0 (university project; python-wikipedia)"


def _wiki_rest_fallback(title: str) -> str:
    """Direct Wikipedia REST API — more permissive than the action API."""
    try:
        import urllib.parse
        import requests
        encoded = urllib.parse.quote(title, safe="")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        resp = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=4)
        if resp.status_code == 200:
            return resp.json().get("extract", "")[:1200]
    except Exception:
        pass
    return ""


def _wiki(query: str) -> str:
    """
    Two-step Wikipedia fetch: search() for a title, then page() for content.
    Falls back to the REST summary API if the python package fails.
    Returns '' only if both paths fail.
    """
    title = query  # used by the REST fallback even if the package path fails early
    try:
        import wikipedia
        wikipedia.set_lang("en")
        wikipedia.set_user_agent(_WIKI_UA)

        titles = wikipedia.search(query, results=3)
        if not titles:
            print(f"  [RAG-Entertainment] Wikipedia search empty for {query!r}, trying REST.")
            return _wiki_rest_fallback(query)
        title = titles[0]

        try:
            page = wikipedia.page(title, auto_suggest=False)
        except wikipedia.exceptions.DisambiguationError as e:
            if not e.options:
                return _wiki_rest_fallback(title)
            page = wikipedia.page(e.options[0], auto_suggest=False)
            title = e.options[0]
        except Exception:
            print(f"  [RAG-Entertainment] wikipedia.page() failed for {title!r}, trying REST.")
            return _wiki_rest_fallback(title)

        summary = wikipedia.summary(title, sentences=4, auto_suggest=False)
        paragraphs = [
            p.strip() for p in page.content.split("\n")
            if len(p.strip()) > 120
        ]
        extra = "\n\n".join(paragraphs[:2])
        combined = f"{summary}\n\n{extra}" if extra else summary
        return combined[:1200]

    except Exception:
        print(f"  [RAG-Entertainment] Wikipedia package failed for {query!r}, trying REST.")
        return _wiki_rest_fallback(title)

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
    ddg_query  = _build_query(query)
    wiki_query = _build_wiki_query(ddg_query)
    print(f"  [RAG-Entertainment] DDG query: {ddg_query!r}  Wiki query: {wiki_query!r}")

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
        wiki_fut = pool.submit(_wiki, wiki_query)
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
