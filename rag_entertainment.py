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

# Unicode letter range covers accented names like Beyoncé, Björk
_UL = r'A-ZÀ-ÖØ-Ý'   # uppercase Unicode letters
_AL = r'a-zA-ZÀ-ÖØ-öø-ÿ'  # all-case Unicode letters

# Multi-word proper noun: two or more Title-Case/mixed-case words
_PROPER_NOUN_RE = re.compile(
    rf'([{_UL}][{_AL}]+(?:\s+[{_UL}][{_AL}]+)+)'
)

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


import urllib.parse
import requests

_WIKI_UA = "PoliMillionaireBot/1.0 (university research project)"


def _wiki(query: str) -> str:
    """Wikipedia REST summary API — fast, no package dependency."""
    title = urllib.parse.quote(query, safe="")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=4)
        if r.status_code == 200:
            return r.json().get("extract", "")[:1200]
    except Exception:
        pass
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


_QUOTED_TITLE_RE = re.compile(r"""(?<!\w)['"]([\w][\w\s,\.\-]{1,58}?)['"]""")
_CAMEL_CASE_RE = re.compile(rf'\b([{_UL}][{_AL.replace("A-Z", "")}]+[{_UL}][{_AL}]+)\b')
_QUESTION_WORDS = re.compile(r'^(Which|What|How|Who|When|Where|Why)$')
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|2[0-9]{3})\b')

# Words that look title-case but are not entity names
TITLE_STOP = {
    "Which", "What", "How", "Who", "When", "Where", "Why",
    "The", "This", "That", "These", "Those", "A", "An",
    "Is", "Are", "Was", "Were", "Has", "Have", "Had",
    "Does", "Do", "Did", "Will", "Would", "Could", "Should",
    "According", "Following", "Best", "Most", "First", "Last",
    "Film", "Song", "Show", "Role", "Style", "Music", "Band",
    "Album", "Movie", "Book", "Character", "Actor", "Director",
    "Describes", "Describe", "Known", "Used", "Made", "Played",
    "Between", "During", "After", "Before", "About", "Into",
}
_TITLE_STOP_LOWER = {w.lower() for w in TITLE_STOP}

# Single title-case word (Unicode-aware), min 3 chars
_SINGLE_PROPER_RE = re.compile(rf'\b([{_UL}][{_AL}]{{2,}})\b')


def _build_query(question: str) -> tuple[str, int | str]:
    """
    Pure-regex query builder — no LLM call.
    Returns (query, priority) where priority is 1, 2, '2b', or 3.

    Priority 1:  quoted titles  e.g. 'Thriller', "E.T."
    Priority 2:  multi-word proper nouns  e.g. "James Cameron", "The Godfather"
    Priority 2c: single title-case proper noun  e.g. Chaplin, Beyoncé (NEW)
    Priority 2b: CamelCase single token  e.g. LazyTown, YouTube
    Priority 3:  significant keyword fallback (improved filtering)

    Any 4-digit year found in the question is appended to the result.
    """
    year_match = _YEAR_RE.search(question)
    year_suffix = f" {year_match.group(1)}" if year_match else ""

    # Priority 1: quoted titles
    quoted = _QUOTED_TITLE_RE.findall(question)
    if quoted:
        title = quoted[0].strip()
        print(f"  [RAG-Entertainment] Quoted title: {title!r}")
        return f"{title}{year_suffix}", 1

    # Priority 2: multi-word proper nouns
    proper = _PROPER_NOUN_RE.findall(question)
    if proper:
        entity = proper[0].strip()
        print(f"  [RAG-Entertainment] Proper noun: {entity!r}")
        return f"{entity}{year_suffix}", 2

    # Priority 2c: single title-case word not in stop set; skip sentence-initial word
    # Prefer words that appear with a possessive 's (strong name signal)
    words = question.split()
    possessive_names = re.findall(
        rf"\b([{_UL}][{_AL}]{{2,}})'s\b", question
    )
    single_candidates = [
        m for m in _SINGLE_PROPER_RE.findall(question)
        if m not in TITLE_STOP and m != words[0]
    ]
    # Possessive names first, then remaining candidates
    ordered = list(dict.fromkeys(possessive_names + single_candidates))
    if ordered:
        entity = ordered[0]
        print(f"  [RAG-Entertainment] Single proper noun: {entity!r}")
        return f"{entity}{year_suffix}", "2c"

    # Priority 2b: CamelCase single token (e.g. LazyTown, YouTube, TikTok)
    camel = [m for m in _CAMEL_CASE_RE.findall(question) if not _QUESTION_WORDS.match(m)]
    if camel:
        entity = camel[0]
        print(f"  [RAG-Entertainment] CamelCase token: {entity!r}")
        return f"{entity}{year_suffix}", "2b"

    # Priority 3: keyword fallback — filter stop words and short tokens aggressively
    keywords = [
        kw for kw in _extract_keywords(question)
        if len(kw) >= 5 and kw not in _TITLE_STOP_LOWER
    ]
    if len(keywords) >= 2:
        q = " ".join(keywords[:5])
        print(f"  [RAG-Entertainment] Keyword fallback: {q!r}")
        return f"{q}{year_suffix}", 3

    # Last resort: raw question capped at 80 chars
    print(f"  [RAG-Entertainment] Raw question fallback")
    return question[:80], 3


def _build_ddg_query(entity: str, question: str) -> str:
    """
    Enrich an entity name with context keywords from the question for DDG.
    Removes words already present in the entity to avoid redundancy.
    """
    entity_words = set(entity.lower().split())
    extra = [kw for kw in _extract_keywords(question) if kw not in entity_words]
    combined = f"{entity} {' '.join(extra[:3])}"
    return combined.strip()[:80]


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
    base_query, priority = _build_query(query)
    wiki_query = _build_wiki_query(base_query)  # entity-only (string)
    if priority in (2, "2c", "2b"):
        ddg_query = _build_ddg_query(wiki_query, query)
    else:
        ddg_query = base_query
    print(f"  [RAG-Entertainment] P{priority} wiki={wiki_query!r}  ddg={ddg_query!r}")

    # ------------------------------------------------------------------ #
    # Stage 2: Wikipedia + DDG in parallel, 4 s timeout each             #
    # ------------------------------------------------------------------ #
    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    wiki_fut = pool.submit(_wiki, wiki_query)
    ddg_fut  = pool.submit(_fetch_ddg, ddg_query, num_results)

    try:
        wiki_result = wiki_fut.result(timeout=4)
    except Exception:
        wiki_result = ""
        print("  [RAG-Entertainment] Wikipedia timed out.")

    try:
        ddg_results = ddg_fut.result(timeout=4)
    except Exception:
        ddg_results = []
        print("  [RAG-Entertainment] DDG timed out.")

    pool.shutdown(wait=False)

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
