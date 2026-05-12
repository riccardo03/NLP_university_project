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

# Single fused prompt: subject identification + query generation in one LLM call.
_SEARCH_STRATEGY_SYSTEM = (
    "You are a search strategist for an entertainment trivia bot. "
    "Given a quiz question, output TWO fields on a single line:\n"
    "  SUBJECT: <the primary named entity the question is about (1-4 words)>\n"
    "  QUERY: <a 3-6 word search query that retrieves the answer page>\n"
    "\n"
    "Rules for SUBJECT:\n"
    "- One specific person, film, song, album, TV show, award, character, or event.\n"
    "- If two entities are compared/related, write both: 'Entity A AND Entity B'.\n"
    "- If there is no named entity, write NONE.\n"
    "\n"
    "Rules for QUERY:\n"
    "- Target the topic page, NOT the answer itself.\n"
    "- Keep: proper nouns, titles, years, 1-2 context words (biography, filmography, "
    "discography, career, history, relationship).\n"
    "- Drop: question words (what, why, how, when, which), verbs, filler words.\n"
    "- 3 to 6 words maximum. No punctuation at end.\n"
    "\n"
    "Output format (EXACTLY — no other text):\n"
    "SUBJECT: <value> | QUERY: <value>\n"
    "\n"
    "Examples:\n"
    "Q: What was the primary reason James Cameron switched from physics to English? → "
    "SUBJECT: James Cameron | QUERY: James Cameron biography early career\n"
    "Q: Who directed The Godfather? → "
    "SUBJECT: The Godfather | QUERY: The Godfather 1972 film\n"
    "Q: In what year did Michael Jackson release Thriller? → "
    "SUBJECT: Michael Jackson Thriller | QUERY: Michael Jackson Thriller album release\n"
    "Q: Which actor played Tony Stark in the Marvel films? → "
    "SUBJECT: Tony Stark Marvel | QUERY: Tony Stark Iron Man Marvel cast\n"
    "Q: How does the blues form relate to the 12-bar structure? → "
    "SUBJECT: NONE | QUERY: 12-bar blues music theory\n"
    "Q: Which film won Best Picture at the 2020 Academy Awards? → "
    "SUBJECT: Academy Awards 2020 | QUERY: Academy Awards 2020 Best Picture winner\n"
)

_STOP_WORDS = {
    "what", "when", "which", "where", "does", "have", "this",
    "that", "from", "with", "about", "into", "their", "there",
    "been", "were", "would", "could", "should", "according"
}

_SUBJECT_TRIGGERS = re.compile(
    r'\b(film|movie|song|album|series|show|band|actor|actress|director|artist|character)\b',
    re.I
)

_PROPER_NOUN_RE = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)')
_POSSESSIVE_PROPER_RE = re.compile(r"\b([A-Z][a-z]{2,})'s\b")
_SINGLE_NAME_RE = re.compile(
    r'\b(?:between|behind|describes|about|by|for|of|with|from|in|on)\s+([A-Z][a-z]{2,})\b'
)

def _needs_subject_id(question: str) -> bool:
    if _SUBJECT_TRIGGERS.search(question):
        return True
    if _PROPER_NOUN_RE.findall(question):
        return True
    if _POSSESSIVE_PROPER_RE.findall(question):
        return True
    if _SINGLE_NAME_RE.findall(question):
        return True
    return False

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

def _parse_search_strategy(raw: str) -> tuple[str, str]:
    """
    Parse 'SUBJECT: <val> | QUERY: <val>' from LLM output.
    Returns (subject, query). Falls back gracefully on malformed output.
    """
    raw = raw.strip()
    subject = ""
    query = ""

    subject_m = re.search(r'SUBJECT\s*:\s*(.+?)(?:\s*\|\s*QUERY|\Z)', raw, re.I)
    query_m = re.search(r'QUERY\s*:\s*(.+)', raw, re.I)

    if subject_m:
        subject = subject_m.group(1).strip().strip('"').strip("'")
    if query_m:
        query = query_m.group(1).strip().strip('"').strip("'")

    return subject, query

def _get_search_decision(question: str, generate_answer_fn) -> str:
    """
    Single LLM call that returns both subject and distilled query.
    Returns the best ddg_query string to use.
    """
    try:
        raw = generate_answer_fn(
            _SEARCH_STRATEGY_SYSTEM,
            f"Q: {question}",
            max_new_tokens=30
        )
        subject, query = _parse_search_strategy(raw)

        if not subject or subject.upper() == "NONE":
            print(f"  [RAG-Entertainment] No subject. Query from LLM: {query!r}")
            return query if query and len(query) > 3 else question

        if " AND " in subject.upper():
            parts = re.split(r'\s+AND\s+', subject, flags=re.I)
            combined = " ".join(p.strip() for p in parts)
            print(f"  [RAG-Entertainment] Multi-entity subject: {parts}")
            # prefer LLM query; fall back to combined+question
            return query if query and len(query) > 3 else f"{combined} {question}"[:120]

        print(f"  [RAG-Entertainment] Subject: {subject!r}, Query: {query!r}")
        return query if query and len(query) > 3 else f"{subject} {question}"[:120]

    except Exception as e:
        print(f"  [RAG-Entertainment] Strategy LLM call failed: {e}")
        return question


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """
    Wikipedia + DuckDuckGo RAG for entertainment quiz questions.
    """
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Entertainment] Article-reference question — skipping search.")
        return ""

    # ------------------------------------------------------------------ #
    # Stage 1: single fused LLM call for subject + query                  #
    # ------------------------------------------------------------------ #
    if generate_answer_fn is not None and _needs_subject_id(query):
        ddg_query = _get_search_decision(query, generate_answer_fn)
    else:
        ddg_query = query
        print(f"  [RAG-Entertainment] No subject ID needed. Query: {ddg_query!r}")

    print(f"  [RAG-Entertainment] Final search query: {ddg_query!r}")

    # ------------------------------------------------------------------ #
    # Stage 2: Wikipedia first, DDG fallback                              #
    # ------------------------------------------------------------------ #
    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    print(f"  [RAG-Entertainment] Trying Wikipedia...")
    wiki_result = _wiki(ddg_query)
    if _wiki_is_useful(wiki_result) and _is_relevant(wiki_result, query):
        print(f"  [RAG-Entertainment] Wikipedia hit ({len(wiki_result)} chars), skipping DDG.")
        _add(wiki_result)
    else:
        if wiki_result and not _is_relevant(wiki_result, query):
            print(f"  [RAG-Entertainment] Wikipedia result not relevant, trying DDG too.")
            _add(wiki_result)
        else:
            print(f"  [RAG-Entertainment] Wikipedia miss, falling back to DDG.")

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(ddg_query, max_results=num_results, timeout=3):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("href", "")
                    if _is_quality_snippet(body, url):
                        _add(f"[{title}]{body}" if title else body)
        except Exception as exc:
            print(f"  [RAG-Entertainment] DDG failed: {exc}")

    # ------------------------------------------------------------------ #
    # Stage 3: relevance filter with fail-safe                            #
    # ------------------------------------------------------------------ #
    if snippets:
        relevant = [s for s in snippets if _is_relevant(s, query)]
        if relevant:
            snippets = relevant
        else:
            # fail-safe: return best-effort top-2 rather than nothing
            print(f"  [RAG-Entertainment] No relevant snippets — using best-effort top-2.")
            snippets = snippets[:2]

    return "\n\n".join(snippets)[:3000] if snippets else ""
