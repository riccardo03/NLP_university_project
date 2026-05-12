
import re

_LOW_QUALITY_SIGNALS = [
    r'\d{4}\s*[·•]\s',   # "Jun 12, 2018 ·" — dated blog format
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
    "been", "were", "would", "could", "should", "according",
    "following", "best", "describes", "describe", "primary",
    "fundamental", "principle", "term", "following",
}

_QUERY_DECISION_SYSTEM = (
    "You are a search assistant for an entertainment trivia bot. "
    "Given a multiple-choice quiz question, decide whether web search is needed "
    "and if so, what to search for. "
    "\n\n"
    "Output EXACTLY one of these three formats and nothing else:\n"
    "\n"
    "SKIP\n"
    "  — Use when the question references source material that cannot be found "
    "online: 'according to the article', 'as described in the text', "
    "'as mentioned in the article', 'as stated', 'based on the passage', "
    "'in his/her own words', OR when the question says 'the film/show/book' "
    "without naming it (e.g. 'the film portrays...' with no film title given).\n"
    "\n"
    "PARAMETRIC\n"
    "  — Use when the question is about a general concept or definition with "
    "no specific named entity (e.g. 'what is sonata form', 'define jazz fusion', "
    "'which term describes a melody'). The LLM can answer from its own knowledge.\n"
    "\n"
    "SEARCH: <query>\n"
    "  — Use for all other questions. <query> must be 3-6 words, targeting the "
    "topic not the answer.\n"
    "\n"
    "Rules for SEARCH queries:\n"
    "- Use proper names and 1-2 context words (biography, career, filmography, "
    "history, relationship, album, film).\n"
    "- For relationship questions between TWO named entities, include both names.\n"
    "- Include specific years or titles when they narrow which page is needed.\n"
    "- Drop: question words (what, why, how, when, which), verbs, filler words.\n"
    "- Drop: consequence words (impact, legacy, mental health, effect, result, "
    "controversy, reason, primary).\n"
    "- Always put a space between every word. Never merge words.\n"
    "- Never include non-English characters.\n"
    "\n"
    "Examples:\n"
    "Q: What was the primary reason James Cameron switched from physics to English? "
    "→ SEARCH: James Cameron biography early career\n"
    "Q: According to the article, what does the author argue? → SKIP\n"
    "Q: Which term describes the way the film portrays antebellum life? → SKIP\n"
    "Q: Which of the following best describes the fundamental principle of sonata "
    "form? → PARAMETRIC\n"
    "Q: How does 'Hey Joe' relate to 'Purple Haze'? "
    "→ SEARCH: Hey Joe Purple Haze Jimi Hendrix\n"
    "Q: What was Kanye West's 2002 car accident? "
    "→ SEARCH: Kanye West 2002 car accident\n"
    "Q: Which term best describes Jack Nicholson's relationship with his mother "
    "as mentioned in the article? → SKIP\n"
    "Q: What is jazz fusion? → PARAMETRIC\n"
    "Q: Which of the following best describes the significance of the Tramp "
    "character in Chaplin's films? "
    "→ SEARCH: Charlie Chaplin Tramp character significance\n"
    "Q: What was the primary reason Spielberg decided to direct Schindler's List? "
    "→ SEARCH: Spielberg Schindler List motivation Jewish heritage\n"
    "Q: How did Taylor's relationship with Richard Burton impact her public image? "
    "→ SEARCH: Elizabeth Taylor Richard Burton relationship\n"
    "Q: What is Mr. Bean's profession in the first film adaptation? "
    "→ SEARCH: Bean 1997 film Mr Bean\n"
    "Q: Which of Hitchcock's films is known for the dolly zoom effect? "
    "→ SEARCH: Hitchcock dolly zoom Vertigo\n"
    "Q: What is the primary instrument played by The Edge in U2? "
    "→ SEARCH: The Edge U2 guitarist\n"
    "Q: Which of the following best describes the fundamental principle of "
    "classical Hollywood continuity editing? → PARAMETRIC\n"
    "Q: What was the primary reason for the Beatles' decision to retire from "
    "live performances in 1966? "
    "→ SEARCH: Beatles 1966 retire live performances\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize_query(query: str) -> str:
    """Fix common LLM query generation artifacts."""
    # fix missing spaces before capital letters: "earlyCareer" → "early Career"
    query = re.sub(r'([a-z])([A-Z])', r'\1 \2', query)
    # remove non-ASCII characters (Chinese, Arabic, etc.)
    query = re.sub(r'[^\x00-\x7F]+', ' ', query)
    # normalize multiple spaces
    query = re.sub(r'\s+', ' ', query).strip()
    return query


def _get_search_decision(query: str, generate_answer_fn) -> tuple[str, str]:
    """
    Single LLM call that decides: SKIP / PARAMETRIC / SEARCH: <query>
    Returns (decision, search_query).
    """
    try:
        raw = generate_answer_fn(
            _QUERY_DECISION_SYSTEM,
            f"Q: {query}",
            max_new_tokens=20
        ).strip()

        upper = raw.upper()

        if upper.startswith("SKIP"):
            return "SKIP", ""
        elif upper.startswith("PARAMETRIC"):
            return "PARAMETRIC", ""
        elif upper.startswith("SEARCH:"):
            search_query = _sanitize_query(raw[7:].strip())
            if search_query and len(search_query) > 2:
                return "SEARCH", search_query
            else:
                # malformed SEARCH output — fall back to raw question
                return "SEARCH", query
        else:
            # unexpected format — treat entire output as query
            search_query = _sanitize_query(raw)
            return "SEARCH", search_query if search_query else query

    except Exception as e:
        print(f"  [RAG-Entertainment] Decision LLM failed: {e}")
        return "SEARCH", query


def _is_quality_snippet(text: str, url: str = "") -> bool:
    """Return False for snippets from low-quality or unreliable sources."""
    if not text or len(text.strip()) < 80:
        return False

    text_lower = text.lower()
    for pattern in _LOW_QUALITY_SIGNALS:
        if re.search(pattern, text_lower):
            return False

    # fast-pass trusted domains regardless of content
    if url:
        domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain and any(t in domain.group(1) for t in _TRUSTED_DOMAINS):
            return True

    return True


def _wiki(query: str, sentences: int = 5) -> str:
    """
    Fetch Wikipedia content: summary + first 6 substantive paragraphs.
    Returns empty string on any failure.
    """
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)

            # deeper paragraphs cover: early life, career details,
            # production notes, legacy, achievements — not just the intro
            paragraphs = [
                p.strip() for p in page.content.split("\n")
                if len(p.strip()) > 100
            ]
            extra = "\n\n".join(paragraphs[:6])

            return f"{summary}\n\n{extra}"[:4000]

        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


def _wiki_is_useful(text: str) -> bool:
    """Return True if Wikipedia result is substantial and not a disambiguation page."""
    if not text or len(text.strip()) < 200:
        return False
    if "may refer to:" in text.lower():
        return False
    return True


def _extract_keywords(text: str) -> list[str]:
    """Extract content words of 4+ chars, excluding stop words."""
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return [w for w in words if w not in _STOP_WORDS]


def _is_relevant(snippet: str, question: str) -> bool:
    keywords = _extract_keywords(question)
    if not keywords:
        return True

    snippet_lower = snippet.lower()

    # check 1: quoted titles
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", question)
    for match in quoted:
        title = (match[0] or match[1]).lower()
        if len(title) > 3 and title not in snippet_lower:
            return False

    # check 2: year anchoring
    years = re.findall(r'\b(19[0-9]{2}|20[0-9]{2})\b', question)
    if years:
        if not any(year in snippet_lower for year in years):
            return False

    # check 3: keyword overlap
    matches = sum(1 for kw in keywords if kw in snippet_lower)
    return matches >= 2

def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """
    Pipeline:
        1. Single LLM call → SKIP / PARAMETRIC / SEARCH: <query>
        2. Wikipedia first; DDG if Wikipedia misses or is irrelevant
        3. Quality filter on DDG snippets
        4. Relevance filter on all snippets before returning
    """

    # ── Stage 1: decide whether and what to search ───────────────────────────
    if generate_answer_fn is not None:
        decision, search_query = _get_search_decision(query, generate_answer_fn)
    else:
        # no LLM available — search with raw query
        decision, search_query = "SEARCH", query

    if decision == "SKIP":
        print("  [RAG-Entertainment] Skipping search (article reference or unnamed subject).")
        return ""

    if decision == "PARAMETRIC":
        print("  [RAG-Entertainment] Parametric question — no search needed.")
        return ""

    # decision == "SEARCH"
    print(f"  [RAG-Entertainment] Query: {search_query!r}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    # ── Stage 2: Wikipedia first ─────────────────────────────────────────────
    print(f"  [RAG-Entertainment] Trying Wikipedia...")
    wiki_result = _wiki(search_query)

    if _wiki_is_useful(wiki_result) and _is_relevant(wiki_result, query):
        print(f"  [RAG-Entertainment] Wikipedia hit ({len(wiki_result)} chars), skipping DDG.")
        _add(wiki_result)

    else:
        if wiki_result and not _is_relevant(wiki_result, query):
            print(f"  [RAG-Entertainment] Wikipedia result not relevant, trying DDG too.")
            _add(wiki_result)
        else:
            print(f"  [RAG-Entertainment] Wikipedia miss, falling back to DDG.")

        # ── Stage 3: DDG fallback ─────────────────────────────────────────────
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(search_query, max_results=num_results, timeout=8):
                    body  = r.get("body",  "")
                    title = r.get("title", "")
                    url   = r.get("href",  "")
                    if _is_quality_snippet(body, url):
                        _add(f"[{title}] {body}" if title else body)
        except Exception as exc:
            print(f"  [RAG-Entertainment] DDG failed: {exc}")

    # ── Stage 4: relevance filter — always runs regardless of source ──────────
    if snippets:
        relevant = [s for s in snippets if _is_relevant(s, query)]
        snippets = relevant if relevant else snippets

    return "\n\n".join(snippets)[:3500] if snippets else ""