
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
    "Given a quiz question, follow these steps in order and output "
    "your final decision on the last line.\n"
    "\n"
    "STEP 1 — Check for unanswerable references.\n"
    "Does the question contain any of these phrases ANYWHERE, even at the end?\n"
    "  - 'according to the article/text/passage'\n"
    "  - 'as described/mentioned/stated in the article'\n"
    "  - 'based on the passage'\n"
    "  - 'in his/her own words'\n"
    "  - 'the film/show/book' with NO title given (e.g. 'the film portrays' "
    "but not 'the film Titanic portrays')\n"
    "If YES → output: SKIP\n"
    "\n"
    "STEP 2 — Check for named entities.\n"
    "Is there a specific named person, film, song, album, show, or event?\n"
    "If NO (pure concept question like 'what is sonata form') → output: PARAMETRIC\n"
    "\n"
    "STEP 3 — Build the search query.\n"
    "Follow these rules:\n"
    "  a) Start with the named entity (person, film, song, show).\n"
    "  b) If the question asks about a RELATIONSHIP between two named entities, "
    "include both.\n"
    "  c) If the question implies a well-known work without naming it "
    "(e.g. 'Tom Hanks integrated into archival footage' implies Forrest Gump, "
    "'the shark film' implies Jaws), infer the title and include it.\n"
    "  d) Add 1-2 context words that narrow which PAGE is needed: "
    "biography, career, filmography, formation, production, early life, "
    "relationship, discography.\n"
    "  e) Drop everything else: question words, verbs, consequence words "
    "(impact, legacy, reason, mental health, effect, result), filler.\n"
    "  f) 3-6 words total. Spaces between every word. No non-English characters.\n"
    "Output: SEARCH: <query>\n"
    "\n"
    "Examples of the reasoning:\n"
    "Q: What was the primary reason James Cameron switched from physics to English?\n"
    "Step1: no article reference → continue\n"
    "Step2: 'James Cameron' is a named person → continue\n"
    "Step3: entity=James Cameron, context=biography early career → "
    "SEARCH: James Cameron biography early career\n"
    "\n"
    "Q: According to the article, what does the author argue?\n"
    "Step1: 'according to the article' found → SKIP\n"
    "\n"
    "Q: Which term describes the way the film portrays antebellum life?\n"
    "Step1: 'the film' with no title → SKIP\n"
    "\n"
    "Q: What is the fundamental principle of sonata form?\n"
    "Step1: no article reference → continue\n"
    "Step2: no named entity → PARAMETRIC\n"
    "\n"
    "Q: Which visual effects technique was used to integrate Tom Hanks "
    "into archival footage?\n"
    "Step1: no article reference → continue\n"
    "Step2: 'Tom Hanks' named, 'archival footage' implies Forrest Gump → continue\n"
    "Step3: entity=Forrest Gump Tom Hanks, context=visual effects production → "
    "SEARCH: Forrest Gump visual effects production\n"
    "\n"
    "Q: Which of the following was a key challenge faced by Coldplay "
    "during their early years?\n"
    "Step1: no article reference → continue\n"
    "Step2: 'Coldplay' is a named band → continue\n"
    "Step3: entity=Coldplay, context=biography formation → "
    "SEARCH: Coldplay band biography formation\n"
    "\n"
    "Q: How did Freddie Mercury's relationship with Mary Austin evolve "
    "over time according to the article?\n"
    "Step1: 'according to the article' found at end of question → SKIP\n"
    "\n"
    "Output format: write the steps briefly, then the final decision on the last line.\n"
    "The last line must be exactly SKIP, PARAMETRIC, or SEARCH: <query>.\n"
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
            max_new_tokens=80
        ).strip()

        lines = [l.strip() for l in raw.split('\n') if l.strip()]
        decision_line = lines[-1] if lines else ""
        upper = decision_line.upper()


        if upper.startswith("SKIP"):
            return "SKIP", ""
        elif upper.startswith("PARAMETRIC"):
            return "PARAMETRIC", ""
        elif upper.startswith("SEARCH:"):
            search_query = _sanitize_query(decision_line[7:].strip())
            return "SEARCH", search_query if len(search_query) > 2 else query
        else:
            # fallback — scan all lines for a SEARCH decision
            for line in reversed(lines):
                if line.upper().startswith("SEARCH:"):
                    search_query = _sanitize_query(line[7:].strip())
                    return "SEARCH", search_query if len(search_query) > 2 else query
            return "SEARCH", query

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