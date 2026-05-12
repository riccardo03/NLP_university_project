"""
RAG pipeline for the Entertainment competition.
Wikipedia first, DuckDuckGo to supplement.
"""

import re
import concurrent.futures

_ARTICLE_REF_RE = re.compile(
    r'\baccording to (the article|the text|the passage)\b', re.I
)

_LOW_QUALITY_SIGNALS = [
    r'\d{4}\s*[·•]\s',   # "Jun 12, 2018 ·" — dated blog format
    r'click here',
    r'subscribe',
    r'read more',
    r'sign up',
    r'\.\.\.read',
    r'youtube\.com',
    r'goo\.gl',
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

_QUERY_GEN_SYSTEM = (
    "You are a search-query generator for an entertainment trivia bot. "
    "Given a question (already anchored to its subject), output a SHORT search query "
    "of 3 to 6 words that will retrieve a Wikipedia or web page "
    "WHERE THE ANSWER CAN BE READ. "
    "\n\n"
    "Rules:\n"
    "- Target the TOPIC, not the answer. Generate a query to find the right PAGE, "
    "not to pre-encode the answer.\n"
    "- 3 to 6 words maximum. Shorter is better.\n"
    "- Keep: proper nouns, titles, years, and 1-2 context words (biography, filmography, "
    "discography, career, history, relationship).\n"
    "- Drop: question words (what, why, how, when, which), verbs, filler words.\n"
    "- Output ONLY the query string. No punctuation at the end. No explanation.\n"
    "\n"
    "Examples:\n"
    "Question: James Cameron James Cameron primary reason switched physics English → "
    "James Cameron biography early career\n"
    "Question: The Godfather The Godfather director who directed → "
    "The Godfather 1972 film\n"
    "Question: Michael Jackson Thriller In what year did Michael Jackson release Thriller → "
    "Michael Jackson Thriller album release\n"
    "Question: 12-bar blues How does the blues form relate to 12-bar blues structure → "
    "12-bar blues music theory\n"
    "Question: Academy Awards Which film won Best Picture at the 2020 Academy Awards → "
    "Academy Awards 2020 Best Picture\n"
)

_SUBJECT_IDENTIFICATION_SYSTEM = (
    "You are a search assistant for an entertainment quiz bot. "
    "Given a quiz question, extract the PRIMARY named entity — "
    "the specific person, film, song, album, TV show, award, character, or event "
    "that the question is ABOUT. "
    "\n\n"
    "Rules:\n"
    "- Output ONLY the entity name, as short as possible (1-4 words).\n"
    "- If the question is about a concept, relationship, or process with NO named entity, "
    "output exactly: NONE\n"
    "- Never output a full sentence. Never explain.\n"
    "- If multiple entities appear, pick the one the question is primarily asking about.\n"
    "\n"
    "Examples:\n"
    "Q: What was the primary reason James Cameron switched from physics to English? → James Cameron\n"
    "Q: Who directed The Godfather? → The Godfather\n"
    "Q: How does the blues form relate to the 12-bar structure? → NONE\n"
    "Q: Which actor played Tony Stark in the Marvel films? → Tony Stark Marvel\n"
    "Q: In what year did Michael Jackson release Thriller? → Michael Jackson Thriller\n"
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
    r'\b(?:between|behind|describes|about|by|for|of|with|from)\s+([A-Z][a-z]{2,})\b'
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

    # fast-pass trusted domains
    if url:
        domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain and any(t in domain.group(1) for t in _TRUSTED_DOMAINS):
            return True

    return True



def _wiki(query: str, sentences: int = 5) -> str:
    """Fetch a Wikipedia summary; returns '' on any failure."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
            paragraphs = [
                p.strip() for p in page.content.split("\n")
                if len(p.strip()) > 100
            ]
            extra = "\n\n".join(paragraphs[:4])

            combined = f"{summary}\n\n{extra}"
            return combined[:3000]
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""

def _wiki_is_useful(text: str) -> bool:
    """
    Return True if the Wikipedia result is substantial enough to use.
    Filters out disambiguation pages, stubs, and empty results.
    """
    if not text or len(text.strip()) < 200:
        return False
    if "may refer to:" in text.lower():
        return False
    return True

def _extract_keywords(text: str) -> list[str]:
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    return [w for w in words if w not in _STOP_WORDS]

def _is_relevant(snippet: str, question: str) -> bool:
    keywords = _extract_keywords(question)
    if not keywords:
        return True  # no keywords to check, keep everything
    snippet_lower = snippet.lower()
    matches = sum(1 for kw in keywords if kw in snippet_lower)
    # require at least 2 question keywords to appear
    return matches >= 2
def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """
    Wikipedia + DuckDuckGo RAG for entertainment quiz questions.
    """
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Entertainment] Article-reference question — skipping search.")
        return ""
    # ------------------------------------------------------------------ #
    # Stage 1: LLM query distillation                                      #
    # ------------------------------------------------------------------ #
    ddg_query = query
    if generate_answer_fn is not None and _needs_subject_id(query):
            try:
                subject = generate_answer_fn(
                    _SUBJECT_IDENTIFICATION_SYSTEM,
                    f"Q: {query}",
                    max_new_tokens=15
                ).strip()
                
                if subject.upper() == "NONE" or len(subject) < 3:
                     print("  [RAG-Entertainment] No subject identified, using raw query.")
                     ddg_query = query
                else:
                     subject = subject.strip('"').strip("'")
                     print(f"  [RAG-Entertainment] Identified subject: {subject!r}")

                anchored_query = f"{subject} {query}"[:120]
                user_msg = f"Question: {anchored_query}"
                raw = generate_answer_fn(_QUERY_GEN_SYSTEM, user_msg, max_new_tokens=15)
                distilled = raw.strip().strip('"').strip("'")

                if distilled and len(distilled) > 3:
                    ddg_query = distilled
                else: 
                    ddg_query = anchored_query

            except Exception as e:
                print(f"  [RAG-Entertainment] Query distillation failed: {e}")
                ddg_query = query

            print(f"  [RAG-Entertainment] Query: {ddg_query!r}")
    else:
        ddg_query = query
        print(f"  [RAG-Entertainment] No subject ID needed. Query: {ddg_query!r}")
    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)


    print(f"  [RAG-Entertainment] Trying Wikipedia...")
    wiki_result = _wiki(ddg_query)
    if _wiki_is_useful(wiki_result):
        print(f"  [RAG-Entertainment] Wikipedia hit ({len(wiki_result)} chars), skipping DDG.")
        _add(wiki_result)
    else:
        print(f"  [RAG-Entertainment] Wikipedia miss, falling back to DDG.")
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(ddg_query, max_results=num_results, timeout=8):
                    title = r.get("title", "")
                    body = r.get("body", "")
                    url = r.get("href", "")
                    if _is_quality_snippet(body, url):
                        _add(f"[{title}]{body}" if title else {body})
        except Exception as exc:
            print(f"  [RAG-Entertainment] DDG failed: {exc}")

        if snippets:
            relevant = [s for s in snippets if _is_relevant(s, query)]
            snippets = relevant if relevant else snippets

    return "\n\n".join(snippets)[:3500] if snippets else ""
