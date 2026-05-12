"""
RAG pipeline for the Entertainment competition.
Wikipedia first, DuckDuckGo to supplement.
"""

import re
import concurrent.futures

_ARTICLE_REF_RE = re.compile(
    r'\baccording to (the article|the text|the passage)\b', re.I
)

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

def _needs_subject_id(question: str) -> bool:
    """Only run subject ID if the question is about a named entertainment entity."""
    return bool(_SUBJECT_TRIGGERS.search(question))

def _wiki(query: str, sentences: int = 5) -> str:
    """Fetch a Wikipedia summary; returns '' on any failure."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            return wikipedia.summary(query, sentences=sentences, auto_suggest=True)
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


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

    # ------------------------------------------------------------------ #
    # Stage 2a: Wikipedia — main query + one lookup per named-entity option#
    # ------------------------------------------------------------------ #
    def _fetch_wiki_main():
        return _wiki(ddg_query, sentences=10)

    # ------------------------------------------------------------------ #
    # Stage 2b: DDG — main query + one search per option                   #
    # ------------------------------------------------------------------ #
    def _fetch_ddg_main():
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(ddg_query, max_results=num_results, timeout=8):
                title = r.get("title", "")
                body  = r.get("body",  "")
                if body:
                    results.append(f"{body} (fonte: {title})" if title else body)
        return results


    # ------------------------------------------------------------------ #
    # Stage 3: run everything in parallel                                  #
    # ------------------------------------------------------------------ #
    futures = []
    try:
        # +2 for main wiki + main DDG; +len(entity_options) wiki; +len(options) DDG
        max_workers = 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures.append(pool.submit(_fetch_wiki_main))
            futures.append(pool.submit(_fetch_ddg_main))

            for fut in concurrent.futures.as_completed(futures, timeout=15):
                try:
                    result = fut.result()
                except Exception:
                    continue
                if isinstance(result, list):
                    for s in result:
                        _add(s)
                else:
                    _add(result)

    except concurrent.futures.TimeoutError:
        print("  [RAG-Entertainment] Timed out.")
    except Exception as exc:
        print(f"  [RAG-Entertainment] Failed: {exc}")

    # ------------------------------------------------------------------ #
    # Stage 4: filter — keep only snippets relevant to at least one option #
    # ------------------------------------------------------------------ #
    if option_texts:
        relevant = [s for s in snippets if _is_relevant(s, option_texts)]
        # Fall back to all snippets if filtering removed everything
        snippets = relevant if relevant else snippets

    context = "\n\n".join(snippets)
    return context[:4000] if context else ""