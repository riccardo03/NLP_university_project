"""
RAG pipeline for the Entertainment competition.
Wikipedia first, DuckDuckGo to supplement.

Improvements over v1:
- LLM distillation prompt explicitly instructs to use options as guide
- Wikipedia lookup runs for each option that looks like a named entity
- DDG runs a "base_query + option" search for every option
- Post-retrieval filtering keeps only snippets relevant to at least one option
"""

import re
import concurrent.futures

_ARTICLE_REF_RE = re.compile(
    r'\baccording to (the article|the text|the passage)\b', re.I
)

_QUERY_GEN_SYSTEM = (
    "You are a search-query optimizer. "
    "Given a quiz question and its possible answers, output a concise DuckDuckGo "
    "search query of at most 10 words that would help determine the correct answer. "
    "Focus on proper nouns, names, titles, and years. "
    "Return ONLY the query string — no explanation, no punctuation at the end."
)

_SUBJECT_IDENTIFICATION_SYSTEM = (
    "You are an entertainment expert. "
    "Given a quiz question about a film, song, artist, or show, "
    "identify the specific title or subject being referred to, even if not explicitly named. "
    "Output ONLY the identified subject — no explanation."
)

def _wiki(query: str, sentences: int = 5) -> str:
    """Fetch a Wikipedia summary; returns '' on any failure."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query, auto_suggest=True)
            # Cerca il paragrafo più rilevante invece delle prime frasi
            content = page.content
            paragraphs = [p for p in content.split("\n") if len(p) > 100]
            # Restituisci i primi 3 paragrafi rilevanti
            return "\n\n".join(paragraphs[:3])
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


def _is_relevant(text: str, options: list[str]) -> bool:
    """Return True if the snippet mentions at least one of the answer options."""
    text_lower = text.lower()
    return any(opt.lower() in text_lower for opt in options)


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """
    Wikipedia + DuckDuckGo RAG for entertainment quiz questions.


    Pipeline:
        1. Skip document-reference questions (no web source can substitute).
        2. [Optional] Distil the query with an LLM, using options as guidance.
        3a. Wikipedia — main query lookup.
        3b. DuckDuckGo — main query search.
        4. Filter snippets to those that mention at least one option.
        5. Return up to 2000 characters of deduplicated context.
    """
    # Guard: document-reference questions can't be answered by web search  #
    # ------------------------------------------------------------------ #
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Entertainment] Article-reference question — skipping search.")
        return ""
    # ------------------------------------------------------------------ #
    # Stage 1: LLM query distillation                                      #
    # ------------------------------------------------------------------ #
    ddg_query = query
    if generate_answer_fn is not None:
            try:
                subject = generate_answer_fn(
                    _SUBJECT_IDENTIFICATION_SYSTEM,
                    query,
                    max_new_tokens=20
                ).strip()
                print(f"  [RAG-Entertainment] Identified subject: {subject!r}")
                anchored_query = f"{subject} {query}" if subject else query

                user_msg = (
                    f"Question: {anchored_query}\n"
                    f"Possible answers: {', '.join(option_texts)}\n"
                    "Generate a search query to find which answer is correct."
                ) if option_texts else anchored_query

                raw = generate_answer_fn(_QUERY_GEN_SYSTEM, user_msg, 20)
                distilled = raw.strip().strip('"').strip("'")
                if distilled:
                    ddg_query = distilled
            except Exception:
                pass
            print(f"  [RAG-Entertainment] Query: {ddg_query!r}")

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