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
    "You are a search-query optimizer. "
    "Given a quiz question, output a concise DuckDuckGo search query of at most 10 words. "
    "Focus on proper nouns, names, titles, and years. "
    "Return ONLY the query string — no explanation, no punctuation at the end."
)


def _wiki(query: str, sentences: int = 5) -> str:
    """Fetch a Wikipedia summary; returns '' on any failure."""
    try:
        import wikipedia
        wiki_q = query if len(query) <= 280 else query[:280].rsplit(" ", 1)[0]
        wikipedia.set_lang("en")
        try:
            return wikipedia.summary(wiki_q, sentences=sentences, auto_suggest=True)
        except wikipedia.exceptions.DisambiguationError as e:
            return wikipedia.summary(e.options[0], sentences=sentences)
        except wikipedia.exceptions.PageError:
            return ""
    except Exception:
        return ""


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None) -> str:
    """
    Wikipedia + DuckDuckGo search for entertainment context.
    If generate_answer_fn is provided, distil query to ≤10 focused words first.
    Document-reference questions ("according to the article/text/passage") are
    skipped entirely — no web result can substitute for the original article.
    """
    # Document-reference questions can't be answered by web search — the article
    # isn't online. Return empty so the model falls back to its own knowledge.
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Entertainment] Article-reference question — skipping search.")
        return ""

    # Stage 1: optional LLM query distillation
    ddg_query = query
    if generate_answer_fn is not None:
        try:
            raw = generate_answer_fn(_QUERY_GEN_SYSTEM, query, 20)
            distilled = raw.strip().strip('"').strip("'")
            if distilled:
                ddg_query = distilled
        except Exception:
            pass
        print(f"  [RAG-Entertainment] Query: {ddg_query!r}")

    snippets: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        if text and text not in seen:
            seen.add(text)
            snippets.append(text)

    # Stage 2a: Wikipedia — structured facts for actors, films, TV shows
    def _fetch_wiki():
        return _wiki(ddg_query)

    # Stage 2b: DDG — covers recent releases, charts, box office
    def _fetch_ddg():
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(ddg_query, max_results=num_results, timeout=8):
                title = r.get("title", "")
                body  = r.get("body",  "")
                if body:
                    results.append(f"[{title}] {body}" if title else body)
        return results

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_wiki = pool.submit(_fetch_wiki)
            f_ddg  = pool.submit(_fetch_ddg)
            for fut in concurrent.futures.as_completed(
                [f_wiki, f_ddg], timeout=12
            ):
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

    context = "\n\n".join(snippets)
    return context[:2000] if context else ""
