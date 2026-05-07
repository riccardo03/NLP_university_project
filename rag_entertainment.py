"""
RAG pipeline for the Entertainment competition.
DuckDuckGo search with optional LLM query distillation.
"""

import concurrent.futures

_QUERY_GEN_SYSTEM = (
    "You are a search-query optimizer. "
    "Given a quiz question, output a concise DuckDuckGo search query of at most 10 words. "
    "Focus on proper nouns, names, titles, and years. "
    "Return ONLY the query string — no explanation, no punctuation at the end."
)


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None) -> str:
    """
    Search DuckDuckGo for entertainment context.
    If generate_answer_fn is provided, distil query to ≤10 focused words first.
    """
    # Stage 1: optional LLM query distillation
    ddg_query = query
    if generate_answer_fn is not None:
        try:
            raw = generate_answer_fn(_QUERY_GEN_SYSTEM, query, max_new_tokens=20)
            distilled = raw.strip().strip('"').strip("'")
            if distilled:
                ddg_query = distilled
        except Exception:
            pass
        print(f"  [RAG-Entertainment] Query: {ddg_query!r}")

    # Stage 2: DDG inside a 12s executor; ddgs.text itself gets timeout=8
    def _search():
        from ddgs import DDGS
        snippets = []
        with DDGS() as ddgs:
            for r in ddgs.text(ddg_query, max_results=num_results, timeout=8):
                title = r.get("title", "")
                body  = r.get("body",  "")
                if body:
                    snippets.append(f"[{title}] {body}" if title else body)
        return snippets

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_search)
            try:
                snippets = next(
                    concurrent.futures.as_completed([fut], timeout=12)
                ).result()
            except concurrent.futures.TimeoutError:
                print("  [RAG-Entertainment] Timed out.")
                snippets = []
    except Exception as exc:
        print(f"  [RAG-Entertainment] Failed: {exc}")
        snippets = []

    context = "\n\n".join(snippets)
    return context[:2000] if context else ""
