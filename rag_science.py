"""
RAG pipeline for the Science & Nature competition.
Multi-stage: query generation → parallel retrieval → cross-encoder reranking.
"""

import re
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rag_history import rag_history
from rag_entertainment import rag_entertainment

# Cross-encoder loaded lazily on first Science question, heavy it need not be at import
_reranker = None


def _get_reranker():
    """Load (and cache) the cross-encoder model. Once loaded, remember it we do."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        print("  [RAG-Science] Loading cross-encoder, patience you must have...")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("  [RAG-Science] Cross-encoder ready, it is.")
    return _reranker


def warmup_reranker() -> None:
    """Force the cross-encoder to load before the game timer starts. Warm it we must."""
    _rerank("warmup", ["warmup"])


# Stage 1 — Query Generation
_QUERY_GEN_SYSTEM = (
    "You are a search-query generator. "
    "Given a quiz question and its options, return a JSON object with exactly this structure:\n"
    '{"queries": ["query1", "query2", "query3"]}\n'
    "The three queries must be distinct, complementary Wikipedia/web search queries "
    "that together cover the key concepts needed to answer the question. "
    "Return raw JSON only — no markdown, no explanation."
)

# Phrases that signal document-specific questions — mislead web search they do
_ARTICLE_REF_RE = re.compile(
    r"\b(according to (the )?(article|text|passage|paragraph|excerpt))\b",
    re.I,
)


def _generate_search_queries(question_text: str, option_texts: list,
                              generate_answer_fn) -> list:
    """
    Ask the LLM for three search queries, with robust fallback parsing.
    Strip document-reference phrases first; mislead the search, they would.
    """
    # Strip "according to the article/text/passage" phrases
    clean_q = _ARTICLE_REF_RE.sub("", question_text).strip(" ,;")
    if clean_q != question_text:
        print(f"  [RAG-Science] Document-reference stripped: '{question_text[:60]}...'")

    opts_str = ", ".join(f'"{t}"' for t in option_texts)
    user_msg = f'Question: {clean_q}\nOptions: [{opts_str}]'

    t_llm = time.time()
    try:
        raw = generate_answer_fn(_QUERY_GEN_SYSTEM, user_msg, max_new_tokens=80)
        print(f"  [RAG-Science] Query LLM output ({time.time()-t_llm:.1f}s): {raw[:150]!r}")

        # Primary: JSON parse
        json_match = re.search(r'\{.*?"queries".*?\}', raw, re.S)
        if json_match:
            data = json.loads(json_match.group())
            queries = [q.strip() for q in data.get("queries", []) if q.strip()]
            if queries:
                return queries[:3]

        # Fallback: line-by-line extraction when JSON is malformed
        lines = raw.splitlines()
        queries = []
        for line in lines:
            line = re.sub(r'^[\d\.\-\*\)\s]+', '', line).strip().strip('"').strip("'")
            if 5 <= len(line) <= 120:
                queries.append(line)
        if queries:
            return queries[:3]

    except Exception as exc:
        print(f"  [RAG-Science] Query generation failed: {exc}")

    # Last-resort fallback: truncated question text so Wikipedia doesn't choke
    fallback = clean_q[:80].rsplit(' ', 1)[0]
    return [fallback]


# Stage 2 — Parallel Multi-Source Retrieval
def _retrieve_parallel(queries: list) -> list:
    """
    Run Wikipedia and DuckDuckGo in parallel for every query.
    Fast retrieval across sources, ThreadPoolExecutor provides.
    """
    tasks = []
    for q in queries:
        tasks.append(("wiki", q))
        tasks.append(("ddg",  q))

    snippets = []
    seen = set()

    def _fetch(source, query):
        if source == "wiki":
            result = rag_history(query, sentences=4)
            # Wikipedia parse failures return ""; fall through to DuckDuckGo
            if not result:
                result = rag_entertainment(query, num_results=2)
            return result
        else:
            return rag_entertainment(query, num_results=2)

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, src, q): (src, q) for src, q in tasks}
        for fut in as_completed(futures, timeout=15):
            try:
                text = fut.result()
            except Exception:
                text = ""
            if text and text not in seen:
                seen.add(text)
                snippets.append(text)

    return snippets


# Stage 3 — Cross-Encoder Re-ranking
def _rerank(question_text: str, snippets: list, top_k: int = 3) -> str:
    """
    Score each snippet against the question; keep the top-k.
    Relevant context rises to the top, irrelevant falls away.
    """
    if not snippets:
        return ""
    reranker = _get_reranker()
    pairs = [(question_text, s) for s in snippets]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, snippets), reverse=True)
    top_snippets = [s for _, s in ranked[:top_k]]
    return "\n\n".join(top_snippets)


def rag_science(question_text: str, option_texts: list = None,
                generate_answer_fn=None) -> str:
    """
    Multi-stage science RAG: query generation → parallel retrieval → re-ranking.
    Wikidata dropped; cross-encoder reranking added, better coverage achieved.
    """
    if option_texts is None:
        option_texts = []

    # Stage 1: generate diverse queries
    t1 = time.time()
    queries = _generate_search_queries(question_text, option_texts, generate_answer_fn)
    print(f"  [RAG-Science] Stage1 query-gen: {time.time()-t1:.1f}s → {queries}")

    # Stage 2: parallel multi-source retrieval
    t2 = time.time()
    snippets = _retrieve_parallel(queries)
    print(f"  [RAG-Science] Stage2 retrieval: {time.time()-t2:.1f}s → {len(snippets)} snippets")

    if not snippets:
        return ""

    # Stage 3: re-rank and return top context
    t3 = time.time()
    context = _rerank(question_text, snippets, top_k=3)
    print(f"  [RAG-Science] Stage3 reranking: {time.time()-t3:.1f}s")
    return context
