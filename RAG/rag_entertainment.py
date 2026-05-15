"""
RAG pipeline for Entertainment — v2.
Key improvement: option entity extraction + person+work dual query.
"""

import re
import concurrent.futures
from rag_shared import (wiki_fetch, ddg_search, best_paragraphs,
                        score_paragraph, extract_keywords,
                        _PROPER_RE, _YEAR_RE, TITLE_STOP, _STOP_WORDS)

_ARTICLE_REF_RE = re.compile(
    r"\b(according to|as described in|based on|per)\s+"
    r"(the article|the text|the passage|the excerpt)\b", re.I
)
_QUOTED_RE = re.compile(r"[\'\"]([ \w]{2,50}?)[\'\"]")
_ENT_STOP = _STOP_WORDS | {
    "film","movie","song","show","role","style","music","band","album",
    "book","character","actor","director","singer","artist","group",
    "known","starred","released","debut","popular","famous","best",
    "primary","reason","which","describe","describes","called","first",
}
_PERSON_RE = re.compile(
    r"\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)
_WORK_RE = re.compile(r"[\'\"]([ \w]{2,40}?)[\'\"]")


def _build_ent_queries(question: str, option_texts: list) -> list:
    """Build multiple search queries using question + option entities."""
    queries = []
    year = _YEAR_RE.search(question)
    year_str = f" {year.group(1)}" if year else ""

    # 1. Quoted titles from question (highest priority)
    for m in _QUOTED_RE.findall(question):
        m = m.strip()
        if len(m) > 2 and m not in TITLE_STOP:
            queries.append(m + year_str)

    # 2. Person names from question
    persons = [p for p in _PERSON_RE.findall(question)
               if p.split()[0] not in TITLE_STOP]
    if persons and len(queries) < 3:
        p = persons[0]
        # Enrich with context kws
        p_words = set(p.lower().split())
        extra = [w for w in extract_keywords(question, min_len=5)
                 if w not in p_words and w not in _ENT_STOP]
        queries.append(f"{p} {' '.join(extra[:2])} {year_str}".strip())

    # 3. Option entities — the answer is one of them!
    for opt in option_texts:
        if len(opt.strip()) < 4: continue
        # Quoted works in options
        for m in _WORK_RE.findall(opt):
            if len(m.strip()) > 2 and m.strip() not in queries:
                queries.append(m.strip())
        # Person names in options
        for p in _PERSON_RE.findall(opt):
            if p.split()[0] not in TITLE_STOP and p not in queries:
                queries.append(p)
        if len(queries) >= 5:
            break

    # 4. Multi-word proper nouns from question
    if len(queries) < 2:
        proper = _PROPER_RE.findall(question)
        for p in proper:
            if p.split()[0] not in TITLE_STOP and p not in queries:
                queries.append(p + year_str)

    # 5. Keyword fallback
    if not queries:
        kws = [w for w in extract_keywords(question, min_len=5) if w not in _ENT_STOP]
        if kws: queries.append(" ".join(kws[:4]) + year_str)

    # Deduplicate, max 4
    seen, result = set(), []
    for q in queries:
        q = q.strip()
        if q and q not in seen and len(q) > 2:
            seen.add(q); result.append(q)
    return result[:4]


def _fetch_wiki_ent(query: str, question: str) -> str:
    text = wiki_fetch(query, max_chars=5000)
    if not text or len(text.strip()) < 100: return ""
    return best_paragraphs(text, question, top_n=2, min_len=80)


def _ddg_ent(query: str) -> list:
    _TRUSTED = {"wikipedia.org","imdb.com","britannica.com","rollingstone.com",
                "billboard.com","biography.com","variety.com","theguardian.com"}
    raw = ddg_search(query, max_results=5, timeout=4)
    trusted = [s for s in raw if any(t in s for t in _TRUSTED)]
    other   = [s for s in raw if s not in trusted]
    return (trusted + other)[:3]


def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """Entertainment RAG v2: question + option entity extraction → Wikipedia + DDG."""
    if _ARTICLE_REF_RE.search(query): return ""
    print("  [RAG-Ent] Pipeline started...")
    option_texts = option_texts or []

    queries = _build_ent_queries(query, option_texts)
    print(f"  [RAG-Ent] Queries: {queries}")

    snippets, seen = [], set()
    def _add(text):
        if text and text.strip() and text not in seen:
            seen.add(text); snippets.append(text)

    # Parallel Wikipedia fetches (primary) + DDG on best query
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        wiki_futs = {ex.submit(_fetch_wiki_ent, q, query): q for q in queries[:3]}
        ddg_fut   = ex.submit(_ddg_ent, queries[0] if queries else query[:80])

        for fut in concurrent.futures.as_completed(wiki_futs, timeout=8):
            try:
                res = fut.result()
                if res: _add(res)
            except Exception: pass

        try:
            for s in ddg_fut.result(timeout=5): _add(s)
        except Exception: pass

    if not snippets: return ""

    scored = sorted([(score_paragraph(s, query), s) for s in snippets], reverse=True)
    best = [s for sc, s in scored if sc > 0] or [s for _, s in scored[:2]]
    context = "\n\n".join(best[:3])
    print(f"  [RAG-Ent] Done. Context: {len(context)} chars.")
    return context[:5000]
