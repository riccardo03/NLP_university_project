"""
RAG pipeline for Science & Nature — v2.
Key improvement: option texts used as query hints.
No LLM calls — fast and memory-safe.
"""

import re
import concurrent.futures
from rag_shared import (wiki_fetch, wiki_search, ddg_search,
                        extract_entity, best_paragraphs, score_paragraph,
                        extract_keywords, _STOP_WORDS, _PROPER_RE, TITLE_STOP)

_ARTICLE_REF_RE = re.compile(
    r"\b(according to (the )?(article|text|passage|paragraph|excerpt))\b", re.I
)
_SCI_STOP = _STOP_WORDS | {
    "science","scientific","natural","nature","describe","describes","explain",
    "example","following","process","result","study","term","concept","best",
    "primary","mainly","which","what","type","form","called","known",
}
_QUOTED_RE = re.compile(r"[\'\"]([ \w]{2,40}?)[\'\"]")


def _extract_sci_entities(question: str, option_texts: list) -> list:
    """Extract science-specific entities from question + options."""
    entities = []

    # Quoted terms (e.g. "Krebs cycle", "Newton's law")
    for m in _QUOTED_RE.findall(question):
        if len(m.strip()) > 2: entities.append(m.strip())

    # Scientific named terms in question
    sci_terms = re.findall(
        r"\b(photosynthesis|mitosis|meiosis|osmosis|respiration|evolution|"
        r"relativity|thermodynamics|electromagnetism|chromosome|enzyme|catalyst|"
        r"isotope|DNA|RNA|ATP|CRISPR|PCR|mRNA|amino acid|protein|ribosome|"
        r"mitochondria|chloroplast|neuron|synapse|genome|allele|genotype|"
        r"phenotype|homeostasis|transpiration|fermentation|diffusion|"
        r"[A-Z][a-z]+ law|[A-Z][a-z]+ theorem|[A-Z][a-z]+ effect|"
        r"[A-Z][a-z]+\'s law|[A-Z][a-z]+\'s principle|"
        r"[A-Z][a-z]+ cycle|[A-Z][a-z]+ reaction|"
        r"[A-Z][a-z]+ constant|[A-Z][a-z]+ number)\b",
        question
    )
    entities.extend(sci_terms)

    # Multi-word proper nouns from question
    for p in _PROPER_RE.findall(question):
        if p.split()[0] not in TITLE_STOP: entities.append(p.strip())

    # Options — the answer is in there!
    for opt in option_texts:
        if len(opt.strip()) < 3: continue
        # Scientific terms in options
        sci_in_opt = re.findall(
            r"\b([A-Z][a-z]+(?:\s+[a-z]+){0,2}|[a-z]{6,})\b", opt
        )
        for t in sci_in_opt:
            if t.lower() not in _SCI_STOP and len(t) > 4:
                entities.append(t)
        # Short options are often the answer label itself
        words = opt.strip().split()
        if 2 <= len(words) <= 5:
            kws = [w for w in words if w.lower() not in _SCI_STOP and len(w) > 4]
            if kws: entities.append(" ".join(words[:4]))

    return list(dict.fromkeys(e for e in entities if e))[:6]


def _build_science_queries(question: str, option_texts: list) -> list:
    entities = _extract_sci_entities(question, option_texts)
    queries = []
    for e in entities[:3]:
        if e not in queries: queries.append(e)
    # Fallback
    if not queries:
        kws = [w for w in extract_keywords(question, min_len=5) if w not in _SCI_STOP]
        if kws: queries.append(" ".join(kws[:3]))
    seen, result = set(), []
    for q in queries:
        q = q.strip()
        if q and q not in seen and len(q) > 2:
            seen.add(q); result.append(q)
    return result[:4]


def _fetch_wiki_science(query: str, question: str) -> str:
    text = wiki_fetch(query, max_chars=4000)
    if not text or len(text.strip()) < 100:
        titles = wiki_search(query, max_results=2)
        for t in titles:
            text = wiki_fetch(t, max_chars=4000)
            if text and len(text.strip()) >= 100: break
    if not text: return ""
    return best_paragraphs(text, question, top_n=2, min_len=60)


def rag_science(question_text: str, option_texts: list = None,
                generate_answer_fn=None) -> str:
    """Science RAG v2: question + option entity extraction → Wikipedia + DDG."""
    if _ARTICLE_REF_RE.search(question_text):
        return ""
    print("  [RAG-Science] Pipeline started...")
    option_texts = option_texts or []

    queries = _build_science_queries(question_text, option_texts)
    print(f"  [RAG-Science] Queries: {queries}")

    snippets, seen = [], set()
    def _add(text):
        if text and text.strip() and text not in seen:
            seen.add(text); snippets.append(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_wiki_science, q, question_text): q for q in queries}
        for fut in concurrent.futures.as_completed(futures, timeout=9):
            try:
                res = fut.result()
                if res: _add(res)
            except Exception:
                pass

    if not snippets:
        print("  [RAG-Science] Wikipedia miss — DDG fallback...")
        for s in ddg_search(queries[0] if queries else question_text[:80], max_results=3):
            _add(s)

    if not snippets: return ""

    scored = sorted([(score_paragraph(s, question_text), s) for s in snippets], reverse=True)
    context = "\n\n".join(s for _, s in scored[:3])
    print(f"  [RAG-Science] Done. Context: {len(context)} chars.")
    return context[:5000]
