"""
RAG pipeline for History & Politics — v2.
Key improvement: option texts used as query hints (correct answer is one of them!).
No LLM calls — fast and memory-safe.
"""

import re
import concurrent.futures
from rag_shared import (wiki_fetch, wiki_search, ddg_search,
                        extract_entity, best_paragraphs, score_paragraph,
                        extract_keywords, _STOP_WORDS, _PROPER_RE, TITLE_STOP)

_HIST_STOP = _STOP_WORDS | {
    "history","ancient","period","empire","dynasty","century",
    "describe","describes","explains","significant","primary","term",
    "best","following","which","known","used","refers","called",
    "type","form","process","practice","concept","system",
}

_DISAMBIGUATION = {
    r"\bthe city\b":      "Ancient Athens",
    r"\bthe republic\b":  "Roman Republic",
    r"\bthe empire\b":    "Roman Empire",
    r"\bthe senate\b":    "Roman Senate",
    r"\bthe pharaoh\b":   "Ancient Egypt pharaoh",
    r"\bthe church\b":    "Catholic Church history",
}

_QUOTED_RE = re.compile(r"[\'\"]([ \w]{2,40}?)[\'\"]")
_YEAR_RE   = re.compile(r"\b(1[0-9]{3}|2[0-9]{3}|[0-9]{1,3}\s*BC|[0-9]{1,3}\s*AD)\b", re.I)


def _extract_question_entities(question: str) -> list:
    """Extract specific named entities from question text."""
    entities = []

    # 1. Quoted terms — highest priority ("Linear B", "devotio", etc.)
    for m in _QUOTED_RE.findall(question):
        m = m.strip()
        if len(m) > 2 and m not in TITLE_STOP:
            entities.append(m)

    # 2. Multi-word proper nouns
    for p in _PROPER_RE.findall(question):
        words = p.strip().split()
        if words[0] not in TITLE_STOP and len(p) > 3:
            entities.append(p.strip())

    # 3. Specific history terms
    hist_terms = re.findall(
        r"\b(Linear B|Mycenaean|Herodian|Sassanid|Ptolemaic|Achaemenid|"
        r"Carolingian|Byzantine|Umayyad|Abbasid|Fatimid|Merovingian|"
        r"Diocletian|Augustan|Flavian|Julio-Claudian|Antonine|Severan|"
        r"Tetrarchy|Principate|Dominate|Pax Romana|Res Publica|"
        r"Ius civile|devotio|contubernium|maniple|phalanx|trireme|"
        r"[A-Z][a-z]+ dynasty|[A-Z][a-z]+ Empire|[A-Z][a-z]+ Republic|"
        r"[A-Z][a-z]+ Kingdom|[A-Z][a-z]+ civilization)\b",
        question, re.I
    )
    entities.extend(hist_terms)

    return list(dict.fromkeys(e for e in entities if e))


def _extract_option_entities(option_texts: list) -> list:
    """Extract searchable entities from answer options.
    
    KEY INSIGHT: The correct answer IS one of the options.
    Searching for option entities massively improves recall.
    """
    entities = []
    for opt in option_texts:
        # Skip generic options like "Yes/No", "True/False", numbers
        if len(opt.strip()) < 4:
            continue
        # Extract proper nouns from option text
        for p in _PROPER_RE.findall(opt):
            if p.split()[0] not in TITLE_STOP:
                entities.append(p.strip())
        # Extract quoted terms
        for m in _QUOTED_RE.findall(opt):
            if len(m.strip()) > 2:
                entities.append(m.strip())
        # Use the full option if short and specific (< 5 words)
        words = opt.strip().split()
        if 2 <= len(words) <= 5:
            kws = [w for w in words if w.lower() not in _HIST_STOP and len(w) > 3]
            if kws:
                entities.append(" ".join(words[:4]))
    return list(dict.fromkeys(entities))


def _build_history_queries(question: str, option_texts: list = None) -> list:
    """Build targeted Wikipedia search queries using question + options."""
    queries = []
    option_texts = option_texts or []

    # 1. Check disambiguation map
    q_lower = question.lower()
    for pattern, replacement in _DISAMBIGUATION.items():
        if re.search(pattern, q_lower):
            queries.append(replacement)
            break

    # 2. Extract entities from question
    q_entities = _extract_question_entities(question)
    for ent in q_entities[:2]:
        if ent not in queries:
            queries.append(ent)

    # 3. Extract entities from OPTIONS — the answer is in there!
    opt_entities = _extract_option_entities(option_texts)
    for ent in opt_entities[:2]:
        if ent not in queries:
            queries.append(ent)

    # 4. Fallback: civilization keyword + topic keyword
    if len(queries) < 2:
        civs = re.findall(
            r"\b(Roman|Greek|Egyptian|Babylonian|Persian|Ottoman|Byzantine|"
            r"Spartan|Athenian|Macedonian|Hittite|Assyrian|Sumerian|"
            r"Carthaginian|Phoenician|Etruscan|Ptolemaic|Sassanid)\b",
            question, re.I
        )
        topic_kws = [w for w in extract_keywords(question, min_len=5)
                     if w not in _HIST_STOP]
        if civs and topic_kws:
            queries.append(f"{civs[0].title()} {topic_kws[0]}")
        elif topic_kws:
            queries.append(" ".join(topic_kws[:3]))

    # Deduplicate, max 4 queries
    seen, result = set(), []
    for q in queries:
        q = q.strip()
        if q and q not in seen and len(q) > 2:
            seen.add(q)
            result.append(q)
    return result[:4]


def _fetch_wiki_history(query: str, question: str) -> str:
    text = wiki_fetch(query, max_chars=6000)
    if not text or len(text.strip()) < 200:
        titles = wiki_search(query, max_results=2)
        for t in titles:
            text = wiki_fetch(t, max_chars=6000)
            if text and len(text.strip()) >= 200:
                break
    if not text:
        return ""
    return best_paragraphs(text, question, top_n=3, min_len=80)


def rag_history(question_text: str, option_texts: list = None,
                generate_answer_fn=None) -> str:
    """History RAG v2: question + option entity extraction → Wikipedia + DDG."""
    print("  [RAG-History] Pipeline started...")
    option_texts = option_texts or []

    queries = _build_history_queries(question_text, option_texts)
    print(f"  [RAG-History] Queries: {queries}")

    snippets, seen = [], set()
    def _add(text):
        if text and text.strip() and text not in seen:
            seen.add(text); snippets.append(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_wiki_history, q, question_text): q for q in queries}
        for fut in concurrent.futures.as_completed(futures, timeout=9):
            try:
                res = fut.result()
                if res: _add(res)
            except Exception:
                pass

    if not snippets:
        print("  [RAG-History] Wikipedia miss — DDG fallback...")
        ddg_q = queries[0] if queries else question_text[:80]
        for s in ddg_search(ddg_q, max_results=3): _add(s)

    if not snippets:
        return ""

    scored = sorted([(score_paragraph(s, question_text), s) for s in snippets], reverse=True)
    context = "\n\n".join(s for _, s in scored[:3])
    print(f"  [RAG-History] Done. Context: {len(context)} chars.")
    return context[:7000]
