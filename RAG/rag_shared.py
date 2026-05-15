"""
Shared RAG utilities — MediaWiki API, DDG search, relevance scoring.
No LLM calls — fast and memory-safe.
"""

import re, urllib.parse, time
import requests

_WIKI_UA = "PoliMillionaireBot/2.0 (polimi NLP project)"
_CITE_RE = re.compile(r"\[\d+\]|\[citation needed\]|\[clarification needed\]", re.I)
_SECTION_END_RE = re.compile(r"^==\s*", re.M)

_STOP_WORDS = {
    "what","when","which","where","does","have","this","that","from","with",
    "about","into","their","there","been","were","would","could","should",
    "according","following","describe","called","known","used","term",
    "best","most","first","last","also","many","some","such","they",
    "refers","refers","refer","between","during","after","before",
    "primary","mainly","mainly","often","always","never","each",
}

_UL = "A-ZÀ-ÖØ-Ý"
_AL = "a-zA-ZÀ-ÖØ-öø-ÿ"
_PROPER_RE = re.compile(rf"([{_UL}][{_AL}]+(?:\s+[{_UL}][{_AL}]+)+)")
_SINGLE_PROPER_RE = re.compile(rf"\b([{_UL}][{_AL}]{{2,}})\b")
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|2[0-9]{3})\b")
_QUOTED_RE = re.compile(r"""(?<!\w)['\"]([\w][\w\s,\.\-]{{1,58}}?)['\"]""")

TITLE_STOP = {
    "Which","What","How","Who","When","Where","Why","The","This","That","These",
    "Those","A","An","Is","Are","Was","Were","Has","Have","Had","Does","Do","Did",
    "Will","Would","Could","Should","According","Following","Best","Most","First",
    "Last","Both","Each","Some","Many","Often",
}


def wiki_fetch(title: str, max_chars: int = 4000) -> str:
    """Fetch Wikipedia article via MediaWiki API. Returns plain text."""
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&explaintext=true&redirects=1"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )
    for attempt in range(2):
        try:
            r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=5)
            if r.status_code != 200:
                continue
            pages = r.json()["query"]["pages"]
            pid = next(iter(pages))
            if pid == "-1":
                return ""
            text = pages[pid].get("extract", "")
            text = _CITE_RE.sub("", text)
            return text[:max_chars]
        except Exception:
            time.sleep(0.3)
    return ""


def wiki_search(query: str, max_results: int = 3) -> list:
    """Wikipedia OpenSearch — returns list of page titles."""
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=opensearch&search={urllib.parse.quote(query)}"
        f"&limit={max_results}&namespace=0&format=json"
    )
    try:
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=4)
        if r.status_code == 200:
            data = r.json()
            return data[1] if len(data) > 1 else []
    except Exception:
        pass
    return []


def ddg_search(query: str, max_results: int = 3, timeout: int = 4) -> list:
    """DuckDuckGo search. Returns list of snippet strings."""
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results + 3, timeout=timeout):
                body = r.get("body", "")
                title = r.get("title", "")
                url = r.get("href", "")
                if not body or len(body.strip()) < 80:
                    continue
                # Skip low quality
                if re.search(r"click here|subscribe|sign up|read more", body, re.I):
                    continue
                text = f"[{title}] {body}" if title else body
                results.append(text)
                if len(results) >= max_results:
                    break
    except Exception as e:
        pass
    return results


def extract_keywords(text: str, min_len: int = 4) -> list:
    """Extract meaningful keywords from text."""
    words = []
    for w in re.findall(rf"\b[a-z]{{{min_len},}}\b", text.lower()):
        if w not in _STOP_WORDS:
            words.append(w)
    # Proper nouns and acronyms
    for w in re.findall(r"\b[A-Z]{2,}\b", text):
        words.append(w.lower())
    return list(dict.fromkeys(words))


def score_paragraph(para: str, question: str) -> float:
    """Score a paragraph by keyword overlap with question."""
    q_words = set(extract_keywords(question, min_len=4))
    if not q_words:
        return 0.0
    p_lower = para.lower()
    hits = sum(1 for w in q_words if w in p_lower)
    return hits / len(q_words)


def best_paragraphs(text: str, question: str, top_n: int = 3, min_len: int = 100) -> str:
    """Extract and rank paragraphs by relevance to question."""
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= min_len]
    if not paras:
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) >= min_len]
    scored = [(score_paragraph(p, question), p) for p in paras]
    scored.sort(reverse=True)
    best = [p for sc, p in scored[:top_n] if sc > 0]
    if not best:
        best = [p for _, p in scored[:2]]
    return "\n\n".join(best)


def extract_entity(question: str) -> str:
    """Extract best search entity from question using regex (no LLM)."""
    year = _YEAR_RE.search(question)
    year_str = f" {year.group(1)}" if year else ""

    # Quoted titles first
    quoted = _QUOTED_RE.findall(question)
    if quoted:
        return max(quoted, key=len).strip() + year_str

    # Multi-word proper nouns
    proper = _PROPER_RE.findall(question)
    filtered = [p for p in proper if p.split()[0] not in TITLE_STOP]
    if filtered:
        return filtered[0].strip() + year_str

    # Single title-case words
    words = question.split()
    candidates = [
        m for m in _SINGLE_PROPER_RE.findall(question)
        if m not in TITLE_STOP and m != words[0]
    ]
    if candidates:
        return candidates[0] + year_str

    # Keyword fallback
    kws = extract_keywords(question, min_len=5)
    if kws:
        return " ".join(kws[:4]) + year_str

    return question[:70]
