"""
Pipeline:
  1. Extract subjects from the question (named entities / proper nouns).
  2. Fetch Wikipedia (main subject) + DuckDuckGo (per-option queries) in parallel.
  3. Assemble deduplicated context and return it for the LLM to reason over.
"""

import re
import concurrent.futures
import urllib.parse
import requests
from functools import lru_cache

_WIKI_UA = "QuizBot/1.0 (research)"
_TIMEOUT = 4

_STOP_WORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "own",
    "film", "movie", "song", "show", "album", "band", "role", "character",
    "single", "track", "series", "actor", "actress", "director", "article",
}

_GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
_GLINER_LABELS = [
    "movie", "film", "TV show", "TV series",
    "person", "actor", "musician", "director",
    "band", "music group",
    "album", "song",
    "character",
]
# Lower index = higher priority as Wikipedia search anchor
_GLINER_LABEL_PRIORITY: dict[str, int] = {
    "movie": 0, "film": 0, "TV show": 0, "TV series": 0,
    "album": 1, "song": 1,
    "person": 2, "actor": 2, "musician": 2, "director": 2,
    "band": 2, "music group": 2,
    "character": 3,
}
_TITLE_LABELS  = frozenset({"movie", "film", "TV show", "TV series"})
_PERSON_LABELS = frozenset({"person", "actor", "musician", "director", "band", "music group"})

_QUOTED_RE        = re.compile(r"""['\"‘’“”]([\w][\w\s,.\-&!]{1,58}?)['\"‘’“”]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')
_TOKEN_RE         = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_CITE_RE          = re.compile(r"\[\d+\]")
_SECTION_HEADER   = re.compile(r"^=+\s*[^=]+\s*=+$")

# GLiNER lazy singleton
_gliner_model       = None
_gliner_model_tried = False


def _get_gliner_model():
    global _gliner_model, _gliner_model_tried
    if _gliner_model_tried:
        return _gliner_model
    _gliner_model_tried = True
    try:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
    except Exception as e:
        print(f"  [RAG] GLiNER model unavailable: {e}")
    return _gliner_model


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}


def _clean_query_text(text: str) -> str:
    kept = [w for w in text.split() if w.lower().rstrip(".,!?:;'\"") not in _STOP_WORDS]
    return " ".join(kept) if kept else text


def _extract_subjects_regex(question: str) -> list[str]:
    """Quoted titles first, then multi-word proper nouns, then single proper nouns."""
    subjects: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        s = raw.strip()
        sl = s.lower()
        if s and sl not in seen and sl not in _STOP_WORDS:
            seen.add(sl)
            subjects.append(s)

    for q in _QUOTED_RE.findall(question):
        _add(q)
    for m in _PROPER_MULTI_RE.findall(question):
        _add(m)
    for w in question.split()[1:]:
        clean = re.sub(r"[^\w]+$", "", w)
        if (_PROPER_SINGLE_RE.match(clean)
                and clean.lower() not in seen
                and not any(clean in s.split() for s in subjects)):
            _add(clean)

    return subjects


def _extract_subjects_gliner(question: str) -> list[tuple[str, str]]:
    """Returns [(text, label), ...] sorted by label priority, deduped."""
    model = _get_gliner_model()
    if model is None:
        return []
    try:
        entities = model.predict_entities(question, _GLINER_LABELS, threshold=0.5)
        entities.sort(key=lambda e: (_GLINER_LABEL_PRIORITY.get(e["label"], 99), e["start"]))
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for e in entities:
            text = e["text"].strip()
            tl = text.lower()
            if text and tl not in seen and tl not in _STOP_WORDS:
                seen.add(tl)
                result.append((text, e["label"]))
        return result
    except Exception:
        return []


def _pick_main_term(labeled: list[tuple[str, str]]) -> str:
    """Title entities first, then person/band, then first entity."""
    for preferred in (_TITLE_LABELS, _PERSON_LABELS):
        for text, label in labeled:
            if label in preferred:
                return text
    return labeled[0][0] if labeled else ""


@lru_cache(maxsize=64)
def _wiki_lookup(query: str) -> str:
    try:
        search_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&list=search&srsearch={urllib.parse.quote(query)}"
            f"&srlimit=3&srnamespace=0&format=json"
        )
        resp = requests.get(search_url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return ""
        results = resp.json().get("query", {}).get("search", [])
        if not results:
            return ""
        candidates = [item["title"] for item in results]
        title = next(
            (c for c in candidates if "disambiguation" not in c.lower()),
            candidates[0],
        )
        extract_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=false&explaintext=true"
            f"&titles={urllib.parse.quote(title)}&format=json"
        )
        resp = requests.get(extract_url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT)
        if resp.status_code != 200:
            return ""
        pages = resp.json()["query"]["pages"]
        text = next(iter(pages.values())).get("extract", "")
        text = _CITE_RE.sub("", text)
        return text if "may refer to:" not in text.lower() else ""
    except Exception:
        return ""


def _wiki_relevant_passages(wiki_text: str, question: str, max_chars: int = 1500) -> str:
    if not wiki_text:
        return ""
    paragraphs = [
        p.strip() for p in re.split(r"\n+", wiki_text)
        if len(p.strip()) > 50 and not _SECTION_HEADER.match(p.strip())
    ]
    if not paragraphs:
        return wiki_text[:max_chars]
    q_kws = _keywords(question)
    if not q_kws:
        return paragraphs[0][:max_chars]

    intro  = paragraphs[0]
    scored = sorted(
        ((len(q_kws & set(_tokenize(p))), p) for p in paragraphs[1:]),
        key=lambda x: x[0],
        reverse=True,
    )
    out: list[str] = [intro]
    budget = max_chars - len(intro)
    for score, p in scored:
        if score < 2 or budget <= 100:
            break
        snippet = p if len(p) <= budget else p[:budget].rsplit(" ", 1)[0] + "…"
        out.append(snippet)
        budget -= len(snippet) + 2

    return "\n\n".join(out)


@lru_cache(maxsize=128)
def _ddg_lookup(query: str, max_results: int = 2) -> list[str]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            out = []
            for r in ddgs.text(query, max_results=max_results, timeout=_TIMEOUT):
                body  = r.get("body", "")
                title = r.get("title", "")
                if body and len(body) >= 60:
                    out.append(f"{title}. {body}" if title else body)
            return out
    except Exception as e:
        print(f"  [RAG] DDG error: {e}")
        return []


def rag_entertainment(query: str, num_results: int = 3, option_texts: list = None) -> str:
    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = _pick_main_term(labeled)
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""

    if not main_term:
        kws = [w for w in _tokenize(query) if len(w) >= 4 and w not in _STOP_WORDS]
        main_term = " ".join(kws[:4]) if kws else query[:60]

    subj_str = " ".join(subjects[:2]) if subjects else main_term

    if not option_texts:
        wiki = _wiki_relevant_passages(_wiki_lookup(main_term), query, max_chars=1200)
        ddg  = _ddg_lookup(subj_str, num_results)
        return "\n\n".join(filter(None, [wiki, *ddg]))[:1500]

    n_opts = min(len(option_texts), 4)
    cand_queries = [
        f"{_clean_query_text(option_texts[i])[:50]} {subj_str}".strip()[:90]
        for i in range(n_opts)
    ]

    pool        = concurrent.futures.ThreadPoolExecutor(max_workers=n_opts + 2)
    wiki_fut    = pool.submit(_wiki_lookup, main_term)
    general_fut = pool.submit(_ddg_lookup, f"{subj_str} {query[:50]}".strip()[:90], num_results)
    opt_futs    = [pool.submit(_ddg_lookup, q, 2) for q in cand_queries]

    def _safe(fut, default):
        try:
            return fut.result(timeout=_TIMEOUT + 1)
        except Exception:
            return default

    wiki_full    = _safe(wiki_fut, "")
    general_snip = _safe(general_fut, [])
    opt_snips    = [_safe(f, []) for f in opt_futs]
    pool.shutdown(wait=False)

    wiki_text = _wiki_relevant_passages(wiki_full, query, max_chars=1400)

    all_snippets: list[str] = []
    seen_key: set[str] = set()
    for s in filter(None, [wiki_text, *general_snip, *(s for snips in opt_snips for s in snips)]):
        k = s[:120]
        if k not in seen_key:
            seen_key.add(k)
            all_snippets.append(s)

    return "\n\n".join(all_snippets)[:2500]