"""
Entertainment RAG: Wikipedia (cached) + DDG article fetch → LLM prompt.
Subject extraction uses GLiNER with entertainment-specific entity labels.
"""

import re
import concurrent.futures
import urllib.parse
import html as _html_module
import requests

_WIKI_UA           = "QuizBot/1.0 (research)"
_TIMEOUT           = 4
_ARTICLE_MAX_CHARS = 7000
_MIN_ARTICLE_CHARS = 300
_MAX_DDG_RESULTS   = 6

_SKIP_DOMAINS: frozenset[str] = frozenset({
    "youtube.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "reddit.com",
    "vk.com",
    "t.me",
})

_SKIP_URL_PATTERNS: frozenset[str] = frozenset({
    "/tag/", "/tags/", "/category/", "/categories/",
    "/topic/", "/topics/", "/search/", "/archive/",
})

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

_QUOTED_RE        = re.compile(r"""['\"''“”]([\w][\w\s,.\-&!]{1,58}?)['\"''“”]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')
_YEAR_RE          = re.compile(r'\b(?:19|20)\d{2}s?\b')
_TOKEN_RE         = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_CITE_RE          = re.compile(r"\[\d+\]")
_SECTION_HEADER   = re.compile(r"^=+\s*[^=]+\s*=+$")
_BLOCK_TAG_RE     = re.compile(
    r'<(script|style|nav|header|footer|aside|noscript)[^>]*>.*?</\1>',
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE   = re.compile(r'<[^>]+>')
_WS_RE    = re.compile(r'\s+')
_P_TAG_RE = re.compile(r'<p[^>]*>(.*?)</p>', re.DOTALL | re.IGNORECASE)

# GLiNER lazy singleton
_gliner_model       = None
_gliner_model_tried = False


# ---------------------------------------------------------------------------
# GLiNER
# ---------------------------------------------------------------------------

def _get_gliner_model():
    global _gliner_model, _gliner_model_tried
    if _gliner_model_tried:
        return _gliner_model
    _gliner_model_tried = True
    try:
        from gliner import GLiNER
        import torch
        _gliner_model = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
        _gliner_model.eval()
        torch.set_grad_enabled(False)
    except Exception as e:
        print(f"  [RAG] GLiNER model unavailable: {e}")
    return _gliner_model


def set_entertainment_rag() -> None:
    """Eagerly load GLiNER. Idempotent."""
    global _gliner_model_tried
    if _gliner_model_tried:
        return
    print("[Entertainment RAG] Loading GLiNER model...")
    _get_gliner_model()
    if _gliner_model is not None:
        print(f"[Entertainment RAG] GLiNER ready ({_GLINER_MODEL_NAME}).")
    else:
        print("[Entertainment RAG] GLiNER unavailable; will fall back to regex extraction.")


# ---------------------------------------------------------------------------
# Subject extraction
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}



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
        import torch
        with torch.no_grad():
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


# ---------------------------------------------------------------------------
# HTML stripping
# ---------------------------------------------------------------------------

def _strip_html(raw: str) -> str:
    text = _BLOCK_TAG_RE.sub(" ", raw)
    text = _TAG_RE.sub(" ", text)
    text = _html_module.unescape(text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _extract_body(raw_html: str, max_chars: int = _ARTICLE_MAX_CHARS) -> str:
    paragraphs = _P_TAG_RE.findall(raw_html)
    body_parts: list[str] = []
    total = 0

    for p in paragraphs:
        clean = _WS_RE.sub(" ", _TAG_RE.sub("", _html_module.unescape(p))).strip()
        if len(clean) >= 80:
            body_parts.append(clean)
            total += len(clean) + 1
            if total >= max_chars:
                break

    if body_parts:
        return " ".join(body_parts)[:max_chars]

    return _strip_html(raw_html)[:max_chars]


# ---------------------------------------------------------------------------
# Wikipedia (cached — entertainment facts are stable)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# DDG search + article fetch
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = _MAX_DDG_RESULTS) -> list[str]:
    try:
        from ddgs import DDGS
        urls = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, timeout=_TIMEOUT):
                url = r.get("href") or r.get("url", "")
                if url:
                    urls.append(url)
        return urls
    except Exception as e:
        print(f"  [RAG-Entertainment] DDG error: {e}")
        return []


def _fetch_article(url: str) -> str:
    try:
        r = requests.get(
            url,
            timeout=(_TIMEOUT, _TIMEOUT),
            headers={"User-Agent": "Mozilla/5.0 (compatible; EntertainmentBot/1.0)"},
        )
        if r.status_code != 200:
            return ""
        return _extract_body(r.text)
    except Exception:
        return ""


def _search_and_fetch(query: str, q_keywords: set[str], target: int = 3) -> list[tuple[str, str]]:
    results = []
    for url in _ddg_search(query):
        if len(results) >= target:
            break
        if any(domain in url for domain in _SKIP_DOMAINS):
            print(f"  [Entertainment] skipping: {url[:80]}")
            continue
        if any(pat in url for pat in _SKIP_URL_PATTERNS):
            print(f"  [Entertainment] skipping: {url[:80]}")
            continue
        text = _fetch_article(url)
        if text and len(text) >= _MIN_ARTICLE_CHARS and any(kw in text.lower() for kw in q_keywords):
            results.append((text, url))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_entertainment_rag() -> None:
    """Alias kept for callers that use the newer name."""
    set_entertainment_rag()


def rag_entertainment(query: str, option_texts: list = None) -> str:
    # --- subject extraction ---
    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = _pick_main_term(labeled)
        print(f"  [Entertainment] GLiNER entities: {labeled}")
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""
        print(f"  [Entertainment] regex subjects: {subjects}")

    if not main_term:
        kws = [w for w in _tokenize(query) if len(w) >= 4 and w not in _STOP_WORDS]
        main_term = " ".join(kws[:4]) if kws else query[:60]

    # --- keyword sets for scoring ---
    q_keywords = {
        t for t in _tokenize(query)
        if len(t) >= 4 and t not in _STOP_WORDS
    }
    option_kw = {
        t for opt in (option_texts or [])
        for t in _tokenize(opt)
        if len(t) >= 3 and t not in _STOP_WORDS
    }

    # --- extract dates from query ---
    dates = _YEAR_RE.findall(query)
    date_str = " ".join(dict.fromkeys(dates))
    if date_str:
        print(f"  [Entertainment] dates: {date_str!r}")

    # --- build queries ---
    subject_tokens  = {s.lower() for s in subjects}
    q_content_words = [
        t for t in _tokenize(query)
        if len(t) >= 5 and t not in _STOP_WORDS and t not in subject_tokens
    ]
    q1 = " ".join(filter(None, [main_term, date_str]))
    q2 = " ".join(filter(None, [main_term, " ".join(q_content_words[:3]), date_str]))

    queries = list(dict.fromkeys(q for q in [q1, q2] if q.strip()))
    print(f"  [Entertainment] queries: {queries}")

    # --- parallel: Wikipedia + DDG article fetch ---
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries) + 1) as pool:
        wiki_fut  = pool.submit(_wiki_lookup, main_term)
        fetch_futs = {
            pool.submit(_search_and_fetch, q, q_keywords): q for q in queries
        }

        wiki_full = ""
        try:
            wiki_full = wiki_fut.result(timeout=_TIMEOUT + 1)
        except Exception:
            pass

        candidates: list[tuple[int, str, str]] = []
        seen_candidate_urls: set[str] = set()
        for fut in concurrent.futures.as_completed(fetch_futs):
            try:
                results = fut.result(timeout=_TIMEOUT + 2)
            except Exception:
                continue
            for text, url in results:
                if url in seen_candidate_urls:
                    print(f"  [Entertainment] duplicate skipped: {url[:80]}")
                    continue
                seen_candidate_urls.add(url)
                score = (
                    sum(2 for kw in option_kw  if kw in text.lower()) +
                    sum(1 for kw in q_keywords if kw in text.lower())
                )
                candidates.append((score, text, url))
                print(f"  [Entertainment] candidate (score={score}, {len(text)} chars): {url[:80]}")

    # --- select best article ---
    wiki_text = _wiki_relevant_passages(wiki_full, query, max_chars=1400)

    article_text = ""
    article_url  = ""
    if candidates:
        # tiebreak: same score → prefer non-Wikipedia (Wikipedia already in wiki_text)
        candidates.sort(key=lambda x: (x[0], "wikipedia.org" not in x[2]), reverse=True)
        _, article_text, article_url = candidates[0]
        print(f"  [Entertainment] selected article: {article_url[:80]}")
    else:
        print("  [Entertainment] No article found — Wikipedia only")

    # --- assemble structured prompt ---
    lines: list[str] = []
    if wiki_text:
        lines += [f"WIKIPEDIA (source: https://en.wikipedia.org/wiki/{urllib.parse.quote(main_term)}):", wiki_text, ""]
    if article_text:
        lines += [f"ARTICLE (source: {article_url}):", article_text, ""]

    if not lines:
        print("  [Entertainment] No context found — LLM fallback")
        return ""

    lines += [f"QUESTION: {query}", ""]
    if option_texts:
        lines.append("OPTIONS:")
        for i, opt in enumerate(option_texts[:4]):
            lines.append(f"[{i}] {opt}")
        lines.append("")
    lines.append("Answer (0/1/2/3):")

    return "\n".join(lines)
