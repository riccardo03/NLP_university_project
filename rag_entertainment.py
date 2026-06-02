"""
Entertainment RAG: Wikipedia (cached) + DDG article fetch → LLM prompt.
Subject extraction uses GLiNER with entertainment-specific entity labels.
"""

import re
import concurrent.futures
import urllib.parse

import requests

from rag_utils import (
    STOP_WORDS_BASE,
    search_and_fetch,
    setup_gliner, extract_subjects_gliner, extract_subjects_regex,
)

_WIKI_UA           = "QuizBot/1.0 (research)"
_TIMEOUT           = 4
_MIN_ARTICLE_CHARS = 300
_MAX_DDG_RESULTS   = 6

_STOP_WORDS: frozenset[str] = STOP_WORDS_BASE | {
    "own", "film", "movie", "song", "show", "album", "band", "role", "character",
    "single", "track", "series", "actor", "actress", "director",
}

_GLINER_LABELS = [
    "movie", "film", "TV show", "TV series",
    "person", "actor", "musician", "director",
    "band", "music group",
    "album", "song",
    "character",
]
_GLINER_LABEL_PRIORITY: dict[str, int] = {
    "movie": 0, "film": 0, "TV show": 0, "TV series": 0,
    "album": 1, "song": 1,
    "person": 2, "actor": 2, "musician": 2, "director": 2,
    "band": 2, "music group": 2,
    "character": 3,
}
_TITLE_LABELS  = frozenset({"movie", "film", "TV show", "TV series"})
_PERSON_LABELS = frozenset({"person", "actor", "musician", "director", "band", "music group"})

# entertainment needs $!& for names like P!nk, R&B, $uicideboy$
_TOKEN_RE       = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_YEAR_RE        = re.compile(r'\b(?:19|20)\d{2}s?\b')
_CITE_RE        = re.compile(r"\[\d+\]")
_SECTION_HEADER = re.compile(r"^=+\s*[^=]+\s*=+$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}



def _pick_main_term(labeled: list[tuple[str, str]]) -> str:
    """Title entities first, then person/band, then first entity."""
    for preferred in (_TITLE_LABELS, _PERSON_LABELS):
        for text, label in labeled:
            if label in preferred:
                return text
    return labeled[0][0] if labeled else ""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def set_entertainment_rag() -> None:
    """Eagerly load GLiNER. Idempotent."""
    setup_gliner()


def setup_entertainment_rag() -> None:
    """Alias kept for callers that use the newer name."""
    set_entertainment_rag()


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
# DDG article fetch
# ---------------------------------------------------------------------------

def _search_and_fetch(query: str, q_keywords: set[str], target: int = 3) -> list[tuple[str, str]]:
    return search_and_fetch(
        query, q_keywords,
        max_results=_MAX_DDG_RESULTS, target=target, min_chars=_MIN_ARTICLE_CHARS,
        timeout=_TIMEOUT, user_agent="EntertainmentBot/1.0", module_tag="Entertainment",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rag_entertainment(query: str, option_texts: list = None) -> str:
    # --- subject extraction ---
    labeled = extract_subjects_gliner(query, _GLINER_LABELS, _GLINER_LABEL_PRIORITY)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = _pick_main_term(labeled)
        print(f"  [Entertainment] GLiNER entities: {labeled}")
    else:
        subjects  = extract_subjects_regex(query, _STOP_WORDS)
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
        wiki_fut   = pool.submit(_wiki_lookup, main_term)
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
