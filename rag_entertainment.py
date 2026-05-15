"""RAG pipeline for the Entertainment competition."""

import re
import concurrent.futures
import urllib.parse
import requests

# ── constants ─────────────────────────────────────────────────────────────────

_ARTICLE_REF_RE = re.compile(
    r'\b(according to|as described in|as stated in|in his own words|based on|per) '
    r'(the article|the text|the passage|the excerpt)\b', re.I
)
_LOW_QUALITY_SIGNALS = [
    r'\d{4}\s*[·•]\s', r'click here', r'subscribe', r'read more',
    r'sign up', r'\.\.\.read', r'youtube\.com', r'goo\.gl',
    r'fandom\.com', r'wikia\.com',
]
_TRUSTED_DOMAINS = {
    "wikipedia.org", "britannica.com", "imdb.com", "allmusic.com",
    "rottentomatoes.com", "biography.com", "rollingstone.com", "billboard.com",
}
_STOP_WORDS = {
    "what", "when", "which", "where", "does", "have", "this",
    "that", "from", "with", "about", "into", "their", "there",
    "been", "were", "would", "could", "should", "according",
}
_TITLE_STOP_LOWER = {
    "which", "what", "how", "who", "when", "where", "why",
    "the", "this", "that", "these", "those", "a", "an",
    "is", "are", "was", "were", "has", "have", "had",
    "does", "do", "did", "will", "would", "could", "should",
    "according", "following", "best", "most", "first", "last",
    "film", "song", "show", "role", "style", "music", "band",
    "album", "movie", "book", "character", "actor", "director",
    "describes", "describe", "known", "used", "made", "played",
    "between", "during", "after", "before", "about", "into",
}
_SOURCE_WEIGHT = {"wiki": 0.7, "ddg": 1.3}   # DDG precision > Wikipedia verbosity

_RELATION_VERBS = frozenset({
    "starred", "starring", "stars", "directed", "directing", "directs",
    "released", "releasing", "wrote", "written", "writes",
    "produced", "producing", "appeared", "appearing",
    "performed", "performing", "played", "playing",
    "sang", "singing", "recorded", "recording",
    "featuring", "featured", "voiced", "voicing",
    "hosted", "hosting", "created", "creating", "won", "nominated",
})

_UL = r'A-ZÀ-ÖØ-Ý'
_AL = r'a-zA-ZÀ-ÖØ-öø-ÿ'
_PROPER_NOUN_RE   = re.compile(rf'([{_UL}][{_AL}]+(?:\s+[{_UL}][{_AL}]+)+)')
_SPECIAL_NAME_RE  = re.compile(r'[A-Za-z]+[\$!@&\.][A-Za-z]+')
_QUOTED_TITLE_RE  = re.compile(r"""(?<!\w)['"]([\w][\w\s,\.\-]{1,58}?)['"]""")
_CAMEL_CASE_RE    = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')
_QUESTION_WORDS   = re.compile(r'^(Which|What|How|Who|When|Where|Why)$')
_YEAR_RE          = re.compile(r'\b(1[0-9]{3}|2[0-9]{3})\b')
_SINGLE_PROPER_RE = re.compile(rf'\b([{_UL}][{_AL}]{{2,}})\b')
_Q_WHO            = re.compile(r'^\s*who\b', re.I)
_Q_WHAT_WORK      = re.compile(
    r'^\s*what\s+(film|movie|show|series|song|album|track|record|role)\b', re.I
)
_SENT_SPLIT_RE    = re.compile(r'(?<=[.!?])\s+')
_CITE_RE          = re.compile(r'\[\d+\]')
_WIKI_UA          = "PoliMillionaireBot/1.0 (university research project)"


# ── snippet quality ───────────────────────────────────────────────────────────

def _is_quality_snippet(text: str, url: str = "") -> bool:
    if not text:
        return False
    s, sl = text.strip(), text.strip().lower()
    if len(s) < 80:
        return False
    for p in _LOW_QUALITY_SIGNALS:
        if re.search(p, sl):
            return False
    if url:
        m = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if m and any(t in m.group(1) for t in _TRUSTED_DOMAINS):
            return True
    return len(s) >= 80 and len(sl.split()) >= 12


# ── Wikipedia ─────────────────────────────────────────────────────────────────

def _wiki(title: str) -> str:
    url = (
        "https://en.wikipedia.org/w/api.php"
        f"?action=query&prop=extracts&exintro=false&explaintext=true"
        f"&titles={urllib.parse.quote(title)}&format=json"
    )
    try:
        r = requests.get(url, headers={"User-Agent": _WIKI_UA}, timeout=4)
        if r.status_code == 200:
            pages = r.json()["query"]["pages"]
            text = _CITE_RE.sub("", pages[next(iter(pages))].get("extract", ""))
            return text[:5000] if text else ""
    except Exception:
        pass
    return ""


def _wiki_is_useful(text: str) -> bool:
    return bool(text) and len(text.strip()) >= 100 and "may refer to:" not in text.lower()


# ── keyword extraction ────────────────────────────────────────────────────────

def _extract_keywords(text: str) -> list[str]:
    """Tiered extraction: named entities first, generic topic words last."""
    seen: set[str] = set()
    kws: list[str] = []

    def _add(w: str) -> None:
        if w and w not in seen:
            seen.add(w); kws.append(w)

    for entity in _PROPER_NOUN_RE.findall(text):        # Tier 1: proper noun components
        for word in entity.split():
            wl = word.lower()
            if len(wl) >= 3 and wl not in _STOP_WORDS:
                _add(wl)
    for w in _SPECIAL_NAME_RE.findall(text):            # Tier 2: A$AP, P!nk, etc.
        _add(w.lower())
    for w in re.findall(r'\b[A-Z]{2,}\b', text):        # Tier 3: MCU, HBO, etc.
        _add(w.lower())
    for w in re.findall(r'\b(1[0-9]{3}|2[0-9]{3})\b', text):  # Tier 4: years
        _add(w)
    for w in re.findall(r'\b[a-z]{4,}\b', text.lower()):       # Tier 5: generic words
        if w not in _STOP_WORDS:
            _add(w)
    return kws


def _is_relevant(snippet: str, question: str, threshold: int = 2) -> bool:
    kws = _extract_keywords(question)
    if not kws:
        return True
    snl = snippet.lower()
    return sum(1 for kw in kws if kw in snl) >= threshold


# ── query building ────────────────────────────────────────────────────────────

def _build_wiki_query(query: str, question: str = "") -> str:
    """Pick the most useful Wikipedia entity when multiple proper nouns exist."""
    entities = _PROPER_NOUN_RE.findall(query)
    if not entities:
        return query
    if len(entities) == 1:
        return entities[0].strip()

    _ARTS = {"the", "a", "an", "of"}
    persons = [e for e in entities
               if len(e.split()) == 2 and not any(w.lower() in _ARTS for w in e.split())]
    works   = [e for e in entities if e.split()[0].lower() in _ARTS]
    others  = [e for e in entities if e not in persons and e not in works]

    if _Q_WHO.match(question):
        ranked = works or persons or others        # "Who" → search the work for its cast
    elif _Q_WHAT_WORK.match(question):
        ranked = persons or works or others        # "What film" → search the person
    else:
        ranked = works or persons or others
    return (ranked[0] if ranked else max(entities, key=len)).strip()


def _build_query(question: str) -> tuple[str, int | str]:
    ysuf = f" {m.group(1)}" if (m := _YEAR_RE.search(question)) else ""

    quoted = _QUOTED_TITLE_RE.findall(question)
    if quoted:
        title  = max(quoted, key=len).strip()
        proper = _PROPER_NOUN_RE.findall(question)
        if proper:
            entity = proper[0].strip()
            print(f"  [RAG-Ent] Relation: {entity!r} + {title!r}")
            return f"{entity} {title}{ysuf}", "1a"
        print(f"  [RAG-Ent] Title: {title!r}")
        return f"{title}{ysuf}", 1

    proper = _PROPER_NOUN_RE.findall(question)
    if proper:
        entity = proper[0].strip()
        print(f"  [RAG-Ent] Proper: {entity!r}")
        return f"{entity}{ysuf}", 2

    words       = question.split()
    possessives = re.findall(rf"\b([{_UL}][{_AL}]{{2,}})'s\b", question)
    singles     = [m for m in _SINGLE_PROPER_RE.findall(question)
                   if m.lower() not in _TITLE_STOP_LOWER and m != words[0]]
    ordered = list(dict.fromkeys(possessives + singles))
    if ordered:
        print(f"  [RAG-Ent] Single: {ordered[0]!r}")
        return f"{ordered[0]}{ysuf}", "2c"

    camel = [m for m in _CAMEL_CASE_RE.findall(question) if not _QUESTION_WORDS.match(m)]
    if camel:
        print(f"  [RAG-Ent] CamelCase: {camel[0]!r}")
        return f"{camel[0]}{ysuf}", "2b"

    kws = [kw for kw in _extract_keywords(question)
           if len(kw) >= 5 and kw not in _TITLE_STOP_LOWER]
    if len(kws) >= 2:
        q = " ".join(kws[:5])
        print(f"  [RAG-Ent] Keywords: {q!r}")
        return f"{q}{ysuf}", 3

    print("  [RAG-Ent] Raw fallback")
    return question[:80], 3


def _build_ddg_query(entity: str, question: str) -> str:
    ew    = set(entity.lower().split())
    extra = [kw for kw in _extract_keywords(question) if kw not in ew]
    return f"{entity} {' '.join(extra[:3])}".strip()[:80]


# ── fetch ─────────────────────────────────────────────────────────────────────

def _fetch_ddg(ddg_query: str, num_results: int) -> list[str]:
    results = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(ddg_query, max_results=num_results, timeout=3):
                body, url = r.get("body", ""), r.get("href", "")
                if _is_quality_snippet(body, url):
                    title = r.get("title", "")
                    results.append(f"[{title}]{body}" if title else body)
    except Exception as exc:
        print(f"  [RAG-Ent] DDG error: {exc}")
    return results


# ── scoring ───────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) > 20]


def _score_sentence(words: list[str], cand_kws: list[str], entity_kws: list[str]) -> float:
    """Returns 0 if assertion gate fails (either token absent in sentence)."""
    cand_pos   = [i for i, w in enumerate(words) if any(kw in w for kw in cand_kws)]
    entity_pos = [i for i, w in enumerate(words)
                  if any(kw in w for kw in entity_kws)] if entity_kws else []
    if not cand_pos or (entity_kws and not entity_pos):
        return 0.0
    if entity_pos:
        lo, hi = min(min(cand_pos), min(entity_pos)), max(max(cand_pos), max(entity_pos))
        vb     = min(sum(1 for w in words[lo:hi+1] if w in _RELATION_VERBS) * 0.75, 1.5)
        prox   = max(0.0, (10 - min(abs(c-e) for c in cand_pos for e in entity_pos)) / 10.0)
    else:
        vb, prox = min(sum(1 for w in words if w in _RELATION_VERBS) * 0.75, 1.5), 0.0
    return 1.0 + vb + prox + (0.3 if cand_pos[0] <= 3 else 0.0)


def _relation_score(snippet: str, candidate: str, entity: str) -> float:
    """Normalized score: mean_quality × (1 + density). Dense DDG facts beat sparse Wiki dumps."""
    cand_kws, entity_kws = _extract_keywords(candidate), _extract_keywords(entity)
    if not cand_kws:
        return 0.0
    sentences = _split_sentences(snippet)
    if not sentences:
        return 0.0
    scores    = [_score_sentence(s.lower().split(), cand_kws, entity_kws) for s in sentences]
    asserting = [s for s in scores if s > 0.0]
    if not asserting:
        return 0.0
    return (sum(asserting) / len(asserting)) * (1.0 + len(asserting) / len(sentences))


# ── main entry point ──────────────────────────────────────────────────────────

def rag_entertainment(query: str, num_results: int = 3,
                      generate_answer_fn=None, option_texts: list = None) -> str:
    """Candidate-centric RAG: evidence bucketed and scored per option independently."""
    if _ARTICLE_REF_RE.search(query):
        print("  [RAG-Ent] Article-reference — skipping.")
        return ""

    base_query, priority = _build_query(query)
    wiki_query = _build_wiki_query(base_query, question=query)
    ddg_query  = _build_ddg_query(wiki_query, query) if priority in (2, "2c", "2b") else base_query
    print(f"  [RAG-Ent] P{priority} wiki={wiki_query!r} ddg={ddg_query!r}")

    n_opts       = min(len(option_texts), 4) if option_texts else 0
    cand_queries = [f"{option_texts[i].strip()[:35]} {wiki_query}"[:80] for i in range(n_opts)]

    pool      = concurrent.futures.ThreadPoolExecutor(max_workers=2 + n_opts)
    wiki_fut  = pool.submit(_wiki, wiki_query)
    ddg_fut   = pool.submit(_fetch_ddg, ddg_query, num_results)
    cand_futs = [pool.submit(_fetch_ddg, cq, 1) for cq in cand_queries]

    try:    wiki_result = wiki_fut.result(timeout=4)
    except Exception: wiki_result = ""; print("  [RAG-Ent] Wikipedia timed out.")
    try:    ddg_results = ddg_fut.result(timeout=4)
    except Exception: ddg_results = []; print("  [RAG-Ent] DDG timed out.")

    cand_raw: list[list[str]] = []
    for i, fut in enumerate(cand_futs):
        try:
            hits = fut.result(timeout=3)
            cand_raw.append(hits)
            if hits: print(f"  [RAG-Ent] Cand[{i}] hit: {cand_queries[i]!r}")
        except Exception:
            cand_raw.append([])
    pool.shutdown(wait=False)

    # Build global snippet pool — tagged by source for downstream weighting
    seen_g: set[str] = set()
    global_snippets: list[tuple[str, str]] = []   # (source, text)

    if _wiki_is_useful(wiki_result):
        seen_g.add(wiki_result)
        global_snippets.append(("wiki", wiki_result))
        print(f"  [RAG-Ent] Wikipedia hit ({len(wiki_result)} chars).")
    for text in ddg_results:
        if text and text not in seen_g:
            seen_g.add(text); global_snippets.append(("ddg", text))

    # Flat fallback when options not provided
    if not n_opts:
        relevant = [t for _, t in global_snippets if _is_relevant(t, query)]
        return "\n\n".join(relevant or [t for _, t in global_snippets[:2]])[:1500]

    # Isolated per-candidate buckets — each entry is (source, text)
    buckets:     dict[int, list[tuple[str, str]]] = {i: [] for i in range(n_opts)}
    seen_bucket: set[tuple]                        = set()

    def _put(i: int, src: str, text: str) -> None:
        k = (i, text[:60])
        if k not in seen_bucket:
            seen_bucket.add(k); buckets[i].append((src, text))

    for i, hits in enumerate(cand_raw):
        for s in hits: _put(i, "ddg", s)

    for src, snippet in global_snippets:
        snl = snippet.lower()
        for i in range(n_opts):
            opt_kws = _extract_keywords(option_texts[i])
            if opt_kws and any(kw in snl for kw in opt_kws):
                _put(i, src, snippet)

    # Score buckets — DDG snippets weighted higher than Wikipedia
    scores = {
        i: sum(
            _relation_score(text, option_texts[i], wiki_query) * _SOURCE_WEIGHT[src]
            for src, text in buckets[i]
        )
        for i in range(n_opts)
    }
    sorted_opts = sorted(range(n_opts), key=lambda i: scores[i], reverse=True)
    print(f"  [RAG-Ent] Scores: { {i: f'{scores[i]:.1f}' for i in sorted_opts} }")

    # Format structured context
    parts: list[str] = []
    shared = [t for _, t in global_snippets if _is_relevant(t, query)]
    if shared:
        parts.append(f"CONTEXT:\n{shared[0][:500]}")
    for i in sorted_opts:
        label = f"[{i}] {option_texts[i]}"
        parts.append(f"{label}:\n{buckets[i][0][1][:300]}" if buckets[i]
                     else f"{label}: (no evidence)")

    return "\n\n".join(parts)[:2000]
