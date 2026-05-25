"""
Entertainment RAG pipeline.

  1. Extract named entities from the question (GLiNER, regex fallback).
  2. Classify the question type (biography vs factual) to build semantic queries.
  3. Fetch up to 2 Wikipedia pages + DuckDuckGo snippets in parallel.
  4. Score options via BM25 + negation-aware polarity over a shared snippet pool.
  5. Return a structured evidence block for the LLM to reason over.
"""

import re
import concurrent.futures
import urllib.parse
import requests
from dataclasses import dataclass
from functools import lru_cache
from rank_bm25 import BM25Okapi

_WIKI_UA = "QuizBot/1.0 (research)"
_TIMEOUT = 4

# Pure linguistic noise — stop in both keyword extraction and query cleaning.
_STOP_WORDS_BASE: frozenset[str] = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "as", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "when", "where", "why", "how", "does", "do", "did",
    "has", "have", "had", "will", "would", "could", "should", "can", "may",
    "this", "that", "these", "those", "their", "there", "according", "following",
    "describes", "describe", "best", "most", "called", "named", "own", "article",
})

# Domain terms: meaningful in search queries but too generic for keyword extraction.
_DOMAIN_TERMS: frozenset[str] = frozenset({
    "film", "movie", "song", "show", "album", "band", "role",
    "character", "single", "track", "series", "actor", "actress", "director",
})

# Used by _keywords() — suppresses domain terms so entity names dominate.
_STOP_WORDS: frozenset[str] = _STOP_WORDS_BASE | _DOMAIN_TERMS

# Used by _clean_query_text() — keeps domain terms so queries stay specific.
_STOP_WORDS_QUERY: frozenset[str] = _STOP_WORDS_BASE

_GLINER_MODEL_NAME = "urchade/gliner_medium-v2.1"
_GLINER_LABELS = [
    "movie", "film", "TV show", "TV series",
    "person", "actor", "musician", "director",
    "band", "music group",
    "album", "song",
    "family member", "historical person",
    "character",
]
# Lower index = higher priority as Wikipedia search anchor.
_GLINER_LABEL_PRIORITY: dict[str, int] = {
    "movie": 0, "film": 0, "TV show": 0, "TV series": 0,
    "album": 1, "song": 1,
    "person": 2, "actor": 2, "musician": 2, "director": 2,
    "band": 2, "music group": 2,
    "family member": 2, "historical person": 2,
    "character": 3,
}
_TITLE_LABELS  = frozenset({"movie", "film", "TV show", "TV series"})
_PERSON_LABELS = frozenset({
    "person", "actor", "musician", "director",
    "band", "music group", "family member", "historical person",
})

_QUOTED_RE        = re.compile(r"""['\"''""]([\w][\w\s,.\-&!]{1,58}?)['\"''""]""")
_PROPER_MULTI_RE  = re.compile(r'\b[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+(?:\s+[A-ZÀ-Ý][a-zA-ZÀ-ÿ]+)+\b')
_PROPER_SINGLE_RE = re.compile(r'^[A-ZÀ-Ý][a-zA-ZÀ-ÿ]{2,}$')
_TOKEN_RE         = re.compile(r"[a-zA-ZÀ-ÿ0-9$!&]+")
_CITE_RE          = re.compile(r"\[\d+\]")
_SECTION_HEADER   = re.compile(r"^=+\s*[^=]+\s*=+$")

_NOT_EXCEPT_RE = re.compile(
    r'\b(NOT|EXCEPT|least|never|false|incorrect|wrong)\b',
    re.IGNORECASE,
)

# Individual tokens (post-_tokenize) that carry negation signal.
# Contracted forms like "didn't" tokenize to ["didn", "t"] and are excluded.
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not", "never", "no", "nor", "neither",
    "denied", "failed", "refused", "unrelated", "unconnected",
})

# max(votes) below this threshold → no reliable retrieval signal; rank hints suppressed.
_VOTE_CONFIDENCE_MIN: float = 0.5

# Relationship/life-event words that signal a biography question.
_BIO_KEYWORDS_RE = re.compile(
    r"\b(father|mother|parent|sibling|brother|sister|wife|husband|"
    r"relation|relationship|childhood|born|raised|family|married|"
    r"early life|upbringing|ancestry)\b",
    re.IGNORECASE,
)

# GLiNER singleton — populated by setup_entertainment_rag().
_gliner_model: object = None


@dataclass(frozen=True)
class WeightedQuery:
    text:     str
    weight:   float
    strategy: str


def setup_entertainment_rag() -> None:
    global _gliner_model
    print("  [RAG-Entertainment] Loading GLiNER model...")
    try:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained(_GLINER_MODEL_NAME)
        print("  [RAG-Entertainment] GLiNER ready.")
    except Exception as e:
        print(f"  [RAG-Entertainment] GLiNER unavailable: {e}")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _keywords(text: str) -> set[str]:
    return {t for t in _tokenize(text) if len(t) >= 3 and t not in _STOP_WORDS}


def _clean_query_text(text: str) -> str:
    kept = [w for w in text.split()
            if w.lower().rstrip(".,!?:;'\"") not in _STOP_WORDS_QUERY]
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
    if _gliner_model is None:
        return []
    try:
        # Primary pass: all entity types at standard confidence.
        entities = _gliner_model.predict_entities(question, _GLINER_LABELS, threshold=0.45)

        # Secondary pass: lower threshold for person-type labels to catch family
        # members and secondary characters that primary pass misses.
        person_labels = [
            "person", "actor", "musician", "director",
            "family member", "historical person",
        ]
        secondary     = _gliner_model.predict_entities(question, person_labels, threshold=0.30)
        primary_texts = {e["text"].strip().lower() for e in entities}
        for e in secondary:
            if e["text"].strip().lower() not in primary_texts:
                entities.append(e)

        entities.sort(key=lambda e: (_GLINER_LABEL_PRIORITY.get(e["label"], 99), e["start"]))
        seen: set[str] = set()
        result: list[tuple[str, str]] = []
        for e in entities:
            text = e["text"].strip()
            tl   = text.lower()
            if text and tl not in seen and tl not in _STOP_WORDS:
                seen.add(tl)
                result.append((text, e["label"]))
        return result
    except Exception:
        return []


def _extract_option_entity(option: str) -> str:
    """Single primary-pass GLiNER on one option string.

    Returns the highest-priority entity text, or '' when GLiNER is unavailable
    or the option contains no recognisable entity (e.g. 'Indifferent and distant').
    """
    if _gliner_model is None:
        return ""
    try:
        entities = _gliner_model.predict_entities(option, _GLINER_LABELS, threshold=0.35)
        if not entities:
            return ""
        entities.sort(key=lambda e: (_GLINER_LABEL_PRIORITY.get(e["label"], 99), e["start"]))
        return entities[0]["text"].strip()
    except Exception:
        return ""


def _pick_main_term(labeled: list[tuple[str, str]]) -> str:
    persons = [(t, l) for t, l in labeled if l in _PERSON_LABELS]
    titles  = [(t, l) for t, l in labeled if l in _TITLE_LABELS]
    # When multiple titles exist, the person is a better disambiguation anchor.
    if persons and len(titles) > 1:
        return persons[0][0]
    for preferred in (_TITLE_LABELS, _PERSON_LABELS):
        for text, label in labeled:
            if label in preferred:
                return text
    return labeled[0][0] if labeled else ""


def _pick_wiki_anchors(labeled: list[tuple[str, str]]) -> list[str]:
    """Returns 1-2 Wikipedia search terms: prefer one TITLE + one PERSON entity."""
    titles  = [t for t, l in labeled if l in _TITLE_LABELS]
    persons = [t for t, l in labeled if l in _PERSON_LABELS]

    anchors: list[str] = []
    if titles:
        anchors.append(titles[0])
    if persons:
        anchors.append(persons[0])

    if len(anchors) < 2 and labeled:
        for text, _ in labeled:
            if text not in anchors:
                anchors.append(text)
                break

    return anchors[:2]


@lru_cache(maxsize=64)
def _wiki_lookup(query: str) -> str:
    try:
        # Step 1: resolve the best matching article title.
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

        # Step 2: fetch the full article text (up to 8000 chars).
        extract_url = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&prop=extracts&exintro=false&explaintext=true"
            f"&exchars=8000&titles={urllib.parse.quote(title)}&format=json"
        )
        resp = requests.get(extract_url, headers={"User-Agent": _WIKI_UA}, timeout=_TIMEOUT + 2)
        if resp.status_code != 200:
            return ""
        pages = resp.json()["query"]["pages"]
        text  = next(iter(pages.values())).get("extract", "")
        text  = _CITE_RE.sub("", text)
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
        snippet = p if len(p) <= budget else p[:budget].rsplit(" ", 1)[0] + "..."
        out.append(snippet)
        budget -= len(snippet) + 2

    return "\n\n".join(out)


def _fetch_multi_wiki(anchors: list[str], question: str, max_chars: int = 600) -> str:
    """Fetch up to 2 Wikipedia pages concurrently, tagged with [WIKI: title] headers."""
    if not anchors:
        return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futs = {pool.submit(_wiki_lookup, a): a for a in anchors}
        results: dict[str, str] = {}
        for fut, anchor in futs.items():
            try:
                results[anchor] = fut.result(timeout=_TIMEOUT + 1)
            except Exception:
                results[anchor] = ""

    parts = []
    for anchor in anchors:
        raw = results.get(anchor, "")
        if not raw:
            continue
        passage = _wiki_relevant_passages(raw, question, max_chars=max_chars)
        if passage:
            parts.append(f"[WIKI: {anchor}]\n{passage}")

    return "\n\n".join(parts)


@lru_cache(maxsize=128)
def _ddg_lookup(query: str, max_results: int = 2) -> tuple[str, ...]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            out: list[str] = []
            for r in ddgs.text(query, max_results=max_results, timeout=_TIMEOUT):
                body  = r.get("body", "")
                title = r.get("title", "")
                if body and len(body) >= 60:
                    out.append(f"{title}. {body}" if title else body)
            return tuple(out)
    except Exception as e:
        print(f"  [RAG] DDG error: {e}")
        return ()


def _bm25_rank(corpus: list[str], query: str, top_k: int = 3) -> list[tuple[float, str]]:
    if not corpus:
        return []
    tokenized = [_tokenize(doc) for doc in corpus]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(_tokenize(query))
    ranked    = sorted(zip(scores, corpus), reverse=True)
    return ranked[:top_k]


def _snippet_polarity(snippet: str, option: str) -> float:
    """Returns -1.0 if a negation token appears within 8 positions of an option keyword, else +1.0."""
    tokens  = _tokenize(snippet)
    opt_kws = _keywords(option)
    for i, tok in enumerate(tokens):
        if tok in opt_kws:
            window = tokens[max(0, i - 8): i + 8]
            if any(w in _NEGATION_TOKENS for w in window):
                return -1.0
    return 1.0


def _extract_bio_keywords(question: str) -> str:
    """Extract relationship/context words + named persons from a biography question."""
    matches = _BIO_KEYWORDS_RE.findall(question)
    named   = _PROPER_MULTI_RE.findall(question)
    parts   = list(dict.fromkeys(matches + named))  # deduplicated, order-preserved
    return " ".join(parts[:4])


def _detect_question_type(question: str) -> str:
    """Returns 'biography' if the question is about personal life/relationships, else 'factual'."""
    return "biography" if _BIO_KEYWORDS_RE.search(question) else "factual"


def _build_queries(
    question: str,
    option_texts: list[str],
    labeled: list[tuple[str, str]],
    option_entities: list[str],
    subj_str: str,
    main_term: str,
    question_type: str,
) -> list[WeightedQuery]:
    """
    Build semantic search queries based on question type.

    Global queries (from question entities) feed both Wikipedia and DDG.
    Per-option queries (from option_entities) feed DDG only for that option.

    Biography questions — option texts like "Indifferent and distant" have no
    encyclopedic footprint so we use entity + bio keywords for global queries,
    and per-option entity queries only when GLiNER extracted something.

    Factual questions — option texts are film/song titles that DO appear in
    articles; per-option entity queries are the primary signal.
    """
    queries: list[WeightedQuery] = []
    clean_q = _clean_query_text(question)

    # Q1: main entity + question keywords — always useful.
    queries.append(WeightedQuery(
        text=f"{main_term} {clean_q}"[:120],
        weight=1.2,
        strategy="entity_question",
    ))

    # Q2: all extracted entities together — catches multi-entity questions.
    all_entities = " ".join(t for t, _ in labeled[:3])
    if all_entities and all_entities != main_term:
        queries.append(WeightedQuery(
            text=all_entities[:120],
            weight=1.0,
            strategy="all_entities",
        ))

    if question_type == "biography":
        # Q3: entity + relationship keywords extracted from the question itself.
        bio_kws = _extract_bio_keywords(question)
        if bio_kws:
            queries.append(WeightedQuery(
                text=f"{main_term} {bio_kws}"[:120],
                weight=1.4,
                strategy="biography_keywords",
            ))
        # Q4: secondary person entity — gets their Wikipedia page directly.
        secondary = [t for t, _ in labeled if t != main_term]
        if secondary:
            queries.append(WeightedQuery(
                text=f"{secondary[0]} {main_term}"[:120],
                weight=1.3,
                strategy="secondary_entity",
            ))

    # Per-option entity queries: run for all question types.
    # Each option's GLiNER entity (if any) produces a targeted DDG query.
    # Options with no extractable entity (e.g. "Indifferent and distant") are skipped.
    for idx, opt in enumerate(option_texts):
        ent = option_entities[idx] if idx < len(option_entities) else ""
        if ent:
            queries.append(WeightedQuery(
                text=f"{main_term} {ent}"[:120],
                weight=1.1 if question_type == "biography" else 1.0,
                strategy=f"entity_option_{idx}",
            ))
        elif question_type == "factual" and idx < 2:
            # Fallback for factual options when GLiNER found nothing: use cleaned text.
            opt_clean = _clean_query_text(opt)
            if opt_clean:
                queries.append(WeightedQuery(
                    text=f"{main_term} {opt_clean}"[:120],
                    weight=0.8,
                    strategy=f"text_option_{idx}",
                ))

    return queries


def _vote(
    option_texts: list[str],
    global_snips: tuple[str, ...],
    opt_snips: dict[int, tuple[str, ...]],
    subj_str: str,
    question: str,
) -> list[float]:
    votes    = [0.0] * len(option_texts)
    subj_kws = _keywords(subj_str)

    for i, opt_text in enumerate(option_texts):
        # Option-specific snippets first (fetched for this option's query),
        # then global snippets as a fallback pool.
        pool = list(opt_snips.get(i, ())) + list(global_snips)
        if not pool:
            continue
        ranked = _bm25_rank(pool, f"{question} {opt_text}", top_k=3)
        for bm25_score, snippet in ranked:
            tokens    = set(_tokenize(snippet))
            subj_hits = len(subj_kws & tokens)
            if subj_hits == 0:
                continue
            votes[i] += bm25_score * subj_hits * _snippet_polarity(snippet, opt_text)

    return votes


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    window   = text[:max_chars]
    last_end = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if last_end > max_chars // 2:
        return window[:last_end + 1]
    return window.rsplit(" ", 1)[0] + "..."


def _best_snippet_for_option(
    opt_idx: int,
    opt_text: str,
    global_snips: tuple[str, ...],
    opt_snips: dict[int, tuple[str, ...]],
    question: str,
) -> str:
    pool = list(opt_snips.get(opt_idx, ())) + list(global_snips)
    if not pool:
        return ""
    ranked = _bm25_rank(pool, f"{question} {opt_text}", top_k=1)
    return ranked[0][1] if ranked else ""


def _build_llm_prompt(
    question: str,
    option_texts: list[str],
    wiki_text: str,
    global_snips: tuple[str, ...],
    opt_snips: dict[int, tuple[str, ...]],
    votes: list[float],
    is_negative: bool,
    low_confidence: bool = False,
) -> str:
    lines: list[str] = []

    if is_negative:
        lines.append(
            "[NEGATIVE QUESTION: This asks what is NOT true / EXCEPT. "
            "Choose the option with the LEAST supporting evidence.]"
        )

    if low_confidence:
        lines.append(
            "[WEAK EVIDENCE: Retrieved vote scores are near-zero — "
            "rank hints are unreliable. Use your own expertise to answer.]"
        )

    if wiki_text:
        lines.append(f"-- WIKIPEDIA --\n{wiki_text}")

    lines.append("-- EVIDENCE PER OPTION --")

    n           = len(option_texts)
    ranked_opts = sorted(range(n), key=lambda i: votes[i], reverse=not is_negative)

    for i, opt_text in enumerate(option_texts):
        if low_confidence:
            hint = "no reliable evidence"
        else:
            rank_pos = ranked_opts.index(i) + 1
            hint     = "strongest evidence" if rank_pos == 1 else f"rank {rank_pos}"
        lines.append(f"\n[Option {i}] {opt_text}  ({hint}, score={votes[i]:.2f})")
        snip = _best_snippet_for_option(i, opt_text, global_snips, opt_snips, question)
        if snip:
            lines.append(f"  Evidence: {_truncate_at_sentence(snip, 250)}")
        else:
            lines.append("  Evidence: (none retrieved)")

    return "\n".join(lines)


def rag_entertainment(
    query: str,
    num_results: int = 3,
    option_texts: list[str] | None = None,
) -> str:
    labeled = _extract_subjects_gliner(query)
    if labeled:
        subjects  = [text for text, _ in labeled]
        main_term = _pick_main_term(labeled)
        print(f"  [ENT] GLiNER entities: {labeled}")
    else:
        subjects  = _extract_subjects_regex(query)
        main_term = subjects[0] if subjects else ""
        print(f"  [ENT] regex subjects: {subjects}")

    if not main_term:
        kws = [w for w in _tokenize(query) if len(w) >= 4 and w not in _STOP_WORDS]
        main_term = " ".join(kws[:4]) if kws else query[:60]

    subj_str = " ".join(subjects[:2]) if subjects else main_term
    anchors  = _pick_wiki_anchors(labeled) if labeled else [main_term]
    print(f"  [ENT] main_term={main_term!r}  subj_str={subj_str!r}  anchors={anchors}")

    if not option_texts:
        wiki = _fetch_multi_wiki(anchors, query, max_chars=1200)
        ddg  = _ddg_lookup(subj_str, num_results)
        return "\n\n".join(filter(None, [wiki, *ddg]))[:1500]

    n_opts          = min(len(option_texts), 4)
    question_type   = _detect_question_type(query)
    option_entities = [_extract_option_entity(opt) for opt in option_texts[:n_opts]]
    print(f"  [ENT] option_entities={list(zip(option_texts[:n_opts], option_entities))}")
    queries         = _build_queries(
        query, list(option_texts[:n_opts]), labeled, option_entities, subj_str, main_term, question_type
    )
    print(f"  [ENT] question_type={question_type!r}  queries={[(q.strategy, q.text) for q in queries]}")

    def _safe(fut, default):
        try:
            return fut.result(timeout=_TIMEOUT + 1)
        except Exception:
            return default

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries) + 1) as pool:
        wiki_fut    = pool.submit(_fetch_multi_wiki, anchors, query, 600)
        query_futs  = [pool.submit(_ddg_lookup, q.text, 2) for q in queries]
        wiki_text   = _safe(wiki_fut, "")
        query_snips = [_safe(f, ()) for f in query_futs]

    # Partition snippets: option-specific queries → opt_snips[idx],
    # everything else (entity_question, all_entities, bio, secondary) → global_snips.
    seen_keys: set[str] = set()
    _global: list[str] = []
    _opt: dict[int, list[str]] = {}

    for wq, snips in zip(queries, query_snips):
        for s in snips:
            k = s[:120]
            if k in seen_keys:
                continue
            seen_keys.add(k)
            if wq.strategy.startswith(("entity_option_", "text_option_")):
                opt_idx = int(wq.strategy.rsplit("_", 1)[-1])
                _opt.setdefault(opt_idx, []).append(s)
            else:
                _global.append(s)

    global_snips = tuple(_global)
    opt_snips    = {k: tuple(v) for k, v in _opt.items()}

    n_opt_snips = sum(len(v) for v in opt_snips.values())
    print(f"  [ENT] wiki chars={len(wiki_text)}  global_snips={len(global_snips)}  opt_snips={n_opt_snips}")

    has_title_entity = any(label in _TITLE_LABELS for _, label in labeled)
    if not has_title_entity and not wiki_text:
        print("  [ENT] Low confidence -> LLM fallback")
        return ""

    votes          = _vote(list(option_texts[:n_opts]), global_snips, opt_snips, subj_str, query)
    is_negative    = bool(_NOT_EXCEPT_RE.search(query))
    max_vote       = max(votes) if votes else 0.0
    low_confidence = max_vote < _VOTE_CONFIDENCE_MIN
    winner         = (min if is_negative else max)(range(n_opts), key=lambda i: votes[i])
    print(f"  [ENT] votes={votes}  is_negative={is_negative}  winner=[{winner}]  low_confidence={low_confidence}")

    return _build_llm_prompt(
        question=query,
        option_texts=list(option_texts[:n_opts]),
        wiki_text=wiki_text,
        global_snips=global_snips,
        opt_snips=opt_snips,
        votes=votes,
        is_negative=is_negative,
        low_confidence=low_confidence,
    )
